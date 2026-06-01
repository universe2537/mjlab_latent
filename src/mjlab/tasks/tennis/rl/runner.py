"""Runner helpers for tennis latent-control tasks."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tennis.mdp import (
  FrozenDecoderLatentJointPositionAction,
  apply_latent_action_barrier,
)
from mjlab.tasks.tennis.rl.config import TennisLatentOnPolicyRunnerCfg  # noqa: F401


def expand_actor_action_head_for_wrist_residual(
  actor_state_dict: dict[str, torch.Tensor],
  *,
  latent_dim: int,
  target_dim: int,
  wrist_init_std: float,
) -> bool:
  """Expand a latent-only actor head to include wrist residual outputs."""
  current_dim = _actor_state_action_dim(actor_state_dict)
  if current_dim == target_dim:
    return False
  if current_dim != latent_dim:
    raise ValueError(
      f"Cannot migrate actor action dim {current_dim} to {target_dim}; "
      f"expected old latent dim {latent_dim}."
    )

  weight_key, bias_key = _actor_output_layer_keys(actor_state_dict, current_dim)
  old_weight = actor_state_dict[weight_key]
  old_bias = actor_state_dict[bias_key]
  new_weight = old_weight.new_zeros((target_dim, old_weight.shape[1]))
  new_bias = old_bias.new_zeros(target_dim)
  new_weight[:current_dim] = old_weight
  new_bias[:current_dim] = old_bias
  actor_state_dict[weight_key] = new_weight
  actor_state_dict[bias_key] = new_bias

  for std_key in ("distribution.std_param", "distribution.log_std_param"):
    if std_key not in actor_state_dict:
      continue
    old_std = actor_state_dict[std_key]
    if old_std.numel() != current_dim:
      raise ValueError(
        f"Cannot migrate {std_key} with shape {tuple(old_std.shape)}; "
        f"expected {current_dim} entries."
      )
    new_std = old_std.new_empty(target_dim)
    new_std[:current_dim] = old_std.reshape(-1)
    fill_value = (
      math.log(wrist_init_std) if std_key.endswith("log_std_param") else wrist_init_std
    )
    new_std[current_dim:] = fill_value
    actor_state_dict[std_key] = new_std
  return True


def expand_mlp_input_for_observation(
  model_state_dict: dict[str, torch.Tensor],
  *,
  target_dim: int,
) -> bool:
  """Resize an MLP input layer for tail-appended or tail-truncated observations."""
  weight_key = _mlp_input_layer_key(model_state_dict)
  old_weight = model_state_dict[weight_key]
  current_dim = int(old_weight.shape[1])
  if current_dim == target_dim:
    return False
  if current_dim > target_dim:
    removed_dim = current_dim - target_dim
    if removed_dim > 2:
      raise ValueError(
        f"Cannot safely migrate observation dim {current_dim} down to "
        f"{target_dim}; only small tail truncations are supported."
      )
    new_weight = old_weight[:, :target_dim].clone()
  else:
    new_weight = old_weight.new_zeros((old_weight.shape[0], target_dim))
    new_weight[:, :current_dim] = old_weight
  model_state_dict[weight_key] = new_weight

  for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
    tensor = model_state_dict.get(key)
    if not isinstance(tensor, torch.Tensor):
      continue
    if tensor.shape[-1] != current_dim:
      continue
    if current_dim > target_dim:
      new_tensor = tensor[..., :target_dim].clone()
    else:
      fill_value = 0.0 if key.endswith("_mean") else 1.0
      new_tensor = tensor.new_full((*tensor.shape[:-1], target_dim), fill_value)
      new_tensor[..., :current_dim] = tensor
    model_state_dict[key] = new_tensor
  return True


def _mlp_input_layer_key(model_state_dict: dict[str, torch.Tensor]) -> str:
  for key in ("mlp.0.weight", "actor.0.weight"):
    tensor = model_state_dict.get(key)
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
      return key
  candidates = [
    key
    for key, tensor in model_state_dict.items()
    if key.endswith(".weight") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2
  ]
  if not candidates:
    raise ValueError("Cannot find MLP input layer in checkpoint.")
  return candidates[0]


def _model_state_obs_dim(model_state_dict: dict[str, torch.Tensor]) -> int:
  weight_key = _mlp_input_layer_key(model_state_dict)
  return int(model_state_dict[weight_key].shape[1])


def _actor_state_action_dim(actor_state_dict: dict[str, torch.Tensor]) -> int:
  for key in (
    "distribution.std_param",
    "distribution.log_std_param",
    "std",
    "log_std",
  ):
    tensor = actor_state_dict.get(key)
    if isinstance(tensor, torch.Tensor):
      return int(tensor.numel())
  bias_keys = [
    key
    for key, tensor in actor_state_dict.items()
    if key.endswith(".bias") and isinstance(tensor, torch.Tensor) and tensor.ndim == 1
  ]
  if not bias_keys:
    raise ValueError("Cannot infer actor action dimension from checkpoint.")
  return int(min(actor_state_dict[key].numel() for key in bias_keys))


def _actor_output_layer_keys(
  actor_state_dict: dict[str, torch.Tensor], output_dim: int
) -> tuple[str, str]:
  candidates: list[tuple[str, str]] = []
  for key, tensor in actor_state_dict.items():
    if not key.endswith(".weight") or not isinstance(tensor, torch.Tensor):
      continue
    if tensor.ndim != 2 or tensor.shape[0] != output_dim:
      continue
    bias_key = f"{key[: -len('.weight')]}.bias"
    bias = actor_state_dict.get(bias_key)
    if isinstance(bias, torch.Tensor) and bias.shape == (output_dim,):
      candidates.append((key, bias_key))
  if not candidates:
    raise ValueError(f"Cannot find actor output layer with output dim {output_dim}.")
  return candidates[-1]


class _OnnxTennisLatentModel(nn.Module):
  """Export wrapper that emits both high-level latent and decoded actions."""

  state_indices: torch.Tensor
  wrist_residual_joint_ids: torch.Tensor
  wrist_residual_scale: torch.Tensor
  wrist_residual_decoder_scale: torch.Tensor

  def __init__(
    self,
    actor,
    decoder_action: FrozenDecoderLatentJointPositionAction,
  ) -> None:
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.decoder = copy.deepcopy(decoder_action.decoder_model).to("cpu")
    self.decoder.eval()
    self.register_buffer("state_indices", decoder_action.state_indices.to("cpu"))
    self.latent_dim = decoder_action.cfg.latent_dim
    self.use_latent_action_barrier = decoder_action.cfg.use_latent_action_barrier
    self.latent_barrier_scale = decoder_action.cfg.latent_barrier_scale
    self.latent_barrier_min_std = decoder_action.cfg.latent_barrier_min_std
    self.latent_barrier_max_std = decoder_action.cfg.latent_barrier_max_std
    self.register_buffer(
      "wrist_residual_joint_ids",
      decoder_action.wrist_residual_joint_ids.to("cpu"),
    )
    self.register_buffer(
      "wrist_residual_scale",
      decoder_action.wrist_residual_scale.to("cpu"),
    )
    self.register_buffer(
      "wrist_residual_decoder_scale",
      decoder_action.wrist_residual_decoder_scale.to("cpu"),
    )

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    high_level_action = self.policy(obs)
    raw_latent = high_level_action[:, : self.latent_dim]
    state_indices = self.state_indices
    assert isinstance(state_indices, torch.Tensor)
    state = obs.index_select(1, state_indices)
    if self.use_latent_action_barrier:
      prior = self.decoder.prior_distribution(state)
      latent = apply_latent_action_barrier(
        raw_latent,
        prior.mean,
        prior.std,
        scale=self.latent_barrier_scale,
        min_std=self.latent_barrier_min_std,
        max_std=self.latent_barrier_max_std,
      )
    else:
      latent = raw_latent
    low_level_action = self.decoder.decode(state, latent)
    wrist_ids = self.wrist_residual_joint_ids
    if wrist_ids.numel() > 0:
      wrist_raw = high_level_action[:, self.latent_dim :]
      wrist_residual = torch.tanh(wrist_raw) * self.wrist_residual_scale
      wrist_decoder_delta = wrist_residual / self.wrist_residual_decoder_scale
      low_level_action = low_level_action.clone()
      low_level_action[:, wrist_ids] += wrist_decoder_delta
    return high_level_action, low_level_action


class TennisLatentOnPolicyRunner(MjlabOnPolicyRunner):
  """RSL-RL runner for tennis high-level latent policies."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.require_decoder_checkpoint = bool(
      train_cfg.pop("require_decoder_checkpoint", True)
    )
    self.reset_resume_progress = bool(train_cfg.pop("reset_resume_progress", False))
    self.decoder_action_name = "latent_joint_pos"
    self._validate_decoder_checkpoint(env)
    super().__init__(env, train_cfg, log_dir, device)

  def _decoder_action(self) -> FrozenDecoderLatentJointPositionAction:
    action = self.env.unwrapped.action_manager.get_term(self.decoder_action_name)
    if not isinstance(action, FrozenDecoderLatentJointPositionAction):
      raise TypeError(
        f"Action term {self.decoder_action_name!r} is not a frozen decoder action."
      )
    return action

  def _validate_decoder_checkpoint(self, env: VecEnv) -> None:
    if not self.require_decoder_checkpoint:
      return
    wrapped_env = cast(RslRlVecEnvWrapper, env)
    action = wrapped_env.unwrapped.action_manager.get_term(self.decoder_action_name)
    if not isinstance(action, FrozenDecoderLatentJointPositionAction):
      raise TypeError(
        f"Action term {self.decoder_action_name!r} is not a frozen decoder action."
      )
    path = action.decoder_checkpoint_path
    if path is None:
      raise FileNotFoundError(
        "Tennis latent training requires a frozen decoder checkpoint. Set "
        "`--env.actions.latent-joint-pos.decoder-checkpoint /path/to/model.pt`."
      )
    if not path.exists():
      raise FileNotFoundError(f"Frozen decoder checkpoint not found: {path}")

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the high-level policy together with the frozen decoder."""
    os.makedirs(path, exist_ok=True)
    model = _OnnxTennisLatentModel(self.alg.get_policy(), self._decoder_action())
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    torch.onnx.export(
      model,
      (obs,),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs"],
      output_names=["latent_actions", "joint_actions"],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    """Save checkpoint and export ONNX with decoder metadata."""
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )  # type: ignore[assignment]
      decoder_action = self._decoder_action()
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      metadata.update(
        {
          "latent_dim": decoder_action.action_dim,
          "decoder_latent_dim": decoder_action.cfg.latent_dim,
          "wrist_residual_dim": decoder_action.wrist_residual_dim,
          "wrist_residual_joint_names": list(
            decoder_action.cfg.wrist_residual_joint_names
          ),
          "decoder_checkpoint": str(decoder_action.loaded_checkpoint or ""),
          "decoder_state_terms": list(decoder_action.cfg.decoder_state_terms),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load checkpoints, skipping optimizer state when action dim is migrated."""
    if load_cfg is None and self._checkpoint_needs_model_migration(path, map_location):
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": True,
        "rnd": True,
      }
      print(
        "[INFO] Migrating tennis checkpoint model shapes; "
        "optimizer state will be reinitialized."
      )
    if self.reset_resume_progress:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
        **(load_cfg or {}),
      }
      load_cfg["optimizer"] = False
      load_cfg["iteration"] = False
      load_cfg["rnd"] = False
      print(
        "[INFO] Warm-starting tennis checkpoint weights; "
        "iteration, optimizer, RND, and env progress will reset."
      )
    return super().load(path, load_cfg, strict, map_location)

  def _preprocess_loaded_dict(self, loaded_dict: dict) -> dict:
    loaded_dict = super()._preprocess_loaded_dict(loaded_dict)
    action = self._decoder_action()
    actor_sd = loaded_dict.get("actor_state_dict")
    if isinstance(actor_sd, dict):
      expand_mlp_input_for_observation(
        actor_sd,
        target_dim=self._target_observation_dim("actor"),
      )
      if action.wrist_residual_dim > 0:
        expand_actor_action_head_for_wrist_residual(
          actor_sd,
          latent_dim=action.cfg.latent_dim,
          target_dim=action.action_dim,
          wrist_init_std=action.cfg.wrist_residual_migration_std,
        )
      loaded_dict["actor_state_dict"] = actor_sd
    critic_sd = loaded_dict.get("critic_state_dict")
    if isinstance(critic_sd, dict):
      expand_mlp_input_for_observation(
        critic_sd,
        target_dim=self._target_observation_dim("critic"),
      )
      loaded_dict["critic_state_dict"] = critic_sd
    return loaded_dict

  def _checkpoint_needs_model_migration(
    self, path: str, map_location: str | None
  ) -> bool:
    action = self._decoder_action()
    checkpoint = torch.load(
      path,
      map_location=map_location or self.device,
      weights_only=False,
    )
    actor_sd = checkpoint.get("actor_state_dict", {})
    if "model_state_dict" in checkpoint:
      actor_sd = checkpoint["model_state_dict"]
    if not isinstance(actor_sd, dict):
      return False
    needs_action_migration = _actor_state_action_dim(actor_sd) != action.action_dim
    needs_actor_obs_migration = _model_state_obs_dim(
      actor_sd
    ) != self._target_observation_dim("actor")
    critic_sd = checkpoint.get("critic_state_dict", {})
    needs_critic_obs_migration = False
    if isinstance(critic_sd, dict) and critic_sd:
      needs_critic_obs_migration = _model_state_obs_dim(
        critic_sd
      ) != self._target_observation_dim("critic")
    return (
      needs_action_migration or needs_actor_obs_migration or needs_critic_obs_migration
    )

  def _target_observation_dim(self, group_name: str) -> int:
    dim = self.env.unwrapped.observation_manager.group_obs_dim[group_name]
    if not isinstance(dim, tuple) or len(dim) != 1:
      raise ValueError(f"Expected flat {group_name!r} observation dim, got {dim}.")
    return int(dim[0])


class TennisTokenOnPolicyRunner(MjlabOnPolicyRunner):
  """RSL-RL runner for token policies without frozen PyTorch decoder export."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    train_cfg.pop("require_decoder_checkpoint", None)
    super().__init__(env, train_cfg, log_dir, device)
