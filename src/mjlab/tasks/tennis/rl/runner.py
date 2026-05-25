"""Runner helpers for tennis latent-control tasks."""

from __future__ import annotations

import copy
import os
from typing import Any, cast

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionAction
from mjlab.tasks.tennis.rl.config import TennisLatentOnPolicyRunnerCfg  # noqa: F401


class _OnnxTennisLatentModel(nn.Module):
  """Export wrapper that emits both high-level latent and decoded actions."""

  state_indices: torch.Tensor

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

  def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    latent = self.policy(obs)
    state_indices = self.state_indices
    assert isinstance(state_indices, torch.Tensor)
    state = obs.index_select(1, state_indices)
    low_level_action = self.decoder.decode(state, latent)
    return latent, low_level_action


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
          "decoder_checkpoint": str(decoder_action.loaded_checkpoint or ""),
          "decoder_state_terms": list(decoder_action.cfg.decoder_state_terms),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")


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
