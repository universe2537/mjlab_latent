"""网球潜变量控制任务的动作项。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.tasks.distillation.rl.models import LatentStudentModel
from mjlab.tasks.distillation.rl.obs_slicer import ObservationSlicer


@dataclass(kw_only=True)
class FrozenDecoderLatentJointPositionActionCfg(ActionTermCfg):
  """将高层潜变量动作解码为低层关节位置动作。"""

  actuator_names: tuple[str, ...] | list[str]
  """冻结解码器输出所控制的执行器名称模式。"""
  scale: float | dict[str, float] = 1.0
  """解码后施加的低层关节动作缩放系数。"""
  offset: float | dict[str, float] = 0.0
  """解码后施加的低层关节动作偏置。"""
  use_default_offset: bool = True
  """以机器人默认关节位置作为低层偏置。"""
  latent_dim: int = 16
  """PPO 传入的高层动作维度。"""
  decoder_checkpoint: str = ""
  """包含解码器的在线蒸馏检查点路径。"""
  decoder_state_terms: tuple[str, ...] = ()
  """切片后传递给 ``LatentStudentModel.decode`` 的观测项。"""
  decoder_obs_group: str = "actor"
  """包含 ``decoder_state_terms`` 的观测组名称。"""
  strict_checkpoint_load: bool = True
  """是否严格加载蒸馏模型检查点。"""
  use_latent_action_barrier: bool = False
  """是否将高层动作解释为围绕冻结 prior 的有界 latent residual。"""
  latent_barrier_scale: float = 1.0
  """LAB residual 半径系数：``z = mu + scale * std * tanh(action)``。"""
  latent_barrier_min_std: float = 0.05
  """LAB 使用的 prior 标准差下界。"""
  latent_barrier_max_std: float = 2.0
  """LAB 使用的 prior 标准差上界。"""
  target_dim: int = 67
  """无检查点时使用的后验目标维度（回退值）。"""
  encoder_hidden_dims: tuple[int, ...] = (512, 256)
  prior_hidden_dims: tuple[int, ...] = (512, 256)
  decoder_hidden_dims: tuple[int, ...] = (512, 256)
  activation: str = "elu"
  min_log_std: float = -5.0
  max_log_std: float = 2.0
  posterior_feature_multiplier: int = 3
  prior_feature_multiplier: int = 1
  z_all: bool = False

  def build(self, env) -> FrozenDecoderLatentJointPositionAction:
    return FrozenDecoderLatentJointPositionAction(self, env)


def apply_latent_action_barrier(
  action: torch.Tensor,
  prior_mean: torch.Tensor,
  prior_std: torch.Tensor,
  *,
  scale: float,
  min_std: float,
  max_std: float,
) -> torch.Tensor:
  """Apply a LATENT-style barrier around the frozen decoder prior."""
  std = torch.clamp(prior_std, min=min_std, max=max_std)
  return prior_mean + float(scale) * std * torch.tanh(action)


class FrozenDecoderLatentJointPositionAction(ActionTerm):
  """暴露潜变量动作并应用解码后关节目标的动作项。"""

  cfg: FrozenDecoderLatentJointPositionActionCfg

  def __init__(self, cfg: FrozenDecoderLatentJointPositionActionCfg, env) -> None:
    super().__init__(cfg=cfg, env=env)
    low_level_cfg = JointPositionActionCfg(
      entity_name=cfg.entity_name,
      actuator_names=cfg.actuator_names,
      scale=cfg.scale,
      offset=cfg.offset,
      clip=cfg.clip,
      use_default_offset=cfg.use_default_offset,
    )
    self._low_level_action = JointPositionAction(low_level_cfg, env)
    low_level_dim = self._low_level_action.action_dim
    self._raw_actions = torch.zeros(self.num_envs, cfg.latent_dim, device=self.device)
    self._decoded_actions = torch.zeros(
      self.num_envs, low_level_dim, device=self.device
    )
    self._barrier_latent_actions = torch.zeros_like(self._raw_actions)
    self._latent_prior_mean = torch.zeros_like(self._raw_actions)
    self._latent_prior_std = torch.zeros_like(self._raw_actions)
    self._prev_decoded_actions = torch.zeros_like(self._decoded_actions)
    self._slicer: ObservationSlicer | None = None
    self._model: LatentStudentModel | None = None
    self._loaded_checkpoint: Path | None = None

  @property
  def action_dim(self) -> int:
    return self.cfg.latent_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def latent_action(self) -> torch.Tensor:
    """Return the latent actually passed into the frozen decoder."""
    if self.cfg.use_latent_action_barrier:
      return self._barrier_latent_actions
    return self._raw_actions

  @property
  def latent_prior_mean(self) -> torch.Tensor:
    return self._latent_prior_mean

  @property
  def latent_prior_std(self) -> torch.Tensor:
    return self._latent_prior_std

  @property
  def low_level_action(self) -> torch.Tensor:
    return self._decoded_actions

  @property
  def prev_low_level_action(self) -> torch.Tensor:
    return self._prev_decoded_actions

  @property
  def low_level_action_dim(self) -> int:
    return self._low_level_action.action_dim

  @property
  def state_indices(self) -> torch.Tensor:
    self.ensure_decoder_ready()
    assert self._slicer is not None
    return self._slicer.state_indices

  @property
  def decoder_model(self) -> LatentStudentModel:
    self.ensure_decoder_ready()
    assert self._model is not None
    return self._model

  @property
  def loaded_checkpoint(self) -> Path | None:
    return self._loaded_checkpoint

  @property
  def decoder_checkpoint_path(self) -> Path | None:
    if not self.cfg.decoder_checkpoint:
      return None
    return Path(os.path.expandvars(self.cfg.decoder_checkpoint)).expanduser()

  def process_actions(self, actions: torch.Tensor) -> None:
    self.ensure_decoder_ready()
    self._raw_actions[:] = actions.to(self.device)
    actor_obs = self._cached_actor_obs()
    assert self._slicer is not None
    assert self._model is not None
    state, _ = self._slicer.split(actor_obs)
    with torch.no_grad():
      latent = self._latent_for_decode(state)
      decoded = self._model.decode(state, latent)
    self._prev_decoded_actions[:] = self._decoded_actions
    self._decoded_actions[:] = decoded
    self._low_level_action.process_actions(self._decoded_actions)

  def _latent_for_decode(self, state: torch.Tensor) -> torch.Tensor:
    assert self._model is not None
    if not self.cfg.use_latent_action_barrier:
      return self._raw_actions
    prior = self._model.prior_distribution(state)
    self._latent_prior_mean[:] = prior.mean
    self._latent_prior_std[:] = prior.std
    latent = apply_latent_action_barrier(
      self._raw_actions,
      prior.mean,
      prior.std,
      scale=self.cfg.latent_barrier_scale,
      min_std=self.cfg.latent_barrier_min_std,
      max_std=self.cfg.latent_barrier_max_std,
    )
    self._barrier_latent_actions[:] = latent
    return self._barrier_latent_actions

  def apply_actions(self) -> None:
    self._low_level_action.apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._barrier_latent_actions[env_ids] = 0.0
    self._latent_prior_mean[env_ids] = 0.0
    self._latent_prior_std[env_ids] = 0.0
    self._decoded_actions[env_ids] = 0.0
    self._prev_decoded_actions[env_ids] = 0.0
    self._low_level_action.reset(env_ids)

  def ensure_decoder_ready(self) -> None:
    if self._slicer is None:
      self._slicer = ObservationSlicer(
        cast(Any, self._env),
        group_name=self.cfg.decoder_obs_group,
        state_terms=tuple(self.cfg.decoder_state_terms),
        target_terms=(),
      )
      self._slicer.to(torch.device(self.device))
    if self._model is not None:
      return
    checkpoint = self._load_checkpoint()
    model_cfg = self._model_cfg_from_checkpoint(checkpoint)
    state_dim = self._state_dim_from_checkpoint(checkpoint)
    target_dim = self._target_dim_from_checkpoint(checkpoint)
    if state_dim != self._slicer.state_dim:
      raise ValueError(
        f"Frozen decoder state dim mismatch: checkpoint expects {state_dim}, "
        f"but env provides {self._slicer.state_dim}."
      )
    self._validate_state_terms_from_checkpoint(checkpoint)
    self._model = LatentStudentModel(
      state_dim=state_dim,
      # base_lin_vel, base_ang_vel, joint_pos, joint_vel, actions =3+3+29+29+29=93
      target_dim=target_dim,
      # motion_anchor_pos_b, motion_anchor_ori_b, command = 3+6+58=67（目标维度注释）
      action_dim=self.low_level_action_dim,
      latent_dim=self.cfg.latent_dim,
      encoder_hidden_dims=tuple(model_cfg["encoder_hidden_dims"]),
      prior_hidden_dims=tuple(model_cfg["prior_hidden_dims"]),
      decoder_hidden_dims=tuple(model_cfg["decoder_hidden_dims"]),
      activation=model_cfg["activation"],
      min_log_std=float(model_cfg["min_log_std"]),
      max_log_std=float(model_cfg["max_log_std"]),
      posterior_feature_multiplier=int(model_cfg["posterior_feature_multiplier"]),
      prior_feature_multiplier=int(model_cfg["prior_feature_multiplier"]),
      z_all=bool(model_cfg["z_all"]),
    ).to(self.device)
    if checkpoint is not None:
      self._model.load_state_dict(
        checkpoint["model_state_dict"], strict=self.cfg.strict_checkpoint_load
      )
    self._model.eval()
    for param in self._model.parameters():
      param.requires_grad_(False)

  def _cached_actor_obs(self) -> torch.Tensor:
    obs = self._env.observation_manager.compute(update_history=False)
    actor_obs = obs[self.cfg.decoder_obs_group]
    if not isinstance(actor_obs, torch.Tensor):
      raise ValueError(
        f"Decoder observation group {self.cfg.decoder_obs_group!r} must be "
        "concatenated."
      )
    return actor_obs.to(self.device)

  def _load_checkpoint(self) -> dict[str, Any] | None:
    path = self.decoder_checkpoint_path
    if path is None:
      return None
    if not path.exists():
      raise FileNotFoundError(f"Frozen decoder checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
    latent_dim = int(checkpoint.get("cfg", {}).get("latent_dim", self.cfg.latent_dim))
    if latent_dim != self.cfg.latent_dim:
      raise ValueError(
        f"Frozen decoder latent dim mismatch: checkpoint expects {latent_dim}, "
        f"but env action dim is {self.cfg.latent_dim}."
      )
    self._loaded_checkpoint = path
    return checkpoint

  def _model_cfg_from_checkpoint(self, checkpoint: dict[str, Any] | None) -> dict:
    checkpoint_cfg = checkpoint.get("cfg", {}) if checkpoint is not None else {}
    return {
      "encoder_hidden_dims": checkpoint_cfg.get(
        "encoder_hidden_dims", self.cfg.encoder_hidden_dims
      ),
      "prior_hidden_dims": checkpoint_cfg.get(
        "prior_hidden_dims", self.cfg.prior_hidden_dims
      ),
      "decoder_hidden_dims": checkpoint_cfg.get(
        "decoder_hidden_dims", self.cfg.decoder_hidden_dims
      ),
      "activation": checkpoint_cfg.get("activation", self.cfg.activation),
      "min_log_std": checkpoint_cfg.get("min_log_std", self.cfg.min_log_std),
      "max_log_std": checkpoint_cfg.get("max_log_std", self.cfg.max_log_std),
      "posterior_feature_multiplier": checkpoint_cfg.get(
        "posterior_feature_multiplier", self.cfg.posterior_feature_multiplier
      ),
      "prior_feature_multiplier": checkpoint_cfg.get(
        "prior_feature_multiplier", self.cfg.prior_feature_multiplier
      ),
      "z_all": checkpoint_cfg.get("z_all", self.cfg.z_all),
    }

  def _state_dim_from_checkpoint(self, checkpoint: dict[str, Any] | None) -> int:
    assert self._slicer is not None
    if checkpoint is None:
      return self._slicer.state_dim
    obs_slicer = checkpoint.get("obs_slicer", {})
    state_indices = obs_slicer.get("state_indices")
    if state_indices is None:
      return int(checkpoint.get("cfg", {}).get("state_dim", self._slicer.state_dim))
    return int(state_indices.numel())

  def _target_dim_from_checkpoint(self, checkpoint: dict[str, Any] | None) -> int:
    if checkpoint is None:
      return self.cfg.target_dim
    obs_slicer = checkpoint.get("obs_slicer", {})
    target_indices = obs_slicer.get("target_indices")
    if target_indices is None:
      return int(checkpoint.get("cfg", {}).get("target_dim", self.cfg.target_dim))
    return int(target_indices.numel())

  def _validate_state_terms_from_checkpoint(
    self, checkpoint: dict[str, Any] | None
  ) -> None:
    if checkpoint is None:
      return
    checkpoint_state_terms = checkpoint.get("cfg", {}).get("state_terms")
    if checkpoint_state_terms is None:
      return
    expected_terms = tuple(checkpoint_state_terms)
    actual_terms = tuple(self.cfg.decoder_state_terms)
    if expected_terms != actual_terms:
      raise ValueError(
        "Frozen decoder state term mismatch: checkpoint expects "
        f"{expected_terms}, but tennis is configured with {actual_terms}."
      )
