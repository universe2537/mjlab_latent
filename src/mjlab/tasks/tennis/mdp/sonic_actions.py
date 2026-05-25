"""SONIC decoder action terms for tennis tasks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.buffers import CircularBuffer


@dataclass(kw_only=True)
class SonicDecoderTokenJointPositionActionCfg(ActionTermCfg):
  """Decode a SONIC token into low-level joint-position actions."""

  actuator_names: tuple[str, ...] | list[str]
  """Actuator name patterns controlled by the decoded joint action."""
  scale: float | dict[str, float] = 1.0
  """Scale applied by the low-level joint-position action."""
  offset: float | dict[str, float] = 0.0
  """Offset applied by the low-level joint-position action."""
  use_default_offset: bool = True
  """Use robot default joint positions as the low-level offset."""
  token_dim: int = 64
  """Dimension of the SONIC token state exposed to the high-level policy."""
  decoder_onnx_path: str = ""
  """Path to SONIC's deployment decoder ONNX file."""
  history_length: int = 10
  """Number of proprioceptive history frames expected by the SONIC decoder."""
  token_clip: tuple[float, float] | None = (-1.0, 1.0)
  """Optional clipping applied to high-level token actions before decoding."""
  providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
  """Preferred ONNX Runtime execution providers."""

  def build(self, env) -> SonicDecoderTokenJointPositionAction:
    return SonicDecoderTokenJointPositionAction(self, env)


class SonicDecoderTokenJointPositionAction(ActionTerm):
  """Expose SONIC token actions and apply the decoded joint targets."""

  cfg: SonicDecoderTokenJointPositionActionCfg

  def __init__(self, cfg: SonicDecoderTokenJointPositionActionCfg, env) -> None:
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

    self._raw_actions = torch.zeros(self.num_envs, cfg.token_dim, device=self.device)
    self._token_actions = torch.zeros_like(self._raw_actions)
    self._decoded_actions = torch.zeros(
      self.num_envs, low_level_dim, device=self.device
    )
    self._prev_decoded_actions = torch.zeros_like(self._decoded_actions)

    self._base_ang_vel_history = CircularBuffer(
      max_len=cfg.history_length, batch_size=self.num_envs, device=self.device
    )
    self._joint_pos_history = CircularBuffer(
      max_len=cfg.history_length, batch_size=self.num_envs, device=self.device
    )
    self._joint_vel_history = CircularBuffer(
      max_len=cfg.history_length, batch_size=self.num_envs, device=self.device
    )
    self._last_action_history = CircularBuffer(
      max_len=cfg.history_length, batch_size=self.num_envs, device=self.device
    )
    self._gravity_history = CircularBuffer(
      max_len=cfg.history_length, batch_size=self.num_envs, device=self.device
    )

    self._session: Any | None = None
    self._session_input_name: str | None = None
    self._session_output_name: str | None = None
    self._loaded_decoder: Path | None = None

  @property
  def action_dim(self) -> int:
    return self.cfg.token_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def token_action(self) -> torch.Tensor:
    return self._token_actions

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
  def decoder_onnx_path(self) -> Path | None:
    if not self.cfg.decoder_onnx_path:
      return None
    return Path(os.path.expandvars(self.cfg.decoder_onnx_path)).expanduser()

  @property
  def loaded_decoder(self) -> Path | None:
    return self._loaded_decoder

  def process_actions(self, actions: torch.Tensor) -> None:
    self.ensure_decoder_ready()
    self._raw_actions[:] = actions.to(self.device)
    if self.cfg.token_clip is None:
      self._token_actions[:] = self._raw_actions
    else:
      low, high = self.cfg.token_clip
      self._token_actions[:] = torch.clamp(self._raw_actions, low, high)

    self._append_robot_history()
    sonic_obs = self._sonic_observation()
    decoded = self._decode(sonic_obs)
    self._prev_decoded_actions[:] = self._decoded_actions
    self._decoded_actions[:] = decoded
    self._low_level_action.process_actions(self._decoded_actions)

  def apply_actions(self) -> None:
    self._low_level_action.apply_actions()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._token_actions[env_ids] = 0.0
    self._decoded_actions[env_ids] = 0.0
    self._prev_decoded_actions[env_ids] = 0.0
    self._low_level_action.reset(env_ids)

    reset_ids = None if isinstance(env_ids, slice) else env_ids
    self._base_ang_vel_history.reset(reset_ids)
    self._joint_pos_history.reset(reset_ids)
    self._joint_vel_history.reset(reset_ids)
    self._last_action_history.reset(reset_ids)
    self._gravity_history.reset(reset_ids)

  def ensure_decoder_ready(self) -> None:
    if self._session is not None:
      return
    path = self.decoder_onnx_path
    if path is None:
      raise FileNotFoundError("SONIC decoder ONNX path is not configured.")
    if not path.exists():
      raise FileNotFoundError(f"SONIC decoder ONNX not found: {path}")

    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    self._patch_dynamic_batch(model)
    onnx.checker.check_model(model)

    available_providers = getattr(
      ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    available = set(available_providers())
    providers = [provider for provider in self.cfg.providers if provider in available]
    if not providers:
      providers = ["CPUExecutionProvider"]
    self._session = ort.InferenceSession(
      model.SerializeToString(),
      providers=providers,
    )
    input_meta = self._session.get_inputs()[0]
    output_meta = self._session.get_outputs()[0]
    self._session_input_name = input_meta.name
    self._session_output_name = output_meta.name
    self._validate_session_shapes(input_meta.shape, output_meta.shape)
    self._loaded_decoder = path

  def _patch_dynamic_batch(self, model: Any) -> None:
    for value_info in (*model.graph.input, *model.graph.output):
      shape = value_info.type.tensor_type.shape
      if not shape.dim:
        continue
      shape.dim[0].ClearField("dim_value")
      shape.dim[0].dim_param = "batch"

  def _validate_session_shapes(
    self, input_shape: list[Any], output_shape: list[Any]
  ) -> None:
    expected_input_dim = self.cfg.token_dim + self.cfg.history_length * (
      6 + 3 * self.low_level_action_dim
    )
    if len(input_shape) != 2 or int(input_shape[1]) != expected_input_dim:
      raise ValueError(
        "SONIC decoder input shape mismatch: expected "
        f"(*, {expected_input_dim}), got {input_shape}."
      )
    if len(output_shape) != 2 or int(output_shape[1]) != self.low_level_action_dim:
      raise ValueError(
        "SONIC decoder output shape mismatch: expected "
        f"(*, {self.low_level_action_dim}), got {output_shape}."
      )

  def _append_robot_history(self) -> None:
    target_ids = self._low_level_action.target_ids
    default_joint_pos = self._entity.data.default_joint_pos
    default_joint_vel = self._entity.data.default_joint_vel
    assert default_joint_pos is not None
    assert default_joint_vel is not None

    joint_pos = (
      self._entity.data.joint_pos[:, target_ids] - default_joint_pos[:, target_ids]
    )
    joint_vel = (
      self._entity.data.joint_vel[:, target_ids] - default_joint_vel[:, target_ids]
    )
    self._base_ang_vel_history.append(self._entity.data.root_link_ang_vel_b)
    self._joint_pos_history.append(joint_pos)
    self._joint_vel_history.append(joint_vel)
    self._last_action_history.append(self._prev_decoded_actions)
    self._gravity_history.append(self._entity.data.projected_gravity_b)

  def _sonic_observation(self) -> torch.Tensor:
    return torch.cat(
      [
        self._token_actions,
        self._base_ang_vel_history.buffer.reshape(self.num_envs, -1),
        self._joint_pos_history.buffer.reshape(self.num_envs, -1),
        self._joint_vel_history.buffer.reshape(self.num_envs, -1),
        self._last_action_history.buffer.reshape(self.num_envs, -1),
        self._gravity_history.buffer.reshape(self.num_envs, -1),
      ],
      dim=-1,
    )

  def _decode(self, sonic_obs: torch.Tensor) -> torch.Tensor:
    assert self._session is not None
    assert self._session_input_name is not None
    assert self._session_output_name is not None

    obs_np = sonic_obs.detach().to("cpu").numpy().astype("float32", copy=False)
    output = self._session.run(
      [self._session_output_name],
      {self._session_input_name: obs_np},
    )[0]
    return torch.as_tensor(output, device=self.device, dtype=torch.float32)
