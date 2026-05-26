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
from mjlab.utils.lab_api.math import matrix_from_quat


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
  encoder_onnx_path: str = ""
  """Path to SONIC's deployment encoder ONNX file."""
  use_encoder_token_prior: bool = False
  """Interpret high-level token actions as residuals around the encoder token."""
  token_residual_scale: float = 0.2
  """Residual radius used when ``use_encoder_token_prior`` is enabled."""
  encoder_history_stride: int = 5
  """Step spacing used for the 10-frame SONIC encoder motion history."""
  encoder_mode_id: int = 0
  """SONIC encoder mode. ``0`` is the native G1 motion-reference mode."""
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
    self._encoder_token_actions = torch.zeros_like(self._raw_actions)
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
    encoder_history_length = cfg.history_length * cfg.encoder_history_stride
    self._encoder_joint_pos_history = CircularBuffer(
      max_len=encoder_history_length, batch_size=self.num_envs, device=self.device
    )
    self._encoder_joint_vel_history = CircularBuffer(
      max_len=encoder_history_length, batch_size=self.num_envs, device=self.device
    )
    self._encoder_root_z_history = CircularBuffer(
      max_len=encoder_history_length, batch_size=self.num_envs, device=self.device
    )
    self._encoder_anchor_ori_history = CircularBuffer(
      max_len=encoder_history_length, batch_size=self.num_envs, device=self.device
    )

    self._session: Any | None = None
    self._session_input_name: str | None = None
    self._session_output_name: str | None = None
    self._encoder_session: Any | None = None
    self._encoder_input_name: str | None = None
    self._encoder_output_name: str | None = None
    self._loaded_decoder: Path | None = None
    self._loaded_encoder: Path | None = None

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
  def encoder_token_action(self) -> torch.Tensor:
    return self._encoder_token_actions

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
  def encoder_onnx_path(self) -> Path | None:
    if not self.cfg.encoder_onnx_path:
      return None
    return Path(os.path.expandvars(self.cfg.encoder_onnx_path)).expanduser()

  @property
  def loaded_decoder(self) -> Path | None:
    return self._loaded_decoder

  @property
  def loaded_encoder(self) -> Path | None:
    return self._loaded_encoder

  def process_actions(self, actions: torch.Tensor) -> None:
    self.ensure_decoder_ready()
    self._raw_actions[:] = actions.to(self.device)
    self._append_robot_history()
    if self.cfg.token_clip is None:
      token = self._token_for_decode()
    else:
      low, high = self.cfg.token_clip
      token = torch.clamp(self._token_for_decode(), low, high)
    self._token_actions[:] = token

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
    self._encoder_token_actions[env_ids] = 0.0
    self._decoded_actions[env_ids] = 0.0
    self._prev_decoded_actions[env_ids] = 0.0
    self._low_level_action.reset(env_ids)

    reset_ids = None if isinstance(env_ids, slice) else env_ids
    self._base_ang_vel_history.reset(reset_ids)
    self._joint_pos_history.reset(reset_ids)
    self._joint_vel_history.reset(reset_ids)
    self._last_action_history.reset(reset_ids)
    self._gravity_history.reset(reset_ids)
    self._encoder_joint_pos_history.reset(reset_ids)
    self._encoder_joint_vel_history.reset(reset_ids)
    self._encoder_root_z_history.reset(reset_ids)
    self._encoder_anchor_ori_history.reset(reset_ids)

  def ensure_decoder_ready(self) -> None:
    if self._session is not None:
      if self.cfg.use_encoder_token_prior:
        self.ensure_encoder_ready()
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
    if self.cfg.use_encoder_token_prior:
      self.ensure_encoder_ready()

  def ensure_encoder_ready(self) -> None:
    if self._encoder_session is not None:
      return
    path = self.encoder_onnx_path
    if path is None:
      raise FileNotFoundError("SONIC encoder ONNX path is not configured.")
    if not path.exists():
      raise FileNotFoundError(f"SONIC encoder ONNX not found: {path}")

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
    self._encoder_session = ort.InferenceSession(
      model.SerializeToString(),
      providers=providers,
    )
    input_meta = self._encoder_session.get_inputs()[0]
    output_meta = self._encoder_session.get_outputs()[0]
    self._encoder_input_name = input_meta.name
    self._encoder_output_name = output_meta.name
    self._validate_encoder_session_shapes(input_meta.shape, output_meta.shape)
    self._loaded_encoder = path

  def _patch_dynamic_batch(self, model: Any) -> None:
    for value_info in (*model.graph.input, *model.graph.output):
      shape = value_info.type.tensor_type.shape
      if not shape.dim:
        continue
      shape.dim[0].ClearField("dim_value")
      shape.dim[0].dim_param = "batch"
    self._patch_reshape_batch_constants(model)

  def _patch_reshape_batch_constants(self, model: Any) -> None:
    from onnx import numpy_helper

    reshape_shape_inputs = {
      node.input[1]
      for node in model.graph.node
      if node.op_type == "Reshape" and len(node.input) > 1
    }
    for node in model.graph.node:
      if node.op_type != "Constant" or not node.output:
        continue
      if node.output[0] not in reshape_shape_inputs:
        continue
      for attr in node.attribute:
        if attr.name != "value" or not attr.HasField("t"):
          continue
        value = numpy_helper.to_array(attr.t)
        if value.ndim == 1 and value.size > 1 and value[0] == 1:
          patched = value.copy()
          patched[0] = -1
          attr.t.CopyFrom(numpy_helper.from_array(patched, name=attr.t.name))

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

  def _validate_encoder_session_shapes(
    self, input_shape: list[Any], output_shape: list[Any]
  ) -> None:
    expected_input_dim = 1762
    if len(input_shape) != 2 or int(input_shape[1]) != expected_input_dim:
      raise ValueError(
        "SONIC encoder input shape mismatch: expected "
        f"(*, {expected_input_dim}), got {input_shape}."
      )
    if len(output_shape) != 2 or int(output_shape[1]) != self.cfg.token_dim:
      raise ValueError(
        "SONIC encoder output shape mismatch: expected "
        f"(*, {self.cfg.token_dim}), got {output_shape}."
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
    root_z = self._entity.data.root_link_pos_w[:, 2:3]
    anchor_ori = self._root_anchor_orientation_6d()
    self._encoder_joint_pos_history.append(joint_pos)
    self._encoder_joint_vel_history.append(joint_vel)
    self._encoder_root_z_history.append(root_z)
    self._encoder_anchor_ori_history.append(anchor_ori)

  def _root_anchor_orientation_6d(self) -> torch.Tensor:
    mat = matrix_from_quat(self._entity.data.root_link_quat_w)
    return mat[..., :2].reshape(self.num_envs, -1)

  def _token_for_decode(self) -> torch.Tensor:
    if not self.cfg.use_encoder_token_prior:
      return self._raw_actions
    encoder_obs = self._sonic_encoder_observation()
    encoder_token = self._encode(encoder_obs)
    self._encoder_token_actions[:] = encoder_token
    residual = torch.tanh(self._raw_actions) * float(self.cfg.token_residual_scale)
    return encoder_token + residual

  def _sonic_encoder_observation(self) -> torch.Tensor:
    obs = torch.zeros(self.num_envs, 1762, device=self.device)
    mode = int(self.cfg.encoder_mode_id)
    if mode < 0 or mode >= 4:
      raise ValueError(f"SONIC encoder mode must be in [0, 3], got {mode}.")
    obs[:, mode] = 1.0

    joint_pos = self._sample_encoder_history(self._encoder_joint_pos_history)
    joint_vel = self._sample_encoder_history(self._encoder_joint_vel_history)
    anchor_ori_history = self._sample_encoder_history(self._encoder_anchor_ori_history)

    obs[:, 4:294] = joint_pos.reshape(self.num_envs, -1)
    obs[:, 294:584] = joint_vel.reshape(self.num_envs, -1)
    obs[:, 601:661] = anchor_ori_history.reshape(self.num_envs, -1)
    return obs

  def _sample_encoder_history(self, history: CircularBuffer) -> torch.Tensor:
    buffer = history.buffer
    stride = self.cfg.encoder_history_stride
    return buffer[:, ::stride]

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

  def _encode(self, encoder_obs: torch.Tensor) -> torch.Tensor:
    assert self._encoder_session is not None
    assert self._encoder_input_name is not None
    assert self._encoder_output_name is not None

    obs_np = encoder_obs.detach().to("cpu").numpy().astype("float32", copy=False)
    output = self._encoder_session.run(
      [self._encoder_output_name],
      {self._encoder_input_name: obs_np},
    )[0]
    return torch.as_tensor(output, device=self.device, dtype=torch.float32)
