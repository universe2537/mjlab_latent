"""PACE-style prediction, observations, and rewards for table tennis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.bounce import (
  PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE,
  PINGPONG_POST_BOUNCE_VERTICAL_SCALE,
)
from mjlab.tasks.pingpong.mdp.state import get_pingpong_rally_state
from mjlab.tasks.pingpong.pace_geometry import G1_PACE_GEOMETRY
from mjlab.tasks.pingpong.scene import (
  BALL_CENTER_TABLE_Z,
  NET_TOP_Z,
  NET_X,
  TABLE_HALF_LENGTH,
  TABLE_HALF_WIDTH,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_PADDLE_CFG = SceneEntityCfg("robot", site_names=("pingpong_paddle_center",))
_PADDLE_GEOM_CFG = SceneEntityCfg("robot", geom_names=("pingpong_paddle_collision",))
_BALL_CFG = SceneEntityCfg("ball")
_PACE_STATE_ATTR = "_pingpong_pace_prediction_state"
_DEFAULT_PADDLE_SENSOR = "paddle_ball_contact"
_DEFAULT_NET_SENSOR = "pingpong_ball_net_contact"
_DEFAULT_BODY_BALL_SENSOR = "robot_ball_contact"
_DEFAULT_FOOT_SENSOR = "pace_foot_contact"

_MIN_TIME = 1.0e-4
_MAX_FUTURE_TIME = 1.5
_MAX_LANDING_TIME = 2.5
_EDGE_CLEARANCE = 0.02
PACE_PREDICTION_MODE_INVALID = 0
PACE_PREDICTION_MODE_PRE_BOUNCE = 1
PACE_PREDICTION_MODE_POST_BOUNCE_DIRECT = 2
PACE_TARGET_INVALID_NONE = 0
PACE_TARGET_INVALID_NOT_MOVING_TO_HIT = 1
PACE_TARGET_INVALID_BAD_BOUNCE = 2
PACE_TARGET_INVALID_SECOND_BOUNCE = 3
PACE_TARGET_INVALID_OUT_OF_BOUNDS = 4
PACE_TARGET_INVALID_LOW_OR_TIME = 5
PACE_TARGET_INVALID_NUMERIC = 6
PACE_TARGET_INVALID_RALLY_DONE = 7
_DEFAULT_TARGET_BASE_OFFSET_XY = G1_PACE_GEOMETRY.target_base_offset_xy
_DEFAULT_NATURAL_HIT_X = G1_PACE_GEOMETRY.natural_hit_x
_DEFAULT_TARGET_ROOT_HEIGHT = G1_PACE_GEOMETRY.target_root_height
_DEFAULT_TARGET_BASE_VEL_GAIN = G1_PACE_GEOMETRY.target_base_vel_gain
_DEFAULT_TARGET_BASE_VEL_MAX = G1_PACE_GEOMETRY.target_base_vel_max
_DEFAULT_FOREHAND_PADDLE_OFFSET = G1_PACE_GEOMETRY.forehand_paddle_offset
_DEFAULT_FOREHAND_PADDLE_OFFSET_STD = G1_PACE_GEOMETRY.forehand_paddle_offset_std
_DEFAULT_TARGET_LANDING_X = -0.45 * TABLE_HALF_LENGTH
_DEFAULT_TARGET_LANDING_Y = 0.0
_DEFAULT_HIT_WINDOW_BEFORE_X = 0.35
_DEFAULT_HIT_WINDOW_AFTER_ROOT_X = 0.20
_DEFAULT_HIT_WINDOW_EXTRA_Y = 0.25
_DEFAULT_PRE_BOUNCE_MIN_LOOKAHEAD = 0.06


def _rally_param_subset(params: dict[str, Any]) -> dict[str, Any]:
  keys = (
    "paddle_sensor_name",
    "net_sensor_name",
    "body_ball_sensor_name",
    "ball_cfg",
    "paddle_cfg",
    "paddle_geom_cfg",
    "robot_cfg",
    "force_threshold",
    "table_z",
    "net_x",
    "net_top_z",
    "gravity",
    "self_x_limits",
    "opponent_x_limits",
    "table_y_limits",
    "x_limits",
    "y_limits",
    "z_limits",
    "bounce_z_tolerance",
    "impact_window_distance",
    "impact_followthrough_steps",
    "impact_target_x",
    "impact_target_y",
    "impact_target_z",
  )
  subset = {key: params[key] for key in keys if key in params}
  subset.setdefault("paddle_sensor_name", _DEFAULT_PADDLE_SENSOR)
  subset.setdefault("net_sensor_name", _DEFAULT_NET_SENSOR)
  subset.setdefault("body_ball_sensor_name", _DEFAULT_BODY_BALL_SENSOR)
  subset.setdefault("ball_cfg", _BALL_CFG)
  subset.setdefault("paddle_cfg", _PADDLE_CFG)
  subset.setdefault("paddle_geom_cfg", _PADDLE_GEOM_CFG)
  subset.setdefault("robot_cfg", _ROBOT_CFG)
  subset.setdefault("table_z", BALL_CENTER_TABLE_Z)
  subset.setdefault("net_x", NET_X)
  subset.setdefault("net_top_z", NET_TOP_Z)
  subset.setdefault("gravity", 9.81)
  return subset


def _get_rally_state(env: ManagerBasedRlEnv, params: dict[str, Any]):
  return get_pingpong_rally_state(env, **cast(Any, _rally_param_subset(params)))


def _tuple_tensor(
  value: object,
  *,
  size: int,
  device: torch.device | str,
  dtype: torch.dtype,
) -> torch.Tensor:
  if isinstance(value, torch.Tensor):
    tensor = value.to(device=device, dtype=dtype)
  elif isinstance(value, (tuple, list)):
    tensor = torch.tensor(value, device=device, dtype=dtype)
  else:
    tensor = torch.full((size,), float(cast(Any, value)), device=device, dtype=dtype)
  if tensor.numel() != size:
    raise ValueError(f"Expected {size} values, got {tensor.numel()}.")
  return tensor.reshape(size)


def _ball_table_pos(env: ManagerBasedRlEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
  ball: Entity = env.scene[ball_cfg.name]
  return ball.data.root_link_pos_w - env.scene.env_origins


def _robot_table_pos(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  return robot.data.root_link_pos_w - env.scene.env_origins


def _time_to_table(
  z: torch.Tensor,
  vz: torch.Tensor,
  *,
  table_z: float,
  gravity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  disc = vz * vz + 2.0 * gravity * (z - table_z)
  valid_disc = disc >= 0.0
  sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
  t = (vz + sqrt_disc) / gravity
  valid = valid_disc & (t > _MIN_TIME)
  return torch.where(valid, t, torch.zeros_like(t)), valid


def _ballistic_pose_at(
  pos: torch.Tensor,
  vel: torch.Tensor,
  t: torch.Tensor,
  *,
  gravity: float,
) -> torch.Tensor:
  pose = pos + vel * t.unsqueeze(-1)
  pose[:, 2] = pose[:, 2] - 0.5 * gravity * t * t
  return pose


def _time_to_hit_x(
  x: torch.Tensor,
  vx: torch.Tensor,
  *,
  hit_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  moving_to_hit = vx > 1.0e-6
  safe_vx = torch.where(moving_to_hit, vx, torch.ones_like(vx))
  t = (hit_x - x) / safe_vx
  valid = moving_to_hit & (t > _MIN_TIME) & (t <= _MAX_FUTURE_TIME)
  return torch.where(valid, t, torch.zeros_like(t)), valid


def _self_table_hit_before(
  pos: torch.Tensor,
  vel: torch.Tensor,
  hit_t: torch.Tensor,
  *,
  table_z: float,
  net_x: float,
  gravity: float,
) -> torch.Tensor:
  table_t, table_valid = _time_to_table(
    pos[:, 2],
    vel[:, 2],
    table_z=table_z,
    gravity=gravity,
  )
  table_pos = _ballistic_pose_at(pos, vel, table_t, gravity=gravity)
  table_pos[:, 2] = table_z
  on_self_table = (
    (table_pos[:, 0] >= net_x)
    & (table_pos[:, 0] <= TABLE_HALF_LENGTH)
    & (table_pos[:, 1] >= -TABLE_HALF_WIDTH)
    & (table_pos[:, 1] <= TABLE_HALF_WIDTH)
  )
  return table_valid & on_self_table & (table_t < (hit_t - _MIN_TIME))


def _target_pose_in_hit_window(
  pos: torch.Tensor,
  vel: torch.Tensor,
  *,
  natural_hit_x: torch.Tensor,
  robot_x: torch.Tensor,
  table_z: float,
  net_x: float,
  gravity: float,
  hit_window_before_x: float,
  hit_window_after_root_x: float,
  hit_window_extra_y: float,
  min_lookahead: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  window_min_x = torch.clamp(natural_hit_x - hit_window_before_x, min=net_x)
  window_max_x = robot_x + hit_window_after_root_x
  moving_to_robot = vel[:, 0] > 1.0e-6
  safe_vx = torch.where(moving_to_robot, vel[:, 0], torch.ones_like(vel[:, 0]))
  time_to_natural = (natural_hit_x - pos[:, 0]) / safe_vx
  max_window_t = torch.clamp((window_max_x - pos[:, 0]) / safe_vx, min=0.0)
  natural_still_ahead = time_to_natural > _MIN_TIME
  raw_target_t = torch.where(
    natural_still_ahead,
    time_to_natural,
    torch.full_like(time_to_natural, min_lookahead),
  )
  target_t = torch.minimum(raw_target_t, max_window_t)
  time_valid = (
    moving_to_robot
    & (target_t <= _MAX_FUTURE_TIME)
    & torch.where(natural_still_ahead, target_t > _MIN_TIME, target_t >= 0.0)
  )
  hit_pose = _ballistic_pose_at(pos, vel, target_t, gravity=gravity)
  table_z_t = torch.full_like(pos[:, 0], table_z)
  y_min = -TABLE_HALF_WIDTH - hit_window_extra_y
  y_max = TABLE_HALF_WIDTH + hit_window_extra_y
  in_bounds = (
    (pos[:, 0] <= window_max_x)
    & (hit_pose[:, 0] >= window_min_x)
    & (hit_pose[:, 0] <= window_max_x)
    & (hit_pose[:, 1] >= y_min)
    & (hit_pose[:, 1] <= y_max)
  )
  high_enough = hit_pose[:, 2] >= table_z_t + _EDGE_CLEARANCE
  second_bounce = _self_table_hit_before(
    pos,
    vel,
    target_t,
    table_z=table_z,
    net_x=net_x,
    gravity=gravity,
  )
  finite = torch.isfinite(hit_pose).all(dim=-1) & torch.isfinite(target_t)
  valid = time_valid & in_bounds & high_enough & ~second_bounce & finite

  reason = torch.full(
    (pos.shape[0],),
    PACE_TARGET_INVALID_NONE,
    dtype=torch.long,
    device=pos.device,
  )
  reason = torch.where(
    time_valid,
    reason,
    torch.full_like(reason, PACE_TARGET_INVALID_NOT_MOVING_TO_HIT),
  )
  reason = torch.where(
    time_valid & ~in_bounds,
    torch.full_like(reason, PACE_TARGET_INVALID_OUT_OF_BOUNDS),
    reason,
  )
  reason = torch.where(
    time_valid & in_bounds & ~high_enough & ~second_bounce,
    torch.full_like(reason, PACE_TARGET_INVALID_LOW_OR_TIME),
    reason,
  )
  reason = torch.where(
    time_valid & in_bounds & second_bounce,
    torch.full_like(reason, PACE_TARGET_INVALID_SECOND_BOUNCE),
    reason,
  )
  reason = torch.where(
    time_valid & in_bounds & high_enough & ~second_bounce & ~finite,
    torch.full_like(reason, PACE_TARGET_INVALID_NUMERIC),
    reason,
  )
  return hit_pose, target_t, valid, reason


def _predict_incoming_future_pose(
  pos: torch.Tensor,
  vel: torch.Tensor,
  *,
  has_self_bounce: torch.Tensor,
  natural_hit_x: float,
  robot_x: torch.Tensor,
  table_z: float,
  net_x: float,
  gravity: float,
  hit_window_before_x: float = _DEFAULT_HIT_WINDOW_BEFORE_X,
  hit_window_after_root_x: float = _DEFAULT_HIT_WINDOW_AFTER_ROOT_X,
  hit_window_extra_y: float = _DEFAULT_HIT_WINDOW_EXTRA_Y,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Predict the robot-side future hitting pose in table-local coordinates."""
  hit_x = torch.full_like(pos[:, 0], natural_hit_x)
  net_x_t = torch.full_like(pos[:, 0], net_x)
  y_min = -TABLE_HALF_WIDTH
  y_max = TABLE_HALF_WIDTH

  direct_pose, direct_t, direct_valid, direct_reason = (
    _target_pose_in_hit_window(
      pos,
      vel,
      natural_hit_x=hit_x,
      robot_x=robot_x,
      table_z=table_z,
      net_x=net_x,
      gravity=gravity,
      hit_window_before_x=hit_window_before_x,
      hit_window_after_root_x=hit_window_after_root_x,
      hit_window_extra_y=hit_window_extra_y,
    )
  )

  bounce_t, bounce_valid_t = _time_to_table(
    pos[:, 2], vel[:, 2], table_z=table_z, gravity=gravity
  )
  bounce_pos = _ballistic_pose_at(pos, vel, bounce_t, gravity=gravity)
  bounce_pos[:, 2] = table_z
  impact_vz = vel[:, 2] - gravity * bounce_t
  bounce_base_valid = (
    bounce_valid_t
    & (bounce_t <= _MAX_FUTURE_TIME)
    & (bounce_pos[:, 0] >= net_x_t)
    & (bounce_pos[:, 0] <= TABLE_HALF_LENGTH)
    & (bounce_pos[:, 1] >= y_min)
    & (bounce_pos[:, 1] <= y_max)
    & (impact_vz < 0.0)
  )
  post_vel = torch.stack(
    (
      vel[:, 0] * PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE,
      vel[:, 1] * PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE,
      -impact_vz * PINGPONG_POST_BOUNCE_VERTICAL_SCALE,
    ),
    dim=-1,
  )
  bounce_pose, bounce_hit_t, bounce_hit_valid, bounce_reason = (
    _target_pose_in_hit_window(
      bounce_pos,
      post_vel,
      natural_hit_x=hit_x,
      robot_x=robot_x,
      table_z=table_z,
      net_x=net_x,
      gravity=gravity,
      hit_window_before_x=hit_window_before_x,
      hit_window_after_root_x=hit_window_after_root_x,
      hit_window_extra_y=hit_window_extra_y,
      min_lookahead=_DEFAULT_PRE_BOUNCE_MIN_LOOKAHEAD,
    )
  )
  total_bounce_t = bounce_t + bounce_hit_t
  bounce_valid = bounce_base_valid & bounce_hit_valid & (
    total_bounce_t <= _MAX_FUTURE_TIME
  )
  bounce_reason = torch.where(
    bounce_base_valid,
    bounce_reason,
    torch.full_like(bounce_reason, PACE_TARGET_INVALID_BAD_BOUNCE),
  )
  bounce_reason = torch.where(
    bounce_base_valid & bounce_hit_valid & (total_bounce_t > _MAX_FUTURE_TIME),
    torch.full_like(bounce_reason, PACE_TARGET_INVALID_LOW_OR_TIME),
    bounce_reason,
  )

  future = torch.where(
    has_self_bounce.unsqueeze(-1),
    direct_pose,
    bounce_pose,
  )
  future_t = torch.where(has_self_bounce, direct_t, total_bounce_t)
  valid = torch.where(has_self_bounce, direct_valid, bounce_valid)
  mode = torch.where(
    has_self_bounce & direct_valid,
    torch.full_like(bounce_reason, PACE_PREDICTION_MODE_POST_BOUNCE_DIRECT),
    torch.full_like(bounce_reason, PACE_PREDICTION_MODE_INVALID),
  )
  mode = torch.where(
    (~has_self_bounce) & bounce_valid,
    torch.full_like(mode, PACE_PREDICTION_MODE_PRE_BOUNCE),
    mode,
  )
  reason = torch.where(has_self_bounce, direct_reason, bounce_reason)
  reason = torch.where(
    valid,
    torch.full_like(reason, PACE_TARGET_INVALID_NONE),
    reason,
  )

  fallback = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
  fallback[:, 2] = torch.clamp(fallback[:, 2], min=table_z + _EDGE_CLEARANCE)
  future = torch.where(valid.unsqueeze(-1), future, fallback)
  future_t = torch.where(valid, future_t, torch.zeros_like(future_t))
  future = torch.where(torch.isfinite(future), future, fallback)
  future_t = torch.where(torch.isfinite(future_t), future_t, torch.zeros_like(future_t))
  finite = torch.isfinite(future).all(dim=-1) & torch.isfinite(future_t)
  valid = valid & finite
  reason = torch.where(
    finite,
    reason,
    torch.full_like(reason, PACE_TARGET_INVALID_NUMERIC),
  )
  mode = torch.where(
    valid,
    mode,
    torch.full_like(mode, PACE_PREDICTION_MODE_INVALID),
  )
  return future, future_t, valid, mode, reason


def _predict_landing_xy(
  ball_pos: torch.Tensor,
  ball_vel: torch.Tensor,
  *,
  table_z: float,
  gravity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  t_land, valid = _time_to_table(
    ball_pos[:, 2], ball_vel[:, 2], table_z=table_z, gravity=gravity
  )
  valid = valid & (t_land <= _MAX_LANDING_TIME)
  t_land = torch.where(valid, t_land, torch.zeros_like(t_land))
  return ball_pos[:, :2] + ball_vel[:, :2] * t_land.unsqueeze(-1), valid


def _predict_net_height(
  ball_pos: torch.Tensor,
  ball_vel: torch.Tensor,
  *,
  net_x: float,
  gravity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  vx = ball_vel[:, 0]
  moving_to_opponent = vx < -1.0e-6
  safe_vx = torch.where(moving_to_opponent, -vx, torch.ones_like(vx))
  t_net = (ball_pos[:, 0] - net_x) / safe_vx
  valid = moving_to_opponent & (t_net > _MIN_TIME) & (t_net <= _MAX_FUTURE_TIME)
  z_at_net = ball_pos[:, 2] + ball_vel[:, 2] * t_net - 0.5 * gravity * t_net**2
  return torch.where(valid, z_at_net, torch.zeros_like(z_at_net)), valid


def _pace_posture_gate(
  env: ManagerBasedRlEnv,
  *,
  robot_cfg: SceneEntityCfg,
  target_root_height: float,
  root_height_minimum: float,
  tilt_full_gate_deg: float,
  tilt_zero_gate_deg: float,
  ang_vel_full_gate: float,
  ang_vel_zero_gate: float,
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  root_z = robot.data.root_link_pos_w[:, 2]
  height_gate = (root_z - root_height_minimum) / max(
    target_root_height - root_height_minimum,
    1.0e-6,
  )
  height_gate = torch.clamp(height_gate, min=0.0, max=1.0)

  gravity_w = torch.zeros_like(robot.data.root_link_pos_w)
  gravity_w[:, 2] = -1.0
  gravity_b = quat_apply_inverse(robot.data.root_link_quat_w, gravity_w)
  tilt_mag = torch.linalg.vector_norm(gravity_b[:, :2], dim=-1)
  tilt_full = torch.sin(
    torch.tensor(
      tilt_full_gate_deg * torch.pi / 180.0,
      device=env.device,
      dtype=tilt_mag.dtype,
    )
  )
  tilt_zero = torch.sin(
    torch.tensor(
      tilt_zero_gate_deg * torch.pi / 180.0,
      device=env.device,
      dtype=tilt_mag.dtype,
    )
  )
  tilt_gate = (tilt_zero - tilt_mag) / torch.clamp(
    tilt_zero - tilt_full,
    min=1.0e-6,
  )
  tilt_gate = torch.clamp(tilt_gate, min=0.0, max=1.0)

  ang_vel_xy = torch.linalg.vector_norm(robot.data.root_link_ang_vel_b[:, :2], dim=-1)
  ang_gate = (ang_vel_zero_gate - ang_vel_xy) / max(
    ang_vel_zero_gate - ang_vel_full_gate,
    1.0e-6,
  )
  ang_gate = torch.clamp(ang_gate, min=0.0, max=1.0)
  return torch.nan_to_num(height_gate * tilt_gate * ang_gate)


class PingpongPacePredictionState:
  """Per-environment PACE future target and learned prediction state."""

  def __init__(self, env: ManagerBasedRlEnv, params: dict[str, Any]) -> None:
    self._env = env
    self._params = dict(params)
    self._last_step = -1
    self._has_learned_prediction = False
    num_envs = env.num_envs
    device = env.device

    self.ball_future_pose = torch.zeros(num_envs, 3, device=device)
    self.ball_future_t = torch.zeros(num_envs, device=device)
    self.ball_future_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.prediction_mode = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.invalid_reason = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.posture_gate = torch.ones(num_envs, device=device)
    self.ball_prediction = torch.zeros(num_envs, 3, device=device)
    self.robot_future_pos = torch.zeros(num_envs, 3, device=device)
    self.robot_future_vel = torch.zeros(num_envs, 3, device=device)
    self.target_base_xy = torch.zeros(num_envs, 2, device=device)
    self.predict_landing_xy = torch.zeros(num_envs, 2, device=device)
    self.predict_landing_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.predict_net_height = torch.zeros(num_envs, device=device)
    self.predict_net_valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.reward_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.return_flight_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def update_learned_prediction(self, prediction: torch.Tensor) -> None:
    self.ball_prediction[:] = prediction.to(self._env.device)
    self._has_learned_prediction = True

  @property
  def actor_prediction(self) -> torch.Tensor:
    if self._has_learned_prediction:
      return self.ball_prediction
    return torch.zeros_like(self.ball_future_pose)

  def update(self) -> None:
    step = int(self._env.common_step_counter)
    episode_length_buf = getattr(self._env, "episode_length_buf", None)
    reset_active = (
      isinstance(episode_length_buf, torch.Tensor)
      and bool((episode_length_buf == 0).any().item())
    )
    if step == self._last_step and not reset_active:
      return

    params = self._params
    ball_cfg = params.get("ball_cfg", _BALL_CFG)
    robot_cfg = params.get("robot_cfg", _ROBOT_CFG)
    assert isinstance(ball_cfg, SceneEntityCfg)
    assert isinstance(robot_cfg, SceneEntityCfg)
    table_z = float(params.get("table_z", BALL_CENTER_TABLE_Z))
    net_x = float(params.get("net_x", NET_X))
    gravity = float(params.get("gravity", 9.81))
    natural_hit_x = float(params.get("natural_hit_x", _DEFAULT_NATURAL_HIT_X))

    ball: Entity = self._env.scene[ball_cfg.name]
    ball_pos = _ball_table_pos(self._env, ball_cfg)
    ball_vel = ball.data.root_link_lin_vel_w
    robot_pos = _robot_table_pos(self._env, robot_cfg)

    rally = _get_rally_state(self._env, params)
    rally.update()
    future, future_t, future_valid, mode, invalid_reason = (
      _predict_incoming_future_pose(
        ball_pos,
        ball_vel,
        has_self_bounce=rally.has_self_bounce,
        natural_hit_x=natural_hit_x,
        robot_x=robot_pos[:, 0],
        table_z=table_z,
        net_x=net_x,
        gravity=gravity,
        hit_window_before_x=float(
          params.get("hit_window_before_x", _DEFAULT_HIT_WINDOW_BEFORE_X)
        ),
        hit_window_after_root_x=float(
          params.get("hit_window_after_root_x", _DEFAULT_HIT_WINDOW_AFTER_ROOT_X)
        ),
        hit_window_extra_y=float(
          params.get("hit_window_extra_y", _DEFAULT_HIT_WINDOW_EXTRA_Y)
        ),
      )
    )
    if isinstance(episode_length_buf, torch.Tensor):
      learned_episode_reset = episode_length_buf == 0
    else:
      learned_episode_reset = torch.zeros(
        self._env.num_envs,
        dtype=torch.bool,
        device=self._env.device,
      )
    if learned_episode_reset.any():
      self.ball_prediction[learned_episode_reset] = 0.0

    rally_invalid = rally.has_paddle_hit | rally.fault_edge
    future_valid = future_valid & ~rally_invalid
    invalid_reason = torch.where(
      rally_invalid,
      torch.full_like(invalid_reason, PACE_TARGET_INVALID_RALLY_DONE),
      invalid_reason,
    )
    mode = torch.where(
      future_valid,
      mode,
      torch.full_like(mode, PACE_PREDICTION_MODE_INVALID),
    )
    active = future_valid
    self.ball_future_pose[:] = future
    self.ball_future_t[:] = future_t
    self.ball_future_valid[:] = future_valid
    self.prediction_mode[:] = mode
    self.invalid_reason[:] = invalid_reason
    self.reward_active[:] = active

    offset_xy = _tuple_tensor(
      params.get("target_base_offset_xy", _DEFAULT_TARGET_BASE_OFFSET_XY),
      size=2,
      device=self._env.device,
      dtype=future.dtype,
    )
    target_root_height = float(
      params.get("target_root_height", _DEFAULT_TARGET_ROOT_HEIGHT)
    )
    target_base_vel_gain = float(
      params.get("target_base_vel_gain", _DEFAULT_TARGET_BASE_VEL_GAIN)
    )
    target_base_vel_max = float(
      params.get("target_base_vel_max", _DEFAULT_TARGET_BASE_VEL_MAX)
    )
    posture_gate = _pace_posture_gate(
      self._env,
      robot_cfg=robot_cfg,
      target_root_height=target_root_height,
      root_height_minimum=float(params.get("root_height_minimum", 0.68)),
      tilt_full_gate_deg=float(params.get("posture_tilt_full_gate_deg", 15.0)),
      tilt_zero_gate_deg=float(params.get("posture_tilt_zero_gate_deg", 40.0)),
      ang_vel_full_gate=float(params.get("posture_ang_vel_full_gate", 1.0)),
      ang_vel_zero_gate=float(params.get("posture_ang_vel_zero_gate", 4.0)),
    )

    self.target_base_xy[:, 0] = future[:, 0] + offset_xy[0]
    self.target_base_xy[:, 1] = future[:, 1] + offset_xy[1]
    self.robot_future_pos[:, :2] = self.target_base_xy
    self.robot_future_pos[:, 2] = target_root_height
    target_delta = self.robot_future_pos - robot_pos
    self.robot_future_vel[:] = torch.clamp(
      target_delta * target_base_vel_gain,
      min=-target_base_vel_max,
      max=target_base_vel_max,
    )
    self.robot_future_vel[~active] = 0.0
    self.posture_gate[:] = posture_gate

    landing_xy, landing_valid = _predict_landing_xy(
      ball_pos, ball_vel, table_z=table_z, gravity=gravity
    )
    net_height, net_valid = _predict_net_height(
      ball_pos, ball_vel, net_x=net_x, gravity=gravity
    )
    self.predict_landing_xy[:] = landing_xy
    self.predict_landing_valid[:] = landing_valid
    self.predict_net_height[:] = net_height
    self.predict_net_valid[:] = net_valid
    self.return_flight_active[:] = (
      rally.hit_valid
      & rally.has_paddle_hit
      & ~rally.has_crossed_net
      & (ball_pos[:, 0] > net_x)
    )
    self._last_step = step


def get_pingpong_pace_prediction_state(
  env: ManagerBasedRlEnv,
  **params,
) -> PingpongPacePredictionState:
  state = getattr(env, _PACE_STATE_ATTR, None)
  if isinstance(state, PingpongPacePredictionState):
    if params:
      state._params.update(params)
    return state
  state = PingpongPacePredictionState(env, dict(params))
  setattr(env, _PACE_STATE_ATTR, state)
  return state


def update_pingpong_pace_prediction(
  env: ManagerBasedRlEnv,
  prediction: torch.Tensor,
  **params,
) -> None:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update_learned_prediction(prediction)


def pace_ball_position_table(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  del params
  return _ball_table_pos(env, ball_cfg)


def pace_robot_position_table(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  del params
  return _robot_table_pos(env, robot_cfg)


def pace_ball_prediction_table(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return state.actor_prediction


def pace_relative_target_base_xy(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  robot_cfg = params.get("robot_cfg", _ROBOT_CFG)
  assert isinstance(robot_cfg, SceneEntityCfg)
  robot_pos = _robot_table_pos(env, robot_cfg)
  return state.target_base_xy - robot_pos[:, :2]


def pace_heading(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[robot_cfg.name]
  return robot.data.heading_w.unsqueeze(-1)


def pace_future_ball_pose_table(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return state.ball_future_pose


def pace_paddle_touch_point_table(
  env: ManagerBasedRlEnv,
  paddle_cfg: SceneEntityCfg = _PADDLE_CFG,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[paddle_cfg.name]
  return (
    robot.data.site_pos_w[:, paddle_cfg.site_ids].squeeze(1) - env.scene.env_origins
  )


def pace_robot_future_delta(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  robot_cfg = params.get("robot_cfg", _ROBOT_CFG)
  assert isinstance(robot_cfg, SceneEntityCfg)
  robot_pos = _robot_table_pos(env, robot_cfg)
  return state.robot_future_pos - robot_pos


def pace_future_time(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return state.ball_future_t.unsqueeze(-1)


def pace_rally_flags(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  rally = _get_rally_state(env, params)
  rally.update()
  return torch.stack(
    (
      rally.has_self_bounce.float(),
      rally.has_paddle_hit.float(),
      rally.has_crossed_net.float(),
      rally.has_opponent_bounce.float(),
      rally.fault_edge.float(),
    ),
    dim=-1,
  )


def pace_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[robot_cfg.name]
  return torch.square(robot.data.root_link_lin_vel_w[:, 2])


def pace_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[robot_cfg.name]
  return torch.sum(torch.square(robot.data.root_link_ang_vel_b[:, :2]), dim=-1)


def pace_ang_vel_z_l2(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[robot_cfg.name]
  return torch.square(robot.data.root_link_ang_vel_b[:, 2])


def pace_joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(
      robot.data.joint_pos[:, asset_cfg.joint_ids]
      - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    ),
    dim=-1,
  )


def pace_robot_table_proximity_x(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  min_distance: float = 0.15,
  **params,
) -> torch.Tensor:
  del params
  robot_pos = _robot_table_pos(env, robot_cfg)
  distance = torch.clamp(robot_pos[:, 0] - TABLE_HALF_LENGTH, min=0.0)
  return torch.clamp(min_distance - distance, min=0.0)


def _contact_force_history(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  data = sensor.data
  force_history = getattr(data, "force_history", None)
  if isinstance(force_history, torch.Tensor):
    return torch.nan_to_num(force_history)

  force = getattr(data, "force", None)
  if isinstance(force, torch.Tensor):
    return torch.nan_to_num(force).unsqueeze(2)

  found = getattr(data, "found", None)
  if isinstance(found, torch.Tensor):
    return torch.zeros(
      (*found.shape, 1, 3),
      dtype=found.dtype,
      device=found.device,
    )

  return torch.zeros((env.num_envs, 1, 1, 3), device=env.device)


def _contact_mask_from_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  force_history = _contact_force_history(env, sensor_name)
  force_mag = torch.linalg.vector_norm(force_history, dim=-1)
  contact_mask = torch.amax(force_mag, dim=-1) > force_threshold

  sensor = env.scene[sensor_name]
  found = getattr(sensor.data, "found", None)
  if isinstance(found, torch.Tensor) and tuple(found.shape) == tuple(contact_mask.shape):
    contact_mask = contact_mask | (found > 0)

  return contact_mask


def _contact_primary_names(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  num_contacts: int,
) -> list[str]:
  sensor = env.scene[sensor_name]
  names_attr = getattr(sensor, "primary_names", None)
  names = names_attr() if callable(names_attr) else names_attr
  if isinstance(names, (tuple, list)) and names:
    primary_names = [str(name) for name in names]
    num_slots = int(getattr(getattr(sensor, "cfg", None), "num_slots", 1))
    if len(primary_names) * num_slots == num_contacts:
      return [name for name in primary_names for _ in range(num_slots)]
    if len(primary_names) == num_contacts:
      return primary_names

  if num_contacts == 2:
    return ["left_foot", "right_foot"]
  half = num_contacts // 2
  return ["left_foot"] * half + ["right_foot"] * (num_contacts - half)


def _foot_side_masks(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  num_contacts: int,
  device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
  names = _contact_primary_names(env, sensor_name, num_contacts)
  left = torch.tensor(
    ["left_" in name or name.startswith("left") for name in names],
    device=device,
    dtype=torch.bool,
  )
  right = torch.tensor(
    ["right_" in name or name.startswith("right") for name in names],
    device=device,
    dtype=torch.bool,
  )
  return left, right


def _any_masked(mask: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
  if bool(selector.any().item()):
    return torch.any(mask[:, selector], dim=-1)
  return torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)


def _contact_side_mask_from_sensor(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  contact_mask = _contact_mask_from_sensor(env, sensor_name, force_threshold)
  left, right = _foot_side_masks(
    env, sensor_name, contact_mask.shape[1], contact_mask.device
  )
  return torch.stack(
    (_any_masked(contact_mask, left), _any_masked(contact_mask, right)),
    dim=-1,
  )


def _contact_side_force_history(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
) -> torch.Tensor:
  force_history = _contact_force_history(env, sensor_name)
  left, right = _foot_side_masks(
    env, sensor_name, force_history.shape[1], force_history.device
  )

  def _sum_side(selector: torch.Tensor) -> torch.Tensor:
    if bool(selector.any().item()):
      return torch.sum(force_history[:, selector], dim=1)
    return torch.zeros(
      force_history.shape[0],
      force_history.shape[2],
      force_history.shape[3],
      dtype=force_history.dtype,
      device=force_history.device,
    )

  return torch.stack((_sum_side(left), _sum_side(right)), dim=1)


def _contact_side_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  current_air_time = getattr(sensor.data, "current_air_time", None)
  if current_air_time is None:
    return torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)
  left, right = _foot_side_masks(
    env, sensor_name, current_air_time.shape[1], current_air_time.device
  )

  def _min_side(selector: torch.Tensor) -> torch.Tensor:
    if bool(selector.any().item()):
      return torch.amin(current_air_time[:, selector], dim=1)
    return torch.zeros(
      current_air_time.shape[0],
      dtype=current_air_time.dtype,
      device=current_air_time.device,
    )

  return torch.stack((_min_side(left), _min_side(right)), dim=1)


def pace_fly(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 1.0,
  **params,
) -> torch.Tensor:
  del params
  contact_mask = _contact_side_mask_from_sensor(env, sensor_name, force_threshold)
  return (~torch.any(contact_mask, dim=-1)).float()


def pace_hit_unstable_support(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 0.1,
  **params,
) -> torch.Tensor:
  rally = _get_rally_state(env, params)
  rally.update()
  contact_mask = _contact_side_mask_from_sensor(env, sensor_name, force_threshold)
  required_contacts = 2
  contact_count = torch.sum(contact_mask.float(), dim=-1)
  unstable = contact_count < float(required_contacts)
  return (rally.paddle_hit_edge & unstable).float()


def pace_feet_slide_contact(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 1.0,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[feet_cfg.name]
  body_ids = _body_id_list(feet_cfg, min_count=1)
  feet_vel = robot.data.body_link_lin_vel_w[:, body_ids, :2]
  contact_mask = _contact_side_mask_from_sensor(
    env, sensor_name, force_threshold
  ).float()
  contact_count = min(feet_vel.shape[1], contact_mask.shape[1])
  slide = torch.sum(torch.square(feet_vel[:, :contact_count]), dim=-1)
  return torch.sum(slide * contact_mask[:, :contact_count], dim=-1)


def pace_feet_force(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  threshold: float = 500.0,
  max_reward: float = 400.0,
  **params,
) -> torch.Tensor:
  del params
  force_history = _contact_side_force_history(env, sensor_name)
  force_z = torch.abs(force_history[..., 2])
  total_z_force = torch.linalg.vector_norm(force_z, dim=1)
  peak_force = torch.amax(total_z_force, dim=-1)
  return torch.clamp(peak_force - threshold, min=0.0, max=max_reward)


def pace_feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 1.0,
  **params,
) -> torch.Tensor:
  del params
  force_history = _contact_side_force_history(env, sensor_name)
  horizontal_force = torch.linalg.vector_norm(force_history[..., :2], dim=-1)
  vertical_force = torch.abs(force_history[..., 2])
  stumble = (horizontal_force > 5.0 * vertical_force) & (
    horizontal_force > force_threshold
  )
  return torch.any(stumble, dim=(1, 2)).float()


def pace_feet_slide(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  contact_height: float = 0.08,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[feet_cfg.name]
  feet_pos = (
    robot.data.body_link_pos_w[:, feet_cfg.body_ids] - env.scene.env_origins[:, None, :]
  )
  feet_vel = robot.data.body_link_lin_vel_w[:, feet_cfg.body_ids]
  in_contact_band = feet_pos[..., 2] <= contact_height
  slide = torch.sum(torch.square(feet_vel[..., :2]), dim=-1)
  return torch.sum(slide * in_contact_band.float(), dim=-1)


def _feet_height(env: ManagerBasedRlEnv, feet_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[feet_cfg.name]
  feet_pos = (
    robot.data.body_link_pos_w[:, feet_cfg.body_ids] - env.scene.env_origins[:, None, :]
  )
  return feet_pos[..., 2]


def _body_id_list(body_cfg: SceneEntityCfg, *, min_count: int) -> list[int]:
  body_ids = body_cfg.body_ids
  if isinstance(body_ids, slice) or len(body_ids) < min_count:
    raise ValueError(
      f"{body_cfg.name!r} body_cfg must resolve to at least {min_count} bodies."
    )
  return body_ids


def pace_fly_height(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  contact_height: float = 0.08,
  **params,
) -> torch.Tensor:
  del params
  feet_z = _feet_height(env, feet_cfg)
  return torch.all(feet_z > contact_height, dim=-1).float()


def pace_hit_unstable_support_height(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  contact_height: float = 0.08,
  **params,
) -> torch.Tensor:
  rally = _get_rally_state(env, params)
  rally.update()
  feet_z = _feet_height(env, feet_cfg)
  both_supported = torch.all(feet_z <= contact_height, dim=-1)
  return (rally.paddle_hit_edge & ~both_supported).float()


def pace_body_orientation_l2(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[body_cfg.name]
  body_ids = _body_id_list(body_cfg, min_count=1)
  body_quat = robot.data.body_link_quat_w[:, body_ids[0], :]
  gravity_w = torch.zeros_like(body_quat[:, :3])
  gravity_w[:, 2] = -1.0
  projected_gravity_b = quat_apply_inverse(body_quat, gravity_w)
  return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


def pace_feet_too_near(
  env: ManagerBasedRlEnv,
  feet_cfg: SceneEntityCfg,
  threshold: float = 0.2,
  **params,
) -> torch.Tensor:
  del params
  robot: Entity = env.scene[feet_cfg.name]
  body_ids = _body_id_list(feet_cfg, min_count=2)
  feet_pos = robot.data.body_link_pos_w[:, body_ids]
  distance = torch.linalg.vector_norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
  return torch.clamp(threshold - distance, min=0.0)


def pace_contact(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  rally = _get_rally_state(env, params)
  rally.update()
  return rally.paddle_hit_edge.float()


def pace_table_success(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  rally = _get_rally_state(env, params)
  rally.update()
  return rally.opponent_bounce_edge.float()


def pace_future_ee_target(
  env: ManagerBasedRlEnv,
  std_ee: float = 0.5,
  threshold: float = 0.15,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  paddle_pos = pace_paddle_touch_point_table(env, **params)
  dist = torch.linalg.vector_norm(state.ball_future_pose - paddle_pos, dim=-1)
  reward = torch.exp(-torch.clamp(dist, min=threshold) / (std_ee * std_ee + 1.0e-12))
  active = state.reward_active.float() * state.posture_gate
  return torch.nan_to_num(reward * active)


def pace_future_paddle_height_target(
  env: ManagerBasedRlEnv,
  z_std: float = 0.25,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  paddle_pos = pace_paddle_touch_point_table(env, **params)
  z_error = paddle_pos[:, 2] - state.ball_future_pose[:, 2]
  reward = torch.exp(-torch.square(z_error) / (z_std * z_std + 1.0e-12))
  active = state.reward_active.float() * state.posture_gate
  return torch.nan_to_num(reward * active)


def pace_future_body_target(
  env: ManagerBasedRlEnv,
  std_ro: float = 0.5,
  threshold: float = 0.05,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  robot_cfg = params.get("robot_cfg", _ROBOT_CFG)
  assert isinstance(robot_cfg, SceneEntityCfg)
  robot_pos = _robot_table_pos(env, robot_cfg)
  dist = torch.linalg.vector_norm(state.target_base_xy - robot_pos[:, :2], dim=-1)
  reward = torch.exp(-torch.clamp(dist, min=threshold) / (std_ro * std_ro + 1.0e-12))
  active = state.reward_active.float() * state.posture_gate
  return torch.nan_to_num(reward * active)


def pace_future_base_vel_target(
  env: ManagerBasedRlEnv,
  vel_std: float = 1.2,
  threshold: float = 0.1,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  robot_cfg = params.get("robot_cfg", _ROBOT_CFG)
  assert isinstance(robot_cfg, SceneEntityCfg)
  robot: Entity = env.scene[robot_cfg.name]
  diff = torch.linalg.vector_norm(
    state.robot_future_vel[:, :2] - robot.data.root_link_lin_vel_w[:, :2],
    dim=-1,
  )
  reward = torch.exp(-torch.clamp(diff, min=threshold) / (vel_std * vel_std + 1.0e-12))
  active = state.reward_active.float() * state.posture_gate
  return torch.nan_to_num(reward * active)


def pace_step_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  threshold_min: float = 0.05,
  threshold_max: float = 0.50,
  future_time_threshold: float = 0.18,
  target_speed_threshold: float = 0.50,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  side_air_time = _contact_side_air_time(env, sensor_name)
  in_range = (side_air_time > threshold_min) & (side_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=-1)
  target_speed = torch.linalg.vector_norm(state.robot_future_vel[:, :2], dim=-1)
  active = (
    state.reward_active
    & (state.ball_future_t > future_time_threshold)
    & (target_speed > target_speed_threshold)
  )
  return torch.nan_to_num(reward * active.float())


def pace_future_landing_distance(
  env: ManagerBasedRlEnv,
  threshold: float = 3.0,
  target_x: float = _DEFAULT_TARGET_LANDING_X,
  target_y: float = _DEFAULT_TARGET_LANDING_Y,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  target = torch.tensor((target_x, target_y), device=env.device)
  dist = torch.linalg.vector_norm(state.predict_landing_xy - target, dim=-1)
  reward = threshold - dist
  active = state.return_flight_active & state.predict_landing_valid
  return torch.nan_to_num(torch.where(active, reward, torch.zeros_like(reward)))


def pace_future_pass_net(
  env: ManagerBasedRlEnv,
  std_h: float = 0.4,
  z_target: float = NET_TOP_Z + 0.35,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  height_err = torch.abs(state.predict_net_height - z_target)
  reward = torch.exp(-height_err / (std_h + 1.0e-12))
  active = state.return_flight_active & state.predict_net_valid
  return torch.nan_to_num(reward * active.float())


def _single_body_pos(
  env: ManagerBasedRlEnv,
  body_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[body_cfg.name]
  body_ids = _body_id_list(body_cfg, min_count=1)
  return robot.data.body_link_pos_w[:, body_ids[0]]


def pace_forehand_paddle_offset(
  env: ManagerBasedRlEnv,
  paddle_cfg: SceneEntityCfg = _PADDLE_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  target_offset: tuple[float, float, float] = _DEFAULT_FOREHAND_PADDLE_OFFSET,
  offset_std: tuple[float, float, float] = _DEFAULT_FOREHAND_PADDLE_OFFSET_STD,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  robot: Entity = env.scene[robot_cfg.name]
  target = _tuple_tensor(
    target_offset,
    size=3,
    device=env.device,
    dtype=robot.data.root_link_pos_w.dtype,
  )
  std = _tuple_tensor(
    offset_std,
    size=3,
    device=env.device,
    dtype=robot.data.root_link_pos_w.dtype,
  )
  paddle_pos = pace_paddle_touch_point_table(env, paddle_cfg=paddle_cfg)
  root_to_paddle_w = paddle_pos + env.scene.env_origins - robot.data.root_link_pos_w
  root_to_paddle_b = quat_apply_inverse(robot.data.root_link_quat_w, root_to_paddle_w)
  error = (root_to_paddle_b[:, :2] - target[:2]) / torch.clamp(
    std[:2], min=1.0e-6
  )
  reward = torch.exp(-torch.sum(torch.square(error), dim=-1))
  return torch.nan_to_num(reward * state.reward_active.float())


def pace_forehand_elbow_extension(
  env: ManagerBasedRlEnv,
  shoulder_cfg: SceneEntityCfg,
  elbow_cfg: SceneEntityCfg,
  wrist_cfg: SceneEntityCfg,
  target_ratio: float = G1_PACE_GEOMETRY.forehand_elbow_target_ratio,
  target_span: float = 0.24,
  std: float = 0.08,
  span_std: float = 0.08,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  shoulder = _single_body_pos(env, shoulder_cfg)
  elbow = _single_body_pos(env, elbow_cfg)
  wrist = _single_body_pos(env, wrist_cfg)
  upper = torch.linalg.vector_norm(elbow - shoulder, dim=-1)
  lower = torch.linalg.vector_norm(wrist - elbow, dim=-1)
  span = torch.linalg.vector_norm(wrist - shoulder, dim=-1)
  ratio = span / torch.clamp(upper + lower, min=1.0e-6)
  shortfall = torch.clamp(target_ratio - ratio, min=0.0)
  span_shortfall = torch.clamp(target_span - span, min=0.0)
  reward = torch.exp(
    -torch.square(shortfall) / (std * std + 1.0e-12)
    - torch.square(span_shortfall) / (span_std * span_std + 1.0e-12)
  )
  return torch.nan_to_num(reward * state.reward_active.float())


def pace_target_valid_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return state.ball_future_valid.float()


def _pace_active_value(
  env: ManagerBasedRlEnv,
  value: torch.Tensor,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return torch.nan_to_num(value * state.reward_active.float())


def pace_active_future_z_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return _pace_active_value(env, state.ball_future_pose[:, 2], **params)


def pace_active_paddle_z_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  paddle_pos = pace_paddle_touch_point_table(env, **params)
  return _pace_active_value(env, paddle_pos[:, 2], **params)


def pace_active_ee_dist_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  paddle_pos = pace_paddle_touch_point_table(env, **params)
  dist = torch.linalg.vector_norm(state.ball_future_pose - paddle_pos, dim=-1)
  return _pace_active_value(env, dist, **params)


def pace_active_target_base_speed_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  speed = torch.linalg.vector_norm(state.robot_future_vel[:, :2], dim=-1)
  return _pace_active_value(env, speed, **params)


def pace_active_root_speed_metric(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  **params,
) -> torch.Tensor:
  robot: Entity = env.scene[robot_cfg.name]
  speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
  return _pace_active_value(env, speed, **params)


def pace_prediction_post_bounce_direct_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return (state.prediction_mode == PACE_PREDICTION_MODE_POST_BOUNCE_DIRECT).float()


def pace_posture_gate_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return state.posture_gate


def pace_target_invalid_reason_metric(
  env: ManagerBasedRlEnv,
  reason: int,
  **params,
) -> torch.Tensor:
  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()
  return (~state.ball_future_valid & (state.invalid_reason == reason)).float()


def pace_target_invalid_not_moving_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_NOT_MOVING_TO_HIT,
    **params,
  )


def pace_target_invalid_bad_bounce_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_BAD_BOUNCE,
    **params,
  )


def pace_target_invalid_second_bounce_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_SECOND_BOUNCE,
    **params,
  )


def pace_target_invalid_out_of_bounds_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_OUT_OF_BOUNDS,
    **params,
  )


def pace_target_invalid_low_or_time_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_LOW_OR_TIME,
    **params,
  )


def pace_target_invalid_numeric_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_NUMERIC,
    **params,
  )


def pace_target_invalid_rally_done_metric(
  env: ManagerBasedRlEnv,
  **params,
) -> torch.Tensor:
  return pace_target_invalid_reason_metric(
    env,
    reason=PACE_TARGET_INVALID_RALLY_DONE,
    **params,
  )


__all__ = [
  "PACE_PREDICTION_MODE_INVALID",
  "PACE_PREDICTION_MODE_POST_BOUNCE_DIRECT",
  "PACE_PREDICTION_MODE_PRE_BOUNCE",
  "PACE_TARGET_INVALID_BAD_BOUNCE",
  "PACE_TARGET_INVALID_LOW_OR_TIME",
  "PACE_TARGET_INVALID_NONE",
  "PACE_TARGET_INVALID_NOT_MOVING_TO_HIT",
  "PACE_TARGET_INVALID_NUMERIC",
  "PACE_TARGET_INVALID_OUT_OF_BOUNDS",
  "PACE_TARGET_INVALID_RALLY_DONE",
  "PACE_TARGET_INVALID_SECOND_BOUNCE",
  "PingpongPacePredictionState",
  "get_pingpong_pace_prediction_state",
  "pace_active_ee_dist_metric",
  "pace_active_future_z_metric",
  "pace_active_paddle_z_metric",
  "pace_active_root_speed_metric",
  "pace_active_target_base_speed_metric",
  "pace_ang_vel_xy_l2",
  "pace_ang_vel_z_l2",
  "pace_ball_position_table",
  "pace_ball_prediction_table",
  "pace_body_orientation_l2",
  "pace_contact",
  "pace_feet_force",
  "pace_feet_slide_contact",
  "pace_feet_stumble",
  "pace_feet_too_near",
  "pace_feet_slide",
  "pace_fly",
  "pace_fly_height",
  "pace_future_ball_pose_table",
  "pace_future_base_vel_target",
  "pace_future_body_target",
  "pace_future_ee_target",
  "pace_future_landing_distance",
  "pace_future_pass_net",
  "pace_forehand_elbow_extension",
  "pace_forehand_paddle_offset",
  "pace_future_time",
  "pace_future_paddle_height_target",
  "pace_heading",
  "pace_hit_unstable_support",
  "pace_hit_unstable_support_height",
  "pace_joint_deviation_l1",
  "pace_lin_vel_z_l2",
  "pace_paddle_touch_point_table",
  "pace_rally_flags",
  "pace_relative_target_base_xy",
  "pace_robot_future_delta",
  "pace_robot_position_table",
  "pace_robot_table_proximity_x",
  "pace_step_air_time",
  "pace_table_success",
  "pace_posture_gate_metric",
  "pace_prediction_post_bounce_direct_metric",
  "pace_target_invalid_bad_bounce_metric",
  "pace_target_invalid_low_or_time_metric",
  "pace_target_invalid_not_moving_metric",
  "pace_target_invalid_numeric_metric",
  "pace_target_invalid_out_of_bounds_metric",
  "pace_target_invalid_rally_done_metric",
  "pace_target_invalid_reason_metric",
  "pace_target_invalid_second_bounce_metric",
  "pace_target_valid_metric",
  "update_pingpong_pace_prediction",
]
