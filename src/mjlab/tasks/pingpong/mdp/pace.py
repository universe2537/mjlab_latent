"""PACE-style prediction, observations, and rewards for table tennis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.state import get_pingpong_rally_state
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
_POST_BOUNCE_HORIZONTAL_SCALE = 0.94
_POST_BOUNCE_VERTICAL_SCALE = 0.90
_EDGE_CLEARANCE = 0.02
_TARGET_BASE_X_OFFSET = 0.10
_TARGET_BASE_Y_OFFSET = 0.60
_TARGET_BODY_HEIGHT = 0.69
_TARGET_BASE_VEL_GAIN = 4.0
_TARGET_BASE_VEL_MAX = 7.0
_DEFAULT_TARGET_LANDING_X = -0.45 * TABLE_HALF_LENGTH
_DEFAULT_TARGET_LANDING_Y = 0.0


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


def _predict_incoming_future_pose(
  pos: torch.Tensor,
  vel: torch.Tensor,
  *,
  table_z: float,
  net_x: float,
  gravity: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Predict the robot-side future hitting pose in table-local coordinates."""
  edge_x = torch.full_like(pos[:, 0], TABLE_HALF_LENGTH)
  net_x_t = torch.full_like(pos[:, 0], net_x)
  table_z_t = torch.full_like(pos[:, 0], table_z)
  y_min = -TABLE_HALF_WIDTH
  y_max = TABLE_HALF_WIDTH

  vx_safe = torch.clamp(vel[:, 0], min=1.0e-6)
  direct_t = (edge_x - pos[:, 0]) / vx_safe
  direct_y = pos[:, 1] + vel[:, 1] * direct_t
  direct_z = pos[:, 2] + vel[:, 2] * direct_t - 0.5 * gravity * direct_t**2
  direct_valid = (
    (pos[:, 0] >= net_x_t)
    & (vel[:, 0] > 0.0)
    & (vel[:, 2] >= -0.05)
    & (direct_t > _MIN_TIME)
    & (direct_t <= _MAX_FUTURE_TIME)
    & (direct_z >= table_z_t + _EDGE_CLEARANCE)
    & (direct_y >= y_min)
    & (direct_y <= y_max)
  )

  bounce_t, bounce_valid_t = _time_to_table(
    pos[:, 2], vel[:, 2], table_z=table_z, gravity=gravity
  )
  bounce_pos = pos + vel * bounce_t.unsqueeze(-1)
  bounce_pos[:, 2] = table_z
  impact_vz = vel[:, 2] - gravity * bounce_t
  post_vx = vel[:, 0] * _POST_BOUNCE_HORIZONTAL_SCALE
  post_vy = vel[:, 1] * _POST_BOUNCE_HORIZONTAL_SCALE
  post_vz = -impact_vz * _POST_BOUNCE_VERTICAL_SCALE
  edge_t_after_bounce = (edge_x - bounce_pos[:, 0]) / post_vx.clamp_min(1.0e-6)
  edge_y = bounce_pos[:, 1] + post_vy * edge_t_after_bounce
  edge_z = (
    table_z_t + post_vz * edge_t_after_bounce - 0.5 * gravity * edge_t_after_bounce**2
  )
  total_bounce_t = bounce_t + edge_t_after_bounce
  bounce_valid = (
    bounce_valid_t
    & (bounce_t <= _MAX_FUTURE_TIME)
    & (bounce_pos[:, 0] >= net_x_t)
    & (bounce_pos[:, 0] <= edge_x)
    & (bounce_pos[:, 1] >= y_min)
    & (bounce_pos[:, 1] <= y_max)
    & (impact_vz < 0.0)
    & (post_vx > 0.0)
    & (edge_t_after_bounce > _MIN_TIME)
    & (total_bounce_t <= _MAX_FUTURE_TIME)
    & (edge_z >= table_z_t + _EDGE_CLEARANCE)
    & (edge_y >= y_min)
    & (edge_y <= y_max)
  )

  future = torch.empty_like(pos)
  future[:, 0] = edge_x
  future[:, 1] = torch.where(direct_valid, direct_y, edge_y)
  future[:, 2] = torch.where(direct_valid, direct_z, edge_z)
  future_t = torch.where(direct_valid, direct_t, total_bounce_t)
  valid = direct_valid | bounce_valid
  fallback = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
  fallback[:, 2] = torch.clamp(fallback[:, 2], min=table_z + _EDGE_CLEARANCE)
  future = torch.where(valid.unsqueeze(-1), future, fallback)
  future_t = torch.where(valid, future_t, torch.zeros_like(future_t))
  future = torch.where(torch.isfinite(future), future, fallback)
  future_t = torch.where(torch.isfinite(future_t), future_t, torch.zeros_like(future_t))
  valid = valid & torch.isfinite(future).all(dim=-1) & torch.isfinite(future_t)
  return future, future_t, valid


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

    ball: Entity = self._env.scene[ball_cfg.name]
    ball_pos = _ball_table_pos(self._env, ball_cfg)
    ball_vel = ball.data.root_link_lin_vel_w
    robot_pos = _robot_table_pos(self._env, robot_cfg)

    future, future_t, future_valid = _predict_incoming_future_pose(
      ball_pos, ball_vel, table_z=table_z, net_x=net_x, gravity=gravity
    )
    if isinstance(episode_length_buf, torch.Tensor):
      learned_episode_reset = episode_length_buf == 0
    else:
      learned_episode_reset = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)
    if learned_episode_reset.any():
      self.ball_prediction[learned_episode_reset] = 0.0

    rally = _get_rally_state(self._env, params)
    rally.update()
    active = future_valid & ~rally.has_paddle_hit & ~rally.fault_edge
    self.ball_future_pose[:] = future
    self.ball_future_t[:] = future_t
    self.ball_future_valid[:] = future_valid
    self.reward_active[:] = active

    self.target_base_xy[:, 0] = future[:, 0] + _TARGET_BASE_X_OFFSET
    self.target_base_xy[:, 1] = future[:, 1] + _TARGET_BASE_Y_OFFSET
    self.robot_future_pos[:, :2] = self.target_base_xy
    self.robot_future_pos[:, 2] = _TARGET_BODY_HEIGHT
    target_delta = self.robot_future_pos - robot_pos
    self.robot_future_vel[:] = torch.clamp(
      target_delta * _TARGET_BASE_VEL_GAIN,
      min=-_TARGET_BASE_VEL_MAX,
      max=_TARGET_BASE_VEL_MAX,
    )
    self.robot_future_vel[~active] = 0.0

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


def pace_fly(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 1.0,
  **params,
) -> torch.Tensor:
  del params
  contact_mask = _contact_mask_from_sensor(env, sensor_name, force_threshold)
  return (~torch.any(contact_mask, dim=-1)).float()


def pace_hit_unstable_support(
  env: ManagerBasedRlEnv,
  sensor_name: str = _DEFAULT_FOOT_SENSOR,
  force_threshold: float = 0.1,
  **params,
) -> torch.Tensor:
  rally = _get_rally_state(env, params)
  rally.update()
  contact_mask = _contact_mask_from_sensor(env, sensor_name, force_threshold)
  required_contacts = min(2, contact_mask.shape[-1])
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
  contact_mask = _contact_mask_from_sensor(env, sensor_name, force_threshold).float()
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
  force_history = _contact_force_history(env, sensor_name)
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
  force_history = _contact_force_history(env, sensor_name)
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
  return torch.nan_to_num(reward * state.reward_active.float())


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
  return torch.nan_to_num(reward * state.reward_active.float())


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
  return torch.nan_to_num(reward * state.reward_active.float())


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


__all__ = [
  "PingpongPacePredictionState",
  "get_pingpong_pace_prediction_state",
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
  "pace_future_time",
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
  "pace_table_success",
  "update_pingpong_pace_prediction",
]
