"""Table-tennis rally state tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.pingpong.scene import (
  BALL_CENTER_TABLE_Z,
  NET_TOP_Z,
  NET_X,
  TABLE_HALF_LENGTH,
  TABLE_HALF_WIDTH,
)
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PHASE_INCOMING = 0
PHASE_AFTER_SELF_BOUNCE = 1
PHASE_RETURN_FLIGHT = 2
PHASE_DONE = 3

FAULT_NONE = 0
FAULT_ILLEGAL_PRE_BOUNCE_HIT = 1
FAULT_EXTRA_PADDLE_CONTACT = 2
FAULT_RETURN_BOUNCE_OUT = 3
FAULT_RETURN_OUT_OF_PLAY = 4
FAULT_NET_CONTACT = 5
FAULT_LOW_NET_CROSS = 6
FAULT_INCOMING_MISS = 7
FAULT_ILLEGAL_BODY_BALL_CONTACT = 8

_DEFAULT_BALL_CFG = SceneEntityCfg("ball")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_PADDLE_CFG = SceneEntityCfg("robot", site_names=("pingpong_paddle_center",))
_DEFAULT_PADDLE_GEOM_CFG = SceneEntityCfg(
  "robot", geom_names=("pingpong_paddle_collision",)
)
_PINGPONG_RALLY_STATE_ATTR = "_pingpong_rally_state"
_MIN_BALLISTIC_TIME = 1.0e-4
_MAX_BALLISTIC_NET_TIME = 1.5
_MAX_BALLISTIC_LANDING_TIME = 2.5


def _sensor_contact_now(
  env: "ManagerBasedRlEnv", sensor_name: str, force_threshold: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  if sensor_data.force_history is not None:
    force_mag = torch.linalg.vector_norm(sensor_data.force_history, dim=-1)
    return (force_mag > force_threshold).any(dim=(1, 2))
  if sensor_data.force is not None:
    force_mag = torch.linalg.vector_norm(sensor_data.force, dim=-1)
    return (force_mag > force_threshold).any(dim=1)
  if sensor_data.found is not None:
    return (sensor_data.found > 0).any(dim=1)
  raise ValueError(f"Contact sensor {sensor_name!r} must expose force or found.")


class PingpongRallyState:
  """Per-environment state for a one-shot legal table-tennis return."""

  def __init__(
    self,
    env: "ManagerBasedRlEnv",
    *,
    paddle_sensor_name: str,
    net_sensor_name: str,
    ball_cfg: SceneEntityCfg,
    paddle_cfg: SceneEntityCfg = _DEFAULT_PADDLE_CFG,
    paddle_geom_cfg: SceneEntityCfg = _DEFAULT_PADDLE_GEOM_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    body_ball_sensor_name: str | None = None,
    force_threshold: float = 1.0,
    table_z: float = BALL_CENTER_TABLE_Z,
    net_x: float = NET_X,
    net_top_z: float = NET_TOP_Z,
    gravity: float = 9.81,
    self_x_limits: tuple[float, float] = (NET_X, TABLE_HALF_LENGTH),
    opponent_x_limits: tuple[float, float] = (-TABLE_HALF_LENGTH, NET_X),
    table_y_limits: tuple[float, float] = (-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH),
    x_limits: tuple[float, float] = (-2.1, 2.4),
    y_limits: tuple[float, float] = (-1.25, 1.25),
    z_limits: tuple[float, float] = (0.05, 2.5),
    bounce_z_tolerance: float = 0.05,
  ) -> None:
    self._env = env
    self.paddle_sensor_name = paddle_sensor_name
    self.net_sensor_name = net_sensor_name
    self.body_ball_sensor_name = body_ball_sensor_name
    self.ball_cfg = ball_cfg
    self.paddle_cfg = paddle_cfg
    self.paddle_geom_cfg = paddle_geom_cfg
    self.robot_cfg = robot_cfg
    self.force_threshold = force_threshold
    self.table_z = table_z
    self.net_x = net_x
    self.net_top_z = net_top_z
    self.gravity = gravity
    self.self_x_limits = self_x_limits
    self.opponent_x_limits = opponent_x_limits
    self.table_y_limits = table_y_limits
    self.x_limits = x_limits
    self.y_limits = y_limits
    self.z_limits = z_limits
    self.bounce_z_tolerance = bounce_z_tolerance
    self._last_step = -1

    num_envs = env.num_envs
    device = env.device

    def zeros_long() -> torch.Tensor:
      return torch.zeros(num_envs, dtype=torch.long, device=device)

    def zeros_bool() -> torch.Tensor:
      return torch.zeros(num_envs, dtype=torch.bool, device=device)

    def zeros_float() -> torch.Tensor:
      return torch.zeros(num_envs, device=device)

    def zeros_vec3() -> torch.Tensor:
      return torch.zeros(num_envs, 3, device=device)

    self.phase = torch.full(
      (num_envs,), PHASE_INCOMING, dtype=torch.long, device=device
    )
    self.fault_reason = zeros_long()
    self.self_bounce_count = zeros_long()
    self.paddle_hit_count = zeros_long()
    self.crossed_net_count = zeros_long()
    self.opponent_bounce_count = zeros_long()
    self.successful_return_count = zeros_long()
    self.episode_fault_count = zeros_long()

    self.self_bounce_edge = zeros_bool()
    self.paddle_hit_edge = zeros_bool()
    self.crossed_net_edge = zeros_bool()
    self.opponent_bounce_edge = zeros_bool()
    self.successful_return_edge = zeros_bool()
    self.net_contact_edge = zeros_bool()
    self.fault_edge = zeros_bool()

    self.has_self_bounce = zeros_bool()
    self.has_paddle_hit = zeros_bool()
    self.has_crossed_net = zeros_bool()
    self.has_opponent_bounce = zeros_bool()

    # Diagnostics recorded at the first legal paddle hit. Coordinates are in the
    # per-env world frame. The opponent side is negative x, table_z is ball-center
    # height at a legal table bounce, and gravity is assumed constant downward.
    self.hit_valid = zeros_bool()
    self.hit_post_vel = zeros_vec3()
    self.hit_post_speed = zeros_float()
    self.hit_post_vx_toward_opponent_ratio = zeros_float()
    self.hit_pred_net_clearance = zeros_float()
    self.hit_pred_net_clearance_positive = zeros_float()
    self.hit_pred_landing_x = zeros_float()
    self.hit_pred_landing_y = zeros_float()
    self.hit_pred_landing_inside_opponent_table = zeros_float()
    self.hit_paddle_speed = zeros_float()
    self.hit_paddle_normal_alignment = zeros_float()
    self.hit_paddle_velocity_along_normal = zeros_float()

    self._prev_paddle_contact = zeros_bool()
    self._prev_net_contact = zeros_bool()
    self._prev_vz = zeros_float()
    self._prev_x = zeros_float()
    self.prev_ball_x = zeros_float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.self_bounce_count[env_ids] = 0
    self.paddle_hit_count[env_ids] = 0
    self.crossed_net_count[env_ids] = 0
    self.opponent_bounce_count[env_ids] = 0
    self.successful_return_count[env_ids] = 0
    self.episode_fault_count[env_ids] = 0
    self.phase[env_ids] = PHASE_INCOMING
    self.fault_reason[env_ids] = FAULT_NONE
    self._clear_step_edges(env_ids)
    self.has_self_bounce[env_ids] = False
    self.has_paddle_hit[env_ids] = False
    self.has_crossed_net[env_ids] = False
    self.has_opponent_bounce[env_ids] = False
    self.hit_valid[env_ids] = False
    self.hit_post_vel[env_ids] = 0.0
    self.hit_post_speed[env_ids] = 0.0
    self.hit_post_vx_toward_opponent_ratio[env_ids] = 0.0
    self.hit_pred_net_clearance[env_ids] = 0.0
    self.hit_pred_net_clearance_positive[env_ids] = 0.0
    self.hit_pred_landing_x[env_ids] = 0.0
    self.hit_pred_landing_y[env_ids] = 0.0
    self.hit_pred_landing_inside_opponent_table[env_ids] = 0.0
    self.hit_paddle_speed[env_ids] = 0.0
    self.hit_paddle_normal_alignment[env_ids] = 0.0
    self.hit_paddle_velocity_along_normal[env_ids] = 0.0
    self._prev_paddle_contact[env_ids] = False
    self._prev_net_contact[env_ids] = False
    self._prev_vz[env_ids] = 0.0
    self._prev_x[env_ids] = 0.0
    self.prev_ball_x[env_ids] = 0.0
    self._last_step = -1

  def update(self) -> None:
    step = int(self._env.common_step_counter)
    if step == self._last_step:
      return

    ball: Entity = self._env.scene[self.ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w - self._env.scene.env_origins
    ball_vel = ball.data.root_link_lin_vel_w
    ball_x = ball_pos[:, 0]
    ball_y = ball_pos[:, 1]
    ball_z = ball_pos[:, 2]
    ball_vz = ball_vel[:, 2]
    prev_x = self._prev_x.clone()

    paddle_contact_now = _sensor_contact_now(
      self._env, self.paddle_sensor_name, self.force_threshold
    )
    net_contact_now = _sensor_contact_now(
      self._env, self.net_sensor_name, self.force_threshold
    )
    if self.body_ball_sensor_name is None:
      body_ball_contact_now = torch.zeros_like(paddle_contact_now)
    else:
      body_ball_contact_now = _sensor_contact_now(
        self._env, self.body_ball_sensor_name, self.force_threshold
      )
    paddle_contact_edge = paddle_contact_now & ~self._prev_paddle_contact
    net_contact_edge = net_contact_now & ~self._prev_net_contact
    bounce_edge = (
      (self._prev_vz < 0.0)
      & (ball_vz >= 0.0)
      & (torch.abs(ball_z - self.table_z) <= self.bounce_z_tolerance)
    )
    out_of_play = self._out_of_play(ball_x, ball_y, ball_z)
    on_self_table = self._in_box(ball_x, ball_y, self.self_x_limits)
    on_opponent_table = self._in_box(ball_x, ball_y, self.opponent_x_limits)

    self_bounce_edge = bounce_edge & on_self_table & ~self.has_self_bounce
    illegal_body_ball_contact = body_ball_contact_now
    legal_paddle_edge = (
      paddle_contact_edge
      & self.has_self_bounce
      & ~self.has_paddle_hit
      & ~illegal_body_ball_contact
    )
    illegal_pre_bounce_hit = paddle_contact_edge & ~self.has_self_bounce
    extra_paddle_contact = paddle_contact_edge & self.has_paddle_hit
    has_paddle_now = self.has_paddle_hit | legal_paddle_edge

    crossed_to_opp = (prev_x > self.net_x) & (ball_x <= self.net_x)
    low_net_cross = crossed_to_opp & has_paddle_now & (ball_z < self.net_top_z)
    crossed_net_edge = (
      crossed_to_opp
      & has_paddle_now
      & (ball_z >= self.net_top_z)
      & ~self.has_crossed_net
    )
    has_crossed_now = self.has_crossed_net | crossed_net_edge

    opponent_bounce_edge = (
      bounce_edge & on_opponent_table & has_crossed_now & ~self.has_opponent_bounce
    )
    failed_bounce = (
      bounce_edge
      & ~self_bounce_edge
      & ~opponent_bounce_edge
      & (self.has_self_bounce | has_paddle_now)
    )
    incoming_miss = out_of_play & ~has_paddle_now
    return_out = out_of_play & has_paddle_now & ~opponent_bounce_edge
    return_net_contact = net_contact_edge & has_paddle_now
    fault_edge = (
      illegal_pre_bounce_hit
      | extra_paddle_contact
      | failed_bounce
      | incoming_miss
      | return_out
      | return_net_contact
      | low_net_cross
      | illegal_body_ball_contact
    )

    self._clear_step_edges()
    self.self_bounce_edge[:] = self_bounce_edge
    self.paddle_hit_edge[:] = legal_paddle_edge
    self.crossed_net_edge[:] = crossed_net_edge
    self.opponent_bounce_edge[:] = opponent_bounce_edge
    self.successful_return_edge[:] = opponent_bounce_edge
    self.net_contact_edge[:] = net_contact_edge
    self.fault_edge[:] = fault_edge

    self.self_bounce_count += self_bounce_edge.long()
    self.paddle_hit_count += legal_paddle_edge.long()
    self.crossed_net_count += crossed_net_edge.long()
    self.opponent_bounce_count += opponent_bounce_edge.long()
    self.successful_return_count += opponent_bounce_edge.long()
    self.episode_fault_count += fault_edge.long()

    self._record_hit_diagnostics(ball_pos, ball_vel, legal_paddle_edge)

    self.has_self_bounce |= self_bounce_edge
    self.has_paddle_hit |= legal_paddle_edge
    self.has_crossed_net |= crossed_net_edge
    self.has_opponent_bounce |= opponent_bounce_edge
    self.phase[self_bounce_edge] = PHASE_AFTER_SELF_BOUNCE
    self.phase[legal_paddle_edge] = PHASE_RETURN_FLIGHT
    self.phase[opponent_bounce_edge | fault_edge] = PHASE_DONE
    fault_reason = self._fault_reason(
      illegal_pre_bounce_hit,
      extra_paddle_contact,
      failed_bounce,
      return_out,
      return_net_contact,
      low_net_cross,
      incoming_miss,
      illegal_body_ball_contact,
    )
    self.fault_reason[fault_edge] = fault_reason[fault_edge]

    self._prev_paddle_contact[:] = paddle_contact_now
    self._prev_net_contact[:] = net_contact_now
    self._prev_vz[:] = ball_vz
    self.prev_ball_x[:] = prev_x
    self._prev_x[:] = ball_x
    self._last_step = step

  def _record_hit_diagnostics(
    self,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
    legal_hit: torch.Tensor,
  ) -> None:
    speed = torch.linalg.vector_norm(ball_vel, dim=-1)
    vx = ball_vel[:, 0]
    toward_ratio = torch.where(
      speed > 1.0e-6,
      torch.clamp(-vx / speed.clamp_min(1.0e-6), min=0.0, max=1.0),
      torch.zeros_like(speed),
    )

    net_clearance, net_valid = self._predict_net_clearance(ball_pos, ball_vel)
    landing_xy, landing_valid = self._predict_landing_xy(ball_pos, ball_vel)
    landing_inside = landing_valid & self._in_box(
      landing_xy[:, 0],
      landing_xy[:, 1],
      self.opponent_x_limits,
    )

    self.hit_valid[legal_hit] = True
    self.hit_post_vel[legal_hit] = ball_vel[legal_hit]
    self.hit_post_speed[legal_hit] = speed[legal_hit]
    self.hit_post_vx_toward_opponent_ratio[legal_hit] = toward_ratio[legal_hit]
    self.hit_pred_net_clearance[legal_hit] = net_clearance[legal_hit]
    self.hit_pred_net_clearance_positive[legal_hit] = (
      net_valid & (net_clearance > 0.0)
    )[legal_hit].float()
    self.hit_pred_landing_x[legal_hit] = landing_xy[:, 0][legal_hit]
    self.hit_pred_landing_y[legal_hit] = landing_xy[:, 1][legal_hit]
    self.hit_pred_landing_inside_opponent_table[legal_hit] = landing_inside[
      legal_hit
    ].float()
    self._record_paddle_hit_diagnostics(legal_hit)

  def _predict_net_clearance(
    self,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    vx = ball_vel[:, 0]
    toward_net = vx < -1.0e-6
    safe_vx = torch.where(toward_net, vx, torch.full_like(vx, -1.0))
    t_net = (self.net_x - ball_pos[:, 0]) / safe_vx
    valid = (
      toward_net & (t_net > _MIN_BALLISTIC_TIME) & (t_net <= _MAX_BALLISTIC_NET_TIME)
    )
    z_at_net = ball_pos[:, 2] + ball_vel[:, 2] * t_net - 0.5 * self.gravity * t_net**2
    clearance = z_at_net - self.net_top_z
    return torch.where(valid, clearance, torch.zeros_like(clearance)), valid

  def _predict_landing_xy(
    self,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    disc = ball_vel[:, 2] ** 2 + 2.0 * self.gravity * (ball_pos[:, 2] - self.table_z)
    valid_disc = disc >= 0.0
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
    t_land = (ball_vel[:, 2] + sqrt_disc) / self.gravity
    valid = (
      valid_disc
      & (t_land > _MIN_BALLISTIC_TIME)
      & (t_land <= _MAX_BALLISTIC_LANDING_TIME)
    )
    t_land = torch.where(valid, t_land, torch.zeros_like(t_land))
    landing_xy = ball_pos[:, :2] + ball_vel[:, :2] * t_land.unsqueeze(-1)
    return landing_xy, valid

  def _record_paddle_hit_diagnostics(self, legal_hit: torch.Tensor) -> None:
    try:
      robot = self._env.scene[self.robot_cfg.name]
    except (KeyError, TypeError, AttributeError):
      return
    geom_ids = getattr(self.paddle_geom_cfg, "geom_ids", None)
    data = getattr(robot, "data", None)
    if data is None or geom_ids is None:
      return

    try:
      geom_vel_w = data.geom_lin_vel_w[:, geom_ids]
      geom_quat_w = data.geom_quat_w[:, geom_ids]
    except (AttributeError, IndexError, TypeError):
      return

    paddle_vel_w = geom_vel_w.mean(dim=1)
    paddle_quat_w = geom_quat_w[:, 0]
    normal_local = torch.zeros_like(paddle_vel_w)
    # MuJoCo cylinder height is local z; the paddle proxy is a thin cylinder.
    normal_local[:, 2] = 1.0
    normal_w = quat_apply(paddle_quat_w, normal_local)
    normal_w = torch.nn.functional.normalize(normal_w, dim=-1)
    opponent_dir = torch.zeros_like(normal_w)
    opponent_dir[:, 0] = -1.0
    normal_dot = torch.sum(normal_w * opponent_dir, dim=-1)
    oriented_normal = torch.where(normal_dot.unsqueeze(-1) >= 0.0, normal_w, -normal_w)
    normal_alignment = torch.sum(oriented_normal * opponent_dir, dim=-1).clamp(0.0, 1.0)
    velocity_along_normal = torch.sum(paddle_vel_w * oriented_normal, dim=-1)
    paddle_speed = torch.linalg.vector_norm(paddle_vel_w, dim=-1)

    self.hit_paddle_speed[legal_hit] = paddle_speed[legal_hit]
    self.hit_paddle_normal_alignment[legal_hit] = normal_alignment[legal_hit]
    self.hit_paddle_velocity_along_normal[legal_hit] = velocity_along_normal[legal_hit]

  def _clear_step_edges(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.self_bounce_edge[env_ids] = False
    self.paddle_hit_edge[env_ids] = False
    self.crossed_net_edge[env_ids] = False
    self.opponent_bounce_edge[env_ids] = False
    self.successful_return_edge[env_ids] = False
    self.net_contact_edge[env_ids] = False
    self.fault_edge[env_ids] = False

  def _in_box(
    self,
    ball_x: torch.Tensor,
    ball_y: torch.Tensor,
    x_limits: tuple[float, float],
  ) -> torch.Tensor:
    x_min, x_max = x_limits
    y_min, y_max = self.table_y_limits
    return (ball_x >= x_min) & (ball_x <= x_max) & (ball_y >= y_min) & (ball_y <= y_max)

  def _out_of_play(
    self, ball_x: torch.Tensor, ball_y: torch.Tensor, ball_z: torch.Tensor
  ) -> torch.Tensor:
    return (
      (ball_x < self.x_limits[0])
      | (ball_x > self.x_limits[1])
      | (ball_y < self.y_limits[0])
      | (ball_y > self.y_limits[1])
      | (ball_z < self.z_limits[0])
      | (ball_z > self.z_limits[1])
    )

  def _fault_reason(
    self,
    illegal_pre_bounce_hit: torch.Tensor,
    extra_paddle_contact: torch.Tensor,
    failed_bounce: torch.Tensor,
    return_out: torch.Tensor,
    return_net_contact: torch.Tensor,
    low_net_cross: torch.Tensor,
    incoming_miss: torch.Tensor,
    illegal_body_ball_contact: torch.Tensor,
  ) -> torch.Tensor:
    reason = torch.full_like(self.fault_reason, FAULT_NONE)
    reason = torch.where(
      illegal_pre_bounce_hit,
      torch.full_like(reason, FAULT_ILLEGAL_PRE_BOUNCE_HIT),
      reason,
    )
    reason = torch.where(
      extra_paddle_contact,
      torch.full_like(reason, FAULT_EXTRA_PADDLE_CONTACT),
      reason,
    )
    reason = torch.where(
      failed_bounce,
      torch.full_like(reason, FAULT_RETURN_BOUNCE_OUT),
      reason,
    )
    reason = torch.where(
      return_out,
      torch.full_like(reason, FAULT_RETURN_OUT_OF_PLAY),
      reason,
    )
    reason = torch.where(
      return_net_contact,
      torch.full_like(reason, FAULT_NET_CONTACT),
      reason,
    )
    reason = torch.where(
      low_net_cross,
      torch.full_like(reason, FAULT_LOW_NET_CROSS),
      reason,
    )
    reason = torch.where(
      incoming_miss,
      torch.full_like(reason, FAULT_INCOMING_MISS),
      reason,
    )
    reason = torch.where(
      illegal_body_ball_contact,
      torch.full_like(reason, FAULT_ILLEGAL_BODY_BALL_CONTACT),
      reason,
    )
    return reason


def get_pingpong_rally_state(
  env: "ManagerBasedRlEnv",
  *,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  paddle_cfg: SceneEntityCfg = _DEFAULT_PADDLE_CFG,
  paddle_geom_cfg: SceneEntityCfg = _DEFAULT_PADDLE_GEOM_CFG,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  body_ball_sensor_name: str | None = None,
  force_threshold: float = 1.0,
  table_z: float = BALL_CENTER_TABLE_Z,
  net_x: float = NET_X,
  net_top_z: float = NET_TOP_Z,
  gravity: float = 9.81,
  self_x_limits: tuple[float, float] = (NET_X, TABLE_HALF_LENGTH),
  opponent_x_limits: tuple[float, float] = (-TABLE_HALF_LENGTH, NET_X),
  table_y_limits: tuple[float, float] = (-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH),
  x_limits: tuple[float, float] = (-2.1, 2.4),
  y_limits: tuple[float, float] = (-1.25, 1.25),
  z_limits: tuple[float, float] = (0.05, 2.5),
  bounce_z_tolerance: float = 0.05,
) -> PingpongRallyState:
  state = getattr(env, _PINGPONG_RALLY_STATE_ATTR, None)
  if isinstance(state, PingpongRallyState):
    return state
  state = PingpongRallyState(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    paddle_cfg=paddle_cfg,
    paddle_geom_cfg=paddle_geom_cfg,
    robot_cfg=robot_cfg,
    body_ball_sensor_name=body_ball_sensor_name,
    force_threshold=force_threshold,
    table_z=table_z,
    net_x=net_x,
    net_top_z=net_top_z,
    gravity=gravity,
    self_x_limits=self_x_limits,
    opponent_x_limits=opponent_x_limits,
    table_y_limits=table_y_limits,
    x_limits=x_limits,
    y_limits=y_limits,
    z_limits=z_limits,
    bounce_z_tolerance=bounce_z_tolerance,
  )
  setattr(env, _PINGPONG_RALLY_STATE_ATTR, state)
  return state


class PingpongRallyStateTerm:
  """Mixin for reward, termination, and metric terms backed by rally state."""

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    self._state = get_pingpong_rally_state(
      env,
      paddle_sensor_name=cfg.params["paddle_sensor_name"],
      net_sensor_name=cfg.params["net_sensor_name"],
      ball_cfg=cfg.params.get("ball_cfg", _DEFAULT_BALL_CFG),
      paddle_cfg=cfg.params.get("paddle_cfg", _DEFAULT_PADDLE_CFG),
      paddle_geom_cfg=cfg.params.get("paddle_geom_cfg", _DEFAULT_PADDLE_GEOM_CFG),
      robot_cfg=cfg.params.get("robot_cfg", _DEFAULT_ROBOT_CFG),
      body_ball_sensor_name=cfg.params.get("body_ball_sensor_name"),
      force_threshold=float(cfg.params.get("force_threshold", 1.0)),
      table_z=float(cfg.params.get("table_z", BALL_CENTER_TABLE_Z)),
      net_x=float(cfg.params.get("net_x", NET_X)),
      net_top_z=float(cfg.params.get("net_top_z", NET_TOP_Z)),
      gravity=float(cfg.params.get("gravity", 9.81)),
      self_x_limits=cfg.params.get("self_x_limits", (NET_X, TABLE_HALF_LENGTH)),
      opponent_x_limits=cfg.params.get(
        "opponent_x_limits", (-TABLE_HALF_LENGTH, NET_X)
      ),
      table_y_limits=cfg.params.get(
        "table_y_limits", (-TABLE_HALF_WIDTH, TABLE_HALF_WIDTH)
      ),
      x_limits=cfg.params.get("x_limits", (-2.1, 2.4)),
      y_limits=cfg.params.get("y_limits", (-1.25, 1.25)),
      z_limits=cfg.params.get("z_limits", (0.05, 2.5)),
      bounce_z_tolerance=float(cfg.params.get("bounce_z_tolerance", 0.05)),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._state.reset(env_ids)

  @property
  def state(self) -> PingpongRallyState:
    self._state.update()
    return self._state


__all__ = [
  "FAULT_EXTRA_PADDLE_CONTACT",
  "FAULT_ILLEGAL_BODY_BALL_CONTACT",
  "FAULT_ILLEGAL_PRE_BOUNCE_HIT",
  "FAULT_INCOMING_MISS",
  "FAULT_LOW_NET_CROSS",
  "FAULT_NET_CONTACT",
  "FAULT_NONE",
  "FAULT_RETURN_BOUNCE_OUT",
  "FAULT_RETURN_OUT_OF_PLAY",
  "PHASE_AFTER_SELF_BOUNCE",
  "PHASE_DONE",
  "PHASE_INCOMING",
  "PHASE_RETURN_FLIGHT",
  "PingpongRallyState",
  "PingpongRallyStateTerm",
  "get_pingpong_rally_state",
]
