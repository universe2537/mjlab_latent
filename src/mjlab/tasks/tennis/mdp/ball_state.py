"""Continuous tennis ball lifecycle and rally state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tennis.mdp.ball_providers import (
  _uniform,
  _write_ball_state,
  spawn_ball_from_provider,
)
from mjlab.tasks.tennis.mdp.observations import (
  _predict_hit_intersection_w,
  racket_to_ball_b,
  racket_velocity_b,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

PHASE_INCOMING = 0
PHASE_RETURN_FLIGHT = 1
PHASE_RECOVERY = 2

FAULT_NONE = 0
FAULT_INCOMING_BOUNCE = 1
FAULT_RETURN_BOUNCE_OUT = 2
FAULT_RETURN_OUT_OF_PLAY = 3
FAULT_NET_CONTACT = 4
FAULT_EXTRA_RACKET_CONTACT = 5
FAULT_LOW_NET_CROSS = 6

_DEFAULT_BALL_CFG = SceneEntityCfg("ball")
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_CONTINUOUS_STATE_ATTR = "_tennis_continuous_ball_state"


@dataclass(kw_only=True)
class OpponentFeederCfg:
  """Generate incoming balls from the opponent half to the robot half.

  ``spawn_x_range`` should stay on the opponent side (negative x), while
  ``target_x_range`` is the robot-side landing/work region. The feeder samples
  bird's-eye start/target points on the ground plane, keeps the ball center at
  ``z = ground_z`` for both endpoints, then solves the upward z velocity from a
  flight time that is bounded below by the required net-clearance height.
  Rejected incoming feeds are not considered robot failures.
  """

  ball_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("ball"))

  spawn_x_range: tuple[float, float] = (-5.0, -0.5)
  spawn_y_range: tuple[float, float] = (-2.0, 2.0)
  target_x_range: tuple[float, float] = (1.0, 4.0)
  target_y_range: tuple[float, float] = (-2.0, 2.0)

  flight_time_range: tuple[float, float] = (0.85, 1.35)
  flight_time_slack_range: tuple[float, float] = (0.05, 0.35)
  spawn_z_range: tuple[float, float] = (0.06, 0.06)

  ground_z: float = 0.06
  net_x: float = 0.0
  net_height: float = 0.914
  net_clearance: float = 0.25
  ball_radius: float = 0.0335
  net_half_thickness: float = 0.012
  max_apex_z: float = 3.9
  gravity: float = 9.81
  max_resample_attempts: int = 64

  def build(self, env: "ManagerBasedRlEnv") -> "OpponentFeeder":
    return OpponentFeeder(self, env)


class OpponentFeeder:
  """Runtime opponent-half feeder with rejection sampling."""

  def __init__(self, cfg: OpponentFeederCfg, env: "ManagerBasedRlEnv") -> None:
    self.cfg = cfg
    self._env = env
    self._ball: Entity = env.scene[cfg.ball_cfg.name]

  @property
  def device(self) -> str | torch.device:
    return self._env.device

  def spawn(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    cfg = self.cfg
    dev = self.device
    n = env_ids.numel()
    px = torch.empty(n, device=dev)
    py = torch.empty(n, device=dev)
    pz = torch.empty(n, device=dev)
    vx = torch.empty(n, device=dev)
    vy = torch.empty(n, device=dev)
    vz = torch.empty(n, device=dev)

    remaining = torch.arange(n, device=dev)
    for _ in range(max(1, int(cfg.max_resample_attempts))):
      if remaining.numel() == 0:
        break
      sample = self._sample_candidate(env_ids[remaining])
      valid = sample[-1]
      if torch.any(valid):
        write_ids = remaining[valid]
        px[write_ids] = sample[0][valid]
        py[write_ids] = sample[1][valid]
        pz[write_ids] = sample[2][valid]
        vx[write_ids] = sample[3][valid]
        vy[write_ids] = sample[4][valid]
        vz[write_ids] = sample[5][valid]
      remaining = remaining[~valid]

    if remaining.numel() > 0:
      fallback = self._fallback_candidate(env_ids[remaining])
      px[remaining] = fallback[0]
      py[remaining] = fallback[1]
      pz[remaining] = fallback[2]
      vx[remaining] = fallback[3]
      vy[remaining] = fallback[4]
      vz[remaining] = fallback[5]

    pos = torch.stack((px, py, pz), dim=-1)
    lin = torch.stack((vx, vy, vz), dim=-1)
    quat = torch.zeros(n, 4, device=dev)
    quat[:, 0] = 1.0
    ang = torch.zeros(n, 3, device=dev)
    _write_ball_state(self._env, self._ball, env_ids, pos, quat, lin, ang)

  def _sample_candidate(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cfg = self.cfg
    dev = self.device
    px = _uniform(env_ids, cfg.spawn_x_range[0], cfg.spawn_x_range[1], dev)
    py = _uniform(env_ids, cfg.spawn_y_range[0], cfg.spawn_y_range[1], dev)
    tx = _uniform(env_ids, cfg.target_x_range[0], cfg.target_x_range[1], dev)
    ty = _uniform(env_ids, cfg.target_y_range[0], cfg.target_y_range[1], dev)
    base_flight_t = _uniform(
      env_ids, cfg.flight_time_range[0], cfg.flight_time_range[1], dev
    )
    slack_t = _uniform(
      env_ids, cfg.flight_time_slack_range[0], cfg.flight_time_slack_range[1], dev
    )

    g = cfg.gravity
    pz = torch.full_like(base_flight_t, cfg.ground_z)
    dx = tx - px
    net_fraction = ((cfg.net_x - px) / dx.clamp_min(1.0e-6)).clamp(1.0e-4, 1.0 - 1.0e-4)
    required_clearance = max(
      1.0e-4,
      cfg.net_height + cfg.net_clearance + cfg.ball_radius - cfg.ground_z,
    )
    min_flight_t = torch.sqrt(
      torch.full_like(base_flight_t, 2.0 * required_clearance / g)
      / (net_fraction * (1.0 - net_fraction)).clamp_min(1.0e-6)
    )
    flight_t = torch.maximum(base_flight_t, min_flight_t + slack_t)
    vz = (cfg.ground_z - pz + 0.5 * g * flight_t * flight_t) / flight_t
    vx = (tx - px) / flight_t
    vy = (ty - py) / flight_t

    t_net = (cfg.net_x - px) / vx.clamp_min(1.0e-6)
    z_net = pz + vz * t_net - 0.5 * g * t_net * t_net
    z_apex = pz + torch.square(vz) / (2.0 * g)
    valid = (
      (dx > 0.0)
      & (vx > 0.0)
      & (t_net > 0.0)
      & (t_net < flight_t)
      & (z_net >= cfg.net_height + cfg.net_clearance + cfg.ball_radius)
      & (z_apex <= cfg.max_apex_z)
    )
    clearance_x = cfg.net_half_thickness + cfg.ball_radius
    for x_check in (
      cfg.net_x - clearance_x,
      cfg.net_x + clearance_x,
    ):
      t_check = (x_check - px) / vx.clamp_min(1.0e-6)
      within_flight = (t_check > 0.0) & (t_check < flight_t)
      z_check = pz + vz * t_check - 0.5 * g * t_check * t_check
      valid &= ~within_flight | (
        z_check >= cfg.net_height + cfg.net_clearance + cfg.ball_radius
      )
    return px, py, pz, vx, vy, vz, valid

  def _fallback_candidate(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cfg = self.cfg
    dev = self.device
    n = env_ids.numel()
    px = torch.full(
      (n,),
      0.5 * (cfg.spawn_x_range[0] + cfg.spawn_x_range[1]),
      device=dev,
    )
    py = torch.full(
      (n,),
      0.5 * (cfg.spawn_y_range[0] + cfg.spawn_y_range[1]),
      device=dev,
    )
    tx = torch.full(
      (n,),
      0.5 * (cfg.target_x_range[0] + cfg.target_x_range[1]),
      device=dev,
    )
    ty = torch.full(
      (n,),
      0.5 * (cfg.target_y_range[0] + cfg.target_y_range[1]),
      device=dev,
    )
    safe_t = max(cfg.flight_time_range[0], min(cfg.flight_time_range[1], 1.1))
    flight_t = torch.full((n,), safe_t, device=dev)
    pz = torch.full((n,), cfg.ground_z, device=dev)
    dx = tx - px
    net_fraction = ((cfg.net_x - px) / dx.clamp_min(1.0e-6)).clamp(1.0e-4, 1.0 - 1.0e-4)
    required_clearance = max(
      1.0e-4,
      cfg.net_height + cfg.net_clearance + cfg.ball_radius - cfg.ground_z,
    )
    min_flight_t = torch.sqrt(
      torch.full_like(flight_t, 2.0 * required_clearance / cfg.gravity)
      / (net_fraction * (1.0 - net_fraction)).clamp_min(1.0e-6)
    )
    flight_t = torch.maximum(flight_t, min_flight_t + 0.1)
    vz = (cfg.ground_z - pz + 0.5 * cfg.gravity * flight_t * flight_t) / flight_t
    vx = (tx - px) / flight_t
    vy = (ty - py) / flight_t
    return px, py, pz, vx, vy, vz


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


class TennisContinuousBallState:
  """State machine for one or more continuous tennis rallies."""

  def __init__(
    self,
    env: "ManagerBasedRlEnv",
    *,
    racket_sensor_name: str,
    net_sensor_name: str,
    ball_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    net_height: float = 0.914,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] = (-5.8, 3.6),
    y_limits: tuple[float, float] = (-2.7, 2.7),
    z_limits: tuple[float, float] = (0.02, 4.0),
  ) -> None:
    self._env = env
    self.racket_sensor_name = racket_sensor_name
    self.net_sensor_name = net_sensor_name
    self.ball_cfg = ball_cfg
    self.force_threshold = force_threshold
    self.ground_z = ground_z
    self.net_x = net_x
    self.net_height = net_height
    self.landing_x_limits = landing_x_limits
    self.landing_y_limits = landing_y_limits
    self.x_limits = x_limits
    self.y_limits = y_limits
    self.z_limits = z_limits
    self._last_step = -1

    num_envs = env.num_envs
    device = env.device

    def zeros_long() -> torch.Tensor:
      return torch.zeros(num_envs, dtype=torch.long, device=device)

    def zeros_bool() -> torch.Tensor:
      return torch.zeros(num_envs, dtype=torch.bool, device=device)

    def zeros_float() -> torch.Tensor:
      return torch.zeros(num_envs, device=device)

    self.phase = torch.full(
      (num_envs,), PHASE_INCOMING, dtype=torch.long, device=device
    )
    self.fault_reason = zeros_long()
    self.racket_hit_count = zeros_long()
    self.bounce_count = zeros_long()
    self.successful_return_count = zeros_long()
    self.episode_racket_hit_count = zeros_long()
    self.episode_crossed_net_count = zeros_long()
    self.episode_landing_in_bounds_count = zeros_long()
    self.episode_net_contact_count = zeros_long()
    self.episode_invalid_feed_count = zeros_long()
    self.episode_invalid_feed_net_count = zeros_long()
    self.episode_invalid_feed_out_count = zeros_long()
    self.episode_invalid_feed_opponent_bounce_count = zeros_long()
    self.episode_fault_count = zeros_long()
    self.episode_fault_incoming_bounce_count = zeros_long()
    self.episode_fault_return_bounce_out_count = zeros_long()
    self.episode_fault_return_out_count = zeros_long()
    self.episode_fault_net_contact_count = zeros_long()
    self.episode_fault_extra_racket_count = zeros_long()
    self.episode_fault_low_net_cross_count = zeros_long()
    self.episode_recovery_ready_count = zeros_long()

    self.racket_hit_edge = zeros_bool()
    self.bounce_edge = zeros_bool()
    self.crossed_net_edge = zeros_bool()
    self.landing_in_bounds_edge = zeros_bool()
    self.successful_return_edge = zeros_bool()
    self.net_contact_edge = zeros_bool()
    self.invalid_feed_edge = zeros_bool()
    self.fault_edge = zeros_bool()
    self.recovery_ready_edge = zeros_bool()
    self.respawn_edge = zeros_bool()

    self.has_racket_hit = zeros_bool()
    self.has_crossed_net = zeros_bool()
    self.has_landed_in_bounds = zeros_bool()
    self.has_recovery_ready = zeros_bool()
    self.recovery_min_steps_left = zeros_long()
    self.recovery_steps_left = zeros_long()
    self.recovery_steps_total = zeros_long()

    self._prev_racket_contact = zeros_bool()
    self._prev_net_contact = zeros_bool()
    self._prev_vz = zeros_float()
    self._prev_x = zeros_float()
    self._prev_z = zeros_float()
    self.prev_ball_x = zeros_float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.successful_return_count[env_ids] = 0
    self.episode_racket_hit_count[env_ids] = 0
    self.episode_crossed_net_count[env_ids] = 0
    self.episode_landing_in_bounds_count[env_ids] = 0
    self.episode_net_contact_count[env_ids] = 0
    self.episode_invalid_feed_count[env_ids] = 0
    self.episode_invalid_feed_net_count[env_ids] = 0
    self.episode_invalid_feed_out_count[env_ids] = 0
    self.episode_invalid_feed_opponent_bounce_count[env_ids] = 0
    self.episode_fault_count[env_ids] = 0
    self.episode_fault_incoming_bounce_count[env_ids] = 0
    self.episode_fault_return_bounce_out_count[env_ids] = 0
    self.episode_fault_return_out_count[env_ids] = 0
    self.episode_fault_net_contact_count[env_ids] = 0
    self.episode_fault_extra_racket_count[env_ids] = 0
    self.episode_fault_low_net_cross_count[env_ids] = 0
    self.episode_recovery_ready_count[env_ids] = 0
    self.reset_ball(env_ids)
    self._last_step = -1

  def reset_ball(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.phase[env_ids] = PHASE_INCOMING
    self.fault_reason[env_ids] = FAULT_NONE
    self.racket_hit_count[env_ids] = 0
    self.bounce_count[env_ids] = 0
    self.racket_hit_edge[env_ids] = False
    self.bounce_edge[env_ids] = False
    self.crossed_net_edge[env_ids] = False
    self.landing_in_bounds_edge[env_ids] = False
    self.successful_return_edge[env_ids] = False
    self.net_contact_edge[env_ids] = False
    self.invalid_feed_edge[env_ids] = False
    self.fault_edge[env_ids] = False
    self.recovery_ready_edge[env_ids] = False
    self.respawn_edge[env_ids] = False
    self.has_racket_hit[env_ids] = False
    self.has_crossed_net[env_ids] = False
    self.has_landed_in_bounds[env_ids] = False
    self.has_recovery_ready[env_ids] = False
    self.recovery_min_steps_left[env_ids] = 0
    self.recovery_steps_left[env_ids] = 0
    self.recovery_steps_total[env_ids] = 0
    self._prev_racket_contact[env_ids] = False
    self._prev_net_contact[env_ids] = False
    self._prev_vz[env_ids] = 0.0
    self._prev_x[env_ids] = 0.0
    self._prev_z[env_ids] = 0.0
    self.prev_ball_x[env_ids] = 0.0

  @property
  def in_recovery(self) -> torch.Tensor:
    return self.phase == PHASE_RECOVERY

  @property
  def recovery_fraction_remaining(self) -> torch.Tensor:
    total = self.recovery_steps_total.clamp_min(1).float()
    return self.recovery_steps_left.float() / total

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
    prev_z = self._prev_z.clone()

    racket_contact_now = _sensor_contact_now(
      self._env,
      self.racket_sensor_name,
      self.force_threshold,
    )
    net_contact_now = _sensor_contact_now(
      self._env,
      self.net_sensor_name,
      self.force_threshold,
    )
    racket_contact_edge = racket_contact_now & ~self._prev_racket_contact
    net_contact_edge = net_contact_now & ~self._prev_net_contact
    bounce_edge = (
      (self._prev_vz < 0.0) & (ball_vz >= 0.0) & (ball_z < self.ground_z + 0.05)
    )
    out_of_play = self._out_of_play(ball_x, ball_y, ball_z)

    incoming = self.phase == PHASE_INCOMING
    return_flight = self.phase == PHASE_RETURN_FLIGHT
    active_ball = incoming | return_flight

    racket_hit_edge = racket_contact_edge & active_ball
    first_hit_edge = racket_hit_edge & incoming
    incoming_without_hit = incoming & ~first_hit_edge
    return_flight_now = return_flight | first_hit_edge
    extra_racket_edge = racket_hit_edge & return_flight & self.has_racket_hit
    net_contact_active_edge = net_contact_edge & active_ball

    crossed_to_opp = (prev_x > self.net_x) & (ball_x <= self.net_x)
    z_at_net = self._interpolate_net_z(prev_x, ball_x, prev_z, ball_z)
    low_net_cross = crossed_to_opp & return_flight_now & (z_at_net < self.net_height)
    crossed_net_edge = (
      crossed_to_opp
      & return_flight_now
      & (z_at_net >= self.net_height)
      & ~self.has_crossed_net
    )
    has_crossed_now = self.has_crossed_net | crossed_net_edge

    landing_in_bounds = self._landing_in_bounds(ball_x, ball_y)
    successful_return_edge = (
      bounce_edge
      & return_flight_now
      & has_crossed_now
      & landing_in_bounds
      & ~self.has_landed_in_bounds
    )
    landing_failed = bounce_edge & return_flight_now & ~successful_return_edge

    incoming_bounce_fault = bounce_edge & incoming_without_hit & (ball_x > self.net_x)
    incoming_net_invalid = net_contact_active_edge & incoming_without_hit
    incoming_out_invalid = out_of_play & incoming_without_hit
    opponent_side_bounce = bounce_edge & incoming_without_hit & (ball_x <= self.net_x)
    invalid_feed_edge = (
      incoming_net_invalid | incoming_out_invalid | opponent_side_bounce
    )

    return_out = out_of_play & return_flight_now & ~successful_return_edge
    return_net_contact = net_contact_active_edge & return_flight_now
    fault_edge = (
      incoming_bounce_fault
      | landing_failed
      | return_out
      | return_net_contact
      | extra_racket_edge
      | low_net_cross
    )

    self._clear_step_edges()
    self.racket_hit_edge[:] = racket_hit_edge
    self.bounce_edge[:] = bounce_edge & active_ball
    self.crossed_net_edge[:] = crossed_net_edge
    self.landing_in_bounds_edge[:] = successful_return_edge
    self.successful_return_edge[:] = successful_return_edge
    self.net_contact_edge[:] = net_contact_active_edge
    self.invalid_feed_edge[:] = invalid_feed_edge
    self.fault_edge[:] = fault_edge

    self.racket_hit_count += racket_hit_edge.long()
    self.bounce_count += (bounce_edge & active_ball).long()
    self.successful_return_count += successful_return_edge.long()
    self.episode_racket_hit_count += racket_hit_edge.long()
    self.episode_crossed_net_count += crossed_net_edge.long()
    self.episode_landing_in_bounds_count += successful_return_edge.long()
    self.episode_net_contact_count += net_contact_active_edge.long()
    self.episode_invalid_feed_count += invalid_feed_edge.long()
    self.episode_invalid_feed_net_count += incoming_net_invalid.long()
    self.episode_invalid_feed_out_count += incoming_out_invalid.long()
    self.episode_invalid_feed_opponent_bounce_count += opponent_side_bounce.long()
    self.episode_fault_count += fault_edge.long()
    self.episode_fault_incoming_bounce_count += incoming_bounce_fault.long()
    self.episode_fault_return_bounce_out_count += landing_failed.long()
    self.episode_fault_return_out_count += return_out.long()
    self.episode_fault_net_contact_count += return_net_contact.long()
    self.episode_fault_extra_racket_count += extra_racket_edge.long()
    self.episode_fault_low_net_cross_count += low_net_cross.long()

    self.has_racket_hit |= first_hit_edge
    self.has_crossed_net |= crossed_net_edge
    self.has_landed_in_bounds |= successful_return_edge
    self.phase[first_hit_edge] = PHASE_RETURN_FLIGHT
    self.phase[successful_return_edge] = PHASE_RECOVERY
    fault_reason = self._fault_reason(
      incoming_bounce_fault,
      landing_failed,
      return_out,
      return_net_contact,
      extra_racket_edge,
      low_net_cross,
    )
    self.fault_reason[fault_edge] = fault_reason[fault_edge]

    self._prev_racket_contact[:] = racket_contact_now
    self._prev_net_contact[:] = net_contact_now
    self._prev_vz[:] = ball_vz
    self.prev_ball_x[:] = prev_x
    self._prev_x[:] = ball_x
    self._prev_z[:] = ball_z
    self._last_step = step

  def _clear_step_edges(self) -> None:
    self.racket_hit_edge[:] = False
    self.bounce_edge[:] = False
    self.crossed_net_edge[:] = False
    self.landing_in_bounds_edge[:] = False
    self.successful_return_edge[:] = False
    self.net_contact_edge[:] = False
    self.invalid_feed_edge[:] = False
    self.fault_edge[:] = False
    self.recovery_ready_edge[:] = False
    self.respawn_edge[:] = False

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

  def _landing_in_bounds(
    self, ball_x: torch.Tensor, ball_y: torch.Tensor
  ) -> torch.Tensor:
    landing_in_bounds = torch.ones_like(ball_x, dtype=torch.bool)
    if self.landing_x_limits is not None:
      landing_in_bounds &= (ball_x >= self.landing_x_limits[0]) & (
        ball_x <= self.landing_x_limits[1]
      )
    if self.landing_y_limits is not None:
      landing_in_bounds &= (ball_y >= self.landing_y_limits[0]) & (
        ball_y <= self.landing_y_limits[1]
      )
    return landing_in_bounds

  def _interpolate_net_z(
    self,
    prev_x: torch.Tensor,
    ball_x: torch.Tensor,
    prev_z: torch.Tensor,
    ball_z: torch.Tensor,
  ) -> torch.Tensor:
    dx = ball_x - prev_x
    safe_dx = torch.where(
      torch.abs(dx) > 1.0e-6,
      dx,
      torch.full_like(dx, -1.0e-6),
    )
    alpha = ((self.net_x - prev_x) / safe_dx).clamp(0.0, 1.0)
    return prev_z + alpha * (ball_z - prev_z)

  def _fault_reason(
    self,
    incoming_bounce_fault: torch.Tensor,
    landing_failed: torch.Tensor,
    return_out: torch.Tensor,
    net_contact: torch.Tensor,
    extra_racket: torch.Tensor,
    low_net_cross: torch.Tensor,
  ) -> torch.Tensor:
    reason = torch.full_like(self.fault_reason, FAULT_NONE)
    reason = torch.where(
      incoming_bounce_fault,
      torch.full_like(reason, FAULT_INCOMING_BOUNCE),
      reason,
    )
    reason = torch.where(
      landing_failed,
      torch.full_like(reason, FAULT_RETURN_BOUNCE_OUT),
      reason,
    )
    reason = torch.where(
      return_out,
      torch.full_like(reason, FAULT_RETURN_OUT_OF_PLAY),
      reason,
    )
    reason = torch.where(
      net_contact,
      torch.full_like(reason, FAULT_NET_CONTACT),
      reason,
    )
    reason = torch.where(
      extra_racket,
      torch.full_like(reason, FAULT_EXTRA_RACKET_CONTACT),
      reason,
    )
    reason = torch.where(
      low_net_cross,
      torch.full_like(reason, FAULT_LOW_NET_CROSS),
      reason,
    )
    return reason

  def start_recovery(
    self,
    env_ids: torch.Tensor,
    recovery_time_range: tuple[float, float],
    min_recovery_time: float = 1.0,
  ) -> None:
    if env_ids.numel() == 0:
      return
    lo_s, hi_s = recovery_time_range
    lo_steps = max(1, int(math.ceil(lo_s / self._env.step_dt)))
    hi_steps = max(lo_steps, int(math.ceil(hi_s / self._env.step_dt)))
    min_steps = max(1, int(math.ceil(min_recovery_time / self._env.step_dt)))
    min_steps = min(min_steps, hi_steps)
    steps = torch.randint(
      lo_steps,
      hi_steps + 1,
      (env_ids.numel(),),
      dtype=torch.long,
      device=self._env.device,
    )
    steps = torch.maximum(steps, torch.full_like(steps, min_steps))
    self.recovery_min_steps_left[env_ids] = min_steps
    self.recovery_steps_left[env_ids] = steps
    self.recovery_steps_total[env_ids] = steps

  def step_recovery(self, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    if active_mask is None:
      active_mask = self.in_recovery
    active_mask = active_mask & self.in_recovery & (self.recovery_steps_left > 0)
    if torch.any(active_mask):
      self.recovery_min_steps_left[active_mask] = torch.clamp(
        self.recovery_min_steps_left[active_mask] - 1,
        min=0,
      )
      self.recovery_steps_left[active_mask] -= 1
    ready = active_mask & (self.recovery_steps_left <= 0)
    self.recovery_steps_left[ready] = 0
    return ready

  def mark_recovery_ready(self, ready_mask: torch.Tensor) -> torch.Tensor:
    edge = ready_mask & self.in_recovery & ~self.has_recovery_ready
    self.recovery_ready_edge[:] = edge
    self.has_recovery_ready |= edge
    self.episode_recovery_ready_count += edge.long()
    return edge


def get_tennis_continuous_ball_state(
  env: "ManagerBasedRlEnv",
  *,
  racket_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  net_height: float = 0.914,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
  x_limits: tuple[float, float] = (-5.8, 3.6),
  y_limits: tuple[float, float] = (-2.7, 2.7),
  z_limits: tuple[float, float] = (0.02, 4.0),
) -> TennisContinuousBallState:
  state = getattr(env, _CONTINUOUS_STATE_ATTR, None)
  if isinstance(state, TennisContinuousBallState):
    return state
  state = TennisContinuousBallState(
    env,
    racket_sensor_name=racket_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    net_height=net_height,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
    x_limits=x_limits,
    y_limits=y_limits,
    z_limits=z_limits,
  )
  setattr(env, _CONTINUOUS_STATE_ATTR, state)
  return state


class TennisContinuousBallStateTerm:
  """Mixin for continuous-rally terms backed by TennisContinuousBallState."""

  def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:
    self._state = get_tennis_continuous_ball_state(
      env,
      racket_sensor_name=cfg.params["racket_sensor_name"],
      net_sensor_name=cfg.params["net_sensor_name"],
      ball_cfg=cfg.params.get("ball_cfg", _DEFAULT_BALL_CFG),
      force_threshold=float(cfg.params.get("force_threshold", 1.0)),
      ground_z=float(cfg.params.get("ground_z", 0.06)),
      net_x=float(cfg.params.get("net_x", 0.0)),
      net_height=float(cfg.params.get("net_height", 0.914)),
      landing_x_limits=cfg.params.get("landing_x_limits"),
      landing_y_limits=cfg.params.get("landing_y_limits"),
      x_limits=cfg.params.get("x_limits", (-5.8, 3.6)),
      y_limits=cfg.params.get("y_limits", (-2.7, 2.7)),
      z_limits=cfg.params.get("z_limits", (0.02, 4.0)),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._state.reset(env_ids)

  @property
  def state(self) -> TennisContinuousBallState:
    self._state.update()
    return self._state


def continuous_ball_phase(
  env: "ManagerBasedRlEnv",
  racket_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  net_height: float = 0.914,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
  x_limits: tuple[float, float] = (-5.8, 3.6),
  y_limits: tuple[float, float] = (-2.7, 2.7),
  z_limits: tuple[float, float] = (0.02, 4.0),
) -> torch.Tensor:
  """Expose the current continuous-rally phase as a 3-way one-hot vector."""
  state = get_tennis_continuous_ball_state(
    env,
    racket_sensor_name=racket_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    net_height=net_height,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
    x_limits=x_limits,
    y_limits=y_limits,
    z_limits=z_limits,
  )
  state.update()
  incoming = (state.phase == PHASE_INCOMING).float()
  return_flight = (state.phase == PHASE_RETURN_FLIGHT).float()
  recovery = (state.phase == PHASE_RECOVERY).float()
  return torch.stack(
    (
      incoming,
      return_flight,
      recovery,
    ),
    dim=-1,
  )


class continuous_racket_to_predicted_hit_point_dense(TennisContinuousBallStateTerm):
  """Approach the predicted incoming intercept point outside recovery."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    racket_sensor_name: str,
    net_sensor_name: str,
    std: float,
    racket_cfg: SceneEntityCfg = _DEFAULT_RACKET_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    hit_height_offset: float = 0.05,
    gravity: float = 9.81,
    max_horizon: float = 1.5,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    net_height: float = 0.914,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] = (-5.8, 3.6),
    y_limits: tuple[float, float] = (-2.7, 2.7),
    z_limits: tuple[float, float] = (0.02, 4.0),
  ) -> torch.Tensor:
    del (
      racket_sensor_name,
      net_sensor_name,
      force_threshold,
      ground_z,
      net_x,
      net_height,
      landing_x_limits,
      landing_y_limits,
      x_limits,
      y_limits,
      z_limits,
    )
    state = self.state
    robot: Entity = env.scene[robot_cfg.name]
    racket_pos_w = robot.data.site_pos_w[:, racket_cfg.site_ids].squeeze(1)
    hit_w, _, valid = _predict_hit_intersection_w(
      env,
      ball_cfg,
      robot_cfg,
      hit_height_offset=hit_height_offset,
      gravity=gravity,
      max_horizon=max_horizon,
    )
    error = torch.sum(torch.square(hit_w - racket_pos_w), dim=-1)
    reward = torch.exp(-error / std**2) * valid.float()
    active = state.phase == PHASE_INCOMING
    return reward * active.float()


class continuous_racket_towards_ball_velocity(TennisContinuousBallStateTerm):
  """Reward moving the racket toward the incoming ball before contact."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    racket_sensor_name: str,
    net_sensor_name: str,
    racket_cfg: SceneEntityCfg = _DEFAULT_RACKET_CFG,
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    speed_scale: float = 2.0,
    distance_std: float = 0.8,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    net_height: float = 0.914,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
    x_limits: tuple[float, float] = (-5.8, 3.6),
    y_limits: tuple[float, float] = (-2.7, 2.7),
    z_limits: tuple[float, float] = (0.02, 4.0),
  ) -> torch.Tensor:
    del (
      racket_sensor_name,
      net_sensor_name,
      force_threshold,
      ground_z,
      net_x,
      net_height,
      landing_x_limits,
      landing_y_limits,
      x_limits,
      y_limits,
      z_limits,
    )
    state = self.state
    delta_b = racket_to_ball_b(env, racket_cfg, ball_cfg, robot_cfg)
    racket_vel_b = racket_velocity_b(env, racket_cfg, robot_cfg)
    distance = torch.linalg.vector_norm(delta_b, dim=-1).clamp_min(1.0e-6)
    direction_to_ball = delta_b / distance.unsqueeze(-1)
    toward_speed = torch.sum(racket_vel_b * direction_to_ball, dim=-1)
    toward_speed = torch.clamp(toward_speed, min=0.0)
    distance_weight = torch.exp(-(distance**2) / distance_std**2)
    reward = torch.tanh(toward_speed / speed_scale) * distance_weight
    return reward * (state.phase == PHASE_INCOMING).float()


class continuous_racket_hit_event(TennisContinuousBallStateTerm):
  """Sparse reward for the first contact with an incoming ball."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(self, env: "ManagerBasedRlEnv", **_: object) -> torch.Tensor:
    del env
    state = self.state
    return state.racket_hit_edge.float()


class continuous_crossed_net_event(TennisContinuousBallStateTerm):
  """Sparse reward when a robot return first crosses to opponent half."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(self, env: "ManagerBasedRlEnv", **_: object) -> torch.Tensor:
    del env
    return self.state.crossed_net_edge.float()


class continuous_landing_in_bounds_event(TennisContinuousBallStateTerm):
  """Sparse reward when a return first lands in the opponent court."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(self, env: "ManagerBasedRlEnv", **_: object) -> torch.Tensor:
    del env
    return self.state.successful_return_edge.float()


class continuous_post_hit_x_progress(TennisContinuousBallStateTerm):
  """Reward the returned ball moving toward negative x before net crossing."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    net_x: float = 0.0,
    max_progress: float = 0.08,
    **_: object,
  ) -> torch.Tensor:
    state = self.state
    ball: Entity = env.scene[ball_cfg.name]
    ball_x = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = torch.clamp(state.prev_ball_x - ball_x, min=0.0, max=max_progress)
    reward = progress / max_progress
    active = (
      (state.phase == PHASE_RETURN_FLIGHT) & state.has_racket_hit & (ball_x > net_x)
    )
    return reward * active.float()


class continuous_post_hit_ball_velocity_direction(TennisContinuousBallStateTerm):
  """Reward return velocity toward the opponent half."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
    net_x: float = 0.0,
    x_speed_scale: float = 4.0,
    lateral_speed_std: float = 1.5,
    **_: object,
  ) -> torch.Tensor:
    state = self.state
    ball: Entity = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w - env.scene.env_origins
    ball_vel = ball.data.root_link_lin_vel_w
    x_reward = torch.clamp(-ball_vel[:, 0] / x_speed_scale, min=0.0, max=1.0)
    lateral_weight = torch.exp(-(ball_vel[:, 1] ** 2) / lateral_speed_std**2)
    active = (
      (state.phase == PHASE_RETURN_FLIGHT)
      & state.has_racket_hit
      & (ball_pos[:, 0] > net_x)
    )
    return x_reward * lateral_weight * active.float()


def _continuous_recovery_home_terms(
  env: "ManagerBasedRlEnv",
  *,
  racket_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg,
  target_x: float,
  target_y: float,
  target_heading: float,
  base_pos_std: float,
  heading_std: float,
  lin_vel_std: float,
  upright_std: float,
  move_speed_scale: float,
  racket_target_b: tuple[float, float, float],
  racket_std: float,
  ready_pos_threshold: float,
  ready_heading_threshold: float,
  ready_speed_threshold: float,
  ready_upright_threshold: float,
  ready_racket_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  robot: Entity = env.scene[robot_cfg.name]
  base_pos_l = robot.data.root_link_pos_w - env.scene.env_origins
  target_xy = torch.tensor((target_x, target_y), dtype=torch.float32, device=env.device)
  to_home_xy = target_xy.unsqueeze(0) - base_pos_l[:, :2]
  base_xy_error = torch.sum(torch.square(to_home_xy), dim=1)
  base_reward = torch.exp(-base_xy_error / base_pos_std**2)

  heading_error = wrap_to_pi(robot.data.heading_w - target_heading)
  heading_reward = torch.exp(-torch.square(heading_error) / heading_std**2)

  lin_vel_xy_b = torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1)
  still_reward = torch.exp(-lin_vel_xy_b / lin_vel_std**2)
  dist_to_home = torch.sqrt(base_xy_error.clamp_min(1.0e-8))
  home_dir = to_home_xy / dist_to_home.unsqueeze(-1).clamp_min(1.0e-4)
  vel_to_home = torch.sum(robot.data.root_link_lin_vel_w[:, :2] * home_dir, dim=1)
  move_reward = torch.clamp(vel_to_home / move_speed_scale, min=0.0, max=1.0)

  upright_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
  upright_reward = torch.exp(-upright_error / upright_std**2)

  racket_pos_w = robot.data.site_pos_w[:, racket_cfg.site_ids].squeeze(1)
  racket_delta_w = racket_pos_w - robot.data.root_link_pos_w
  racket_pos_b = quat_apply_inverse(robot.data.root_link_quat_w, racket_delta_w)
  racket_target = torch.tensor(racket_target_b, dtype=torch.float32, device=env.device)
  racket_error = torch.sum(torch.square(racket_pos_b - racket_target), dim=1)
  racket_reward = torch.exp(-racket_error / racket_std**2)

  ready_reward = (
    base_reward + heading_reward + still_reward + upright_reward + racket_reward
  ) / 5.0
  far_reward = 0.75 * move_reward + 0.25 * base_reward
  reward = (1.0 - base_reward) * far_reward + base_reward * ready_reward
  ready = (
    (dist_to_home <= ready_pos_threshold)
    & (torch.abs(heading_error) <= ready_heading_threshold)
    & (torch.sqrt(lin_vel_xy_b) <= ready_speed_threshold)
    & (torch.sqrt(upright_error) <= ready_upright_threshold)
    & (torch.sqrt(racket_error) <= ready_racket_threshold)
  )
  return reward, ready


class continuous_recovery_ready_pose_state(TennisContinuousBallStateTerm):
  """Reward returning to a stable ready pose between successful returns."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    racket_cfg: SceneEntityCfg = _DEFAULT_RACKET_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    target_x: float = 3.4,
    target_y: float = 0.0,
    target_heading: float = math.pi,
    base_pos_std: float = 0.6,
    heading_std: float = 0.7,
    lin_vel_std: float = 0.8,
    upright_std: float = 0.35,
    move_speed_scale: float = 1.0,
    racket_target_b: tuple[float, float, float] = (0.35, -0.35, 0.25),
    racket_std: float = 0.6,
    ready_pos_threshold: float = 0.45,
    ready_heading_threshold: float = 0.5,
    ready_speed_threshold: float = 0.45,
    ready_upright_threshold: float = 0.3,
    ready_racket_threshold: float = 0.65,
    **_: object,
  ) -> torch.Tensor:
    state = self.state
    active = state.in_recovery.float()
    if not torch.any(active > 0.0):
      return active

    reward_values, _ready_unused = _continuous_recovery_home_terms(
      env,
      racket_cfg=racket_cfg,
      robot_cfg=robot_cfg,
      target_x=target_x,
      target_y=target_y,
      target_heading=target_heading,
      base_pos_std=base_pos_std,
      heading_std=heading_std,
      lin_vel_std=lin_vel_std,
      upright_std=upright_std,
      move_speed_scale=move_speed_scale,
      racket_target_b=racket_target_b,
      racket_std=racket_std,
      ready_pos_threshold=ready_pos_threshold,
      ready_heading_threshold=ready_heading_threshold,
      ready_speed_threshold=ready_speed_threshold,
      ready_upright_threshold=ready_upright_threshold,
      ready_racket_threshold=ready_racket_threshold,
    )
    return reward_values * active


class continuous_recovery_ready_event(TennisContinuousBallStateTerm):
  """Sparse reward when the robot first reaches the ready pose in recovery."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    racket_cfg: SceneEntityCfg = _DEFAULT_RACKET_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    target_x: float = 3.4,
    target_y: float = 0.0,
    target_heading: float = math.pi,
    base_pos_std: float = 0.6,
    heading_std: float = 0.7,
    lin_vel_std: float = 0.8,
    upright_std: float = 0.35,
    move_speed_scale: float = 1.0,
    racket_target_b: tuple[float, float, float] = (0.35, -0.35, 0.25),
    racket_std: float = 0.6,
    ready_pos_threshold: float = 0.45,
    ready_heading_threshold: float = 0.5,
    ready_speed_threshold: float = 0.45,
    ready_upright_threshold: float = 0.3,
    ready_racket_threshold: float = 0.65,
    **_: object,
  ) -> torch.Tensor:
    state = self.state
    if not torch.any(state.in_recovery):
      return torch.zeros(env.num_envs, device=env.device)
    _reward_unused, ready = _continuous_recovery_home_terms(
      env,
      racket_cfg=racket_cfg,
      robot_cfg=robot_cfg,
      target_x=target_x,
      target_y=target_y,
      target_heading=target_heading,
      base_pos_std=base_pos_std,
      heading_std=heading_std,
      lin_vel_std=lin_vel_std,
      upright_std=upright_std,
      move_speed_scale=move_speed_scale,
      racket_target_b=racket_target_b,
      racket_std=racket_std,
      ready_pos_threshold=ready_pos_threshold,
      ready_heading_threshold=ready_heading_threshold,
      ready_speed_threshold=ready_speed_threshold,
      ready_upright_threshold=ready_upright_threshold,
      ready_racket_threshold=ready_racket_threshold,
    )
    return state.mark_recovery_ready(ready).float()


class advance_continuous_rally_ball(TennisContinuousBallStateTerm):
  """Start recovery and respawn opponent feeds without marking bad feeds failed."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    provider_cfg: OpponentFeederCfg,
    max_successful_returns: int = 8,
    recovery_time_range: tuple[float, float] = (3.0, 5.0),
    min_recovery_time: float = 1.0,
    racket_cfg: SceneEntityCfg = _DEFAULT_RACKET_CFG,
    robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
    target_x: float = 3.4,
    target_y: float = 0.0,
    target_heading: float = math.pi,
    base_pos_std: float = 0.6,
    heading_std: float = 0.7,
    lin_vel_std: float = 0.8,
    upright_std: float = 0.35,
    move_speed_scale: float = 1.0,
    racket_target_b: tuple[float, float, float] = (0.35, -0.35, 0.25),
    racket_std: float = 0.6,
    ready_pos_threshold: float = 0.45,
    ready_heading_threshold: float = 0.5,
    ready_speed_threshold: float = 0.45,
    ready_upright_threshold: float = 0.3,
    ready_racket_threshold: float = 0.65,
    **_: object,
  ) -> torch.Tensor:
    state = self.state
    state.respawn_edge[:] = False
    active_before = state.in_recovery.clone()
    should_start = (
      state.successful_return_edge
      & (state.successful_return_count < int(max_successful_returns))
      & (state.recovery_steps_left <= 0)
    )
    start_env_ids = should_start.nonzero(as_tuple=False).flatten()
    if start_env_ids.numel() > 0:
      state.start_recovery(start_env_ids, recovery_time_range, min_recovery_time)

    timed_out = state.step_recovery(active_before)
    _reward_unused, ready_pose = _continuous_recovery_home_terms(
      env,
      racket_cfg=racket_cfg,
      robot_cfg=robot_cfg,
      target_x=target_x,
      target_y=target_y,
      target_heading=target_heading,
      base_pos_std=base_pos_std,
      heading_std=heading_std,
      lin_vel_std=lin_vel_std,
      upright_std=upright_std,
      move_speed_scale=move_speed_scale,
      racket_target_b=racket_target_b,
      racket_std=racket_std,
      ready_pos_threshold=ready_pos_threshold,
      ready_heading_threshold=ready_heading_threshold,
      ready_speed_threshold=ready_speed_threshold,
      ready_upright_threshold=ready_upright_threshold,
      ready_racket_threshold=ready_racket_threshold,
    )
    can_respawn_ready = (
      active_before
      & state.in_recovery
      & (state.recovery_min_steps_left <= 0)
      & ready_pose
    )
    state.mark_recovery_ready(can_respawn_ready)
    respawn = can_respawn_ready | timed_out | state.invalid_feed_edge
    respawn_env_ids = respawn.nonzero(as_tuple=False).flatten()
    if respawn_env_ids.numel() > 0:
      spawn_ball_from_provider(env, respawn_env_ids, provider_cfg=provider_cfg)
      state.reset_ball(respawn_env_ids)
      state.respawn_edge[respawn_env_ids] = True
    return torch.zeros(env.num_envs, device=env.device)


class continuous_ball_fault(TennisContinuousBallStateTerm):
  """Terminate on robot-side rally faults, but not invalid opponent feeds."""

  def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(self, env: "ManagerBasedRlEnv", **_: object) -> torch.Tensor:
    del env
    return self.state.fault_edge


class continuous_rally_complete_state(TennisContinuousBallStateTerm):
  """Terminate when the configured number of successful returns is reached."""

  def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    max_successful_returns: int = 8,
    **_: object,
  ) -> torch.Tensor:
    del env
    return self.state.successful_return_count >= int(max_successful_returns)


def continuous_racket_hit_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_racket_hit_count.float()


def continuous_crossed_net_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_crossed_net_count.float()


def continuous_landing_in_bounds_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_landing_in_bounds_count.float()


def continuous_successful_return_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.successful_return_count.float()


def continuous_net_contact_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_net_contact_count.float()


def continuous_invalid_feed_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_invalid_feed_count.float()


def continuous_invalid_feed_net_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_invalid_feed_net_count.float()


def continuous_invalid_feed_out_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_invalid_feed_out_count.float()


def continuous_invalid_feed_opponent_bounce_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_invalid_feed_opponent_bounce_count.float()


def continuous_fault_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_count.float()


def continuous_fault_incoming_bounce_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_incoming_bounce_count.float()


def continuous_fault_return_bounce_out_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_return_bounce_out_count.float()


def continuous_fault_return_out_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_return_out_count.float()


def continuous_fault_net_contact_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_net_contact_count.float()


def continuous_fault_extra_racket_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_extra_racket_count.float()


def continuous_fault_low_net_cross_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_fault_low_net_cross_count.float()


def continuous_recovery_ready_count_metric(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.episode_recovery_ready_count.float()


def continuous_success_ratio_metric_state(
  env: "ManagerBasedRlEnv",
  max_successful_returns: int = 8,
  **params: object,
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  max_returns = max(1, int(max_successful_returns))
  return (state.successful_return_count.float() / float(max_returns)).clamp(max=1.0)


def continuous_in_recovery_metric_state(
  env: "ManagerBasedRlEnv", **params: object
) -> torch.Tensor:
  state = get_tennis_continuous_ball_state(env, **params)  # type: ignore[arg-type]
  state.update()
  return state.in_recovery.float()


__all__ = [
  "PHASE_INCOMING",
  "PHASE_RETURN_FLIGHT",
  "PHASE_RECOVERY",
  "FAULT_NONE",
  "FAULT_INCOMING_BOUNCE",
  "FAULT_RETURN_BOUNCE_OUT",
  "FAULT_RETURN_OUT_OF_PLAY",
  "FAULT_NET_CONTACT",
  "FAULT_EXTRA_RACKET_CONTACT",
  "FAULT_LOW_NET_CROSS",
  "OpponentFeeder",
  "OpponentFeederCfg",
  "TennisContinuousBallState",
  "TennisContinuousBallStateTerm",
  "advance_continuous_rally_ball",
  "continuous_ball_fault",
  "continuous_ball_phase",
  "continuous_crossed_net_count_metric",
  "continuous_crossed_net_event",
  "continuous_fault_count_metric",
  "continuous_fault_extra_racket_count_metric",
  "continuous_fault_incoming_bounce_count_metric",
  "continuous_fault_low_net_cross_count_metric",
  "continuous_fault_net_contact_count_metric",
  "continuous_fault_return_bounce_out_count_metric",
  "continuous_fault_return_out_count_metric",
  "continuous_in_recovery_metric_state",
  "continuous_invalid_feed_count_metric",
  "continuous_invalid_feed_net_count_metric",
  "continuous_invalid_feed_opponent_bounce_count_metric",
  "continuous_invalid_feed_out_count_metric",
  "continuous_landing_in_bounds_count_metric",
  "continuous_landing_in_bounds_event",
  "continuous_net_contact_count_metric",
  "continuous_post_hit_ball_velocity_direction",
  "continuous_post_hit_x_progress",
  "continuous_racket_hit_count_metric",
  "continuous_racket_hit_event",
  "continuous_racket_to_predicted_hit_point_dense",
  "continuous_racket_towards_ball_velocity",
  "continuous_rally_complete_state",
  "continuous_recovery_ready_count_metric",
  "continuous_recovery_ready_event",
  "continuous_recovery_ready_pose_state",
  "continuous_success_ratio_metric_state",
  "continuous_successful_return_count_metric",
  "get_tennis_continuous_ball_state",
]
