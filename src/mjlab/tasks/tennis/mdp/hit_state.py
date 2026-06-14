"""网球击球任务的共享击球追踪状态。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_BALL_CFG = SceneEntityCfg("ball")
_HIT_TRACKER_ATTR = "_tennis_hit_tracker"


class TennisHitTracker:
  """
  按环境追踪简化击球任务需要的接触与越网事件。

  该追踪器只维护当前 hit 任务真正使用的信号：
  第一次球拍击球、地面弹跳、击球后越网，以及总接触计数。
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str,
    ball_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
  ) -> None:
    self._env = env
    self.sensor_name = sensor_name
    self.ball_cfg = ball_cfg
    self.force_threshold = force_threshold
    self.ground_z = ground_z
    self.net_x = net_x
    self.landing_x_limits = landing_x_limits
    self.landing_y_limits = landing_y_limits
    self._last_step = -1

    num_envs = env.num_envs
    device = env.device

    def zeros_long() -> torch.Tensor:
      return torch.zeros(num_envs, dtype=torch.long, device=device)

    def zeros_bool() -> torch.Tensor:
      return torch.zeros(num_envs, dtype=torch.bool, device=device)

    def zeros_float() -> torch.Tensor:
      return torch.zeros(num_envs, device=device)

    self.racket_hit_count = zeros_long()
    self.bounce_count = zeros_long()
    self.successful_return_count = zeros_long()
    self.episode_racket_hit_count = zeros_long()
    self.episode_bounce_count = zeros_long()
    self.episode_crossed_net_count = zeros_long()
    self.episode_landing_in_bounds_count = zeros_long()
    self.episode_first_bounce_after_hit_count = zeros_long()
    self.episode_fast_landing_reward_count = zeros_long()
    self.episode_time_to_landing_count = zeros_long()
    self.episode_time_to_landing_sum = zeros_float()
    self.episode_time_to_landing_min = torch.full((num_envs,), torch.inf, device=device)
    self.episode_time_to_landing_max = zeros_float()
    self.episode_fast_landing_score_sum = zeros_float()

    self.racket_hit_edge = zeros_bool()
    self.bounce_edge = zeros_bool()
    self.crossed_net_edge = zeros_bool()
    self.landing_in_bounds_edge = zeros_bool()
    self.first_bounce_after_hit_edge = zeros_bool()

    self.has_racket_hit = zeros_bool()
    self.has_crossed_net = zeros_bool()
    self.has_landed_in_bounds = zeros_bool()
    self.has_first_bounce_after_hit = zeros_bool()
    self.has_rewarded_fast_landing = zeros_bool()

    self.recovery_steps_left = zeros_long()
    self.recovery_steps_total = zeros_long()
    self.hit_step = zeros_long()
    self.first_bounce_after_hit_step = zeros_long()
    self.time_to_landing = zeros_float()
    self.last_fast_landing_score = zeros_float()

    self._prev_contact = zeros_bool()
    self._prev_vz = zeros_float()
    self._prev_x = zeros_float()
    self.prev_ball_x = zeros_float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.successful_return_count[env_ids] = 0
    self.episode_racket_hit_count[env_ids] = 0
    self.episode_bounce_count[env_ids] = 0
    self.episode_crossed_net_count[env_ids] = 0
    self.episode_landing_in_bounds_count[env_ids] = 0
    self.episode_first_bounce_after_hit_count[env_ids] = 0
    self.episode_fast_landing_reward_count[env_ids] = 0
    self.episode_time_to_landing_count[env_ids] = 0
    self.episode_time_to_landing_sum[env_ids] = 0.0
    self.episode_time_to_landing_min[env_ids] = torch.inf
    self.episode_time_to_landing_max[env_ids] = 0.0
    self.episode_fast_landing_score_sum[env_ids] = 0.0
    self.reset_rally(env_ids)
    self._last_step = -1

  def reset_rally(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset per-ball contact state while keeping episode-level rally counts."""
    if env_ids is None:
      env_ids = slice(None)
    self.racket_hit_count[env_ids] = 0
    self.bounce_count[env_ids] = 0
    self.racket_hit_edge[env_ids] = False
    self.bounce_edge[env_ids] = False
    self.crossed_net_edge[env_ids] = False
    self.landing_in_bounds_edge[env_ids] = False
    self.first_bounce_after_hit_edge[env_ids] = False
    self.has_racket_hit[env_ids] = False
    self.has_crossed_net[env_ids] = False
    self.has_landed_in_bounds[env_ids] = False
    self.has_first_bounce_after_hit[env_ids] = False
    self.has_rewarded_fast_landing[env_ids] = False
    self.recovery_steps_left[env_ids] = 0
    self.recovery_steps_total[env_ids] = 0
    self.hit_step[env_ids] = 0
    self.first_bounce_after_hit_step[env_ids] = 0
    self.time_to_landing[env_ids] = 0.0
    self.last_fast_landing_score[env_ids] = 0.0
    self._prev_contact[env_ids] = False
    self._prev_vz[env_ids] = 0.0
    self._prev_x[env_ids] = 0.0
    self.prev_ball_x[env_ids] = 0.0

  def update(self) -> None:
    step = int(self._env.common_step_counter)
    if step == self._last_step:
      return

    sensor: ContactSensor = self._env.scene[self.sensor_name]
    sensor_data = sensor.data
    if sensor_data.force_history is not None:
      force_mag = torch.linalg.vector_norm(sensor_data.force_history, dim=-1)
      contact_now = (force_mag > self.force_threshold).any(dim=(1, 2))
    elif sensor_data.force is not None:
      force_mag = torch.linalg.vector_norm(sensor_data.force, dim=-1)
      contact_now = (force_mag > self.force_threshold).any(dim=1)
    elif sensor_data.found is not None:
      contact_now = (sensor_data.found > 0).any(dim=1)
    else:
      raise ValueError(
        f"Contact sensor {self.sensor_name!r} must expose 'force' or 'found'."
      )

    ball: Entity = self._env.scene[self.ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w - self._env.scene.env_origins
    ball_vel = ball.data.root_link_lin_vel_w
    ball_x = ball_pos[:, 0]
    ball_y = ball_pos[:, 1]
    ball_z = ball_pos[:, 2]
    ball_vz = ball_vel[:, 2]
    prev_x = self._prev_x.clone()

    racket_hit_edge = contact_now & ~self._prev_contact
    bounce_edge = (
      (self._prev_vz < 0.0) & (ball_vz >= 0.0) & (ball_z < self.ground_z + 0.05)
    )

    has_hit_now = self.has_racket_hit | racket_hit_edge
    crossed_to_opp = (self._prev_x > self.net_x) & (ball_x <= self.net_x)
    crossed_net_edge = crossed_to_opp & has_hit_now & ~self.has_crossed_net
    has_crossed_now = self.has_crossed_net | crossed_net_edge
    landing_in_bounds = torch.ones_like(bounce_edge)
    if self.landing_x_limits is not None:
      landing_in_bounds &= (ball_x >= self.landing_x_limits[0]) & (
        ball_x <= self.landing_x_limits[1]
      )
    if self.landing_y_limits is not None:
      landing_in_bounds &= (ball_y >= self.landing_y_limits[0]) & (
        ball_y <= self.landing_y_limits[1]
      )
    landing_in_bounds_edge = (
      bounce_edge
      & has_hit_now
      & has_crossed_now
      & landing_in_bounds
      & ~self.has_landed_in_bounds
    )
    first_bounce_after_hit_edge = (
      bounce_edge & has_hit_now & has_crossed_now & ~self.has_first_bounce_after_hit
    )
    time_to_landing = torch.clamp(
      torch.full_like(self.hit_step, step) - self.hit_step,
      min=0,
    ).float() * float(self._env.step_dt)

    self.racket_hit_edge[:] = racket_hit_edge
    self.bounce_edge[:] = bounce_edge
    self.crossed_net_edge[:] = crossed_net_edge
    self.landing_in_bounds_edge[:] = landing_in_bounds_edge
    self.first_bounce_after_hit_edge[:] = first_bounce_after_hit_edge
    self.racket_hit_count += racket_hit_edge.long()
    self.bounce_count += bounce_edge.long()
    self.successful_return_count += landing_in_bounds_edge.long()
    self.episode_racket_hit_count += racket_hit_edge.long()
    self.episode_bounce_count += bounce_edge.long()
    self.episode_crossed_net_count += crossed_net_edge.long()
    self.episode_landing_in_bounds_count += landing_in_bounds_edge.long()
    self.episode_first_bounce_after_hit_count += first_bounce_after_hit_edge.long()
    self.has_racket_hit |= racket_hit_edge
    self.has_crossed_net |= crossed_net_edge
    self.has_landed_in_bounds |= landing_in_bounds_edge
    self.has_first_bounce_after_hit |= first_bounce_after_hit_edge
    self.hit_step[racket_hit_edge] = step
    self.first_bounce_after_hit_step[first_bounce_after_hit_edge] = step
    self.time_to_landing[first_bounce_after_hit_edge] = time_to_landing[
      first_bounce_after_hit_edge
    ]
    self.episode_time_to_landing_count += first_bounce_after_hit_edge.long()
    self.episode_time_to_landing_sum += torch.where(
      first_bounce_after_hit_edge, time_to_landing, torch.zeros_like(time_to_landing)
    )
    self.episode_time_to_landing_min = torch.where(
      first_bounce_after_hit_edge,
      torch.minimum(self.episode_time_to_landing_min, time_to_landing),
      self.episode_time_to_landing_min,
    )
    self.episode_time_to_landing_max = torch.where(
      first_bounce_after_hit_edge,
      torch.maximum(self.episode_time_to_landing_max, time_to_landing),
      self.episode_time_to_landing_max,
    )

    self._prev_contact[:] = contact_now
    self._prev_vz[:] = ball_vz
    self.prev_ball_x[:] = prev_x
    self._prev_x[:] = ball_x
    self._last_step = step

  @property
  def total_contact_count(self) -> torch.Tensor:
    """返回本回合累计的球拍击球与地面弹跳次数。"""
    return self.racket_hit_count + self.bounce_count

  @property
  def in_recovery(self) -> torch.Tensor:
    """Whether each env is between successful returns waiting for the next feed."""
    return self.recovery_steps_left > 0

  @property
  def recovery_fraction_remaining(self) -> torch.Tensor:
    """Recovery timer normalized to ``[0, 1]`` for policy observations."""
    total = self.recovery_steps_total.clamp_min(1).float()
    return self.recovery_steps_left.float() / total

  def start_recovery(self, env_ids: torch.Tensor, recovery_steps: int) -> None:
    """Start a between-ball recovery window for selected environments."""
    if env_ids.numel() == 0:
      return
    steps = max(1, int(recovery_steps))
    self.recovery_steps_left[env_ids] = steps
    self.recovery_steps_total[env_ids] = steps

  def step_recovery(self, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Advance active recovery timers and return envs that are ready to respawn."""
    if active_mask is None:
      active_mask = self.in_recovery
    active_mask = active_mask & self.in_recovery
    if torch.any(active_mask):
      self.recovery_steps_left[active_mask] -= 1
    ready = active_mask & (self.recovery_steps_left <= 0)
    self.recovery_steps_left[ready] = 0
    return ready

  def fast_landing_score(
    self,
    *,
    t_min: float,
    t_max: float,
    require_in_bounds: bool = True,
  ) -> torch.Tensor:
    """Score short post-hit flight time once the returned ball lands.

    The hit-to-first-landing time is mostly vertical motion:
    z(t) = z0 + vz0 * t - 0.5 * g * t^2.  This score rewards low-arc returns
    after a valid hit and net crossing; it is not a horizontal landing-depth
    reward.
    """
    if t_max <= t_min:
      raise ValueError("t_max must be greater than t_min.")
    landing_edge = (
      self.landing_in_bounds_edge
      if require_in_bounds
      else self.first_bounce_after_hit_edge
    )
    active = landing_edge & ~self.has_rewarded_fast_landing
    score = (float(t_max) - self.time_to_landing) / (float(t_max) - float(t_min))
    score = torch.clamp(score, min=0.0, max=1.0)
    score = torch.where(active, score, torch.zeros_like(score))
    if torch.any(active):
      self.has_rewarded_fast_landing |= active
      self.last_fast_landing_score[active] = score[active]
      self.episode_fast_landing_reward_count += active.long()
      self.episode_fast_landing_score_sum += score
    return score


def get_tennis_hit_tracker(
  env: ManagerBasedRlEnv,
  *,
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> TennisHitTracker:
  tracker = getattr(env, _HIT_TRACKER_ATTR, None)
  if isinstance(tracker, TennisHitTracker):
    return tracker
  tracker = TennisHitTracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  setattr(env, _HIT_TRACKER_ATTR, tracker)
  return tracker


class TennisHitTrackerTerm:
  """依赖 TennisHitTracker 的奖励/终止项的混入类。"""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._tracker = get_tennis_hit_tracker(
      env,
      sensor_name=cfg.params["sensor_name"],
      ball_cfg=cfg.params.get("ball_cfg", _DEFAULT_BALL_CFG),
      force_threshold=float(cfg.params.get("force_threshold", 1.0)),
      ground_z=float(cfg.params.get("ground_z", 0.06)),
      net_x=float(cfg.params.get("net_x", 0.0)),
      landing_x_limits=cfg.params.get("landing_x_limits"),
      landing_y_limits=cfg.params.get("landing_y_limits"),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._tracker.reset(env_ids)

  @property
  def tracker(self) -> TennisHitTracker:
    self._tracker.update()
    return self._tracker
