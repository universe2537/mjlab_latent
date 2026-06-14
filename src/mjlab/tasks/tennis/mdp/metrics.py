"""Unweighted tennis task metrics for interpreting sparse events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.hit_state import get_tennis_hit_tracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BALL_CFG = SceneEntityCfg("ball")


def racket_hit_count_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Unweighted count of racket-hit edges in the current episode."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.episode_racket_hit_count.float()


def crossed_net_count_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Unweighted count of post-hit net crossings in the current episode."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.episode_crossed_net_count.float()


def landing_in_bounds_count_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Unweighted count of in-bounds opponent-court landings in the episode."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.episode_landing_in_bounds_count.float()


def successful_return_count_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Current episode successful return count, preserved across continuous rallies."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.successful_return_count.float()


def first_bounce_after_hit_count_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Count first post-hit bounces after the return has crossed the net."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.episode_first_bounce_after_hit_count.float()


def fast_landing_reward_mean_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Mean raw low-arc quality score over rewarded post-hit landings."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  count = tracker.episode_fast_landing_reward_count.clamp_min(1).float()
  return tracker.episode_fast_landing_score_sum / count


def time_to_landing_mean_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Mean hit-to-first-bounce time for post-hit over-net returns."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  count = tracker.episode_time_to_landing_count.clamp_min(1).float()
  return tracker.episode_time_to_landing_sum / count


def time_to_landing_min_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Minimum hit-to-first-bounce time, or zero if no sample exists."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  has_sample = tracker.episode_time_to_landing_count > 0
  return torch.where(
    has_sample,
    tracker.episode_time_to_landing_min,
    torch.zeros_like(tracker.episode_time_to_landing_min),
  )


def time_to_landing_max_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Maximum hit-to-first-bounce time for post-hit over-net returns."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.episode_time_to_landing_max


def time_to_landing_valid_count_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Number of valid post-hit over-net first-bounce time samples."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.episode_time_to_landing_count.float()


def continuous_success_ratio_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
  max_successful_returns: int = 8,
) -> torch.Tensor:
  """Fraction of the configured rally target completed by episode end."""
  returns = successful_return_count_metric(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  return (returns / float(max(1, int(max_successful_returns)))).clamp(max=1.0)


def in_recovery_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Per-step indicator used as an episode-average recovery occupancy metric."""
  tracker = get_tennis_hit_tracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
    landing_x_limits=landing_x_limits,
    landing_y_limits=landing_y_limits,
  )
  tracker.update()
  return tracker.in_recovery.float()
