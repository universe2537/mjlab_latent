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
