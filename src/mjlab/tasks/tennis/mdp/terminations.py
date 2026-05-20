"""网球潜变量控制任务的终止项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tennis.mdp.hit_state import TennisHitTrackerTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BALL_CFG = SceneEntityCfg("ball")
BALL_MIN_HEIGHT = 0.05


def ball_in_play(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  x_limits: tuple[float, float] = (-5.8, 3.6),
  y_limits: tuple[float, float] = (-2.7, 2.7),
  z_limits: tuple[float, float] = (BALL_MIN_HEIGHT, 2.6),
) -> torch.Tensor:
  """当球离开球场工作区间时终止。"""
  ball: Entity = env.scene[ball_cfg.name]
  pos = ball.data.root_link_pos_w - env.scene.env_origins
  out_x = (pos[:, 0] < x_limits[0]) | (pos[:, 0] > x_limits[1])
  out_y = (pos[:, 1] < y_limits[0]) | (pos[:, 1] > y_limits[1])
  out_z = (pos[:, 2] < z_limits[0]) | (pos[:, 2] > z_limits[1])
  return out_x | out_y | out_z


#
# 简化击球任务终止项（基于 TennisHitTracker）。
#


class first_racket_hit(TennisHitTrackerTerm):
  """在首次有效球拍接触的步骤结束回合。"""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    del env, sensor_name, ball_cfg, force_threshold, ground_z, net_x
    del landing_x_limits, landing_y_limits
    tracker = self.tracker
    return tracker.racket_hit_edge & (tracker.racket_hit_count == 1)


class second_contact(TennisHitTrackerTerm):
  """当球首次落地或发生额外接触后结束回合。

  对于当前 contact 任务，该项主要作为失败兜底：
  若机器人未能在空中截击，球首次落地即结束；若出现额外接触，
  也视为当前回合结束。
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    del env, sensor_name, ball_cfg, force_threshold, ground_z, net_x
    del landing_x_limits, landing_y_limits
    tracker = self.tracker
    return (tracker.bounce_count >= 1) | (tracker.racket_hit_count >= 2)


class crossed_net_after_hit(TennisHitTrackerTerm):
  """在击球后球首次过网的步骤结束回合。"""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    del env, sensor_name, ball_cfg, force_threshold, ground_z, net_x
    del landing_x_limits, landing_y_limits
    return self.tracker.crossed_net_edge


class landing_in_bounds_after_hit(TennisHitTrackerTerm):
  """击球过网后首次落在对方界内时标记成功。"""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    del (
      env,
      sensor_name,
      ball_cfg,
      force_threshold,
      ground_z,
      net_x,
      landing_x_limits,
      landing_y_limits,
    )
    return self.tracker.landing_in_bounds_edge
