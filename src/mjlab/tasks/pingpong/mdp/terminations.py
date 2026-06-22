"""Termination terms for table-tennis tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.pingpong.mdp.state import PingpongRallyStateTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BALL_CFG = SceneEntityCfg("ball")


class first_paddle_hit(PingpongRallyStateTerm):
  """End a hit task when the first legal paddle contact occurs."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.paddle_hit_edge


class legal_return_success(PingpongRallyStateTerm):
  """End a return task when the ball legally lands on the opponent table."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.successful_return_edge


class pingpong_ball_fault(PingpongRallyStateTerm):
  """Terminate on a table-tennis rally fault."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.fault_edge


__all__ = [
  "first_paddle_hit",
  "legal_return_success",
  "pingpong_ball_fault",
]
