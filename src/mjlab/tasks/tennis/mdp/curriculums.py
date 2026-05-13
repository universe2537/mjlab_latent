"""网球回球任务的课程脂手架。

这些函数旨在基础策略达到一定成功率后，接入到
``ManagerBasedRlEnvCfg.curriculum``。它们读取 ``RallyCommand`` 累积的指标
（得分数、有效击球次数、越网率），并调用
``rally.provider.bump_difficulty(key)`` 来拓宽采样范围。

每个课程函数返回 ``None`` 或小型 ``dict`` 标量；课程管理器将记录返回内容。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _get_rally(env: "ManagerBasedRlEnv", command_name: str):
  return env.command_manager.get_term(command_name)


def ball_speed_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,  # noqa: ARG001
  *,
  command_name: str = "rally",
  success_threshold: float = 0.5,
  bump_size: float = 0.05,
) -> dict[str, float]:
  """有效击球率足够高时拓宽入球速度范围。

  读取 ``rally.metrics["valid_hits"]``；若近期均値超过
  ``success_threshold`` 次/回合，则调用 ``bump_difficulty``。
  """
  rally = _get_rally(env, command_name)
  vh_mean = float(rally.metrics["valid_hits"].mean().item())
  if vh_mean >= success_threshold:
    rally.provider.bump_difficulty("ball_speed", bump_size)
  return {"ball_speed_difficulty": rally.provider.difficulty}


def ball_angle_spread_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,  # noqa: ARG001
  *,
  command_name: str = "rally",
  success_threshold: float = 0.5,
  bump_size: float = 0.05,
) -> dict[str, float]:
  """有效击球率足够高时拓宽入球横向分布。"""
  rally = _get_rally(env, command_name)
  vh_mean = float(rally.metrics["valid_hits"].mean().item())
  if vh_mean >= success_threshold:
    rally.provider.bump_difficulty("ball_lateral", bump_size)
  return {"ball_lateral_difficulty": rally.provider.difficulty}


def opponent_level_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,  # noqa: ARG001
  *,
  command_name: str = "rally",
  win_threshold: float = 0.6,
  bump_size: float = 0.05,
) -> dict[str, float]:
  """玩家得分率足够高时压缩对手的飞行时间。"""
  rally = _get_rally(env, command_name)
  pw_mean = float(rally.metrics["points_won"].mean().item())
  if pw_mean >= win_threshold:
    rally.provider.bump_difficulty("opponent_level", bump_size)
  return {"opponent_difficulty": rally.provider.difficulty}


__all__ = [
  "ball_speed_curriculum",
  "ball_angle_spread_curriculum",
  "opponent_level_curriculum",
]
