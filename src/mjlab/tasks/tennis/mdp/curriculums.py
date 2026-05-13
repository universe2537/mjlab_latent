"""Curriculum scaffolding for the tennis return task.

These functions are intended to be wired into ``ManagerBasedRlEnvCfg.curriculum``
once the base policy reaches a non-trivial success rate. They read metrics
that ``RallyCommand`` accumulates (points won, valid hits, over-net rate)
and call into ``rally.provider.bump_difficulty(key)`` to widen sampling
ranges.

Each curriculum function returns either ``None`` or a small ``dict`` of
scalar metrics; the curriculum manager logs whatever is returned.
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
  """Widen the incoming ball-speed range when valid-hit rate is high enough.

  Reads ``rally.metrics["valid_hits"]``; if the recent mean exceeds
  ``success_threshold`` shots-per-episode, calls ``bump_difficulty``.
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
  """Widen the lateral spread of incoming balls."""
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
  """Tighten the opponent's flight time when the player is winning enough."""
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
