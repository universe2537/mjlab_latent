"""Unweighted table-tennis episode metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.state import get_pingpong_rally_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BALL_CFG = SceneEntityCfg("ball")


def _contact_substep_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.linalg.vector_norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  if data.force is not None:
    force_mag = torch.linalg.vector_norm(data.force, dim=-1)
    return (force_mag > force_threshold).any(dim=1).float()
  if data.found is not None:
    return (data.found > 0).any(dim=1).float()
  raise ValueError(f"Contact sensor {sensor_name!r} must expose force or found.")


def _state(env: ManagerBasedRlEnv, **params):
  state = get_pingpong_rally_state(env, **params)
  state.update()
  return state


def self_table_bounce_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  state = _state(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    **params,
  )
  return state.self_bounce_count.float()


def paddle_hit_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  state = _state(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    **params,
  )
  return state.paddle_hit_count.float()


def crossed_net_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  state = _state(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    **params,
  )
  return state.crossed_net_count.float()


def opponent_table_bounce_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  state = _state(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    **params,
  )
  return state.opponent_bounce_count.float()


def legal_return_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  state = _state(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    **params,
  )
  return state.successful_return_count.float()


def fault_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  **params,
) -> torch.Tensor:
  state = _state(
    env,
    paddle_sensor_name=paddle_sensor_name,
    net_sensor_name=net_sensor_name,
    ball_cfg=ball_cfg,
    **params,
  )
  return state.episode_fault_count.float()


def robot_table_contact_count_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 5.0,
) -> torch.Tensor:
  return _contact_substep_count(env, sensor_name, force_threshold)


__all__ = [
  "crossed_net_count_metric",
  "fault_count_metric",
  "legal_return_count_metric",
  "opponent_table_bounce_count_metric",
  "paddle_hit_count_metric",
  "robot_table_contact_count_metric",
  "self_table_bounce_count_metric",
]
