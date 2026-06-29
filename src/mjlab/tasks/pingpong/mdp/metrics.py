"""Unweighted table-tennis episode metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.state import (
  FAULT_EXTRA_PADDLE_CONTACT,
  FAULT_ILLEGAL_BODY_BALL_CONTACT,
  FAULT_ILLEGAL_PRE_BOUNCE_HIT,
  FAULT_LOW_NET_CROSS,
  FAULT_NET_CONTACT,
  FAULT_RETURN_BOUNCE_OUT,
  FAULT_RETURN_OUT_OF_PLAY,
  get_pingpong_rally_state,
)

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


def robot_ball_contact_count_metric(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.05,
) -> torch.Tensor:
  return _contact_substep_count(env, sensor_name, force_threshold)


def fault_reason_count_metric(
  env: ManagerBasedRlEnv,
  paddle_sensor_name: str,
  net_sensor_name: str,
  reason: int,
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
  return (state.fault_reason == reason).float()


def fault_reason_body_ball_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(
    env, reason=FAULT_ILLEGAL_BODY_BALL_CONTACT, **params
  )


def fault_reason_low_net_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(env, reason=FAULT_LOW_NET_CROSS, **params)


def fault_reason_net_contact_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(env, reason=FAULT_NET_CONTACT, **params)


def fault_reason_return_out_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(env, reason=FAULT_RETURN_OUT_OF_PLAY, **params)


def fault_reason_failed_bounce_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(env, reason=FAULT_RETURN_BOUNCE_OUT, **params)


def fault_reason_double_paddle_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(env, reason=FAULT_EXTRA_PADDLE_CONTACT, **params)


def fault_reason_early_hit_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return fault_reason_count_metric(env, reason=FAULT_ILLEGAL_PRE_BOUNCE_HIT, **params)


def hit_post_vx_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_post_vel[:, 0]


def hit_post_vy_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_post_vel[:, 1]


def hit_post_vz_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_post_vel[:, 2]


def hit_post_speed_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_post_speed


def hit_post_vx_toward_opponent_ratio_metric(
  env: ManagerBasedRlEnv, **params
) -> torch.Tensor:
  return _state(env, **params).hit_post_vx_toward_opponent_ratio


def hit_pred_net_clearance_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_pred_net_clearance


def hit_pred_net_clearance_positive_metric(
  env: ManagerBasedRlEnv, **params
) -> torch.Tensor:
  return _state(env, **params).hit_pred_net_clearance_positive


def hit_pred_landing_x_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_pred_landing_x


def hit_pred_landing_y_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_pred_landing_y


def hit_pred_landing_inside_opponent_table_metric(
  env: ManagerBasedRlEnv, **params
) -> torch.Tensor:
  return _state(env, **params).hit_pred_landing_inside_opponent_table


def hit_paddle_speed_metric(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
  return _state(env, **params).hit_paddle_speed


def hit_paddle_normal_alignment_metric(
  env: ManagerBasedRlEnv, **params
) -> torch.Tensor:
  return _state(env, **params).hit_paddle_normal_alignment


def hit_paddle_velocity_along_normal_metric(
  env: ManagerBasedRlEnv, **params
) -> torch.Tensor:
  return _state(env, **params).hit_paddle_velocity_along_normal


__all__ = [
  "crossed_net_count_metric",
  "fault_count_metric",
  "fault_reason_body_ball_metric",
  "fault_reason_count_metric",
  "fault_reason_double_paddle_metric",
  "fault_reason_early_hit_metric",
  "fault_reason_failed_bounce_metric",
  "fault_reason_low_net_metric",
  "fault_reason_net_contact_metric",
  "fault_reason_return_out_metric",
  "hit_paddle_normal_alignment_metric",
  "hit_paddle_speed_metric",
  "hit_paddle_velocity_along_normal_metric",
  "hit_post_speed_metric",
  "hit_post_vx_metric",
  "hit_post_vx_toward_opponent_ratio_metric",
  "hit_post_vy_metric",
  "hit_post_vz_metric",
  "hit_pred_landing_inside_opponent_table_metric",
  "hit_pred_landing_x_metric",
  "hit_pred_landing_y_metric",
  "hit_pred_net_clearance_metric",
  "hit_pred_net_clearance_positive_metric",
  "legal_return_count_metric",
  "opponent_table_bounce_count_metric",
  "paddle_hit_count_metric",
  "robot_ball_contact_count_metric",
  "robot_table_contact_count_metric",
  "self_table_bounce_count_metric",
]
