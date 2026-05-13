"""网球潜变量控制任务的观测项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")


def neutral_motion_anchor_pos_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """返回与追踪解码器状态兼容的中性根目标位置。"""
  return torch.zeros(env.num_envs, 3, device=env.device)


def neutral_motion_anchor_ori_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  """返回 6D 旋转表示下的中性根目标朝向。"""
  ori = torch.tensor((1.0, 0.0, 0.0, 1.0, 0.0, 0.0), device=env.device)
  return ori.repeat(env.num_envs, 1)


def low_level_action(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """返回潜变量动作项解码后的低层动作。"""
  term = env.action_manager.get_term(action_name)
  action = getattr(term, "low_level_action", None)
  if action is None:
    raise ValueError(f"Action term {action_name!r} does not expose low_level_action.")
  return action


def racket_to_ball_b(
  env: ManagerBasedRlEnv,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """从球拍中心到球的向量，以机器人基座坐标系表示。"""
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  racket_pos_w = robot.data.site_pos_w[:, racket_cfg.site_ids].squeeze(1)
  delta_w = ball.data.root_link_pos_w - racket_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, delta_w)


def ball_velocity_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """球的线速度，以机器人基座坐标系表示。"""
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  return quat_apply_inverse(robot.data.root_link_quat_w, ball.data.root_link_lin_vel_w)


def racket_velocity_b(
  env: ManagerBasedRlEnv,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """球拍中心的线速度，以机器人基座坐标系表示。"""
  robot: Entity = env.scene[robot_cfg.name]
  racket_vel_w = robot.data.site_lin_vel_w[:, racket_cfg.site_ids].squeeze(1)
  return quat_apply_inverse(robot.data.root_link_quat_w, racket_vel_w)


def ball_predicted_landing_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  ground_z: float = 0.06,
  gravity: float = 9.81,
  max_horizon: float = 1.5,
) -> torch.Tensor:
  """预测球在简单重力下的落点 (x, y, t_to_land)，以机器人基座坐标系表示。

  使用弹道前向积分，忽略空气阻力和弹跳。对于球已低于 ``ground_z`` 的
  环境，返回零向量。
  """
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  pos = ball.data.root_link_pos_w  # (B, 3)
  vel = ball.data.root_link_lin_vel_w  # (B, 3)

  pz = pos[:, 2]
  vz = vel[:, 2]
  # 求解 pz + vz*t - 0.5*g*t^2 = ground_z  =>  0.5g t^2 - vz t - (pz - gz) = 0
  a = 0.5 * gravity
  b = -vz
  c = -(pz - ground_z)
  disc = torch.clamp(b * b - 4.0 * a * c, min=0.0)
  t = (-b + torch.sqrt(disc)) / (2.0 * a)
  t = torch.clamp(t, min=0.0, max=max_horizon)

  landing = pos.clone()
  landing[:, 0] = pos[:, 0] + vel[:, 0] * t
  landing[:, 1] = pos[:, 1] + vel[:, 1] * t
  landing[:, 2] = ground_z

  delta_w = landing - robot.data.root_link_pos_w
  delta_b = quat_apply_inverse(robot.data.root_link_quat_w, delta_w)
  return torch.cat([delta_b[:, :2], t.unsqueeze(-1)], dim=-1)
