"""网球潜变量控制任务的观测项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.hit_state import get_tennis_hit_tracker
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")
_DEFAULT_HIT_HEIGHT_OFFSET = 0.05


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


def torso_to_ball_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """从机器人 torso/base 到球的向量，以机器人基座坐标系表示。"""
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  delta_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
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


def _predict_hit_intersection_w(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  *,
  hit_height_offset: float = _DEFAULT_HIT_HEIGHT_OFFSET,
  gravity: float = 9.81,
  max_horizon: float = 1.5,
  min_time: float = 1.0e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """预测球与腰部高度击球平面的未来交点。

  击球平面高度采用 ``robot.root_link_pos_w[:, 2] + hit_height_offset``，
  默认比 G1 pelvis 高约 5 cm，接近腰部 / 下胸位置。

  返回:
    hit_w: 世界系击球点，shape ``(B, 3)``
    t_hit: 到达该点的时间，shape ``(B,)``
    valid: 是否存在未来正时间交点且在 ``max_horizon`` 内，shape ``(B,)``
  """
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  pos = ball.data.root_link_pos_w
  vel = ball.data.root_link_lin_vel_w

  hit_height = robot.data.root_link_pos_w[:, 2] + hit_height_offset
  a = -0.5 * gravity
  b = vel[:, 2]
  c = pos[:, 2] - hit_height

  disc = b * b - 4.0 * a * c
  has_real_root = disc >= 0.0
  sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
  denom = 2.0 * a
  t0 = (-b - sqrt_disc) / denom
  t1 = (-b + sqrt_disc) / denom
  candidates = torch.stack([t0, t1], dim=-1)
  future_candidates = candidates > min_time
  vz_at_candidates = vel[:, 2].unsqueeze(-1) - gravity * candidates

  inf = torch.full_like(candidates, float("inf"))
  descending_candidates = torch.where(
    future_candidates & (vz_at_candidates <= 0.0), candidates, inf
  )
  fallback_candidates = torch.where(future_candidates, candidates, inf)
  t_descending = descending_candidates.amin(dim=-1)
  t_fallback = fallback_candidates.amin(dim=-1)
  t_hit = torch.where(torch.isfinite(t_descending), t_descending, t_fallback)

  valid = has_real_root & torch.isfinite(t_hit) & (t_hit <= max_horizon)
  t_hit = torch.where(valid, t_hit, torch.zeros_like(t_hit))

  hit_w = pos + vel * t_hit.unsqueeze(-1)
  hit_w[:, 2] = hit_height
  return hit_w, t_hit, valid


def ball_predicted_hit_point_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  hit_height_offset: float = _DEFAULT_HIT_HEIGHT_OFFSET,
  gravity: float = 9.81,
  max_horizon: float = 1.5,
) -> torch.Tensor:
  """预测腰部高度的击球点，返回 ``(x, y, z, time_to_hit)``，以机器人基座坐标系表示。"""
  robot: Entity = env.scene[robot_cfg.name]
  hit_w, t_hit, valid = _predict_hit_intersection_w(
    env,
    ball_cfg,
    robot_cfg,
    hit_height_offset=hit_height_offset,
    gravity=gravity,
    max_horizon=max_horizon,
  )
  delta_w = hit_w - robot.data.root_link_pos_w
  hit_b = quat_apply_inverse(robot.data.root_link_quat_w, delta_w)
  hit_b = torch.where(valid.unsqueeze(-1), hit_b, torch.zeros_like(hit_b))
  return torch.cat([hit_b, t_hit.unsqueeze(-1)], dim=-1)


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


def continuous_rally_phase(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Expose the legacy continuous-rally recovery flag."""
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
  in_recovery = tracker.in_recovery.float()
  return in_recovery.unsqueeze(-1)
