"""网球潜变量控制任务的奖励项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.ball_providers import (
  RandomFeederCfg,
  spawn_ball_from_provider,
)
from mjlab.tasks.tennis.mdp.hit_state import (
  TennisHitTrackerTerm,
  get_tennis_hit_tracker,
)
from mjlab.tasks.tennis.mdp.observations import (
  _predict_hit_intersection_w,
  racket_to_ball_b,
  racket_velocity_b,
)
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")


def racket_ball_distance_exp(
  env: ManagerBasedRlEnv,
  std: float,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """将球拍中心带近球的密集奖励。"""
  delta_b = racket_to_ball_b(env, racket_cfg, ball_cfg, robot_cfg)
  error = torch.sum(torch.square(delta_b), dim=-1)
  return torch.exp(-error / std**2)


def termination_terms_any(
  env: ManagerBasedRlEnv, term_names: tuple[str, ...]
) -> torch.Tensor:
  """如果本步任何指定终止项激活，则返回 1。"""
  if len(term_names) == 0:
    return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
  stacked = torch.stack(
    [env.termination_manager.get_term(name) for name in term_names], dim=0
  )
  return stacked.any(dim=0).float()


def low_level_action_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """惩罚解码后低层关节动作的变化。"""
  term = env.action_manager.get_term(action_name)
  action = getattr(term, "low_level_action", None)
  prev_action = getattr(term, "prev_low_level_action", None)
  if action is None or prev_action is None:
    raise ValueError(
      f"Action term {action_name!r} does not expose low-level action history."
    )
  return torch.sum(torch.square(action - prev_action), dim=1)


def wrist_residual_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """惩罚高层直接施加的右腕 residual 幅度。"""
  term = env.action_manager.get_term(action_name)
  action = getattr(term, "wrist_residual_action", None)
  if action is None:
    raise ValueError(
      f"Action term {action_name!r} does not expose wrist residual actions."
    )
  if action.shape[-1] == 0:
    return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
  return torch.sum(torch.square(action), dim=1)


def wrist_residual_rate_l2(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
  """惩罚高层右腕 residual 的帧间变化。"""
  term = env.action_manager.get_term(action_name)
  action = getattr(term, "wrist_residual_action", None)
  prev_action = getattr(term, "prev_wrist_residual_action", None)
  if action is None or prev_action is None:
    raise ValueError(
      f"Action term {action_name!r} does not expose wrist residual history."
    )
  if action.shape[-1] == 0:
    return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
  return torch.sum(torch.square(action - prev_action), dim=1)


# ---------------------------------------------------------------------------
# 击球任务奖励
# ---------------------------------------------------------------------------


def racket_to_ball_distance_dense(
  env: ManagerBasedRlEnv,
  std: float,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """常开的球拍到球距离密集奖励。

  该项在整个回合中始终开启，适用于只关心
  「接近 -> 击球 -> 越网 / 首次落地或再次碰拍」的简化击球任务。
  """
  return racket_ball_distance_exp(env, std, racket_cfg, ball_cfg, robot_cfg)


def racket_to_predicted_hit_point_dense(
  env: ManagerBasedRlEnv,
  std: float,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  hit_height_offset: float = 0.05,
  gravity: float = 9.81,
  max_horizon: float = 1.5,
  sensor_name: str | None = None,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
  landing_x_limits: tuple[float, float] | None = None,
  landing_y_limits: tuple[float, float] | None = None,
) -> torch.Tensor:
  """鼓励球拍接近预计击球点，而不是追逐球的当前位置。"""
  robot: Entity = env.scene[robot_cfg.name]
  racket_pos_w = robot.data.site_pos_w[:, racket_cfg.site_ids].squeeze(1)
  hit_w, _, valid = _predict_hit_intersection_w(
    env,
    ball_cfg,
    robot_cfg,
    hit_height_offset=hit_height_offset,
    gravity=gravity,
    max_horizon=max_horizon,
  )
  error = torch.sum(torch.square(hit_w - racket_pos_w), dim=-1)
  reward = torch.exp(-error / std**2)
  reward = reward * valid.float()
  if sensor_name is None:
    return reward
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
  active = ~tracker.has_racket_hit & ~tracker.in_recovery
  return reward * active.float()


class racket_towards_ball_velocity(TennisHitTrackerTerm):
  """鼓励球拍在线速度上朝向球移动，在首次击球前生效。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    racket_cfg: SceneEntityCfg = _RACKET_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    speed_scale: float = 2.0,
    distance_std: float = 0.8,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
  ) -> torch.Tensor:
    del sensor_name, force_threshold, ground_z, net_x
    del landing_x_limits, landing_y_limits
    tracker = self.tracker

    delta_b = racket_to_ball_b(env, racket_cfg, ball_cfg, robot_cfg)
    racket_vel_b = racket_velocity_b(env, racket_cfg, robot_cfg)
    distance = torch.linalg.vector_norm(delta_b, dim=-1).clamp_min(1e-6)
    direction_to_ball = delta_b / distance.unsqueeze(-1)

    toward_speed = torch.sum(racket_vel_b * direction_to_ball, dim=-1)
    toward_speed = torch.clamp(toward_speed, min=0.0)
    distance_weight = torch.exp(-(distance**2) / distance_std**2)
    reward = torch.tanh(toward_speed / speed_scale) * distance_weight

    return reward * (~tracker.has_racket_hit).float()


class racket_hit_event(TennisHitTrackerTerm):
  """首次球拍接触的稀疏一次性奖励。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
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
    # 仅奖励第一次球拍击球（本步计数递增）。
    return (tracker.racket_hit_edge & (tracker.racket_hit_count == 1)).float()


class post_hit_x_progress(TennisHitTrackerTerm):
  """击球后奖励球在己方半场内朝 -x 方向推进。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
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
    max_progress: float = 0.08,
  ) -> torch.Tensor:
    del sensor_name, force_threshold, ground_z
    del landing_x_limits, landing_y_limits
    tracker = self.tracker
    ball: Entity = env.scene[ball_cfg.name]
    ball_x = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = torch.clamp(tracker.prev_ball_x - ball_x, min=0.0, max=max_progress)
    reward = progress / max_progress
    active = tracker.has_racket_hit & (ball_x > net_x)
    return reward * active.float()


class post_hit_ball_velocity_direction(TennisHitTrackerTerm):
  """击球后奖励球朝对方半场飞行，并抑制过大的横向速度。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
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
    x_speed_scale: float = 4.0,
    lateral_speed_std: float = 1.5,
  ) -> torch.Tensor:
    del sensor_name, force_threshold, ground_z
    del landing_x_limits, landing_y_limits
    tracker = self.tracker
    ball: Entity = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w - env.scene.env_origins
    ball_vel = ball.data.root_link_lin_vel_w
    ball_x = ball_pos[:, 0]
    x_reward = torch.clamp(-ball_vel[:, 0] / x_speed_scale, min=0.0, max=1.0)
    lateral_weight = torch.exp(-(ball_vel[:, 1] ** 2) / lateral_speed_std**2)
    active = tracker.has_racket_hit & (ball_x > net_x)
    return x_reward * lateral_weight * active.float()


class crossed_net_event(TennisHitTrackerTerm):
  """球在击球后首次过网的稀疏一次性奖励。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
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
    return self.tracker.crossed_net_edge.float()


class landing_in_bounds_event(TennisHitTrackerTerm):
  """球击过网后首次落在目标界内的一次性奖励。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
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
    return self.tracker.landing_in_bounds_edge.float()


class respawn_successful_continuous_rally_ball(TennisHitTrackerTerm):
  """连续接球中成功回球后延迟重新发球，并开始下一小回合。"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    provider_cfg: RandomFeederCfg,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
    max_successful_returns: int = 8,
    recovery_steps: int = 40,
  ) -> torch.Tensor:
    del sensor_name, ball_cfg, force_threshold, ground_z, net_x
    del landing_x_limits, landing_y_limits
    tracker = self.tracker
    active_before = tracker.in_recovery.clone()
    should_start_recovery = (
      tracker.landing_in_bounds_edge
      & (tracker.successful_return_count < int(max_successful_returns))
      & ~active_before
    )
    start_env_ids = should_start_recovery.nonzero(as_tuple=False).flatten()
    if start_env_ids.numel() > 0:
      tracker.start_recovery(start_env_ids, recovery_steps=recovery_steps)

    should_respawn = tracker.step_recovery(active_before)
    respawn_env_ids = should_respawn.nonzero(as_tuple=False).flatten()
    if respawn_env_ids.numel() > 0:
      spawn_ball_from_provider(env, respawn_env_ids, provider_cfg=provider_cfg)
      tracker.reset_rally(respawn_env_ids)
    return torch.zeros(env.num_envs, device=env.device)


class continuous_recovery_ready_pose(TennisHitTrackerTerm):
  """Reward returning to a stable ready position between successful returns."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    racket_cfg: SceneEntityCfg = _RACKET_CFG,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
    landing_x_limits: tuple[float, float] | None = None,
    landing_y_limits: tuple[float, float] | None = None,
    target_x: float = 3.4,
    target_y: float = 0.0,
    target_heading: float = 3.141592653589793,
    base_pos_std: float = 0.6,
    heading_std: float = 0.7,
    lin_vel_std: float = 0.8,
    upright_std: float = 0.35,
    racket_target_b: tuple[float, float, float] = (0.35, -0.35, 0.25),
    racket_std: float = 0.6,
  ) -> torch.Tensor:
    del (
      sensor_name,
      ball_cfg,
      force_threshold,
      ground_z,
      net_x,
      landing_x_limits,
      landing_y_limits,
    )
    tracker = self.tracker
    active = tracker.in_recovery.float()
    if not torch.any(active > 0.0):
      return active

    robot: Entity = env.scene[robot_cfg.name]
    base_pos_l = robot.data.root_link_pos_w - env.scene.env_origins
    base_xy_error = torch.square(base_pos_l[:, 0] - target_x) + torch.square(
      base_pos_l[:, 1] - target_y
    )
    base_reward = torch.exp(-base_xy_error / base_pos_std**2)

    heading_error = wrap_to_pi(robot.data.heading_w - target_heading)
    heading_reward = torch.exp(-torch.square(heading_error) / heading_std**2)

    lin_vel_xy = torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1)
    still_reward = torch.exp(-lin_vel_xy / lin_vel_std**2)

    upright_error = torch.sum(
      torch.square(robot.data.projected_gravity_b[:, :2]), dim=1
    )
    upright_reward = torch.exp(-upright_error / upright_std**2)

    racket_pos_w = robot.data.site_pos_w[:, racket_cfg.site_ids].squeeze(1)
    racket_delta_w = racket_pos_w - robot.data.root_link_pos_w
    racket_pos_b = quat_apply_inverse(robot.data.root_link_quat_w, racket_delta_w)
    racket_target = torch.tensor(
      racket_target_b, dtype=torch.float32, device=env.device
    )
    racket_error = torch.sum(torch.square(racket_pos_b - racket_target), dim=1)
    racket_reward = torch.exp(-racket_error / racket_std**2)

    reward = (
      base_reward + heading_reward + still_reward + upright_reward + racket_reward
    ) / 5.0
    return reward * active
