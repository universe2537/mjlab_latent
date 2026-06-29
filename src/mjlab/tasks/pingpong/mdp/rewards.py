"""Reward terms for table-tennis latent-control tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.rewards import action_rate_l2
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.state import PingpongRallyStateTerm
from mjlab.tasks.tennis.mdp.observations import racket_to_ball_b, racket_velocity_b
from mjlab.tasks.tennis.mdp.rewards import low_level_action_rate_l2

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_PADDLE_CFG = SceneEntityCfg("robot", site_names=("pingpong_paddle_center",))
_BALL_CFG = SceneEntityCfg("ball")


def _contact_substep_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  max_count: float | None = None,
) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.linalg.vector_norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    count = hit.sum(dim=-1).float()
  elif data.force is not None:
    force_mag = torch.linalg.vector_norm(data.force, dim=-1)
    count = (force_mag > force_threshold).any(dim=1).float()
  elif data.found is not None:
    count = (data.found > 0).any(dim=1).float()
  else:
    raise ValueError(f"Contact sensor {sensor_name!r} must expose force or found.")
  if max_count is not None:
    count = torch.clamp(count, max=max_count)
  return count


def robot_table_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 5.0,
  max_count: float = 4.0,
) -> torch.Tensor:
  """Penalty signal for robot or paddle contacts with the table/net."""
  return _contact_substep_count(env, sensor_name, force_threshold, max_count)


def robot_ball_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.05,
  max_count: float = 4.0,
) -> torch.Tensor:
  """Penalty signal for non-paddle robot contacts with the ball."""
  return _contact_substep_count(env, sensor_name, force_threshold, max_count)


def _return_flight_active(
  state,
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
) -> torch.Tensor:
  ball: Entity = env.scene[ball_cfg.name]
  ball_x = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
  return (
    state.hit_valid
    & state.has_paddle_hit
    & ~state.has_crossed_net
    & (ball_x > state.net_x)
  )


class paddle_to_ball_after_bounce_dense(PingpongRallyStateTerm):
  """Reward keeping the paddle near the ball after the legal self-table bounce."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    paddle_sensor_name: str,
    net_sensor_name: str,
    std: float,
    paddle_cfg: SceneEntityCfg = _PADDLE_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    body_ball_sensor_name: str | None = None,
    force_threshold: float = 1.0,
    table_z: float = 0.78,
    net_x: float = 0.0,
    net_top_z: float = 0.9125,
    self_x_limits: tuple[float, float] = (0.0, 1.37),
    opponent_x_limits: tuple[float, float] = (-1.37, 0.0),
    table_y_limits: tuple[float, float] = (-0.7625, 0.7625),
    x_limits: tuple[float, float] = (-2.1, 2.4),
    y_limits: tuple[float, float] = (-1.25, 1.25),
    z_limits: tuple[float, float] = (0.05, 2.5),
    bounce_z_tolerance: float = 0.05,
    gravity: float = 9.81,
  ) -> torch.Tensor:
    del (
      paddle_sensor_name,
      net_sensor_name,
      body_ball_sensor_name,
      force_threshold,
      table_z,
      net_x,
      net_top_z,
      gravity,
      self_x_limits,
      opponent_x_limits,
      table_y_limits,
      x_limits,
      y_limits,
      z_limits,
      bounce_z_tolerance,
    )
    state = self.state
    delta_b = racket_to_ball_b(env, paddle_cfg, ball_cfg, robot_cfg)
    error = torch.sum(torch.square(delta_b), dim=-1)
    reward = torch.exp(-error / std**2)
    active = state.has_self_bounce & ~state.has_paddle_hit
    return reward * active.float()


class paddle_towards_ball_velocity(PingpongRallyStateTerm):
  """Reward moving the paddle toward the ball after the self-table bounce."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    paddle_sensor_name: str,
    net_sensor_name: str,
    paddle_cfg: SceneEntityCfg = _PADDLE_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
    body_ball_sensor_name: str | None = None,
    speed_scale: float = 2.0,
    distance_std: float = 0.55,
    force_threshold: float = 1.0,
    table_z: float = 0.78,
    net_x: float = 0.0,
    net_top_z: float = 0.9125,
    self_x_limits: tuple[float, float] = (0.0, 1.37),
    opponent_x_limits: tuple[float, float] = (-1.37, 0.0),
    table_y_limits: tuple[float, float] = (-0.7625, 0.7625),
    x_limits: tuple[float, float] = (-2.1, 2.4),
    y_limits: tuple[float, float] = (-1.25, 1.25),
    z_limits: tuple[float, float] = (0.05, 2.5),
    bounce_z_tolerance: float = 0.05,
    gravity: float = 9.81,
  ) -> torch.Tensor:
    del (
      paddle_sensor_name,
      net_sensor_name,
      body_ball_sensor_name,
      force_threshold,
      table_z,
      net_x,
      net_top_z,
      gravity,
      self_x_limits,
      opponent_x_limits,
      table_y_limits,
      x_limits,
      y_limits,
      z_limits,
      bounce_z_tolerance,
    )
    state = self.state
    delta_b = racket_to_ball_b(env, paddle_cfg, ball_cfg, robot_cfg)
    paddle_vel_b = racket_velocity_b(env, paddle_cfg, robot_cfg)
    distance = torch.linalg.vector_norm(delta_b, dim=-1).clamp_min(1.0e-6)
    direction_to_ball = delta_b / distance.unsqueeze(-1)
    toward_speed = torch.sum(paddle_vel_b * direction_to_ball, dim=-1)
    toward_speed = torch.clamp(toward_speed, min=0.0)
    distance_weight = torch.exp(-(distance**2) / distance_std**2)
    reward = torch.tanh(toward_speed / speed_scale) * distance_weight
    active = state.has_self_bounce & ~state.has_paddle_hit
    return reward * active.float()


class self_table_bounce_event(PingpongRallyStateTerm):
  """Sparse reward when the incoming feed legally bounces on the robot side."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.self_bounce_edge.float()


class paddle_hit_event(PingpongRallyStateTerm):
  """Sparse reward for the first legal paddle-ball contact."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.paddle_hit_edge.float()


class post_hit_x_progress(PingpongRallyStateTerm):
  """Reward moving the returned ball toward the opponent half before crossing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    paddle_sensor_name: str,
    net_sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    max_progress: float = 0.04,
    **params,
  ) -> torch.Tensor:
    del paddle_sensor_name, net_sensor_name, params
    state = self.state
    ball: Entity = env.scene[ball_cfg.name]
    ball_x = ball.data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = torch.clamp(state.prev_ball_x - ball_x, min=0.0, max=max_progress)
    active = state.has_paddle_hit & ~state.has_crossed_net & (ball_x > state.net_x)
    return (progress / max_progress) * active.float()


class post_hit_ball_velocity_direction(PingpongRallyStateTerm):
  """Reward a returned ball velocity aimed across the net."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    paddle_sensor_name: str,
    net_sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    x_speed_scale: float = 2.5,
    lateral_speed_std: float = 0.8,
    **params,
  ) -> torch.Tensor:
    del paddle_sensor_name, net_sensor_name, params
    state = self.state
    ball: Entity = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w - env.scene.env_origins
    ball_vel = ball.data.root_link_lin_vel_w
    x_reward = torch.clamp(-ball_vel[:, 0] / x_speed_scale, min=0.0, max=1.0)
    lateral_weight = torch.exp(-(ball_vel[:, 1] ** 2) / lateral_speed_std**2)
    active = state.has_paddle_hit & ~state.has_crossed_net & (ball_pos[:, 0] > 0.0)
    return x_reward * lateral_weight * active.float()


class strike_outgoing_ball_velocity(PingpongRallyStateTerm):
  """Reward a legal hit whose outgoing velocity points toward the opponent."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    target_x_speed: float = 2.5,
    **params,
  ) -> torch.Tensor:
    del params
    state = self.state
    active = _return_flight_active(state, env, ball_cfg)
    x_speed = torch.clamp(
      -state.hit_post_vel[:, 0] / target_x_speed,
      min=0.0,
      max=1.0,
    )
    return x_speed * state.hit_post_vx_toward_opponent_ratio * active.float()


class strike_pred_net_clearance(PingpongRallyStateTerm):
  """Reward legal hits whose ballistic path is predicted to clear the net."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    clearance_margin: float = 0.03,
    clearance_scale: float = 0.18,
    **params,
  ) -> torch.Tensor:
    del params
    state = self.state
    active = _return_flight_active(state, env, ball_cfg)
    reward = torch.clamp(
      (state.hit_pred_net_clearance + clearance_margin) / clearance_scale,
      min=0.0,
      max=1.0,
    )
    return reward * active.float()


class strike_pred_landing_inside(PingpongRallyStateTerm):
  """Reward legal hits predicted to land on the opponent table half."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    **params,
  ) -> torch.Tensor:
    del params
    state = self.state
    active = _return_flight_active(state, env, ball_cfg)
    return state.hit_pred_landing_inside_opponent_table * active.float()


class strike_post_hit_speed(PingpongRallyStateTerm):
  """Reward enough legal-hit ball speed without rewarding wrong-way hits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    speed_scale: float = 4.0,
    **params,
  ) -> torch.Tensor:
    del params
    state = self.state
    active = _return_flight_active(state, env, ball_cfg)
    speed_reward = torch.tanh(state.hit_post_speed / speed_scale)
    return speed_reward * state.hit_post_vx_toward_opponent_ratio * active.float()


class pre_hit_action_rate_l2(PingpongRallyStateTerm):
  """Action-rate penalty that relaxes only after the legal paddle hit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del params
    active = ~self.state.has_paddle_hit
    return action_rate_l2(env) * active.float()


class pre_hit_low_level_action_rate_l2(PingpongRallyStateTerm):
  """Low-level action-rate penalty that relaxes only in return flight."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    action_name: str,
    **params,
  ) -> torch.Tensor:
    del params
    active = ~self.state.has_paddle_hit
    return low_level_action_rate_l2(env, action_name=action_name) * active.float()


class crossed_net_event(PingpongRallyStateTerm):
  """Sparse reward when the returned ball clears the net plane."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.crossed_net_edge.float()


class opponent_table_bounce_event(PingpongRallyStateTerm):
  """Sparse reward for the successful opponent-side table bounce."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(self, env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    del env, params
    return self.state.opponent_bounce_edge.float()


__all__ = [
  "crossed_net_event",
  "opponent_table_bounce_event",
  "paddle_hit_event",
  "paddle_to_ball_after_bounce_dense",
  "paddle_towards_ball_velocity",
  "post_hit_ball_velocity_direction",
  "post_hit_x_progress",
  "pre_hit_action_rate_l2",
  "pre_hit_low_level_action_rate_l2",
  "robot_ball_contact_penalty",
  "robot_table_contact_penalty",
  "self_table_bounce_event",
  "strike_outgoing_ball_velocity",
  "strike_post_hit_speed",
  "strike_pred_landing_inside",
  "strike_pred_net_clearance",
]
