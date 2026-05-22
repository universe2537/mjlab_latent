from types import SimpleNamespace
from typing import Any

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.hit_state import TennisHitTracker
from mjlab.tasks.tennis.mdp.rewards import (
  post_hit_ball_velocity_direction,
  post_hit_x_progress,
)


class _Scene(dict):
  def __init__(self, env_origins, items):
    super().__init__(items)
    self.env_origins = env_origins


def _make_env():
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.4, 0.0, 1.0]], dtype=torch.float32),
      root_link_lin_vel_w=torch.tensor([[-1.5, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      force_history=None,
      force=torch.zeros(1, 1, 3, dtype=torch.float32),
      found=torch.zeros(1, 1, dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    common_step_counter=0,
    scene=_Scene(
      torch.zeros(1, 3, dtype=torch.float32),
      {
        "ball": ball,
        "racket_ball_contact": sensor,
      },
    ),
  )
  return env, ball, sensor


def _make_tracker(env: Any) -> TennisHitTracker:
  return TennisHitTracker(
    env,
    sensor_name="racket_ball_contact",
    ball_cfg=SceneEntityCfg("ball"),
    force_threshold=1.0,
    ground_z=0.06,
    net_x=0.0,
    landing_x_limits=(-2.0, 0.0),
    landing_y_limits=(-0.5, 0.5),
  )


def _progress_params() -> dict[str, Any]:
  return {
    "sensor_name": "racket_ball_contact",
    "ball_cfg": SceneEntityCfg("ball"),
    "force_threshold": 1.0,
    "ground_z": 0.06,
    "net_x": 0.0,
    "landing_x_limits": (-2.0, 0.0),
    "landing_y_limits": (-0.5, 0.5),
    "max_progress": 0.08,
  }


def _velocity_params() -> dict[str, Any]:
  return {
    "sensor_name": "racket_ball_contact",
    "ball_cfg": SceneEntityCfg("ball"),
    "force_threshold": 1.0,
    "ground_z": 0.06,
    "net_x": 0.0,
    "landing_x_limits": (-2.0, 0.0),
    "landing_y_limits": (-0.5, 0.5),
    "x_speed_scale": 4.0,
    "lateral_speed_std": 1.5,
  }


def _make_progress_reward(env: Any) -> post_hit_x_progress:
  cfg = RewardTermCfg(func=post_hit_x_progress, weight=1.0, params=_progress_params())
  return post_hit_x_progress(cfg, env)


def _make_velocity_reward(env: Any) -> post_hit_ball_velocity_direction:
  cfg = RewardTermCfg(
    func=post_hit_ball_velocity_direction,
    weight=1.0,
    params=_velocity_params(),
  )
  return post_hit_ball_velocity_direction(cfg, env)


def test_tennis_hit_tracker_marks_in_bounds_landing_success() -> None:
  env, ball, sensor = _make_env()
  tracker = _make_tracker(env)

  tracker.update()
  assert not tracker.racket_hit_edge[0]

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  tracker.update()
  assert tracker.racket_hit_edge[0]
  assert tracker.racket_hit_count[0] == 1
  assert not tracker.landing_in_bounds_edge[0]

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.2, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  tracker.update()
  assert tracker.crossed_net_edge[0]
  assert not tracker.landing_in_bounds_edge[0]

  env.common_step_counter = 3
  ball.data.root_link_pos_w[:] = torch.tensor([[-1.0, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  tracker.update()
  assert tracker.bounce_edge[0]
  assert tracker.landing_in_bounds_edge[0]
  assert tracker.has_landed_in_bounds[0]


def test_tennis_hit_tracker_rejects_out_of_bounds_landing() -> None:
  env, ball, sensor = _make_env()
  tracker = _make_tracker(env)

  tracker.update()
  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  tracker.update()

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.2, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  tracker.update()

  env.common_step_counter = 3
  ball.data.root_link_pos_w[:] = torch.tensor([[-1.0, 1.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  tracker.update()
  assert tracker.bounce_edge[0]
  assert not tracker.landing_in_bounds_edge[0]
  assert not tracker.has_landed_in_bounds[0]


def test_tennis_hit_tracker_requires_crossing_before_landing_success() -> None:
  env, ball, sensor = _make_env()
  tracker = _make_tracker(env)

  tracker.update()
  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-1.0, 0.0, -1.0]])
  tracker.update()

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[0.5, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-1.0, 0.0, 1.0]])
  tracker.update()
  assert tracker.bounce_edge[0]
  assert not tracker.crossed_net_edge[0]
  assert not tracker.landing_in_bounds_edge[0]


def test_post_hit_x_progress_requires_racket_hit() -> None:
  env, ball, _ = _make_env()
  reward = _make_progress_reward(env)
  params = _progress_params()

  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.0]])
  assert reward(env, **params).item() == 0.0

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.95, 0.0, 1.0]])
  assert reward(env, **params).item() == 0.0


def test_post_hit_x_progress_rewards_negative_x_progress() -> None:
  env, ball, sensor = _make_env()
  reward = _make_progress_reward(env)
  params = _progress_params()

  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.0]])
  reward(env, **params)

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  reward(env, **params)

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[0.95, 0.0, 1.0]])
  value = reward(env, **params).item()

  assert value > 0.0
  assert torch.isclose(torch.tensor(value), torch.tensor(0.05 / 0.08))


def test_post_hit_x_progress_ignores_non_negative_x_progress() -> None:
  env, ball, sensor = _make_env()
  reward = _make_progress_reward(env)
  params = _progress_params()

  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.0]])
  reward(env, **params)

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  reward(env, **params)

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[1.02, 0.0, 1.0]])
  assert reward(env, **params).item() == 0.0


def test_post_hit_x_progress_stops_after_net_crossing() -> None:
  env, ball, sensor = _make_env()
  reward = _make_progress_reward(env)
  params = _progress_params()

  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.0]])
  reward(env, **params)

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  reward(env, **params)

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.1, 0.0, 1.0]])
  assert reward(env, **params).item() == 0.0


def test_post_hit_velocity_direction_requires_racket_hit() -> None:
  env, ball, _ = _make_env()
  reward = _make_velocity_reward(env)
  params = _velocity_params()

  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, 1.0]])
  assert reward(env, **params).item() == 0.0


def test_post_hit_velocity_direction_rewards_opponent_direction() -> None:
  env, ball, sensor = _make_env()
  reward = _make_velocity_reward(env)
  params = _velocity_params()

  reward(env, **params)

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  reward(env, **params)

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[0.8, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  value = reward(env, **params).item()

  assert torch.isclose(torch.tensor(value), torch.tensor(0.5))


def test_post_hit_velocity_direction_penalizes_lateral_speed() -> None:
  env, ball, sensor = _make_env()
  reward = _make_velocity_reward(env)
  params = _velocity_params()

  reward(env, **params)

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  reward(env, **params)

  env.common_step_counter = 2
  sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[0.8, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 2.0, 1.0]])
  lateral_value = reward(env, **params).item()

  env.common_step_counter = 3
  ball.data.root_link_pos_w[:] = torch.tensor([[0.7, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  straight_value = reward(env, **params).item()

  assert 0.0 < lateral_value < straight_value
