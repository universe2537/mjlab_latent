from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.ball_state import (
  PHASE_RECOVERY,
  PHASE_RETURN_FLIGHT,
  OpponentFeeder,
  OpponentFeederCfg,
  TennisContinuousBallState,
)
from mjlab.tasks.tennis.mdp.hit_state import TennisHitTracker
from mjlab.tasks.tennis.mdp.rewards import (
  post_hit_ball_velocity_direction,
  post_hit_x_progress,
)


class _Scene(dict):
  def __init__(self, env_origins, items):
    super().__init__(items)
    self.env_origins = env_origins


def _make_env() -> tuple[Any, Any, Any]:
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


def _make_continuous_env() -> tuple[Any, Any, Any, Any]:
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[-1.0, 0.0, 1.2]], dtype=torch.float32),
      root_link_lin_vel_w=torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  racket_sensor = SimpleNamespace(
    data=SimpleNamespace(
      force_history=None,
      force=torch.zeros(1, 1, 3, dtype=torch.float32),
      found=torch.zeros(1, 1, dtype=torch.float32),
    )
  )
  net_sensor = SimpleNamespace(
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
    step_dt=0.02,
    scene=_Scene(
      torch.zeros(1, 3, dtype=torch.float32),
      {
        "ball": ball,
        "racket_ball_contact": racket_sensor,
        "ball_net_contact": net_sensor,
      },
    ),
  )
  return env, ball, racket_sensor, net_sensor


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


def _make_continuous_state(env: Any) -> TennisContinuousBallState:
  return TennisContinuousBallState(
    env,
    racket_sensor_name="racket_ball_contact",
    net_sensor_name="ball_net_contact",
    ball_cfg=SceneEntityCfg("ball"),
    force_threshold=1.0,
    ground_z=0.06,
    net_x=0.0,
    net_height=0.914,
    landing_x_limits=(-2.0, 0.0),
    landing_y_limits=(-0.5, 0.5),
    x_limits=(-3.0, 3.0),
    y_limits=(-2.0, 2.0),
    z_limits=(0.02, 4.0),
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
  assert tracker.successful_return_count[0] == 1
  assert tracker.episode_racket_hit_count[0] == 1
  assert tracker.episode_crossed_net_count[0] == 1
  assert tracker.episode_landing_in_bounds_count[0] == 1

  tracker.reset_rally(torch.tensor([0]))
  assert tracker.successful_return_count[0] == 1
  assert tracker.episode_racket_hit_count[0] == 1
  assert tracker.episode_crossed_net_count[0] == 1
  assert tracker.episode_landing_in_bounds_count[0] == 1
  assert not tracker.has_racket_hit[0]
  assert not tracker.has_crossed_net[0]
  assert not tracker.has_landed_in_bounds[0]

  tracker.reset(torch.tensor([0]))
  assert tracker.successful_return_count[0] == 0
  assert tracker.episode_racket_hit_count[0] == 0
  assert tracker.episode_crossed_net_count[0] == 0
  assert tracker.episode_landing_in_bounds_count[0] == 0


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


def test_tennis_hit_tracker_recovery_timer_counts_down() -> None:
  env, _, _ = _make_env()
  tracker = _make_tracker(env)

  tracker.successful_return_count[0] = 1
  tracker.start_recovery(torch.tensor([0]), recovery_steps=2)

  assert tracker.in_recovery[0]
  assert torch.isclose(tracker.recovery_fraction_remaining[0], torch.tensor(1.0))

  ready = tracker.step_recovery()
  assert not ready[0]
  assert tracker.in_recovery[0]
  assert torch.isclose(tracker.recovery_fraction_remaining[0], torch.tensor(0.5))

  ready = tracker.step_recovery()
  assert ready[0]
  assert not tracker.in_recovery[0]

  tracker.reset_rally(torch.tensor([0]))
  assert tracker.successful_return_count[0] == 1
  assert not tracker.in_recovery[0]


def test_continuous_ball_state_counts_successful_air_return() -> None:
  env, ball, racket_sensor, _ = _make_continuous_env()
  state = _make_continuous_state(env)

  state.update()
  assert not state.racket_hit_edge[0]

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -0.5]])
  racket_sensor.data.force[:] = 5.0
  state.update()
  assert state.racket_hit_edge[0]
  assert state.phase[0] == PHASE_RETURN_FLIGHT

  env.common_step_counter = 2
  racket_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.2, 0.0, 1.1]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  state.update()
  assert state.crossed_net_edge[0]
  assert not state.fault_edge[0]

  env.common_step_counter = 3
  ball.data.root_link_pos_w[:] = torch.tensor([[-1.0, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  state.update()
  assert state.successful_return_edge[0]
  assert state.successful_return_count[0] == 1
  assert state.phase[0] == PHASE_RECOVERY
  assert not state.fault_edge[0]


def test_continuous_ball_state_invalid_incoming_feed_is_not_failure() -> None:
  env, _, _, net_sensor = _make_continuous_env()
  state = _make_continuous_state(env)

  net_sensor.data.force[:] = 5.0
  state.update()

  assert state.invalid_feed_edge[0]
  assert state.episode_invalid_feed_count[0] == 1
  assert not state.fault_edge[0]
  assert state.episode_fault_count[0] == 0


def test_continuous_ball_state_robot_side_incoming_bounce_fails() -> None:
  env, ball, _, _ = _make_continuous_env()
  state = _make_continuous_state(env)

  state.update()
  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()

  assert state.bounce_edge[0]
  assert state.fault_edge[0]
  assert state.episode_fault_count[0] == 1


def test_opponent_feeder_fallback_still_clears_net() -> None:
  ball = SimpleNamespace()
  env = SimpleNamespace(device="cpu", scene={"ball": ball})
  cfg = OpponentFeederCfg(
    ball_cfg=SceneEntityCfg("ball"),
    spawn_x_range=(-5.8, -0.2),
    spawn_y_range=(-1.0, 1.0),
    target_x_range=(3.2, 3.6),
    target_y_range=(-0.2, 0.2),
    flight_time_range=(0.85, 1.35),
    flight_time_slack_range=(0.05, 0.35),
    spawn_z_range=(0.06, 0.06),
    ground_z=0.06,
    net_x=0.0,
    net_height=0.914,
    net_clearance=0.25,
  )
  feeder = OpponentFeeder(cfg, cast(Any, env))
  env_ids = torch.arange(4)

  px, _, pz, vx, _, vz = feeder._fallback_candidate(env_ids)
  t_net = (cfg.net_x - px) / vx
  z_net = pz + vz * t_net - 0.5 * cfg.gravity * t_net * t_net

  assert torch.allclose(pz, torch.full_like(pz, cfg.ground_z))
  assert torch.all(z_net >= cfg.net_height + cfg.net_clearance)


def test_opponent_feeder_samples_ground_endpoints_that_clear_net() -> None:
  ball = SimpleNamespace()
  env = SimpleNamespace(device="cpu", scene={"ball": ball})
  cfg = OpponentFeederCfg(
    ball_cfg=SceneEntityCfg("ball"),
    spawn_x_range=(-5.8, -0.2),
    spawn_y_range=(-1.0, 1.0),
    target_x_range=(0.4, 3.6),
    target_y_range=(-0.2, 0.2),
    flight_time_range=(0.85, 1.35),
    flight_time_slack_range=(0.05, 0.35),
    spawn_z_range=(0.06, 0.06),
    ground_z=0.06,
    net_x=0.0,
    net_height=0.914,
    net_clearance=0.25,
  )
  feeder = OpponentFeeder(cfg, cast(Any, env))
  env_ids = torch.arange(512)

  px, _, pz, vx, _, vz, valid = feeder._sample_candidate(env_ids)
  t_net = (cfg.net_x - px) / vx
  z_net = pz + vz * t_net - 0.5 * cfg.gravity * t_net * t_net

  assert torch.all(valid)
  assert torch.allclose(pz, torch.full_like(pz, cfg.ground_z))
  assert torch.all(z_net >= cfg.net_height + cfg.net_clearance)


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
