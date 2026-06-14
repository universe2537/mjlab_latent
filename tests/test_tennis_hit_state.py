from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.ball_state import (
  PHASE_INCOMING,
  PHASE_RECOVERY,
  PHASE_RETURN_FLIGHT,
  OpponentFeeder,
  OpponentFeederCfg,
  TennisContinuousBallState,
  continuous_ball_phase,
  continuous_recovery_ready_event,
  continuous_recovery_ready_pose_state,
)
from mjlab.tasks.tennis.mdp.hit_state import TennisHitTracker
from mjlab.tasks.tennis.mdp.observations import ball_predicted_hit_point_b
from mjlab.tasks.tennis.mdp.rewards import (
  post_hit_ball_velocity_direction,
  post_hit_low_arc_quality_reward,
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
    step_dt=0.02,
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


def test_predicted_hit_point_prefers_descending_intersection() -> None:
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
      root_link_lin_vel_w=torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float32),
    )
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(1, 3, dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
  )
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    scene=_Scene(
      torch.zeros(1, 3, dtype=torch.float32),
      {
        "ball": ball,
        "robot": robot,
      },
    ),
  )

  hit_point = ball_predicted_hit_point_b(
    cast(Any, env),
    hit_height_offset=0.05,
    gravity=10.0,
    max_horizon=1.0,
  )

  assert hit_point.shape == (1, 4)
  assert hit_point[0, 3] > 0.3
  assert hit_point[0, 0] > 0.3


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


def _make_low_arc_reward(env: Any) -> post_hit_low_arc_quality_reward:
  params = _low_arc_params()
  cfg = RewardTermCfg(
    func=post_hit_low_arc_quality_reward,
    weight=1.0,
    params=params,
  )
  return post_hit_low_arc_quality_reward(cfg, env)


def _low_arc_params() -> dict[str, Any]:
  params = {
    **_progress_params(),
    "fast_landing_t_min": 0.02,
    "fast_landing_t_max": 0.10,
    "require_in_bounds": True,
  }
  params.pop("max_progress")
  return params


def test_tennis_hit_tracker_marks_in_bounds_landing_success() -> None:
  env, ball, sensor = _make_env()
  tracker = _make_tracker(env)
  env._tennis_hit_tracker = tracker
  reward = _make_low_arc_reward(env)

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
  assert tracker.first_bounce_after_hit_edge[0]
  assert tracker.time_to_landing[0] == torch.tensor(0.04)
  score = reward(cast(Any, env), **_low_arc_params())
  assert torch.allclose(score, torch.tensor([0.75]))
  assert reward(cast(Any, env), **_low_arc_params())[0] == 0.0
  assert tracker.has_landed_in_bounds[0]
  assert tracker.successful_return_count[0] == 1
  assert tracker.episode_racket_hit_count[0] == 1
  assert tracker.episode_crossed_net_count[0] == 1
  assert tracker.episode_landing_in_bounds_count[0] == 1
  assert tracker.episode_first_bounce_after_hit_count[0] == 1
  assert tracker.episode_fast_landing_reward_count[0] == 1
  assert torch.allclose(tracker.episode_fast_landing_score_sum, torch.tensor([0.75]))

  tracker.reset_rally(torch.tensor([0]))
  assert tracker.successful_return_count[0] == 1
  assert tracker.episode_racket_hit_count[0] == 1
  assert tracker.episode_crossed_net_count[0] == 1
  assert tracker.episode_landing_in_bounds_count[0] == 1
  assert not tracker.has_racket_hit[0]
  assert not tracker.has_crossed_net[0]
  assert not tracker.has_landed_in_bounds[0]
  assert not tracker.has_rewarded_fast_landing[0]

  tracker.reset(torch.tensor([0]))
  assert tracker.successful_return_count[0] == 0
  assert tracker.episode_racket_hit_count[0] == 0
  assert tracker.episode_crossed_net_count[0] == 0
  assert tracker.episode_landing_in_bounds_count[0] == 0
  assert tracker.episode_fast_landing_reward_count[0] == 0


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
  assert state.first_bounce_after_hit_edge[0]
  assert torch.allclose(state.time_to_landing, torch.tensor([0.04]))
  score = state.fast_landing_score(t_min=0.02, t_max=0.10)
  assert torch.allclose(score, torch.tensor([0.75]))
  assert state.fast_landing_score(t_min=0.02, t_max=0.10)[0] == 0.0
  assert state.successful_return_count[0] == 1
  assert state.episode_first_bounce_after_hit_count[0] == 1
  assert state.episode_fast_landing_reward_count[0] == 1
  assert torch.allclose(state.episode_fast_landing_score_sum, torch.tensor([0.75]))
  assert state.phase[0] == PHASE_RECOVERY
  assert not state.fault_edge[0]


def test_continuous_net_cross_uses_interpolated_height_near_net() -> None:
  env, ball, racket_sensor, _ = _make_continuous_env()
  state = _make_continuous_state(env)

  state.update()

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.1, 0.0, 2.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  racket_sensor.data.force[:] = 5.0
  state.update()
  assert state.racket_hit_edge[0]
  assert state.phase[0] == PHASE_RETURN_FLIGHT

  env.common_step_counter = 2
  racket_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.1, 0.0, 0.5]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  state.update()
  assert state.crossed_net_edge[0]
  assert not state.fault_edge[0]
  assert state.episode_fault_low_net_cross_count[0] == 0

  env.common_step_counter = 3
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.2, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  state.update()
  assert state.successful_return_edge[0]
  assert state.successful_return_count[0] == 1


def test_continuous_hit_and_cross_same_step_counts_as_return() -> None:
  env, ball, racket_sensor, _ = _make_continuous_env()
  state = _make_continuous_state(env)

  state.update()

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.2]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -0.5]])
  state.update()

  env.common_step_counter = 2
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.2, 0.0, 1.1]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -0.5]])
  racket_sensor.data.force[:] = 5.0
  state.update()

  assert state.racket_hit_edge[0]
  assert state.crossed_net_edge[0]
  assert state.phase[0] == PHASE_RETURN_FLIGHT
  assert not state.fault_edge[0]

  env.common_step_counter = 3
  racket_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-1.0, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  state.update()

  assert state.successful_return_edge[0]
  assert state.successful_return_count[0] == 1
  assert not state.fault_edge[0]


def test_continuous_hit_and_net_same_step_is_return_fault() -> None:
  env, ball, racket_sensor, net_sensor = _make_continuous_env()
  state = _make_continuous_state(env)

  state.update()

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.2]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -0.5]])
  state.update()

  env.common_step_counter = 2
  ball.data.root_link_pos_w[:] = torch.tensor([[0.1, 0.0, 0.9]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -0.5]])
  racket_sensor.data.force[:] = 5.0
  net_sensor.data.force[:] = 5.0
  state.update()

  assert state.racket_hit_edge[0]
  assert state.fault_edge[0]
  assert state.episode_fault_net_contact_count[0] == 1
  assert state.episode_invalid_feed_count[0] == 0


def test_continuous_ball_phase_is_three_way_one_hot() -> None:
  env, ball, racket_sensor, _ = _make_continuous_env()

  def phase_obs() -> torch.Tensor:
    return continuous_ball_phase(
      env,
      racket_sensor_name="racket_ball_contact",
      net_sensor_name="ball_net_contact",
      ball_cfg=SceneEntityCfg("ball"),
      landing_x_limits=(-2.0, 0.0),
      landing_y_limits=(-0.5, 0.5),
      x_limits=(-3.0, 3.0),
      y_limits=(-2.0, 2.0),
      z_limits=(0.02, 4.0),
    )

  phase = phase_obs()
  assert phase.shape == (1, 3)
  assert torch.allclose(phase, torch.tensor([[1.0, 0.0, 0.0]]))

  state = env._tennis_continuous_ball_state
  assert state.phase[0] == PHASE_INCOMING

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[1.0, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -0.5]])
  racket_sensor.data.force[:] = 5.0
  phase = phase_obs()
  assert torch.allclose(phase, torch.tensor([[0.0, 1.0, 0.0]]))

  env.common_step_counter = 2
  racket_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.2, 0.0, 1.1]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, -1.0]])
  phase = phase_obs()
  assert torch.allclose(phase, torch.tensor([[0.0, 1.0, 0.0]]))

  env.common_step_counter = 3
  ball.data.root_link_pos_w[:] = torch.tensor([[-1.0, 0.0, 0.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.0]])
  phase = phase_obs()
  assert torch.allclose(phase, torch.tensor([[0.0, 0.0, 1.0]]))


def test_continuous_ball_state_invalid_incoming_feed_is_not_failure() -> None:
  env, _, _, net_sensor = _make_continuous_env()
  state = _make_continuous_state(env)

  net_sensor.data.force[:] = 5.0
  state.update()

  assert state.invalid_feed_edge[0]
  assert state.episode_invalid_feed_count[0] == 1
  assert state.episode_invalid_feed_net_count[0] == 1
  assert state.episode_invalid_feed_out_count[0] == 0
  assert state.episode_invalid_feed_opponent_bounce_count[0] == 0
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
  assert state.episode_fault_incoming_bounce_count[0] == 1


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
  z_apex = pz + torch.square(vz) / (2.0 * cfg.gravity)

  assert torch.allclose(pz, torch.full_like(pz, cfg.ground_z))
  assert torch.all(z_net >= cfg.net_height + cfg.net_clearance + cfg.ball_radius)
  assert torch.all(z_apex <= cfg.max_apex_z)


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
    max_apex_z=3.9,
  )
  feeder = OpponentFeeder(cfg, cast(Any, env))
  env_ids = torch.arange(512)

  px, _, pz, vx, _, vz, valid = feeder._sample_candidate(env_ids)
  t_net = (cfg.net_x - px) / vx
  z_net = pz + vz * t_net - 0.5 * cfg.gravity * t_net * t_net
  z_apex = pz + torch.square(vz) / (2.0 * cfg.gravity)

  assert torch.any(valid)
  assert torch.allclose(pz, torch.full_like(pz, cfg.ground_z))
  assert torch.all(z_net[valid] >= cfg.net_height + cfg.net_clearance + cfg.ball_radius)
  assert torch.all(z_apex[valid] <= cfg.max_apex_z)


def test_continuous_recovery_reward_requires_moving_home_when_far() -> None:
  env, ball, racket_sensor, net_sensor = _make_continuous_env()
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
      root_link_lin_vel_b=torch.zeros(1, 3, dtype=torch.float32),
      root_link_lin_vel_w=torch.zeros(1, 3, dtype=torch.float32),
      projected_gravity_b=torch.zeros(1, 3, dtype=torch.float32),
      heading_w=torch.tensor([torch.pi], dtype=torch.float32),
      site_pos_w=torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32),
    )
  )
  env.scene = _Scene(
    torch.zeros(1, 3, dtype=torch.float32),
    {
      "ball": ball,
      "robot": robot,
      "racket_ball_contact": racket_sensor,
      "ball_net_contact": net_sensor,
    },
  )
  cfg = RewardTermCfg(
    func=continuous_recovery_ready_pose_state,
    weight=1.0,
    params={
      "racket_sensor_name": "racket_ball_contact",
      "net_sensor_name": "ball_net_contact",
      "ball_cfg": SceneEntityCfg("ball"),
      "racket_cfg": SimpleNamespace(site_ids=torch.tensor([0])),
      "robot_cfg": SceneEntityCfg("robot"),
      "target_x": 3.4,
      "target_y": 0.0,
      "target_heading": torch.pi,
      "racket_target_b": (0.0, 0.0, 0.0),
    },
  )
  reward = continuous_recovery_ready_pose_state(cfg, env)
  reward._state.phase[:] = PHASE_RECOVERY
  reward._state._last_step = env.common_step_counter

  far_still = reward(env, **cfg.params)
  robot.data.root_link_lin_vel_w[:] = torch.tensor([[0.8, 0.0, 0.0]])
  moving_home = reward(env, **cfg.params)
  robot.data.root_link_pos_w[:] = torch.tensor([[3.4, 0.0, 0.0]])
  robot.data.site_pos_w[:] = torch.tensor([[[3.4, 0.0, 0.0]]])
  robot.data.root_link_lin_vel_w.zero_()
  near_ready = reward(env, **cfg.params)

  assert far_still.item() < 0.05
  assert moving_home.item() > far_still.item() + 0.5
  assert near_ready.item() > 0.95


def test_continuous_recovery_ready_event_counts_once() -> None:
  env, ball, racket_sensor, net_sensor = _make_continuous_env()
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[3.4, 0.0, 0.0]], dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
      root_link_lin_vel_b=torch.zeros(1, 3, dtype=torch.float32),
      root_link_lin_vel_w=torch.zeros(1, 3, dtype=torch.float32),
      projected_gravity_b=torch.zeros(1, 3, dtype=torch.float32),
      heading_w=torch.tensor([torch.pi], dtype=torch.float32),
      site_pos_w=torch.tensor([[[3.4, 0.0, 0.0]]], dtype=torch.float32),
    )
  )
  env.scene = _Scene(
    torch.zeros(1, 3, dtype=torch.float32),
    {
      "ball": ball,
      "robot": robot,
      "racket_ball_contact": racket_sensor,
      "ball_net_contact": net_sensor,
    },
  )
  cfg = RewardTermCfg(
    func=continuous_recovery_ready_event,
    weight=1.0,
    params={
      "racket_sensor_name": "racket_ball_contact",
      "net_sensor_name": "ball_net_contact",
      "ball_cfg": SceneEntityCfg("ball"),
      "racket_cfg": SimpleNamespace(site_ids=torch.tensor([0])),
      "robot_cfg": SceneEntityCfg("robot"),
      "target_x": 3.4,
      "target_y": 0.0,
      "target_heading": torch.pi,
      "racket_target_b": (0.0, 0.0, 0.0),
    },
  )
  reward = continuous_recovery_ready_event(cfg, env)
  reward._state.phase[:] = PHASE_RECOVERY
  reward._state._last_step = env.common_step_counter

  first = reward(env, **cfg.params)
  second = reward(env, **cfg.params)

  assert first.item() == 1.0
  assert second.item() == 0.0
  assert reward._state.episode_recovery_ready_count[0] == 1


def test_continuous_recovery_timer_has_minimum_ready_time() -> None:
  env, _, _, _ = _make_continuous_env()
  state = _make_continuous_state(env)
  env_ids = torch.tensor([0])
  state.phase[:] = PHASE_RECOVERY

  state.start_recovery(env_ids, recovery_time_range=(3.0, 5.0), min_recovery_time=1.0)

  assert state.recovery_min_steps_left[0] == 50
  assert state.recovery_steps_left[0] >= state.recovery_min_steps_left[0]
  state.step_recovery(state.in_recovery)
  assert state.recovery_min_steps_left[0] == 49


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
