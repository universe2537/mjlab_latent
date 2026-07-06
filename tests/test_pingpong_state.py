from types import SimpleNamespace
from typing import Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.pace import (
  get_pingpong_pace_prediction_state,
  pace_ball_prediction_table,
  pace_future_base_vel_target,
  pace_future_pass_net,
  pace_relative_target_base_xy,
  update_pingpong_pace_prediction,
)
from mjlab.tasks.pingpong.mdp.state import (
  FAULT_ILLEGAL_BODY_BALL_CONTACT,
  FAULT_ILLEGAL_PRE_BOUNCE_HIT,
  FAULT_LOW_NET_CROSS,
  FAULT_RETURN_BOUNCE_OUT,
  PHASE_AFTER_SELF_BOUNCE,
  PHASE_DONE,
  PHASE_RETURN_FLIGHT,
  PingpongRallyState,
)
from mjlab.tasks.pingpong.scene import BALL_CENTER_TABLE_Z, NET_TOP_Z


class _Scene(dict):
  def __init__(self, env_origins, items):
    super().__init__(items)
    self.env_origins = env_origins


def _make_env() -> tuple[Any, Any, Any, Any, Any]:
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.7, 0.0, 1.0]], dtype=torch.float32),
      root_link_lin_vel_w=torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32),
    )
  )
  paddle_sensor = SimpleNamespace(
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
  body_ball_sensor = SimpleNamespace(
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
        "paddle_ball_contact": paddle_sensor,
        "pingpong_ball_net_contact": net_sensor,
        "robot_ball_contact": body_ball_sensor,
      },
    ),
  )
  return env, ball, paddle_sensor, net_sensor, body_ball_sensor


def _make_state(env: Any) -> PingpongRallyState:
  return PingpongRallyState(
    env,
    paddle_sensor_name="paddle_ball_contact",
    net_sensor_name="pingpong_ball_net_contact",
    body_ball_sensor_name="robot_ball_contact",
    ball_cfg=SceneEntityCfg("ball"),
    force_threshold=1.0,
    table_z=BALL_CENTER_TABLE_Z,
    net_top_z=NET_TOP_Z,
    self_x_limits=(0.0, 1.37),
    opponent_x_limits=(-1.37, 0.0),
    table_y_limits=(-0.7625, 0.7625),
    x_limits=(-2.1, 2.4),
    y_limits=(-1.25, 1.25),
    z_limits=(0.05, 2.5),
    bounce_z_tolerance=0.055,
  )


def _make_pace_params() -> dict[str, Any]:
  return {
    "paddle_sensor_name": "paddle_ball_contact",
    "net_sensor_name": "pingpong_ball_net_contact",
    "body_ball_sensor_name": "robot_ball_contact",
    "ball_cfg": SceneEntityCfg("ball"),
    "robot_cfg": SceneEntityCfg("robot"),
    "force_threshold": 1.0,
    "table_z": BALL_CENTER_TABLE_Z,
    "net_top_z": NET_TOP_Z,
    "self_x_limits": (0.0, 1.37),
    "opponent_x_limits": (-1.37, 0.0),
    "table_y_limits": (-0.7625, 0.7625),
    "x_limits": (-2.1, 2.4),
    "y_limits": (-1.25, 1.25),
    "z_limits": (0.05, 2.5),
    "bounce_z_tolerance": 0.055,
  }


def _add_pace_robot(env: Any) -> Any:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.55, 0.0, 0.74]], dtype=torch.float32),
      root_link_lin_vel_w=torch.zeros(1, 3, dtype=torch.float32),
      root_link_ang_vel_b=torch.zeros(1, 3, dtype=torch.float32),
      heading_w=torch.zeros(1, dtype=torch.float32),
    )
  )
  env.scene["robot"] = robot
  env.episode_length_buf = torch.ones(1, dtype=torch.long)
  return robot


def _set_paddle_robot(
  env: Any,
  *,
  pos: tuple[float, float, float],
  vel: tuple[float, float, float],
  quat: tuple[float, float, float, float],
) -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      geom_pos_w=torch.tensor([[pos]], dtype=torch.float32),
      geom_lin_vel_w=torch.tensor([[vel]], dtype=torch.float32),
      geom_quat_w=torch.tensor([[quat]], dtype=torch.float32),
    )
  )
  env.scene["robot"] = robot


def test_pingpong_pace_prediction_state_shapes_and_hook() -> None:
  env, _, _, _, _ = _make_env()
  _add_pace_robot(env)
  params = _make_pace_params()

  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()

  assert state.ball_future_pose.shape == (1, 3)
  assert state.target_base_xy.shape == (1, 2)
  assert state.robot_future_vel.shape == (1, 3)
  assert torch.isfinite(state.ball_future_pose).all()
  assert torch.isfinite(state.target_base_xy).all()
  assert torch.isfinite(state.robot_future_vel).all()
  assert pace_relative_target_base_xy(env, **params).shape == (1, 2)
  torch.testing.assert_close(
    pace_ball_prediction_table(env, **params),
    torch.zeros(1, 3),
  )

  learned_prediction = torch.tensor([[0.25, -0.10, 1.20]], dtype=torch.float32)
  update_pingpong_pace_prediction(env, learned_prediction, **params)
  torch.testing.assert_close(
    pace_ball_prediction_table(env, **params),
    learned_prediction,
  )


def test_pingpong_pace_invalid_rewards_are_finite() -> None:
  env, ball, _, _, _ = _make_env()
  _add_pace_robot(env)
  params = _make_pace_params()
  ball.data.root_link_pos_w[:] = torch.tensor([[1.20, 0.0, 0.8]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-1.0, 0.0, -0.2]])
  env.common_step_counter = 1

  rewards = torch.stack(
    (
      pace_future_base_vel_target(env, **params),
      pace_future_pass_net(env, **params),
    ),
    dim=-1,
  )
  assert torch.isfinite(rewards).all()
  assert rewards.shape == (1, 2)


def test_pingpong_state_counts_legal_single_return() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  state = _make_state(env)

  state.update()
  assert not state.self_bounce_edge[0]

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()
  assert state.self_bounce_edge[0]
  assert state.has_self_bounce[0]
  assert state.phase[0] == PHASE_AFTER_SELF_BOUNCE

  env.common_step_counter = 2
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 2.0]])
  state.update()
  assert state.paddle_hit_edge[0]
  assert state.has_paddle_hit[0]
  assert state.phase[0] == PHASE_RETURN_FLIGHT
  assert state.hit_valid[0]
  assert state.hit_post_vel.shape == (1, 3)
  assert state.hit_post_vel.device.type == "cpu"
  assert state.hit_post_vel.dtype == torch.float32
  assert state.hit_pred_net_clearance.shape == (1,)
  assert state.hit_pred_landing_inside_opponent_table.dtype == torch.float32
  torch.testing.assert_close(
    state.hit_post_vel[0],
    torch.tensor([-2.5, 0.0, 2.0]),
  )
  assert state.hit_post_speed[0] > 3.0
  assert state.hit_post_vx_toward_opponent_ratio[0] > 0.7
  assert state.hit_pred_net_clearance[0] > 0.0
  assert state.hit_pred_net_clearance_positive[0] == 1.0
  assert state.hit_pred_landing_x[0] < 0.0
  assert state.hit_pred_landing_inside_opponent_table[0] == 1.0
  assert state.hit_paddle_speed[0] == 0.0

  env.common_step_counter = 3
  paddle_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.1, 0.0, NET_TOP_Z + 0.08]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, -0.5]])
  state.update()
  assert state.crossed_net_edge[0]
  assert not state.fault_edge[0]

  env.common_step_counter = 4
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.75, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-1.6, 0.0, 0.8]])
  state.update()
  assert state.opponent_bounce_edge[0]
  assert state.successful_return_edge[0]
  assert state.successful_return_count[0] == 1
  assert state.phase[0] == PHASE_DONE
  assert not state.fault_edge[0]


def test_pingpong_state_rejects_pre_bounce_hit() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  state = _make_state(env)

  state.update()
  env.common_step_counter = 1
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.6, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, -0.4]])
  state.update()

  assert state.fault_edge[0]
  assert state.fault_reason[0] == FAULT_ILLEGAL_PRE_BOUNCE_HIT
  assert not state.paddle_hit_edge[0]


def test_pingpong_state_rejects_out_of_bounds_return_bounce() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  state = _make_state(env)

  state.update()
  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()

  env.common_step_counter = 2
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, -0.3]])
  state.update()

  env.common_step_counter = 3
  paddle_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.1, 0.0, NET_TOP_Z + 0.08]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, -0.5]])
  state.update()

  env.common_step_counter = 4
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.75, 1.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-1.6, 0.0, 0.8]])
  state.update()

  assert state.fault_edge[0]
  assert state.fault_reason[0] == FAULT_RETURN_BOUNCE_OUT
  assert not state.successful_return_edge[0]


def test_pingpong_state_rejects_low_net_return_crossing() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  state = _make_state(env)

  state.update()
  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()

  env.common_step_counter = 2
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 0.3]])
  state.update()

  env.common_step_counter = 3
  paddle_sensor.data.force.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.1, 0.0, NET_TOP_Z - 0.01]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, -0.5]])
  state.update()

  assert state.fault_edge[0]
  assert state.fault_reason[0] == FAULT_LOW_NET_CROSS
  assert not state.crossed_net_edge[0]
  assert not state.successful_return_edge[0]


def test_pingpong_state_rejects_body_ball_contact() -> None:
  env, ball, _, _, body_ball_sensor = _make_env()
  state = _make_state(env)

  state.update()
  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()

  env.common_step_counter = 2
  body_ball_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[1.0, 0.0, 0.1]])
  state.update()

  assert state.fault_edge[0]
  assert state.fault_reason[0] == FAULT_ILLEGAL_BODY_BALL_CONTACT
  assert not state.paddle_hit_edge[0]
  assert state.phase[0] == PHASE_DONE


def test_pingpong_state_body_contact_overrides_same_step_paddle_hit() -> None:
  env, ball, paddle_sensor, _, body_ball_sensor = _make_env()
  state = _make_state(env)

  state.update()
  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()

  env.common_step_counter = 2
  paddle_sensor.data.force[:] = 5.0
  body_ball_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 0.3]])
  state.update()

  assert state.fault_edge[0]
  assert state.fault_reason[0] == FAULT_ILLEGAL_BODY_BALL_CONTACT
  assert not state.paddle_hit_edge[0]
  assert not state.has_paddle_hit[0]
  assert state.paddle_hit_count[0] == 0


def test_pingpong_state_records_impact_window_and_followthrough() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  state = _make_state(env)
  paddle_face_toward_opponent = (0.70710677, 0.0, -0.70710677, 0.0)

  _set_paddle_robot(
    env,
    pos=(0.85, 0.0, 1.05),
    vel=(-1.6, 0.0, 0.0),
    quat=paddle_face_toward_opponent,
  )
  state.update()
  assert not state.impact_window_active[0]
  assert state.impact_desired_outgoing_dir.shape == (1, 3)

  env.common_step_counter = 1
  _set_paddle_robot(
    env,
    pos=(1.45, 0.0, 1.40),
    vel=(-1.6, 0.0, 0.0),
    quat=paddle_face_toward_opponent,
  )
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  state.update()
  assert state.self_bounce_edge[0]
  assert not state.impact_window_active[0]

  env.common_step_counter = 2
  _set_paddle_robot(
    env,
    pos=(0.86, 0.0, 1.05),
    vel=(-1.6, 0.0, 0.0),
    quat=paddle_face_toward_opponent,
  )
  ball.data.root_link_pos_w[:] = torch.tensor([[0.88, 0.0, 1.06]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[1.2, 0.0, 0.2]])
  state.update()
  assert state.impact_window_active[0]
  assert state.impact_window_count[0] == 1
  assert state.impact_center_distance[0] < 0.05
  assert state.impact_velocity_to_target[0] > 1.0
  assert state.impact_velocity_along_normal[0] > 1.0
  assert state.impact_normal_to_target[0] > 0.9
  assert state.impact_paddle_speed.shape == (1,)

  env.common_step_counter = 3
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.88, 0.0, 1.06]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 2.0]])
  state.update()
  assert state.paddle_hit_edge[0]
  assert state.impact_window_active[0]
  assert state.followthrough_active[0]
  assert state.impact_followthrough_velocity[0] > 1.0

  env.common_step_counter = 4
  paddle_sensor.data.force.zero_()
  _set_paddle_robot(
    env,
    pos=(0.75, 0.0, 1.05),
    vel=(-1.2, 0.0, 0.0),
    quat=paddle_face_toward_opponent,
  )
  ball.data.root_link_pos_w[:] = torch.tensor([[0.70, 0.0, 1.10]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.0, 0.0, 1.2]])
  state.update()
  assert not state.impact_window_active[0]
  assert state.followthrough_active[0]
  assert state.impact_followthrough_velocity[0] > 0.5
