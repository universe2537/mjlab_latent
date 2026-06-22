from types import SimpleNamespace
from typing import Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.state import (
  FAULT_ILLEGAL_PRE_BOUNCE_HIT,
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


def _make_env() -> tuple[Any, Any, Any, Any]:
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
      },
    ),
  )
  return env, ball, paddle_sensor, net_sensor


def _make_state(env: Any) -> PingpongRallyState:
  return PingpongRallyState(
    env,
    paddle_sensor_name="paddle_ball_contact",
    net_sensor_name="pingpong_ball_net_contact",
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


def test_pingpong_state_counts_legal_single_return() -> None:
  env, ball, paddle_sensor, _ = _make_env()
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
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 0.3]])
  state.update()
  assert state.paddle_hit_edge[0]
  assert state.has_paddle_hit[0]
  assert state.phase[0] == PHASE_RETURN_FLIGHT

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
  env, ball, paddle_sensor, _ = _make_env()
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
  env, ball, paddle_sensor, _ = _make_env()
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
