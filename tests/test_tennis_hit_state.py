from types import SimpleNamespace
from typing import Any, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.hit_state import TennisHitState


class _Scene(dict):
  def __init__(self, env_origins, items):
    super().__init__(items)
    self.env_origins = env_origins


def test_tennis_hit_state_tracks_first_hit_and_rally_parity() -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([[[0.75, 0.0, 1.0]]], dtype=torch.float32),
      site_lin_vel_w=torch.zeros(1, 1, 3, dtype=torch.float32),
    )
  )
  ball = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.4, 0.0, 1.0]], dtype=torch.float32),
      root_link_lin_vel_w=torch.tensor([[-1.5, 0.0, 0.0]], dtype=torch.float32),
    )
  )
  sensor = SimpleNamespace(
    data=SimpleNamespace(
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
        "robot": robot,
        "ball": ball,
        "racket_ball_contact": sensor,
      },
    ),
  )
  racket_cfg = SceneEntityCfg("robot")
  racket_cfg.site_ids = [0]

  state = TennisHitState(
    cast(Any, env),
    sensor_name="racket_ball_contact",
    force_threshold=1.0,
    valid_leftward_speed=2.0,
    valid_ball_speed=2.5,
    target_line_x=-2.2,
    miss_x_offset=0.2,
    miss_x_direction=1.0,
    robot_cfg=SceneEntityCfg("robot"),
    racket_cfg=racket_cfg,
    ball_cfg=SceneEntityCfg("ball"),
  )

  state.update()
  assert not state.first_contact[0]
  assert not state.has_valid_hit[0]

  env.common_step_counter = 1
  sensor.data.force[:] = 5.0
  sensor.data.found[:] = 1.0
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-3.0, 0.0, 0.0]])
  state.update()
  assert state.first_contact[0]
  assert state.first_valid_hit[0]
  assert state.valid_hit_edge[0]
  assert state.has_valid_hit[0]
  assert state.contact_count[0] == 1
  assert state.valid_hit_count[0] == 1
  assert state.first_contact_step[0] == 1
  assert state.first_valid_hit_step[0] == 1
  assert state.rally_parity[0]

  env.common_step_counter = 2
  sensor.data.force.zero_()
  sensor.data.found.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[-2.3, 0.0, 1.0]])
  state.update()
  assert state.target_line_crossed_edge[0]

  env.common_step_counter = 3
  sensor.data.force[:] = 5.0
  sensor.data.found[:] = 1.0
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-4.0, 0.0, 0.0]])
  state.update()
  assert state.repeat_contact_after_valid_hit[0]
  assert state.valid_hit_edge[0]
  assert state.valid_hit_count[0] == 2
  assert not state.rally_parity[0]

  state.reset(torch.tensor([0]))
  assert not state.has_valid_hit[0]
  assert state.contact_count[0] == 0
  assert state.first_contact_step[0] == -1
  assert not state.rally_parity[0]
