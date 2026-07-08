from types import SimpleNamespace
from typing import Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.debug_vis import install_pingpong_debug_overlay
from mjlab.tasks.pingpong.mdp.pace import (
  get_pingpong_pace_prediction_state,
  pace_ball_prediction_table,
  pace_body_orientation_l2,
  pace_feet_force,
  pace_feet_slide_contact,
  pace_feet_stumble,
  pace_feet_too_near,
  pace_fly,
  pace_fly_height,
  pace_forehand_elbow_extension,
  pace_forehand_paddle_offset,
  pace_future_base_vel_target,
  pace_future_pass_net,
  pace_hit_unstable_support,
  pace_hit_unstable_support_height,
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
from mjlab.tasks.pingpong.pace_geometry import G1_PACE_GEOMETRY
from mjlab.tasks.pingpong.scene import (
  BALL_CENTER_TABLE_Z,
  NET_TOP_Z,
  TABLE_HALF_LENGTH,
)

_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)


class _Scene(dict):
  def __init__(self, env_origins, items):
    super().__init__(items)
    self.env_origins = env_origins


class _CollectingVisualizer:
  env_idx = 0
  show_all_envs = False

  def __init__(self) -> None:
    self.spheres: list[tuple[Any, float, Any, str | None]] = []
    self.arrows: list[tuple[Any, Any, Any, float, str | None]] = []
    self.cylinders: list[tuple[Any, Any, float, Any, str | None]] = []

  @property
  def meansize(self) -> float:
    return 1.0

  def get_env_indices(self, num_envs: int):
    del num_envs
    return [0]

  def add_sphere(self, center, radius, color, label=None) -> None:
    self.spheres.append((center, radius, color, label))

  def add_arrow(self, start, end, color, width=0.015, label=None) -> None:
    self.arrows.append((start, end, color, width, label))

  def add_cylinder(self, start, end, radius, color, label=None) -> None:
    self.cylinders.append((start, end, radius, color, label))

  def add_ghost_mesh(self, *args, **kwargs) -> None:
    del args, kwargs

  def add_frame(self, *args, **kwargs) -> None:
    del args, kwargs

  def add_ellipsoid(self, *args, **kwargs) -> None:
    del args, kwargs

  def clear(self) -> None:
    self.spheres.clear()
    self.arrows.clear()
    self.cylinders.clear()


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
  foot_contact_sensor = SimpleNamespace(
    cfg=SimpleNamespace(num_slots=1),
    primary_names=list(_FOOT_GEOM_NAMES),
    data=SimpleNamespace(
      force_history=torch.zeros(1, 14, 4, 3, dtype=torch.float32),
      force=torch.zeros(1, 14, 3, dtype=torch.float32),
      found=torch.zeros(1, 14, dtype=torch.float32),
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
        "pace_foot_contact": foot_contact_sensor,
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
    "target_base_offset_xy": (0.2541, -0.6239),
    "natural_hit_x": G1_PACE_GEOMETRY.natural_hit_x,
    "target_root_height": 0.760,
    "target_base_vel_gain": 4.0,
    "target_base_vel_max": 7.0,
  }


def _add_pace_robot(env: Any) -> Any:
  def find_sites(names: list[str], preserve_order: bool = False):
    del preserve_order
    return [0 for _ in names], names

  def find_geoms(names: list[str], preserve_order: bool = False):
    del preserve_order
    return [0 for _ in names], names

  robot = SimpleNamespace(
    site_names=("pingpong_paddle_center",),
    geom_names=("pingpong_paddle_collision",),
    num_sites=1,
    num_geoms=1,
    find_sites=find_sites,
    find_geoms=find_geoms,
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.55, 0.0, 0.76]], dtype=torch.float32),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
      root_link_lin_vel_w=torch.zeros(1, 3, dtype=torch.float32),
      root_link_ang_vel_b=torch.zeros(1, 3, dtype=torch.float32),
      heading_w=torch.zeros(1, dtype=torch.float32),
      body_link_pos_w=torch.tensor(
        [
          [
            [1.55, -0.08, 0.04],
            [1.55, 0.08, 0.04],
            [1.55, -0.10, 1.00],
            [1.64, -0.12, 0.90],
            [1.82, -0.15, 0.86],
          ]
        ],
        dtype=torch.float32,
      ),
      body_link_lin_vel_w=torch.zeros(1, 5, 3, dtype=torch.float32),
      body_link_quat_w=torch.tensor(
        [
          [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
          ]
        ],
        dtype=torch.float32,
      ),
      site_pos_w=torch.tensor([[[1.8041, -0.6239, 0.8042]]], dtype=torch.float32),
      geom_pos_w=torch.tensor([[[1.8041, -0.6239, 0.8042]]], dtype=torch.float32),
      geom_lin_vel_w=torch.zeros(1, 1, 3, dtype=torch.float32),
      geom_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
    )
  )
  env.scene["robot"] = robot
  env.episode_length_buf = torch.ones(1, dtype=torch.long)
  return robot


def _shift_world_positions(env: Any, robot: Any, origin: torch.Tensor) -> None:
  env.scene.env_origins[:] = origin
  env.scene["ball"].data.root_link_pos_w += origin
  robot.data.root_link_pos_w += origin
  robot.data.body_link_pos_w += origin[:, None, :]
  robot.data.site_pos_w += origin[:, None, :]
  robot.data.geom_pos_w += origin[:, None, :]


def _sphere_center(visualizer: _CollectingVisualizer, label: str) -> torch.Tensor:
  for center, _, _, sphere_label in visualizer.spheres:
    if sphere_label == label:
      return torch.as_tensor(center, dtype=torch.float32)
  raise AssertionError(f"Missing sphere label: {label}")


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
  torch.testing.assert_close(
    state.ball_future_pose[:, 0],
    torch.tensor([G1_PACE_GEOMETRY.natural_hit_x], dtype=torch.float32),
  )
  assert not torch.allclose(
    state.ball_future_pose[:, 0],
    torch.tensor([TABLE_HALF_LENGTH], dtype=torch.float32),
  )
  torch.testing.assert_close(
    state.target_base_xy - state.ball_future_pose[:, :2],
    torch.tensor([[0.2541, -0.6239]], dtype=torch.float32),
  )
  torch.testing.assert_close(
    state.robot_future_pos[:, 2],
    torch.tensor([0.760], dtype=torch.float32),
  )
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


def test_pingpong_pace_future_pose_uses_natural_hit_plane_after_bounce() -> None:
  env, ball, _, _, _ = _make_env()
  _add_pace_robot(env)
  ball.data.root_link_pos_w[:] = torch.tensor([[1.05, 0.02, 0.98]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.10, 0.80]])
  params = _make_pace_params()

  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()

  torch.testing.assert_close(
    state.ball_future_pose[:, 0],
    torch.tensor([G1_PACE_GEOMETRY.natural_hit_x], dtype=torch.float32),
  )
  torch.testing.assert_close(
    state.target_base_xy - state.ball_future_pose[:, :2],
    torch.tensor([[0.2541, -0.6239]], dtype=torch.float32),
  )
  assert state.ball_future_valid.item()


def test_pingpong_pace_future_pose_invalid_case_is_finite_and_inactive() -> None:
  env, ball, _, _, _ = _make_env()
  _add_pace_robot(env)
  ball.data.root_link_pos_w[:] = torch.tensor([[-0.15, 0.0, 0.60]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-1.0, 0.0, -0.20]])
  params = _make_pace_params()

  state = get_pingpong_pace_prediction_state(env, **params)
  state.update()

  assert torch.isfinite(state.ball_future_pose).all()
  assert torch.isfinite(state.target_base_xy).all()
  assert not state.ball_future_valid.item()
  assert not state.reward_active.item()


def test_pingpong_debug_overlay_installs_without_replacing_existing_visualizers() -> None:
  env, _, _, _, _ = _make_env()
  robot = _add_pace_robot(env)
  origin = torch.tensor([[10.0, -2.0, 0.0]], dtype=torch.float32)
  _shift_world_positions(env, robot, origin)
  calls: list[str] = []

  def original_update(_visualizer) -> None:
    calls.append("original")

  env.update_visualizers = original_update
  install_pingpong_debug_overlay(env)
  install_pingpong_debug_overlay(env)

  visualizer = _CollectingVisualizer()
  env.update_visualizers(visualizer)

  assert calls == ["original"]
  sphere_labels = {label for _, _, _, label in visualizer.spheres}
  arrow_labels = {label for _, _, _, _, label in visualizer.arrows}
  cylinder_labels = {label for _, _, _, _, label in visualizer.cylinders}
  assert "ball" in sphere_labels
  assert "paddle_center" in sphere_labels
  assert "forehand_paddle_target" in sphere_labels
  assert "pace_future_ball_pose" in sphere_labels
  assert "paddle_normal" in arrow_labels
  assert "ball_velocity" in arrow_labels
  assert "ball_trajectory" in cylinder_labels
  assert "paddle_to_future_ball_pose" in cylinder_labels
  torch.testing.assert_close(
    _sphere_center(visualizer, "ball"),
    env.scene["ball"].data.root_link_pos_w[0],
  )
  torch.testing.assert_close(
    _sphere_center(visualizer, "paddle_center"),
    robot.data.site_pos_w[0, 0],
  )
  assert _sphere_center(visualizer, "pace_future_ball_pose")[0] > origin[0, 0]


def test_pingpong_pace_forehand_rewards_prefer_g1_paddle_geometry() -> None:
  env, _, _, _, _ = _make_env()
  robot = _add_pace_robot(env)
  params = _make_pace_params()
  paddle_cfg = SceneEntityCfg("robot")
  paddle_cfg.site_ids = [0]
  shoulder_cfg = SceneEntityCfg("robot")
  shoulder_cfg.body_ids = [2]
  elbow_cfg = SceneEntityCfg("robot")
  elbow_cfg.body_ids = [3]
  wrist_cfg = SceneEntityCfg("robot")
  wrist_cfg.body_ids = [4]

  good_offset = pace_forehand_paddle_offset(
    env,
    paddle_cfg=paddle_cfg,
    target_offset=(0.2541, -0.6239, 0.0442),
    offset_std=(0.15, 0.14, 0.08),
    **params,
  )
  extension = pace_forehand_elbow_extension(
    env,
    shoulder_cfg=shoulder_cfg,
    elbow_cfg=elbow_cfg,
    wrist_cfg=wrist_cfg,
    **params,
  )
  robot.data.site_pos_w[:] = torch.tensor([[[1.55, 0.15, 0.90]]])
  robot.data.body_link_pos_w[:, 4] = robot.data.body_link_pos_w[:, 3]
  bad_offset = pace_forehand_paddle_offset(
    env,
    paddle_cfg=paddle_cfg,
    target_offset=(0.2541, -0.6239, 0.0442),
    offset_std=(0.15, 0.14, 0.08),
    **params,
  )
  folded_extension = pace_forehand_elbow_extension(
    env,
    shoulder_cfg=shoulder_cfg,
    elbow_cfg=elbow_cfg,
    wrist_cfg=wrist_cfg,
    **params,
  )

  assert torch.isfinite(good_offset).all()
  assert torch.isfinite(extension).all()
  assert good_offset[0] > bad_offset[0]
  assert extension[0] > folded_extension[0]


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


def test_pingpong_pace_prediction_refreshes_after_auto_reset_same_step() -> None:
  env, ball, _, _, _ = _make_env()
  _add_pace_robot(env)
  params = _make_pace_params()
  state = get_pingpong_pace_prediction_state(env, **params)

  env.common_step_counter = 4
  state.update()
  state.ball_future_pose[:] = torch.nan
  state.target_base_xy[:] = torch.nan
  state.robot_future_pos[:] = torch.nan
  state._last_step = env.common_step_counter

  env.episode_length_buf.zero_()
  ball.data.root_link_pos_w[:] = torch.tensor([[0.7, 0.0, 1.0]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, -1.0]])
  rel_target = pace_relative_target_base_xy(env, **params)

  assert torch.isfinite(rel_target).all()
  assert torch.isfinite(state.ball_future_pose).all()
  assert torch.isfinite(state.target_base_xy).all()


def test_pingpong_pace_height_stability_rewards_are_finite() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  robot = _add_pace_robot(env)
  params = _make_pace_params()
  feet_cfg = SceneEntityCfg("robot")
  feet_cfg.body_ids = [0, 1]
  left_foot_cfg = SceneEntityCfg("robot")
  left_foot_cfg.body_ids = [0]
  right_foot_cfg = SceneEntityCfg("robot")
  right_foot_cfg.body_ids = [1]

  torch.testing.assert_close(
    pace_fly_height(env, feet_cfg=feet_cfg, contact_height=0.08),
    torch.zeros(1),
  )
  assert torch.isfinite(pace_body_orientation_l2(env, body_cfg=left_foot_cfg)).all()
  torch.testing.assert_close(
    pace_feet_too_near(env, feet_cfg=feet_cfg, threshold=0.20),
    torch.tensor([0.04]),
  )

  robot.data.body_link_pos_w[..., 2] = 0.20
  torch.testing.assert_close(
    pace_fly_height(env, feet_cfg=feet_cfg, contact_height=0.08),
    torch.ones(1),
  )

  torch.testing.assert_close(
    pace_hit_unstable_support_height(
      env,
      feet_cfg=feet_cfg,
      contact_height=0.08,
      **params,
    ),
    torch.zeros(1),
  )

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  torch.testing.assert_close(
    pace_hit_unstable_support_height(
      env,
      feet_cfg=feet_cfg,
      contact_height=0.08,
      **params,
    ),
    torch.zeros(1),
  )

  env.common_step_counter = 2
  robot.data.body_link_pos_w[:, 0, 2] = 0.04
  robot.data.body_link_pos_w[:, 1, 2] = 0.20
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 2.0]])
  unstable = pace_hit_unstable_support_height(
    env,
    feet_cfg=feet_cfg,
    contact_height=0.08,
    **params,
  )
  torch.testing.assert_close(unstable, torch.ones(1))
  assert torch.isfinite(
    torch.stack(
      (
        pace_body_orientation_l2(env, body_cfg=left_foot_cfg),
        pace_body_orientation_l2(env, body_cfg=right_foot_cfg),
        pace_feet_too_near(env, feet_cfg=feet_cfg, threshold=0.15),
      ),
      dim=-1,
    )
  ).all()


def test_pingpong_pace_contact_stability_rewards_are_finite() -> None:
  env, ball, paddle_sensor, _, _ = _make_env()
  robot = _add_pace_robot(env)
  params = _make_pace_params()
  sensor_name = "pace_foot_contact"
  foot_sensor = env.scene[sensor_name]
  feet_cfg = SceneEntityCfg("robot")
  feet_cfg.body_ids = [0, 1]

  torch.testing.assert_close(
    pace_fly(env, sensor_name=sensor_name),
    torch.ones(1),
  )

  foot_sensor.data.force_history[:, :, :, 2] = 20.0
  foot_sensor.data.force[:, :, 2] = 20.0
  foot_sensor.data.found[:] = 1.0
  torch.testing.assert_close(
    pace_fly(env, sensor_name=sensor_name),
    torch.zeros(1),
  )

  robot.data.body_link_lin_vel_w[:, :2] = torch.tensor(
    [[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]],
    dtype=torch.float32,
  )
  torch.testing.assert_close(
    pace_feet_slide_contact(env, feet_cfg=feet_cfg, sensor_name=sensor_name),
    torch.tensor([5.0]),
  )
  assert torch.isfinite(
    pace_feet_force(
      env,
      sensor_name=sensor_name,
      threshold=10.0,
      max_reward=400.0,
    )
  ).all()
  assert torch.all(
    pace_feet_force(
      env,
      sensor_name=sensor_name,
      threshold=10.0,
      max_reward=400.0,
    )
    > 0.0
  )

  foot_sensor.data.force_history.zero_()
  foot_sensor.data.force_history[:, :, 0, 0] = 20.0
  foot_sensor.data.force_history[:, :, 0, 2] = 1.0
  torch.testing.assert_close(
    pace_feet_stumble(env, sensor_name=sensor_name),
    torch.ones(1),
  )

  foot_sensor.data.force_history.zero_()
  foot_sensor.data.force.zero_()
  foot_sensor.data.found.zero_()
  torch.testing.assert_close(
    pace_hit_unstable_support(env, sensor_name=sensor_name, **params),
    torch.zeros(1),
  )

  env.common_step_counter = 1
  ball.data.root_link_pos_w[:] = torch.tensor([[0.65, 0.0, BALL_CENTER_TABLE_Z]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[2.0, 0.0, 1.0]])
  torch.testing.assert_close(
    pace_hit_unstable_support(env, sensor_name=sensor_name, **params),
    torch.zeros(1),
  )

  env.common_step_counter = 2
  foot_sensor.data.force_history[:, 0, :, 2] = 20.0
  foot_sensor.data.force[:, 0, 2] = 20.0
  foot_sensor.data.found[:, 0] = 1.0
  paddle_sensor.data.force[:] = 5.0
  ball.data.root_link_pos_w[:] = torch.tensor([[0.85, 0.0, 1.05]])
  ball.data.root_link_lin_vel_w[:] = torch.tensor([[-2.5, 0.0, 2.0]])
  torch.testing.assert_close(
    pace_hit_unstable_support(env, sensor_name=sensor_name, **params),
    torch.ones(1),
  )


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
