import inspect
from typing import cast

import mujoco
import torch

import mjlab.tasks  # noqa: F401
from mjlab.scene import Scene
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg
from mjlab.tasks.pingpong.config.g1.env_cfgs import (
  DEFAULT_DECODER_CHECKPOINT,
  PINGPONG_PADDLE_HANDLE_HALF_LENGTH,
  PINGPONG_PADDLE_HANDLE_RADIUS,
  PINGPONG_PADDLE_RADIUS,
  PINGPONG_PADDLE_SCALE,
  get_g1_w_pingpong_paddle_spec,
)
from mjlab.tasks.pingpong.config.g1.rl_cfg import (
  DEFAULT_CROSS_RESUME_CHECKPOINT,
  DEFAULT_RETURN_RESUME_CHECKPOINT,
)
from mjlab.tasks.pingpong.mdp.ball_providers import TableTennisFeederCfg
from mjlab.tasks.pingpong.pingpong_env_cfg import (
  ACTION_REGULARIZATION_CURRICULUM_STAGE_WEIGHTS,
  BALL_TARGET_X_RANGE,
  BALL_TARGET_Y_RANGE,
  CROSS_IMPACT_REWARD_WEIGHTS,
  CROSS_LOOSE_REGULARIZATION_WEIGHTS,
  CROSS_POST_HIT_BALL_VELOCITY_DIRECTION_WEIGHT,
  CROSS_POST_HIT_X_PROGRESS_WEIGHT,
  CROSS_ROBOT_BALL_CONTACT_WEIGHT,
  CROSS_STRIKE_QUALITY_REWARD_WEIGHTS,
  DECODER_STATE_TERMS,
  PADDLE_BALL_PAIR_CONDIM,
  PADDLE_BALL_PAIR_FRICTION,
  PADDLE_BALL_PAIR_GEOM1,
  PADDLE_BALL_PAIR_GEOM2,
  PADDLE_BALL_PAIR_MARGIN,
  PADDLE_BALL_PAIR_NAME,
  PADDLE_BALL_PAIR_SOLIMP,
  PADDLE_BALL_PAIR_SOLREF,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg
from mjlab.tasks.tennis.rl.runner import reset_actor_distribution_std


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
  assert name is not None
  return name


def _find_paddle_ball_pair_id(model: mujoco.MjModel) -> int:
  expected = {PADDLE_BALL_PAIR_GEOM1, PADDLE_BALL_PAIR_GEOM2}
  matches = []
  for pair_id in range(model.npair):
    geom_names = {
      _geom_name(model, int(model.pair_geom1[pair_id])),
      _geom_name(model, int(model.pair_geom2[pair_id])),
    }
    if geom_names == expected:
      matches.append(pair_id)
  assert len(matches) == 1
  return matches[0]


def test_pingpong_tasks_registered() -> None:
  assert "Mjlab-Pingpong-Hit-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Cross-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Cross-Diag-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Cross-Impact-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Cross-StrikeQualityEnergyRelax-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Return-Unitree-G1" in list_tasks()


def test_pingpong_task_scene_compiles() -> None:
  for task_id in (
    "Mjlab-Pingpong-Hit-Unitree-G1",
    "Mjlab-Pingpong-Cross-Unitree-G1",
  ):
    cfg = load_env_cfg(task_id)
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()

    geom_by_name = {model.geom(i).name: i for i in range(model.ngeom)}
    geom_names = set(geom_by_name)
    sensor_names = {model.sensor(i).name for i in range(model.nsensor)}

    assert "robot/pingpong_paddle_collision" in geom_names
    assert "robot/pingpong_paddle_handle_collision" in geom_names
    handle_id = geom_by_name["robot/pingpong_paddle_handle_collision"]
    assert int(model.geom_contype[handle_id]) == 1
    assert int(model.geom_conaffinity[handle_id]) == 1
    assert int(model.geom_group[handle_id]) == 3
    assert float(model.geom_rgba[handle_id, 3]) == 0.0
    assert "ball/pingpong_ball" in geom_names
    assert "table/pingpong_table_top_collision" in geom_names
    assert "table/pingpong_net_collision" in geom_names
    assert any(name.startswith("paddle_ball_contact") for name in sensor_names)
    assert any(name.startswith("pingpong_ball_net_contact") for name in sensor_names)
    assert any(name.startswith("robot_table_contact") for name in sensor_names)
    assert any(name.startswith("robot_ball_contact") for name in sensor_names)


def test_pingpong_paddle_ball_explicit_contact_pair() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  model = Scene(cfg.scene, device="cpu").compile()
  pair_id = _find_paddle_ball_pair_id(model)

  assert model.pair(pair_id).name == PADDLE_BALL_PAIR_NAME
  assert int(model.pair_dim[pair_id]) == PADDLE_BALL_PAIR_CONDIM
  assert abs(float(model.pair_margin[pair_id]) - PADDLE_BALL_PAIR_MARGIN) < 1.0e-9
  torch.testing.assert_close(
    torch.as_tensor(model.pair_friction[pair_id, :3]),
    torch.as_tensor(PADDLE_BALL_PAIR_FRICTION, dtype=torch.float64),
    atol=1.0e-9,
    rtol=0.0,
  )
  torch.testing.assert_close(
    torch.as_tensor(model.pair_solref[pair_id]),
    torch.as_tensor(PADDLE_BALL_PAIR_SOLREF, dtype=torch.float64),
    atol=1.0e-9,
    rtol=0.0,
  )
  torch.testing.assert_close(
    torch.as_tensor(model.pair_solimp[pair_id]),
    torch.as_tensor(PADDLE_BALL_PAIR_SOLIMP, dtype=torch.float64),
    atol=1.0e-9,
    rtol=0.0,
  )


def test_pingpong_paddle_scales_visual_and_collision() -> None:
  spec = get_g1_w_pingpong_paddle_spec()
  mesh_by_name = {mesh.name: mesh for mesh in spec.meshes}
  assert "pingpong_paddle_visual" in mesh_by_name
  visual_mesh = mesh_by_name["pingpong_paddle_visual"]
  assert all(abs(float(v) - PINGPONG_PADDLE_SCALE) < 1.0e-6 for v in visual_mesh.scale)

  geom_by_name = {}
  bodies = list(spec.worldbody.bodies)
  while bodies:
    body = bodies.pop()
    for geom in body.geoms:
      geom_by_name[geom.name] = geom
    bodies.extend(body.bodies)
  assert "pingpong_paddle_visual" in geom_by_name
  assert geom_by_name["pingpong_paddle_visual"].meshname == "pingpong_paddle_visual"
  assert "pingpong_paddle_collision" in geom_by_name
  collision = geom_by_name["pingpong_paddle_collision"]
  assert abs(float(collision.size[0]) - PINGPONG_PADDLE_RADIUS) < 1.0e-6
  assert float(collision.pos[2]) < 0.4
  assert "pingpong_paddle_handle_collision" in geom_by_name
  handle = geom_by_name["pingpong_paddle_handle_collision"]
  assert abs(float(handle.size[0]) - PINGPONG_PADDLE_HANDLE_RADIUS) < 1.0e-6
  assert int(handle.group) == 3
  assert float(handle.rgba[3]) == 0.0
  handle_fromto = [float(v) for v in handle.fromto]
  handle_length = (
    sum((handle_fromto[i + 3] - handle_fromto[i]) ** 2 for i in range(3)) ** 0.5
  )
  assert abs(handle_length * 0.5 - PINGPONG_PADDLE_HANDLE_HALF_LENGTH) < 1.0e-6


def test_pingpong_paddle_handle_collision_does_not_score() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}

  paddle_ball = sensors["paddle_ball_contact"]
  assert isinstance(paddle_ball, ContactSensorCfg)
  assert paddle_ball.secondary is not None
  assert paddle_ball.secondary.pattern == "pingpong_paddle_collision"
  assert "handle" not in paddle_ball.secondary.pattern

  robot_ball = sensors["robot_ball_contact"]
  assert isinstance(robot_ball, ContactSensorCfg)
  assert "pingpong_paddle_collision" in robot_ball.primary.exclude
  assert "pingpong_paddle_handle_collision" not in robot_ball.primary.exclude


def test_pingpong_env_uses_frozen_decoder_action() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  assert action.latent_dim == 16
  assert action.decoder_checkpoint == DEFAULT_DECODER_CHECKPOINT
  assert tuple(action.decoder_state_terms) == DECODER_STATE_TERMS

  distill_cfg = cast(
    DistillationRunnerCfg,
    load_rl_cfg("Mjlab-Distill-Flat-Unitree-G1"),
  )
  assert tuple(action.decoder_state_terms) == tuple(distill_cfg.state_terms)


def test_pingpong_rl_configs_load() -> None:
  hit_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Hit-Unitree-G1"),
  )
  assert hit_cfg.experiment_name == "g1_pingpong_latent_hit"
  assert hit_cfg.run_name == "pingpong_hit_scratch"
  assert hit_cfg.resume is False
  assert hit_cfg.require_decoder_checkpoint is True

  cross_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Cross-Unitree-G1"),
  )
  assert cross_cfg.experiment_name == "g1_pingpong_latent_cross"
  assert cross_cfg.run_name == "pingpong_cross_from_hit"
  assert cross_cfg.resume is bool(DEFAULT_CROSS_RESUME_CHECKPOINT)
  assert cross_cfg.load_checkpoint_file == (DEFAULT_CROSS_RESUME_CHECKPOINT or None)
  assert cross_cfg.algorithm.entropy_coef == 0.002
  assert cross_cfg.algorithm.learning_rate == 5.0e-4
  assert cross_cfg.algorithm.desired_kl == 0.01
  assert cross_cfg.clip_actions == 2.5
  assert cross_cfg.reset_actor_std == 0.8
  assert cross_cfg.max_iterations == 40000

  cross_diag_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Cross-Diag-Unitree-G1"),
  )
  assert cross_diag_cfg.experiment_name == "g1_pingpong_latent_cross_diag"
  assert cross_diag_cfg.run_name == "pingpong_cross_diag_only_from_hit"
  assert cross_diag_cfg.algorithm.entropy_coef == 0.002
  assert cross_diag_cfg.clip_actions == 2.5
  assert cross_diag_cfg.reset_actor_std == 0.8

  strike_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1"),
  )
  assert strike_cfg.experiment_name == "g1_pingpong_latent_cross_strike_quality"
  assert strike_cfg.run_name == "pingpong_cross_strike_quality_from_hit"
  assert strike_cfg.algorithm.entropy_coef == 0.002
  assert strike_cfg.clip_actions == 2.5
  assert strike_cfg.reset_actor_std == 0.8

  impact_rl_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Cross-Impact-Unitree-G1"),
  )
  assert impact_rl_cfg.experiment_name == "g1_pingpong_latent_cross_impact"
  assert impact_rl_cfg.run_name == "pingpong_cross_impact_from_hit"
  assert impact_rl_cfg.algorithm.entropy_coef == 0.002
  assert impact_rl_cfg.clip_actions == 2.5
  assert impact_rl_cfg.reset_actor_std == 0.8
  assert impact_rl_cfg.load_checkpoint_file == (DEFAULT_CROSS_RESUME_CHECKPOINT or None)

  energy_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Cross-StrikeQualityEnergyRelax-Unitree-G1"),
  )
  assert energy_cfg.experiment_name == (
    "g1_pingpong_latent_cross_strike_quality_energy_relax"
  )
  assert energy_cfg.run_name == "pingpong_cross_strike_quality_energy_relax_from_hit"
  assert energy_cfg.algorithm.entropy_coef == 0.003
  assert energy_cfg.algorithm.learning_rate == 7.5e-4
  assert energy_cfg.clip_actions == 3.5
  assert energy_cfg.reset_actor_std == 1.0

  return_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Return-Unitree-G1"),
  )
  assert return_cfg.experiment_name == "g1_pingpong_latent_return"
  assert return_cfg.run_name == "pingpong_return_from_hit"
  assert return_cfg.resume is False
  assert return_cfg.load_checkpoint_file == (DEFAULT_RETURN_RESUME_CHECKPOINT or None)
  assert return_cfg.algorithm.entropy_coef == 0.002
  assert return_cfg.algorithm.learning_rate == 5.0e-4
  assert return_cfg.algorithm.desired_kl == 0.01
  assert return_cfg.clip_actions == 2.5
  assert return_cfg.reset_actor_std == 0.8
  assert return_cfg.max_iterations == 40000


def test_pingpong_hit_and_cross_success_terms() -> None:
  hit_cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  assert "first_paddle_hit" in hit_cfg.terminations
  assert "legal_return_success" not in hit_cfg.terminations
  assert "robot_table_contact" in hit_cfg.rewards
  assert "robot_ball_contact" in hit_cfg.rewards
  assert "robot_table_contact_count" in hit_cfg.metrics
  assert "robot_ball_contact_count" in hit_cfg.metrics
  assert "fault_reason/body_ball" in hit_cfg.metrics
  assert "hit/post_vx" in hit_cfg.metrics
  assert "hit/pred_net_clearance" in hit_cfg.metrics
  assert "hit/pred_landing_inside_opponent_table" in hit_cfg.metrics
  assert "hit/paddle_speed" in hit_cfg.metrics
  assert "robot_table_contact" not in hit_cfg.terminations
  assert hit_cfg.rewards["approach_ball"].weight == 5.0
  assert hit_cfg.rewards["paddle_towards_ball"].weight == 2.0
  assert hit_cfg.rewards["paddle_hit_event"].weight == 2000.0
  assert hit_cfg.rewards["robot_ball_contact"].weight == -50.0
  assert hit_cfg.rewards["joint_torques_l2"].weight == -2.0e-5
  assert hit_cfg.rewards["joint_acc_l2"].weight == -2.0e-6
  assert hit_cfg.rewards["latent_action_rate_l2"].weight == -0.005
  assert hit_cfg.rewards["low_level_action_rate_l2"].weight == -0.02
  assert hit_cfg.rewards["fall_penalty"].weight == -200.0
  assert hit_cfg.rewards["flat_orientation_l2"].weight == 0.0
  assert hit_cfg.curriculum["ball_target_region"].params["success_term_name"] == (
    "first_paddle_hit"
  )
  assert list(hit_cfg.curriculum) == ["ball_target_region", "action_regularization"]
  action_curriculum = hit_cfg.curriculum["action_regularization"]
  assert action_curriculum.params["success_term_name"] == "first_paddle_hit"
  assert action_curriculum.params["success_threshold"] == 0.8
  assert action_curriculum.params["success_window"] == 50
  assert action_curriculum.params["prerequisite_curriculum_name"] == (
    "ball_target_region"
  )
  assert action_curriculum.params["prerequisite_stage_key"] == "stage"
  assert action_curriculum.params["prerequisite_min_stage"] == 5.0
  assert (
    action_curriculum.params["stage_weights"][0]
    == (ACTION_REGULARIZATION_CURRICULUM_STAGE_WEIGHTS[0])
  )

  cross_cfg = load_env_cfg("Mjlab-Pingpong-Cross-Unitree-G1")
  assert "first_paddle_hit" not in cross_cfg.terminations
  assert "legal_return_success" in cross_cfg.terminations
  assert "crossed_net_event" in cross_cfg.rewards
  assert cross_cfg.rewards["crossed_net_event"].weight == 500.0
  assert "opponent_table_bounce_event" in cross_cfg.rewards
  assert cross_cfg.rewards["opponent_table_bounce_event"].weight == 1000.0
  assert "post_hit_x_progress" in cross_cfg.rewards
  assert (
    cross_cfg.rewards["post_hit_x_progress"].weight == CROSS_POST_HIT_X_PROGRESS_WEIGHT
  )
  assert "post_hit_ball_velocity_direction" in cross_cfg.rewards
  assert (
    cross_cfg.rewards["post_hit_ball_velocity_direction"].weight
    == CROSS_POST_HIT_BALL_VELOCITY_DIRECTION_WEIGHT
  )
  assert "robot_table_contact" in cross_cfg.rewards
  assert "robot_ball_contact" in cross_cfg.rewards
  assert (
    cross_cfg.rewards["robot_ball_contact"].weight == CROSS_ROBOT_BALL_CONTACT_WEIGHT
  )
  for reward_name, loose_weight in CROSS_LOOSE_REGULARIZATION_WEIGHTS.items():
    assert cross_cfg.rewards[reward_name].weight == loose_weight
  assert "robot_ball_contact_count" in cross_cfg.metrics
  assert "fault_reason/low_net" in cross_cfg.metrics
  assert "fault_reason/net_contact" in cross_cfg.metrics
  assert "fault_reason/return_out" in cross_cfg.metrics
  assert "fault_reason/failed_bounce" in cross_cfg.metrics
  assert "fault_reason/double_paddle" in cross_cfg.metrics
  assert "fault_reason/early_hit" in cross_cfg.metrics
  assert "hit/post_vx_toward_opponent_ratio" in cross_cfg.metrics
  assert "hit/pred_net_clearance_positive" in cross_cfg.metrics
  assert "hit/pred_landing_x" in cross_cfg.metrics
  assert "hit/pred_landing_y" in cross_cfg.metrics
  assert "hit/paddle_normal_alignment" in cross_cfg.metrics
  assert "hit/paddle_velocity_along_normal" in cross_cfg.metrics
  assert "strike_pred_net_clearance" not in cross_cfg.rewards
  assert "strike_pred_landing_inside" not in cross_cfg.rewards
  assert "impact_paddle_to_target_velocity" not in cross_cfg.rewards
  assert "robot_table_contact" not in cross_cfg.terminations
  assert cross_cfg.terminations["bad_orientation"].params["limit_angle"] < 1.0
  assert cross_cfg.terminations["root_height"].params["minimum_height"] == 0.55
  assert cross_cfg.rewards["approach_ball"].weight == 5.0
  assert cross_cfg.rewards["paddle_towards_ball"].weight == 2.0
  assert cross_cfg.rewards["paddle_hit_event"].weight == 25.0
  assert cross_cfg.curriculum["ball_target_region"].params["success_term_name"] == (
    "legal_return_success"
  )
  assert "action_regularization" not in cross_cfg.curriculum

  diag_cfg = load_env_cfg("Mjlab-Pingpong-Cross-Diag-Unitree-G1")
  assert set(diag_cfg.rewards) == set(cross_cfg.rewards)
  assert set(diag_cfg.metrics) == set(cross_cfg.metrics)

  strike_cfg = load_env_cfg("Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1")
  for reward_name, reward_weight in CROSS_STRIKE_QUALITY_REWARD_WEIGHTS.items():
    assert strike_cfg.rewards[reward_name].weight == reward_weight
  for reward_name, loose_weight in CROSS_LOOSE_REGULARIZATION_WEIGHTS.items():
    assert strike_cfg.rewards[reward_name].weight == loose_weight
  assert "impact_paddle_to_target_velocity" not in strike_cfg.rewards
  assert "impact/velocity_to_target" not in strike_cfg.metrics

  impact_cfg = load_env_cfg("Mjlab-Pingpong-Cross-Impact-Unitree-G1")
  for reward_name, reward_weight in CROSS_STRIKE_QUALITY_REWARD_WEIGHTS.items():
    assert impact_cfg.rewards[reward_name].weight == reward_weight
  for reward_name, reward_weight in CROSS_IMPACT_REWARD_WEIGHTS.items():
    assert impact_cfg.rewards[reward_name].weight == reward_weight
  assert impact_cfg.rewards["impact_paddle_to_target_velocity"].func.__name__ == (
    "impact_paddle_to_target_velocity"
  )
  assert impact_cfg.rewards["followthrough_velocity"].func.__name__ == (
    "followthrough_velocity"
  )
  assert "impact/window_active" in impact_cfg.metrics
  assert impact_cfg.metrics["impact/window_active"].reduce == "mean"
  assert "impact/window_count" in impact_cfg.metrics
  assert "impact/velocity_to_target" in impact_cfg.metrics
  assert "impact/velocity_along_normal" in impact_cfg.metrics
  assert "impact/normal_to_target" in impact_cfg.metrics
  assert "impact/center_distance" in impact_cfg.metrics
  assert "impact/followthrough_velocity" in impact_cfg.metrics

  energy_cfg = load_env_cfg("Mjlab-Pingpong-Cross-StrikeQualityEnergyRelax-Unitree-G1")
  assert energy_cfg.rewards["latent_action_rate_l2"].func.__name__ == (
    "pre_hit_action_rate_l2"
  )
  assert energy_cfg.rewards["low_level_action_rate_l2"].func.__name__ == (
    "pre_hit_low_level_action_rate_l2"
  )
  for reward_name, reward_weight in CROSS_STRIKE_QUALITY_REWARD_WEIGHTS.items():
    assert energy_cfg.rewards[reward_name].weight == reward_weight


def test_pingpong_state_backed_rewards_accept_shared_state_params() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Cross-Impact-Unitree-G1")

  for reward_name in (
    "approach_ball",
    "paddle_towards_ball",
    "impact_paddle_to_target_velocity",
  ):
    reward = cfg.rewards[reward_name]
    signature = inspect.signature(reward.func.__call__)
    assert "params" in signature.parameters


def test_pingpong_return_alias_matches_cross_success_terms() -> None:
  cross_cfg = load_env_cfg("Mjlab-Pingpong-Cross-Unitree-G1")
  return_cfg = load_env_cfg("Mjlab-Pingpong-Return-Unitree-G1")

  assert set(return_cfg.terminations) == set(cross_cfg.terminations)
  assert set(return_cfg.rewards) == set(cross_cfg.rewards)
  assert return_cfg.curriculum["ball_target_region"].params["success_term_name"] == (
    "legal_return_success"
  )
  assert "action_regularization" not in return_cfg.curriculum


def test_pingpong_feeder_curriculum_ranges() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  provider_cfg = cfg.events["reset_ball"].params["provider_cfg"]
  assert isinstance(provider_cfg, TableTennisFeederCfg)
  assert provider_cfg.spawn_x_range_mode == "opponent_side_margin"
  assert provider_cfg.spawn_y_range_mode == "field_fraction"
  assert provider_cfg.target_x_range_mode == "self_baseline_margin"
  assert provider_cfg.target_y_range_mode == "field_fraction"
  assert provider_cfg.check.require_edge_crossing
  assert provider_cfg.check.require_second_bounce_outside_self_half

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["provider_cfg"] is provider_cfg
  assert curriculum_params["final_target_x_range"] == BALL_TARGET_X_RANGE
  assert curriculum_params["final_target_y_range"] == BALL_TARGET_Y_RANGE


def test_pingpong_predicted_hit_point_uses_edge_target() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  ball_window = cfg.observations["actor"].terms["ball_pos_window"]
  assert ball_window.func.__name__ == "ball_position_b"
  assert ball_window.history_length == 10
  assert ball_window.flatten_history_dim
  assert "racket_cfg" not in ball_window.params

  actor_term = cfg.observations["actor"].terms["predicted_hit_point"]
  critic_term = cfg.observations["critic"].terms["ball_predicted_hit_point"]
  assert actor_term.func.__name__ == "ball_predicted_edge_hit_point_b"
  assert critic_term.func.__name__ == "ball_predicted_edge_hit_point_b"
  assert "edge_x" not in actor_term.params
  assert "edge_x" not in critic_term.params


def test_reset_actor_distribution_std_updates_loaded_checkpoint_state() -> None:
  actor_state_dict = {
    "distribution.std_param": torch.full((16,), 12.0),
    "distribution.log_std_param": torch.full((16,), 4.0),
  }

  assert reset_actor_distribution_std(actor_state_dict, 0.5)
  assert torch.allclose(
    actor_state_dict["distribution.std_param"], torch.full((16,), 0.5)
  )
  assert torch.allclose(
    actor_state_dict["distribution.log_std_param"],
    torch.full((16,), torch.log(torch.tensor(0.5)).item()),
  )
