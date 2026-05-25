from pathlib import Path
from typing import cast

import mujoco
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tennis.config.g1.rl_cfg import (
  DEFAULT_CONTINUOUS_RESUME_CHECKPOINT,
  DEFAULT_CROSS_LAB_RESUME_CHECKPOINT,
  DEFAULT_CROSS_RESUME_CHECKPOINT,
)
from mjlab.tasks.tennis.mdp import (
  FrozenDecoderLatentJointPositionAction,
  FrozenDecoderLatentJointPositionActionCfg,
  SonicDecoderTokenJointPositionAction,
  SonicDecoderTokenJointPositionActionCfg,
  apply_latent_action_barrier,
)
from mjlab.tasks.tennis.mdp.ball_providers import RandomFeederCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg
from mjlab.tasks.tennis.tennis_env_cfg import (
  BALL_SPAWN_X_RANGE,
  BALL_SPAWN_Z_RANGE,
  COURT_HALF_LENGTH,
  COURT_HALF_WIDTH,
  DEFAULT_COURT_SIZE,
  ROBOT_RESET_YAW,
  TennisLatentEnvCfg,
  resolve_court_scale,
)


def test_tennis_task_registered() -> None:
  assert "Mjlab-Tennis-Hit-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Hit-LAB-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Cross-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Cross-LAB-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Continuous-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Hit-SONIC-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Cross-SONIC-Unitree-G1" in list_tasks()


def test_tennis_task_scene_compiles() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()

  geom_names = {model.geom(i).name for i in range(model.ngeom)}
  sensor_names = {model.sensor(i).name for i in range(model.nsensor)}

  assert "robot/tennis_racket_collision" in geom_names
  assert "ball/tennis_ball" in geom_names
  assert "court/tennis_net_collision" in geom_names
  assert any(name.startswith("racket_ball_contact") for name in sensor_names)


def test_tennis_rl_config_loads() -> None:
  cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Hit-Unitree-G1"),
  )
  assert cfg.experiment_name == "g1_tennis_latent_hit"
  assert cfg.require_decoder_checkpoint is True

  hit_lab_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Hit-LAB-Unitree-G1"),
  )
  assert hit_lab_cfg.experiment_name == "g1_tennis_latent_hit_lab"
  assert hit_lab_cfg.run_name == "tennis_hit_lab_scratch"
  assert hit_lab_cfg.resume is False
  assert hit_lab_cfg.load_checkpoint_file is None
  assert hit_lab_cfg.algorithm.entropy_coef == 0.003
  assert hit_lab_cfg.max_iterations == 30000

  cross_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-Unitree-G1"),
  )
  assert cross_cfg.experiment_name == "g1_tennis_latent_cross"
  assert cross_cfg.run_name == "tennis_cross_from_hit"
  assert cross_cfg.resume is True
  assert cross_cfg.load_checkpoint_file == DEFAULT_CROSS_RESUME_CHECKPOINT
  assert cross_cfg.actor.distribution_cfg is not None
  assert "std_range" not in cross_cfg.actor.distribution_cfg
  assert cross_cfg.algorithm.entropy_coef == 0.003

  cross_lab_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-LAB-Unitree-G1"),
  )
  assert cross_lab_cfg.experiment_name == "g1_tennis_latent_cross_lab"
  assert cross_lab_cfg.run_name == "tennis_cross_lab_finetune"
  assert cross_lab_cfg.resume is True
  assert cross_lab_cfg.load_checkpoint_file == DEFAULT_CROSS_LAB_RESUME_CHECKPOINT
  assert cross_lab_cfg.algorithm.entropy_coef == 0.001
  assert cross_lab_cfg.max_iterations == 20000

  continuous_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Continuous-Unitree-G1"),
  )
  assert continuous_cfg.experiment_name == "g1_tennis_latent_continuous"
  assert continuous_cfg.run_name == "tennis_continuous_from_cross"
  assert continuous_cfg.resume is True
  assert continuous_cfg.load_checkpoint_file == DEFAULT_CONTINUOUS_RESUME_CHECKPOINT
  assert continuous_cfg.algorithm.entropy_coef == 0.003

  sonic_hit_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Hit-SONIC-Unitree-G1"),
  )
  assert sonic_hit_cfg.experiment_name == "g1_tennis_sonic_hit"
  assert sonic_hit_cfg.run_name == "tennis_hit_sonic_token"
  assert sonic_hit_cfg.resume is False
  assert sonic_hit_cfg.require_decoder_checkpoint is False
  assert sonic_hit_cfg.clip_actions == 1.0
  assert sonic_hit_cfg.actor.distribution_cfg is not None
  assert sonic_hit_cfg.actor.distribution_cfg["init_std"] == 0.2

  sonic_cross_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-SONIC-Unitree-G1"),
  )
  assert sonic_cross_cfg.experiment_name == "g1_tennis_sonic_cross"
  assert sonic_cross_cfg.run_name == "tennis_cross_sonic_scratch"
  assert sonic_cross_cfg.resume is False
  assert sonic_cross_cfg.require_decoder_checkpoint is False
  assert sonic_cross_cfg.clip_actions == 1.0


def test_tennis_env_uses_latent_actions() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    action = env.action_manager.get_term("latent_joint_pos")
    assert isinstance(action, FrozenDecoderLatentJointPositionAction)
    assert env.action_manager.total_action_dim == 16
    assert action.low_level_action_dim == 29

    obs, _ = env.reset()
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape[-1] == 127
    latent = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    env.step(latent)
    assert action.low_level_action.shape == (env.num_envs, 29)
  finally:
    env.close()


def test_tennis_decoder_state_terms_align_with_distillation() -> None:
  tennis_cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")
  distill_cfg = cast(
    DistillationRunnerCfg,
    load_rl_cfg("Mjlab-Distill-Flat-Unitree-G1"),
  )
  action = tennis_cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  assert tuple(action.decoder_state_terms) == tuple(distill_cfg.state_terms)


def test_tennis_sonic_env_uses_token_actions() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-SONIC-Unitree-G1")
  action_cfg = cfg.actions["latent_joint_pos"]
  assert isinstance(action_cfg, SonicDecoderTokenJointPositionActionCfg)
  assert action_cfg.token_dim == 64
  assert action_cfg.history_length == 10
  assert action_cfg.decoder_onnx_path == "ckpt/GEAR-SONIC/model_decoder.onnx"
  assert action_cfg.scale == 1.0

  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    action = env.action_manager.get_term("latent_joint_pos")
    assert isinstance(action, SonicDecoderTokenJointPositionAction)
    assert env.action_manager.total_action_dim == 64
    assert action.low_level_action_dim == 29
  finally:
    env.close()


def test_tennis_latent_action_barrier_config() -> None:
  hit_cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")
  hit_action = hit_cfg.actions["latent_joint_pos"]
  assert isinstance(hit_action, FrozenDecoderLatentJointPositionActionCfg)
  assert hit_action.use_latent_action_barrier is False

  hit_lab_cfg = load_env_cfg("Mjlab-Tennis-Hit-LAB-Unitree-G1")
  hit_lab_action = hit_lab_cfg.actions["latent_joint_pos"]
  assert isinstance(hit_lab_action, FrozenDecoderLatentJointPositionActionCfg)
  assert hit_lab_action.use_latent_action_barrier is True
  assert hit_lab_action.latent_barrier_scale == 1.0
  assert hit_lab_action.latent_barrier_min_std == 0.05
  assert hit_lab_action.latent_barrier_max_std == 2.0

  cross_cfg = load_env_cfg("Mjlab-Tennis-Cross-Unitree-G1")
  cross_action = cross_cfg.actions["latent_joint_pos"]
  assert isinstance(cross_action, FrozenDecoderLatentJointPositionActionCfg)
  assert cross_action.use_latent_action_barrier is False

  cross_lab_cfg = load_env_cfg("Mjlab-Tennis-Cross-LAB-Unitree-G1")
  cross_lab_action = cross_lab_cfg.actions["latent_joint_pos"]
  assert isinstance(cross_lab_action, FrozenDecoderLatentJointPositionActionCfg)
  assert cross_lab_action.use_latent_action_barrier is True
  assert cross_lab_action.latent_barrier_scale == 1.5
  assert cross_lab_action.latent_barrier_min_std == 0.05
  assert cross_lab_action.latent_barrier_max_std == 2.0

  continuous_cfg = load_env_cfg("Mjlab-Tennis-Continuous-Unitree-G1")
  continuous_action = continuous_cfg.actions["latent_joint_pos"]
  assert isinstance(continuous_action, FrozenDecoderLatentJointPositionActionCfg)
  assert continuous_action.use_latent_action_barrier is False


def test_apply_latent_action_barrier_bounds_residual() -> None:
  action = torch.tensor([[-100.0, 0.0, 100.0]])
  prior_mean = torch.tensor([[1.0, -2.0, 3.0]])
  prior_std = torch.tensor([[0.001, 0.5, 10.0]])

  latent = apply_latent_action_barrier(
    action,
    prior_mean,
    prior_std,
    scale=2.0,
    min_std=0.05,
    max_std=2.0,
  )

  expected = torch.tensor([[0.9, -2.0, 7.0]])
  assert torch.allclose(latent, expected)


def test_tennis_hit_rewards_and_terminations_end_on_first_hit() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")

  assert "approach_point" in cfg.rewards
  assert "racket_towards_ball" in cfg.rewards
  assert "racket_hit_event" in cfg.rewards
  assert "post_hit_x_progress" not in cfg.rewards

  assert "first_racket_hit" in cfg.terminations
  assert "second_contact" in cfg.terminations

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["success_term_name"] == "first_racket_hit"

  ball_bounds = cfg.terminations["ball_out_of_bounds"].params["x_limits"]
  assert ball_bounds[0] <= -3.0


def test_tennis_cross_rewards_and_terminations_target_landing() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Cross-Unitree-G1")

  assert "racket_hit_event" in cfg.rewards
  assert cfg.rewards["approach_point"].weight == 5.0
  assert cfg.rewards["racket_towards_ball"].weight == 2.0
  assert cfg.rewards["racket_hit_event"].weight == 25.0
  assert "post_hit_x_progress" in cfg.rewards
  assert cfg.rewards["post_hit_x_progress"].weight == 50.0
  assert cfg.rewards["post_hit_x_progress"].params["max_progress"] == 0.05
  assert "post_hit_ball_velocity_direction" in cfg.rewards
  assert cfg.rewards["post_hit_ball_velocity_direction"].weight == 20.0
  assert "crossed_net_event" in cfg.rewards
  assert cfg.rewards["crossed_net_event"].weight == 500.0
  assert "landing_in_bounds_event" in cfg.rewards
  assert cfg.rewards["landing_in_bounds_event"].weight == 1000.0

  assert "first_racket_hit" not in cfg.terminations
  assert "second_contact" in cfg.terminations
  assert "landing_in_bounds_after_hit" in cfg.terminations

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["success_term_name"] == "landing_in_bounds_after_hit"

  landing_params = cfg.terminations["landing_in_bounds_after_hit"].params
  assert landing_params["landing_x_limits"][1] == 0.0
  assert landing_params["landing_y_limits"][0] < 0.0
  assert landing_params["landing_y_limits"][1] > 0.0


def test_tennis_cross_lab_rewards_bias_toward_post_hit_return() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Cross-LAB-Unitree-G1")

  assert cfg.rewards["approach_point"].weight == 2.0
  assert cfg.rewards["racket_towards_ball"].weight == 1.0
  assert cfg.rewards["racket_hit_event"].weight == 5.0
  assert cfg.rewards["post_hit_x_progress"].weight == 80.0
  assert cfg.rewards["post_hit_ball_velocity_direction"].weight == 50.0
  assert cfg.rewards["crossed_net_event"].weight == 700.0
  assert cfg.rewards["landing_in_bounds_event"].weight == 1500.0


def test_tennis_continuous_respawns_until_eight_successful_returns() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Continuous-Unitree-G1")

  assert "landing_in_bounds_event" in cfg.rewards
  assert "continuous_rally_complete_bonus" in cfg.rewards
  assert "respawn_successful_continuous_rally_ball" in cfg.rewards
  assert cfg.rewards["respawn_successful_continuous_rally_ball"].weight == 1.0e-9

  assert "landing_in_bounds_after_hit" not in cfg.terminations
  assert "second_contact" not in cfg.terminations
  assert "continuous_rally_failure" in cfg.terminations
  assert "continuous_rally_complete" in cfg.terminations

  complete_params = cfg.terminations["continuous_rally_complete"].params
  assert complete_params["max_successful_returns"] == 8
  respawn_params = cfg.rewards["respawn_successful_continuous_rally_ball"].params
  assert respawn_params["max_successful_returns"] == 8
  assert (
    respawn_params["provider_cfg"] is cfg.events["reset_ball"].params["provider_cfg"]
  )

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["success_term_name"] == "continuous_rally_complete"


def test_tennis_reset_ranges_face_opponent_half() -> None:
  cfg = cast(TennisLatentEnvCfg, load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1"))
  scale = resolve_court_scale(DEFAULT_COURT_SIZE)
  cl = COURT_HALF_LENGTH * scale
  cw = COURT_HALF_WIDTH * scale
  spawn_y_range = (-cw * 0.83, cw * 0.83)
  robot_reset_x_range = (cl * 0.50, cl * 0.64)
  robot_reset_y_range = (-cw * 0.17, cw * 0.17)
  robot_reset_x_center = 0.5 * (robot_reset_x_range[0] + robot_reset_x_range[1])
  target_initial_x_range = (
    robot_reset_x_center - 0.15 * scale,
    robot_reset_x_center + 0.15 * scale,
  )
  target_initial_y_range = (-0.15 * scale, 0.15 * scale)
  target_x_range = (max(0.3, 0.8 * scale), max(0.5, cl - 0.8 * scale))
  target_y_range = (-cw, cw)

  robot_reset = cfg.events["reset_robot_base"].params
  robot_pose = robot_reset["pose_range"]
  assert cfg.court_size == DEFAULT_COURT_SIZE
  assert robot_pose["x"] == robot_reset_x_range
  assert robot_pose["y"] == robot_reset_y_range
  assert robot_pose["yaw"] == (ROBOT_RESET_YAW, ROBOT_RESET_YAW)

  ball_reset = cfg.events["reset_ball"].params
  provider_cfg = ball_reset["provider_cfg"]
  assert isinstance(provider_cfg, RandomFeederCfg)
  assert provider_cfg.spawn_x_range == BALL_SPAWN_X_RANGE
  assert provider_cfg.spawn_y_range == spawn_y_range
  assert provider_cfg.spawn_z_range == BALL_SPAWN_Z_RANGE
  # Curriculum starts with initial ranges
  assert provider_cfg.target_x_range == target_initial_x_range
  assert provider_cfg.target_y_range == target_initial_y_range

  # Curriculum expands from initial to final ranges
  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["initial_target_x_range"] == target_initial_x_range
  assert curriculum_params["initial_target_y_range"] == target_initial_y_range
  assert curriculum_params["final_target_x_range"] == target_x_range
  assert curriculum_params["final_target_y_range"] == target_y_range


def test_standalone_tennis_scene_compiles() -> None:
  xml_path = (
    Path(__file__).parents[1]
    / "src/mjlab/asset_zoo/robots/unitree_g1_w_racket/xml"
    / "scene_mjx_racket_tennis_return.xml"
  )
  model = mujoco.MjModel.from_xml_path(str(xml_path))

  assert model.nq == 43
  assert model.nu == 0
