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
  DEFAULT_CROSS_WRIST_LAB_RESUME_CHECKPOINT,
)
from mjlab.tasks.tennis.mdp import (
  FrozenDecoderLatentJointPositionAction,
  FrozenDecoderLatentJointPositionActionCfg,
  OpponentFeederCfg,
  SonicDecoderTokenJointPositionAction,
  SonicDecoderTokenJointPositionActionCfg,
  apply_latent_action_barrier,
  racket_to_ball_b,
  torso_to_ball_b,
)
from mjlab.tasks.tennis.mdp.ball_providers import RandomFeederCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg
from mjlab.tasks.tennis.rl.runner import (
  expand_actor_action_head_for_wrist_residual,
  expand_mlp_input_for_observation,
)
from mjlab.tasks.tennis.tennis_env_cfg import (
  BALL_SPAWN_X_RANGE,
  BALL_SPAWN_Z_RANGE,
  CONTINUOUS_FEED_MAX_APEX_Z,
  CONTINUOUS_OUT_Z_LIMITS,
  CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS,
  CONTINUOUS_RECOVERY_INITIAL_TIME_RANGE,
  CONTINUOUS_RECOVERY_MIN_READY_TIME,
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
  assert "Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Continuous-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Hit-SONIC-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Hit-SONIC-Encoder-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Cross-SONIC-Unitree-G1" in list_tasks()
  assert "Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1" in list_tasks()


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
  assert cross_lab_cfg.run_name == "tennis_cross_lab_from_cross"
  assert cross_lab_cfg.resume is True
  assert cross_lab_cfg.load_checkpoint_file == DEFAULT_CROSS_LAB_RESUME_CHECKPOINT
  assert cross_lab_cfg.algorithm.entropy_coef == 0.001
  assert cross_lab_cfg.max_iterations == 30000

  cross_wrist_lab_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1"),
  )
  assert cross_wrist_lab_cfg.experiment_name == "g1_tennis_latent_cross_wrist_lab"
  assert cross_wrist_lab_cfg.run_name == "tennis_cross_wrist_from_cross"
  assert cross_wrist_lab_cfg.resume is True
  assert (
    cross_wrist_lab_cfg.load_checkpoint_file
    == DEFAULT_CROSS_WRIST_LAB_RESUME_CHECKPOINT
  )
  assert cross_wrist_lab_cfg.algorithm.entropy_coef == 0.001
  assert cross_wrist_lab_cfg.max_iterations == 30000

  continuous_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Continuous-Unitree-G1"),
  )
  assert continuous_cfg.experiment_name == "g1_tennis_latent_continuous"
  assert continuous_cfg.run_name == "tennis_continuous_from_cross"
  assert continuous_cfg.resume is True
  assert continuous_cfg.load_checkpoint_file == DEFAULT_CONTINUOUS_RESUME_CHECKPOINT
  assert continuous_cfg.reset_resume_progress is True
  assert continuous_cfg.algorithm.entropy_coef == 0.003

  sonic_hit_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Hit-SONIC-Unitree-G1"),
  )
  assert sonic_hit_cfg.experiment_name == "g1_tennis_sonic_hit"
  assert sonic_hit_cfg.run_name == "tennis_hit_sonic_token_wo_encoder"
  assert sonic_hit_cfg.resume is False
  assert sonic_hit_cfg.require_decoder_checkpoint is False
  assert sonic_hit_cfg.clip_actions == 1.0
  assert sonic_hit_cfg.actor.distribution_cfg is not None
  assert sonic_hit_cfg.actor.distribution_cfg["init_std"] == 0.2

  sonic_encoder_hit_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Hit-SONIC-Encoder-Unitree-G1"),
  )
  assert sonic_encoder_hit_cfg.experiment_name == "g1_tennis_sonic_encoder_hit"
  assert sonic_encoder_hit_cfg.run_name == "tennis_hit_sonic_encoder_prior"
  assert sonic_encoder_hit_cfg.require_decoder_checkpoint is False
  assert sonic_encoder_hit_cfg.clip_actions == 1.0

  sonic_cross_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-SONIC-Unitree-G1"),
  )
  assert sonic_cross_cfg.experiment_name == "g1_tennis_sonic_cross"
  assert sonic_cross_cfg.run_name == "tennis_cross_sonic_token_wo_encoder_scratch"
  assert sonic_cross_cfg.resume is False
  assert sonic_cross_cfg.require_decoder_checkpoint is False
  assert sonic_cross_cfg.clip_actions == 1.0

  sonic_encoder_cross_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1"),
  )
  assert sonic_encoder_cross_cfg.experiment_name == "g1_tennis_sonic_encoder_cross"
  assert sonic_encoder_cross_cfg.run_name == "tennis_cross_sonic_encoder_prior_scratch"
  assert sonic_encoder_cross_cfg.require_decoder_checkpoint is False
  assert sonic_encoder_cross_cfg.clip_actions == 1.0


def test_tennis_env_uses_latent_actions() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")
  assert cfg.observations["actor"].terms["ball_pos_window"].func is torso_to_ball_b
  assert cfg.observations["critic"].terms["racket_to_ball"].func is racket_to_ball_b

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


def test_tennis_cross_wrist_lab_env_adds_wrist_residual_actions() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1")
  action_cfg = cfg.actions["latent_joint_pos"]
  assert isinstance(action_cfg, FrozenDecoderLatentJointPositionActionCfg)
  assert action_cfg.use_latent_action_barrier is False
  assert action_cfg.wrist_residual_joint_names == (
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  )
  assert action_cfg.wrist_residual_scale == (0.03, 0.05, 0.05)
  assert cfg.rewards["wrist_residual_l2"].weight == -0.5
  assert cfg.rewards["wrist_residual_rate_l2"].weight == -0.5

  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    action = env.action_manager.get_term("latent_joint_pos")
    assert isinstance(action, FrozenDecoderLatentJointPositionAction)
    assert env.action_manager.total_action_dim == 19
    assert action.low_level_action_dim == 29
    assert action.wrist_residual_dim == 3
    assert action.wrist_residual_joint_ids.numel() == 3

    obs, _ = env.reset()
    assert isinstance(obs["actor"], torch.Tensor)
    raw_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    raw_actions[:, 16:] = 10.0
    env.step(raw_actions)
    expected_wrist = torch.tensor((0.03, 0.05, 0.05))
    assert torch.allclose(
      action.wrist_residual_action.cpu(),
      expected_wrist.expand(env.num_envs, -1),
      atol=1.0e-4,
    )
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
  assert action_cfg.use_encoder_token_prior is False
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


def test_tennis_sonic_encoder_env_uses_encoder_prior() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-SONIC-Encoder-Unitree-G1")
  action_cfg = cfg.actions["latent_joint_pos"]
  assert isinstance(action_cfg, SonicDecoderTokenJointPositionActionCfg)
  assert action_cfg.token_dim == 64
  assert action_cfg.decoder_onnx_path == "ckpt/GEAR-SONIC/model_decoder.onnx"
  assert action_cfg.encoder_onnx_path == "ckpt/GEAR-SONIC/model_encoder.onnx"
  assert action_cfg.use_encoder_token_prior is True
  assert action_cfg.token_residual_scale == 0.2
  assert action_cfg.encoder_history_stride == 5

  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    action = env.action_manager.get_term("latent_joint_pos")
    assert isinstance(action, SonicDecoderTokenJointPositionAction)
    assert env.action_manager.total_action_dim == 64
    assert action.low_level_action_dim == 29

    obs, _ = env.reset()
    assert isinstance(obs["actor"], torch.Tensor)
    raw_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    env.step(raw_actions)
    assert action.encoder_token_action.shape == (env.num_envs, 64)
    assert action.token_action.shape == (env.num_envs, 64)
    assert torch.allclose(action.token_action, action.encoder_token_action)
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
  assert cross_lab_action.wrist_residual_joint_names == ()

  cross_wrist_lab_cfg = load_env_cfg("Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1")
  cross_wrist_lab_action = cross_wrist_lab_cfg.actions["latent_joint_pos"]
  assert isinstance(cross_wrist_lab_action, FrozenDecoderLatentJointPositionActionCfg)
  assert cross_wrist_lab_action.use_latent_action_barrier is False
  assert len(cross_wrist_lab_action.wrist_residual_joint_names) == 3

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
  assert cfg.metrics["racket_hit_count"].reduce == "last"
  assert cfg.metrics["crossed_net_count"].reduce == "last"
  assert cfg.metrics["landing_in_bounds_count"].reduce == "last"
  assert cfg.metrics["successful_return_count"].reduce == "last"

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
  assert "post_hit_low_arc_quality_reward" in cfg.rewards
  assert cfg.rewards["post_hit_low_arc_quality_reward"].weight == 0.05
  assert (
    cfg.rewards["post_hit_low_arc_quality_reward"].params["fast_landing_t_min"] == 0.35
  )
  assert (
    cfg.rewards["post_hit_low_arc_quality_reward"].params["fast_landing_t_max"] == 1.20
  )
  assert cfg.metrics["crossed_net_count"].params["landing_x_limits"][1] == 0.0
  assert cfg.metrics["landing_in_bounds_count"].params["landing_y_limits"][0] < 0.0
  assert cfg.metrics["successful_return_count"].params["landing_y_limits"][1] > 0.0
  assert "first_bounce_after_hit_count" in cfg.metrics
  assert "fast_landing_reward_mean" in cfg.metrics
  assert "time_to_landing_mean" in cfg.metrics
  assert "time_to_landing_min" in cfg.metrics
  assert "time_to_landing_max" in cfg.metrics
  assert "time_to_landing_valid_count" in cfg.metrics

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
  assert cfg.rewards["post_hit_low_arc_quality_reward"].weight == 0.05


def test_tennis_wrist_lab_checkpoint_action_head_migration() -> None:
  actor_sd = {
    "distribution.std_param": torch.arange(16, dtype=torch.float32),
    "mlp.0.weight": torch.ones(512, 127),
    "mlp.0.bias": torch.ones(512),
    "mlp.2.weight": torch.ones(256, 512),
    "mlp.2.bias": torch.ones(256),
    "mlp.4.weight": torch.ones(128, 256),
    "mlp.4.bias": torch.ones(128),
    "mlp.6.weight": torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128),
    "mlp.6.bias": torch.arange(16, dtype=torch.float32),
  }

  migrated = expand_actor_action_head_for_wrist_residual(
    actor_sd,
    latent_dim=16,
    target_dim=19,
    wrist_init_std=0.2,
  )

  assert migrated is True
  assert actor_sd["mlp.6.weight"].shape == (19, 128)
  assert actor_sd["mlp.6.bias"].shape == (19,)
  assert actor_sd["distribution.std_param"].shape == (19,)
  assert torch.allclose(
    actor_sd["mlp.6.weight"][:16],
    torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128),
  )
  assert torch.all(actor_sd["mlp.6.weight"][16:] == 0.0)
  assert torch.all(actor_sd["mlp.6.bias"][16:] == 0.0)
  assert torch.allclose(
    actor_sd["distribution.std_param"][:16],
    torch.arange(16, dtype=torch.float32),
  )
  assert torch.allclose(
    actor_sd["distribution.std_param"][16:],
    torch.full((3,), 0.2),
  )


def test_expand_mlp_input_for_observation_appends_zero_weight_columns() -> None:
  model_sd = {
    "obs_normalizer._mean": torch.tensor([[1.0, 2.0]]),
    "obs_normalizer._var": torch.tensor([[3.0, 4.0]]),
    "obs_normalizer._std": torch.tensor([[5.0, 6.0]]),
    "obs_normalizer.count": torch.tensor(7.0),
    "mlp.0.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
    "mlp.0.bias": torch.zeros(3),
  }

  migrated = expand_mlp_input_for_observation(model_sd, target_dim=5)

  assert migrated is True
  assert model_sd["mlp.0.weight"].shape == (3, 5)
  assert torch.allclose(
    model_sd["mlp.0.weight"][:, :2],
    torch.arange(6, dtype=torch.float32).reshape(3, 2),
  )
  assert torch.all(model_sd["mlp.0.weight"][:, 2:] == 0.0)
  assert torch.allclose(
    model_sd["obs_normalizer._mean"],
    torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0]]),
  )
  assert torch.allclose(
    model_sd["obs_normalizer._var"],
    torch.tensor([[3.0, 4.0, 1.0, 1.0, 1.0]]),
  )
  assert torch.allclose(
    model_sd["obs_normalizer._std"],
    torch.tensor([[5.0, 6.0, 1.0, 1.0, 1.0]]),
  )


def test_expand_mlp_input_for_observation_truncates_small_tail() -> None:
  model_sd = {
    "obs_normalizer._mean": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
    "obs_normalizer._var": torch.tensor([[6.0, 7.0, 8.0, 9.0, 10.0]]),
    "obs_normalizer._std": torch.tensor([[11.0, 12.0, 13.0, 14.0, 15.0]]),
    "mlp.0.weight": torch.arange(15, dtype=torch.float32).reshape(3, 5),
    "mlp.0.bias": torch.zeros(3),
  }

  migrated = expand_mlp_input_for_observation(model_sd, target_dim=3)

  assert migrated is True
  assert model_sd["mlp.0.weight"].shape == (3, 3)
  assert torch.allclose(
    model_sd["mlp.0.weight"],
    torch.arange(15, dtype=torch.float32).reshape(3, 5)[:, :3],
  )
  assert torch.allclose(
    model_sd["obs_normalizer._mean"],
    torch.tensor([[1.0, 2.0, 3.0]]),
  )
  assert torch.allclose(
    model_sd["obs_normalizer._var"],
    torch.tensor([[6.0, 7.0, 8.0]]),
  )
  assert torch.allclose(
    model_sd["obs_normalizer._std"],
    torch.tensor([[11.0, 12.0, 13.0]]),
  )


def test_tennis_continuous_respawns_until_eight_successful_returns() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Continuous-Unitree-G1")

  provider_cfg = cfg.events["reset_ball"].params["provider_cfg"]
  assert isinstance(provider_cfg, OpponentFeederCfg)
  assert provider_cfg.spawn_x_range[0] < provider_cfg.spawn_x_range[1] < 0.0
  assert provider_cfg.target_x_range[0] > 0.0

  assert "landing_in_bounds_event" in cfg.rewards
  assert "continuous_rally_complete_bonus" in cfg.rewards
  assert "continuous_recovery_ready_pose" in cfg.rewards
  assert "continuous_recovery_ready_event" in cfg.rewards
  assert "advance_continuous_rally_ball" in cfg.rewards
  assert "respawn_successful_continuous_rally_ball" not in cfg.rewards
  assert cfg.rewards["post_hit_low_arc_quality_reward"].weight == 0.05
  assert (
    cfg.rewards["post_hit_low_arc_quality_reward"].params["racket_sensor_name"]
    == "racket_ball_contact"
  )
  assert cfg.rewards["post_hit_low_arc_quality_reward"].params["net_sensor_name"] == (
    "ball_net_contact"
  )
  assert "continuous_success_ratio" in cfg.metrics
  assert "in_recovery_rate" in cfg.metrics
  assert "net_contact_count" in cfg.metrics
  assert "invalid_feed_count" in cfg.metrics
  assert "invalid_feed_net_count" in cfg.metrics
  assert "invalid_feed_out_count" in cfg.metrics
  assert "invalid_feed_opponent_bounce_count" in cfg.metrics
  assert "continuous_fault_count" in cfg.metrics
  assert "fault_incoming_bounce_count" in cfg.metrics
  assert "fault_return_bounce_out_count" in cfg.metrics
  assert "fault_return_out_count" in cfg.metrics
  assert "fault_net_contact_count" in cfg.metrics
  assert "fault_extra_racket_count" in cfg.metrics
  assert "fault_low_net_cross_count" in cfg.metrics
  assert "recovery_ready_count" in cfg.metrics
  assert "first_bounce_after_hit_count" in cfg.metrics
  assert "fast_landing_reward_mean" in cfg.metrics
  assert "time_to_landing_valid_count" in cfg.metrics
  assert cfg.metrics["continuous_success_ratio"].reduce == "last"
  assert cfg.metrics["in_recovery_rate"].reduce == "mean"
  assert (
    cfg.metrics["continuous_success_ratio"].params["max_successful_returns"]
    == CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS
  )
  assert "continuous_ball_phase" in cfg.observations["actor"].terms
  assert "continuous_ball_phase" in cfg.observations["critic"].terms
  assert (
    "max_successful_returns"
    not in cfg.observations["actor"].terms["continuous_ball_phase"].params
  )
  assert cfg.rewards["approach_point"].params["racket_sensor_name"] == (
    "racket_ball_contact"
  )
  assert cfg.rewards["approach_point"].params["net_sensor_name"] == "ball_net_contact"
  assert cfg.rewards["continuous_recovery_ready_pose"].weight == 20.0
  assert cfg.rewards["continuous_recovery_ready_event"].weight == 200.0
  assert cfg.rewards["advance_continuous_rally_ball"].weight == 1.0e-9

  assert "ball_out_of_bounds" not in cfg.terminations
  assert "landing_in_bounds_after_hit" not in cfg.terminations
  assert "second_contact" not in cfg.terminations
  assert "continuous_ball_fault" in cfg.terminations
  assert "continuous_rally_failure" not in cfg.terminations
  assert "continuous_rally_complete" in cfg.terminations

  complete_params = cfg.terminations["continuous_rally_complete"].params
  assert (
    complete_params["max_successful_returns"]
    == CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS
  )
  respawn_params = cfg.rewards["advance_continuous_rally_ball"].params
  assert (
    respawn_params["max_successful_returns"]
    == CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS
  )
  assert respawn_params["recovery_time_range"] == CONTINUOUS_RECOVERY_INITIAL_TIME_RANGE
  assert respawn_params["min_recovery_time"] == CONTINUOUS_RECOVERY_MIN_READY_TIME
  assert (
    respawn_params["target_x"]
    == cfg.rewards["continuous_recovery_ready_pose"].params["target_x"]
  )
  assert (
    respawn_params["provider_cfg"] is cfg.events["reset_ball"].params["provider_cfg"]
  )
  assert (
    provider_cfg.max_apex_z
    < cfg.terminations["continuous_ball_fault"].params["z_limits"][1]
  )
  assert cfg.terminations["continuous_ball_fault"].params["z_limits"] == (
    CONTINUOUS_OUT_Z_LIMITS
  )
  assert provider_cfg.max_apex_z == CONTINUOUS_FEED_MAX_APEX_Z

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["success_term_name"] == "continuous_rally_complete"
  length_stages = cfg.curriculum["continuous_rally_length"].params["stages"]
  assert len(length_stages) == 1
  assert length_stages[0]["params"]["max_successful_returns"] == (
    CONTINUOUS_RALLY_INITIAL_SUCCESSFUL_RETURNS
  )
  respawn_stages = cfg.curriculum["continuous_respawn_length"].params["stages"]
  assert respawn_stages == length_stages
  wait_stages = cfg.curriculum["continuous_wait_interval"].params["stages"]
  assert len(wait_stages) == 1
  assert wait_stages[0]["params"]["recovery_time_range"] == (
    CONTINUOUS_RECOVERY_INITIAL_TIME_RANGE
  )


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
