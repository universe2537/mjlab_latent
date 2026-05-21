from pathlib import Path
from typing import cast

import mujoco
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tennis.config.g1.rl_cfg import DEFAULT_CROSS_RESUME_CHECKPOINT
from mjlab.tasks.tennis.mdp import (
  FrozenDecoderLatentJointPositionAction,
  FrozenDecoderLatentJointPositionActionCfg,
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
  assert "Mjlab-Tennis-Cross-Unitree-G1" in list_tasks()


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

  cross_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tennis-Cross-Unitree-G1"),
  )
  assert cross_cfg.experiment_name == "g1_tennis_latent_cross"
  assert cross_cfg.run_name == "tennis_cross_from_hit"
  assert cross_cfg.resume is True
  assert cross_cfg.load_checkpoint_file == DEFAULT_CROSS_RESUME_CHECKPOINT


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
  assert "post_hit_x_progress" in cfg.rewards
  assert cfg.rewards["post_hit_x_progress"].weight == 10.0
  assert "crossed_net_event" in cfg.rewards
  assert "landing_in_bounds_event" in cfg.rewards

  assert "first_racket_hit" not in cfg.terminations
  assert "second_contact" in cfg.terminations
  assert "landing_in_bounds_after_hit" in cfg.terminations

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["success_term_name"] == "landing_in_bounds_after_hit"

  landing_params = cfg.terminations["landing_in_bounds_after_hit"].params
  assert landing_params["landing_x_limits"][1] == 0.0
  assert landing_params["landing_y_limits"][0] < 0.0
  assert landing_params["landing_y_limits"][1] > 0.0


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
