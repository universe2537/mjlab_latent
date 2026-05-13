import math
from pathlib import Path
from typing import cast

import mujoco
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tennis.mdp import (
  FrozenDecoderLatentJointPositionAction,
  FrozenDecoderLatentJointPositionActionCfg,
)
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg


def test_tennis_task_registered() -> None:
  assert "Mjlab-Tennis-Hit-Unitree-G1" in list_tasks()


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
    assert actor_obs.shape[-1] == 102
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


def test_tennis_hit_rewards_and_terminations_are_phase_based() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")

  assert "first_valid_hit" in cfg.rewards
  assert "successful_return" in cfg.rewards
  assert "miss_ball_penalty" in cfg.rewards
  assert "miss_ball" in cfg.terminations
  assert "second_contact_after_valid_hit" in cfg.terminations
  assert "successful_return" in cfg.terminations

  ball_bounds = cfg.terminations["ball_out_of_bounds"].params["x_limits"]
  assert ball_bounds[0] <= -3.0


def test_tennis_reset_ranges_face_opponent_half() -> None:
  cfg = load_env_cfg("Mjlab-Tennis-Hit-Unitree-G1")

  robot_reset = cfg.events["reset_robot_base"].params
  robot_pose = robot_reset["pose_range"]
  assert robot_pose["x"] == (1.55, 1.95)
  assert robot_pose["y"] == (-0.35, 0.35)
  assert robot_pose["yaw"] == (math.pi, math.pi)

  ball_reset = cfg.events["reset_ball"].params
  ball_pose = ball_reset["pose_range"]
  ball_velocity = ball_reset["velocity_range"]
  assert ball_pose["x"] == (0.35, 3.15)
  assert ball_pose["y"] == (-2.1, 2.1)
  assert ball_pose["z"] == (0.65, 1.45)
  assert ball_velocity["x"] == (0.6, 2.2)
  assert ball_velocity["y"] == (-1.2, 1.2)
  assert ball_velocity["z"] == (-0.8, 0.8)

  miss_params = cfg.terminations["miss_ball"].params
  assert miss_params["miss_x_direction"] == 1.0


def test_standalone_tennis_scene_compiles() -> None:
  xml_path = (
    Path(__file__).parents[1]
    / "src/mjlab/asset_zoo/robots/unitree_g1_w_racket/xml"
    / "scene_mjx_racket_tennis_return.xml"
  )
  model = mujoco.MjModel.from_xml_path(str(xml_path))

  assert model.nq == 43
  assert model.nu == 0
