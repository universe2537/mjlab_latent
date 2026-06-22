from typing import cast

import mjlab.tasks  # noqa: F401
from mjlab.scene import Scene
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg
from mjlab.tasks.pingpong.config.g1.env_cfgs import DEFAULT_DECODER_CHECKPOINT
from mjlab.tasks.pingpong.config.g1.rl_cfg import DEFAULT_RETURN_RESUME_CHECKPOINT
from mjlab.tasks.pingpong.mdp.ball_providers import TableTennisFeederCfg
from mjlab.tasks.pingpong.pingpong_env_cfg import (
  BALL_TARGET_X_RANGE,
  BALL_TARGET_Y_RANGE,
  DECODER_STATE_TERMS,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg


def test_pingpong_tasks_registered() -> None:
  assert "Mjlab-Pingpong-Hit-Unitree-G1" in list_tasks()
  assert "Mjlab-Pingpong-Return-Unitree-G1" in list_tasks()


def test_pingpong_task_scene_compiles() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()

  geom_names = {model.geom(i).name for i in range(model.ngeom)}
  sensor_names = {model.sensor(i).name for i in range(model.nsensor)}

  assert "robot/pingpong_paddle_collision" in geom_names
  assert "ball/pingpong_ball" in geom_names
  assert "table/pingpong_table_top_collision" in geom_names
  assert "table/pingpong_net_collision" in geom_names
  assert any(name.startswith("paddle_ball_contact") for name in sensor_names)
  assert any(name.startswith("pingpong_ball_net_contact") for name in sensor_names)


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

  return_cfg = cast(
    TennisLatentOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Pingpong-Return-Unitree-G1"),
  )
  assert return_cfg.experiment_name == "g1_pingpong_latent_return"
  assert return_cfg.run_name == "pingpong_return_from_hit"
  assert return_cfg.resume is False
  assert return_cfg.load_checkpoint_file == (DEFAULT_RETURN_RESUME_CHECKPOINT or None)
  assert return_cfg.algorithm.entropy_coef == 0.003
  assert return_cfg.max_iterations == 40000


def test_pingpong_hit_and_return_success_terms() -> None:
  hit_cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  assert "first_paddle_hit" in hit_cfg.terminations
  assert "legal_return_success" not in hit_cfg.terminations
  assert hit_cfg.curriculum["ball_target_region"].params["success_term_name"] == (
    "first_paddle_hit"
  )

  return_cfg = load_env_cfg("Mjlab-Pingpong-Return-Unitree-G1")
  assert "first_paddle_hit" not in return_cfg.terminations
  assert "legal_return_success" in return_cfg.terminations
  assert "crossed_net_event" in return_cfg.rewards
  assert "opponent_table_bounce_event" in return_cfg.rewards
  assert return_cfg.curriculum["ball_target_region"].params["success_term_name"] == (
    "legal_return_success"
  )


def test_pingpong_feeder_curriculum_ranges() -> None:
  cfg = load_env_cfg("Mjlab-Pingpong-Hit-Unitree-G1")
  provider_cfg = cfg.events["reset_ball"].params["provider_cfg"]
  assert isinstance(provider_cfg, TableTennisFeederCfg)
  assert provider_cfg.spawn_x_range[1] < 0.0
  assert provider_cfg.target_x_range[0] > 0.0
  assert provider_cfg.target_x_range[1] < 1.0

  curriculum_params = cfg.curriculum["ball_target_region"].params
  assert curriculum_params["provider_cfg"] is provider_cfg
  assert curriculum_params["final_target_x_range"] == BALL_TARGET_X_RANGE
  assert curriculum_params["final_target_y_range"] == BALL_TARGET_Y_RANGE
