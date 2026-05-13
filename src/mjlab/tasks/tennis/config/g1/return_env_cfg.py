"""Unitree G1 wiring for the rally-driven tennis return task."""

from __future__ import annotations

import dataclasses

from mjlab.asset_zoo.robots import (
  G1_W_RACKET_ACTION_SCALE,
  get_g1_w_racket_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.tasks.tennis.return_env_cfg import make_tennis_return_env_cfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg
from mjlab.tasks.tennis.scene import get_tennis_ball_cfg, get_tennis_court_cfg

from .rl_cfg import unitree_g1_tennis_latent_ppo_runner_cfg

DEFAULT_DECODER_CHECKPOINT = "logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"


def unitree_g1_tennis_return_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1 tennis return task wired with the P1 RandomFeeder provider."""
  cfg = make_tennis_return_env_cfg()
  cfg.scene.entities = {
    "robot": get_g1_w_racket_robot_cfg(),
    "ball": get_tennis_ball_cfg(),
    "court": get_tennis_court_cfg(),
  }
  cfg.viewer.body_name = "torso_link"

  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.scale = G1_W_RACKET_ACTION_SCALE
  action.decoder_checkpoint = DEFAULT_DECODER_CHECKPOINT

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg


def unitree_g1_tennis_return_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """PPO config for the G1 return task; reuses hit-task hyperparameters."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg()
  return dataclasses.replace(cfg, experiment_name="g1_tennis_latent_return")
