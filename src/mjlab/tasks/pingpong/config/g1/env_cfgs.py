"""Unitree G1 configuration for table-tennis latent-control tasks."""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_W_RACKET_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1_w_pingpong_paddle import (
  PINGPONG_PADDLE_HANDLE_HALF_LENGTH,
  PINGPONG_PADDLE_HANDLE_RADIUS,
  PINGPONG_PADDLE_RADIUS,
  PINGPONG_PADDLE_SCALE,
  get_g1_w_pingpong_paddle_robot_cfg,
  get_g1_w_pingpong_paddle_spec,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.pingpong.pingpong_env_cfg import (
  add_pingpong_paddle_ball_contact_pair,
  make_pingpong_latent_cross_diag_env_cfg,
  make_pingpong_latent_cross_env_cfg,
  make_pingpong_latent_cross_impact_env_cfg,
  make_pingpong_latent_cross_strike_quality_energy_relax_env_cfg,
  make_pingpong_latent_cross_strike_quality_env_cfg,
  make_pingpong_latent_env_cfg,
  make_pingpong_latent_return_env_cfg,
  make_pingpong_pace_env_cfg,
)
from mjlab.tasks.pingpong.scene import get_pingpong_ball_cfg, get_pingpong_table_cfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg

DEFAULT_DECODER_CHECKPOINT = "logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"


def _apply_g1_pingpong_common(cfg, play: bool):
  cfg.scene.entities = {
    "robot": get_g1_w_pingpong_paddle_robot_cfg(),
    "ball": get_pingpong_ball_cfg(),
    "table": get_pingpong_table_cfg(),
  }
  cfg.scene.spec_fn = add_pingpong_paddle_ball_contact_pair
  cfg.viewer.body_name = "torso_link"
  cfg.viewer.elevation = -16.0
  cfg.viewer.azimuth = 135.0

  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.scale = G1_W_RACKET_ACTION_SCALE
  action.decoder_checkpoint = DEFAULT_DECODER_CHECKPOINT

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg


def unitree_g1_pingpong_latent_hit_env_cfg(play: bool = False):
  """Create G1 table-tennis legal-hit task."""
  cfg = make_pingpong_latent_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_return_env_cfg(play: bool = False):
  """Create G1 table-tennis legal-return task."""
  cfg = make_pingpong_latent_return_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_cross_env_cfg(play: bool = False):
  """Create G1 table-tennis over-net return task."""
  cfg = make_pingpong_latent_cross_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_cross_diag_env_cfg(play: bool = False):
  """Create G1 diagnostics-only table-tennis Cross task."""
  cfg = make_pingpong_latent_cross_diag_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_cross_strike_quality_env_cfg(play: bool = False):
  """Create G1 Cross task with strike-quality dense rewards."""
  cfg = make_pingpong_latent_cross_strike_quality_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_cross_impact_env_cfg(play: bool = False):
  """Create G1 Cross task with impact-window paddle behavior rewards."""
  cfg = make_pingpong_latent_cross_impact_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_cross_strike_quality_energy_relax_env_cfg(
  play: bool = False,
):
  """Create G1 Cross task with strike rewards and hit-window energy relax."""
  cfg = make_pingpong_latent_cross_strike_quality_energy_relax_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_pace_env_cfg(play: bool = False):
  """Create G1 PACE-style direct joint-control table-tennis task."""
  cfg = make_pingpong_pace_env_cfg()
  cfg.scene.entities = {
    "robot": get_g1_w_pingpong_paddle_robot_cfg(),
    "ball": get_pingpong_ball_cfg(),
    "table": get_pingpong_table_cfg(),
  }
  cfg.scene.spec_fn = add_pingpong_paddle_ball_contact_pair
  cfg.viewer.body_name = "torso_link"
  cfg.viewer.elevation = -16.0
  cfg.viewer.azimuth = 135.0

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = G1_W_RACKET_ACTION_SCALE

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg


__all__ = [
  "DEFAULT_DECODER_CHECKPOINT",
  "PINGPONG_PADDLE_HANDLE_HALF_LENGTH",
  "PINGPONG_PADDLE_HANDLE_RADIUS",
  "PINGPONG_PADDLE_RADIUS",
  "PINGPONG_PADDLE_SCALE",
  "get_g1_w_pingpong_paddle_robot_cfg",
  "get_g1_w_pingpong_paddle_spec",
  "unitree_g1_pingpong_latent_cross_diag_env_cfg",
  "unitree_g1_pingpong_latent_cross_env_cfg",
  "unitree_g1_pingpong_latent_cross_impact_env_cfg",
  "unitree_g1_pingpong_latent_cross_strike_quality_energy_relax_env_cfg",
  "unitree_g1_pingpong_latent_cross_strike_quality_env_cfg",
  "unitree_g1_pingpong_latent_hit_env_cfg",
  "unitree_g1_pingpong_latent_return_env_cfg",
  "unitree_g1_pingpong_pace_env_cfg",
]
