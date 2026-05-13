"""网球潜变量控制任务的 Unitree G1 配置。"""

from __future__ import annotations

from mjlab.asset_zoo.robots import (
  G1_W_RACKET_ACTION_SCALE,
  get_g1_w_racket_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg
from mjlab.tasks.tennis.scene import get_tennis_ball_cfg, get_tennis_court_cfg
from mjlab.tasks.tennis.tennis_env_cfg import make_tennis_latent_env_cfg

DEFAULT_DECODER_CHECKPOINT = "logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"


def unitree_g1_tennis_latent_hit_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """创建带冻结解码器潜变量动作空间的 G1 网球击球任务。"""
  cfg = make_tennis_latent_env_cfg()
  cfg.scene.entities = {
    "robot": get_g1_w_racket_robot_cfg(),
    "ball": get_tennis_ball_cfg(),
    "court": get_tennis_court_cfg(),
  }
  cfg.viewer.body_name = "torso_link"
  cfg.viewer.distance = 5.0
  cfg.viewer.elevation = -18.0
  cfg.viewer.azimuth = 140.0

  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.scale = G1_W_RACKET_ACTION_SCALE
  action.decoder_checkpoint = DEFAULT_DECODER_CHECKPOINT

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
