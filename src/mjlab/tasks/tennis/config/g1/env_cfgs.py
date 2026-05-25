"""网球潜变量控制任务的 Unitree G1 配置。"""

from __future__ import annotations

from mjlab.asset_zoo.robots import (
  G1_W_RACKET_ACTION_SCALE,
  get_g1_w_racket_robot_cfg,
)
from mjlab.tasks.tennis.mdp import (
  FrozenDecoderLatentJointPositionActionCfg,
  SonicDecoderTokenJointPositionActionCfg,
)
from mjlab.tasks.tennis.scene import (
  get_tennis_ball_cfg,
  get_tennis_court_cfg,
  resolve_court_scale,
)
from mjlab.tasks.tennis.tennis_env_cfg import (
  DEFAULT_COURT_SIZE,
  CourtSizeType,
  TennisLatentEnvCfg,
  make_tennis_continuous_env_cfg,
  make_tennis_latent_cross_env_cfg,
  make_tennis_latent_env_cfg,
)

DEFAULT_DECODER_CHECKPOINT = "logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"
DEFAULT_SONIC_DECODER_ONNX = "ckpt/GEAR-SONIC/model_decoder.onnx"


def unitree_g1_tennis_latent_hit_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建带冻结解码器潜变量动作空间的 G1 网球击球任务。

  参数:
    play: 若为 True，则关闭观测噪声并设置超长 episode。
    court_size: 球场尺寸预设，默认 ``"mini"``。
      可通过 ``--env.court-size <size>`` 在运行时覆盖，无需修改任务名称。
      可选值：``"standard"`` / ``"half"`` / ``"mini"``。
  """
  cfg = make_tennis_latent_env_cfg(court_size=court_size)
  scale = resolve_court_scale(court_size)
  cfg.scene.entities = {
    "robot": get_g1_w_racket_robot_cfg(),
    "ball": get_tennis_ball_cfg(),
    "court": get_tennis_court_cfg(scale=scale),
  }
  cfg.viewer.body_name = "torso_link"
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


def unitree_g1_tennis_hit_lab_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建 G1 潜变量网球 Hit-LAB 任务。"""
  cfg = unitree_g1_tennis_latent_hit_env_cfg(
    play=play,
    court_size=court_size,
  )
  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.use_latent_action_barrier = True
  action.latent_barrier_scale = 1.0
  action.latent_barrier_min_std = 0.05
  action.latent_barrier_max_std = 2.0
  return cfg


def unitree_g1_tennis_latent_cross_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建 G1 潜变量网球过网落界内任务。"""
  cfg = make_tennis_latent_cross_env_cfg(court_size=court_size)
  scale = resolve_court_scale(court_size)
  cfg.scene.entities = {
    "robot": get_g1_w_racket_robot_cfg(),
    "ball": get_tennis_ball_cfg(),
    "court": get_tennis_court_cfg(scale=scale),
  }
  cfg.viewer.body_name = "torso_link"
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


def unitree_g1_tennis_cross_lab_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建 G1 潜变量网球 Cross-LAB 任务。"""
  cfg = unitree_g1_tennis_latent_cross_env_cfg(
    play=play,
    court_size=court_size,
  )
  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.use_latent_action_barrier = True
  action.latent_barrier_scale = 1.5
  action.latent_barrier_min_std = 0.05
  action.latent_barrier_max_std = 2.0
  cfg.rewards["approach_point"].weight = 2.0
  cfg.rewards["racket_towards_ball"].weight = 1.0
  cfg.rewards["racket_hit_event"].weight = 5.0
  cfg.rewards["post_hit_x_progress"].weight = 80.0
  cfg.rewards["post_hit_ball_velocity_direction"].weight = 50.0
  cfg.rewards["crossed_net_event"].weight = 700.0
  cfg.rewards["landing_in_bounds_event"].weight = 1500.0
  return cfg


def unitree_g1_tennis_continuous_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建 G1 潜变量网球连续接多球任务。"""
  cfg = make_tennis_continuous_env_cfg(court_size=court_size)
  scale = resolve_court_scale(court_size)
  cfg.scene.entities = {
    "robot": get_g1_w_racket_robot_cfg(),
    "ball": get_tennis_ball_cfg(),
    "court": get_tennis_court_cfg(scale=scale),
  }
  cfg.viewer.body_name = "torso_link"
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


def unitree_g1_tennis_sonic_hit_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建使用 SONIC token decoder 的 G1 网球 Hit 任务。"""
  cfg = unitree_g1_tennis_latent_hit_env_cfg(
    play=play,
    court_size=court_size,
  )
  cfg.actions["latent_joint_pos"] = SonicDecoderTokenJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    use_default_offset=True,
    token_dim=64,
    decoder_onnx_path=DEFAULT_SONIC_DECODER_ONNX,
  )
  return cfg


def unitree_g1_tennis_sonic_cross_env_cfg(
  play: bool = False,
  court_size: CourtSizeType = DEFAULT_COURT_SIZE,
) -> TennisLatentEnvCfg:
  """创建使用 SONIC token decoder 的 G1 网球 Cross 任务。"""
  cfg = unitree_g1_tennis_latent_cross_env_cfg(
    play=play,
    court_size=court_size,
  )
  cfg.actions["latent_joint_pos"] = SonicDecoderTokenJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    scale=1.0,
    use_default_offset=True,
    token_dim=64,
    decoder_onnx_path=DEFAULT_SONIC_DECODER_ONNX,
  )
  return cfg
