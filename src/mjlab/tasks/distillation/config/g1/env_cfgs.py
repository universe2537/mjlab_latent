"""Unitree G1 的 latent distillation 环境配置。

Specialises the tracking G1 config with the extra wrist encoder bias event
required by LATENT §3.2.2. All robot setup (motion artifacts, body order,
contact sensors, viewer, ...) 全部直接继承 tracking 配置，
避免 teacher/student 因环境定义漂移而产生额外分布差异。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

_WRIST_JOINTS = (
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
_WRIST_BIAS_RANGE = (-0.1, 0.1)


def unitree_g1_flat_distillation_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """在 G1 tracking 环境基础上增加蒸馏专用事件。

  参数:
    has_state_estimation: 是否保留依赖状态估计的观测项。
    play: 是否进入播放/部署友好的配置。
  """
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )
  # 仅新增这一项：让右手腕编码器偏置更大，以匹配 LATENT 的鲁棒性设定。
  cfg.events["wrist_encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WRIST_JOINTS),
      "bias_range": _WRIST_BIAS_RANGE,
    },
  )
  return cfg
