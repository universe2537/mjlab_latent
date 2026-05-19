"""Latent action distillation 的基础环境配置。

Distillation reuses the tracking MDP / environment unchanged; the only
structural difference required by LATENT §3.2.2 is a stronger encoder bias
event applied to the right wrist joints, so the student becomes robust to
wrist perturbations and leaves headroom for §3.3.2 hybrid wrist control.

1. teacher 与 student 在尽可能一致的环境中交互。
2. tracking 侧任何修复都能自动同步到 distillation。
3. distillation 只保留 LATENT 论文要求的最小差异。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# 需要施加额外编码器偏置的右手腕关节。
# 这样做的目的是让学生策略对手腕误差更鲁棒。
_WRIST_JOINTS = (
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
_WRIST_BIAS_RANGE = (-0.1, 0.1)


def make_distillation_env_cfg() -> ManagerBasedRlEnvCfg:
  """创建 distillation 任务的基础环境配置。

  Identical to the tracking env, plus a startup ``wrist_encoder_bias``
  event that injects a larger persistent offset on the right wrist
  joints. Robot-specific configs (e.g. ``unitree_g1_flat_distillation_env_cfg``)
  further specialise this base just like for tracking.

  返回:
    一个已经带有 ``wrist_encoder_bias`` 事件的环境配置对象。
  """
  cfg = make_tracking_env_cfg()
  cfg.events["wrist_encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WRIST_JOINTS),
      "bias_range": _WRIST_BIAS_RANGE,
    },
  )
  return cfg
