"""Unitree G1 的 latent distillation 环境配置。

Specialises the tracking G1 config with training-only racket-hand impulse
events. All robot setup (motion artifacts, body order,
contact sensors, viewer, ...) 全部直接继承 tracking 配置，
避免 teacher/student 因环境定义漂移而产生额外分布差异。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_table_tennis_tracking_env_cfg,
)

# XML site position converted from the wrist body frame into the inertial
# center-of-mass frame expected by ``apply_body_impulse``.
_RACKET_CENTER_OFFSET_FROM_COM = (-0.00138455, -0.02790999, 0.25233888)


def _add_distillation_events(
  cfg: ManagerBasedRlEnvCfg, *, play: bool
) -> ManagerBasedRlEnvCfg:
  if not play:
    cfg.events["right_wrist_force_impulse"] = EventTermCfg(
      mode="step",
      func=apply_body_impulse,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("right_wrist_yaw_link",)),
        "force_range": (-5.0, 5.0),
        "torque_range": (0.0, 0.0),
        "duration_s": (0.05, 0.12),
        "cooldown_s": (0.5, 1.5),
        "body_point_offset": _RACKET_CENTER_OFFSET_FROM_COM,
      },
    )
  return cfg


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
  return _add_distillation_events(cfg, play=play)


def unitree_g1_table_tennis_distillation_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create distillation env for table-tennis tracking motions."""
  cfg = unitree_g1_table_tennis_tracking_env_cfg(
    has_state_estimation=has_state_estimation,
    play=play,
  )
  return _add_distillation_events(cfg, play=play)


__all__ = [
  "unitree_g1_flat_distillation_env_cfg",
  "unitree_g1_table_tennis_distillation_env_cfg",
]
