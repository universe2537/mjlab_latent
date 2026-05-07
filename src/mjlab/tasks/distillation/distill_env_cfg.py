"""Latent action distillation environment configuration.

Distillation reuses the tracking MDP / environment unchanged; the only
structural difference required by LATENT §3.2.2 is a stronger encoder bias
event applied to the right wrist joints, so the student becomes robust to
wrist perturbations and leaves headroom for §3.3.2 hybrid wrist control.

Keeping the env factory as a thin wrapper around ``make_tracking_env_cfg``
avoids duplicating the ~300-line tracking config and ensures any future
fix in the tracking environment automatically propagates here.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

# Joints targeted by the additional bias (LATENT §3.2.2).
_WRIST_JOINTS = (
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
_WRIST_BIAS_RANGE = (-0.1, 0.1)


def make_distillation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base distillation environment config.

  Identical to the tracking env, plus a startup ``wrist_encoder_bias``
  event that injects a larger persistent offset on the right wrist
  joints. Robot-specific configs (e.g. ``unitree_g1_flat_distillation_env_cfg``)
  further specialise this base just like for tracking.
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
