"""Unitree G1 latent distillation environment configuration.

Specialises the tracking G1 config with the extra wrist encoder bias event
required by LATENT §3.2.2. All robot setup (motion artifacts, body order,
contact sensors, viewer, ...) is inherited directly from the tracking config
to avoid configuration drift between the teacher and the student.
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
  """Create the G1 distillation env cfg by augmenting the tracking one."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.events["wrist_encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=_WRIST_JOINTS),
      "bias_range": _WRIST_BIAS_RANGE,
    },
  )
  return cfg
