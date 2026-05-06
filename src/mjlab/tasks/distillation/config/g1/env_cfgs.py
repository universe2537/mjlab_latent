"""Unitree G1 latent distillation environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg


def unitree_g1_distillation_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the G1 environment used for online latent distillation.

  The distillation task deliberately reuses the tracking environment so the
  student sees the same motion command, reset distribution, and action interface
  as the pretrained tracker teacher.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  if not play:
    cfg.observations["actor"].enable_corruption = True
  return cfg
