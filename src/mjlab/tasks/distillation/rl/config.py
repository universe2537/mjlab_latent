"""Configuration dataclass for online latent distillation."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration for online latent action distillation.

  Mirrors the layout of ``mjlab.rl.config.RslRlOnPolicyRunnerCfg`` so CLI / Hydra
  reflection treats it the same way as the PPO runner config.
  """

  # -- Teacher --
  teacher_task_id: str = "Mjlab-Tracking-Flat-Unitree-G1"
  teacher_checkpoint: str = ""
  teacher_strict_load: bool = True

  # -- Observation routing --
  obs_group: str = "actor"
  state_terms: tuple[str, ...] = ()
  target_terms: tuple[str, ...] = ("command",)

  # -- VAE topology (PHC / PULSE faithful defaults) --
  latent_dim: int = 16
  encoder_hidden_dims: tuple[int, ...] = (512, 256)
  prior_hidden_dims: tuple[int, ...] = (512, 256)
  decoder_hidden_dims: tuple[int, ...] = (512, 256)
  activation: str = "elu"
  min_log_std: float = -5.0
  max_log_std: float = 2.0
  # Posterior bottleneck size = ``latent_dim * posterior_feature_multiplier``
  # before the mu / log_std heads (PHC default: 5). Prior typically uses 1.
  posterior_feature_multiplier: int = 5
  prior_feature_multiplier: int = 1
  # When True, decoder consumes only ``z`` (mirrors PHC's ``z_all`` flag).
  z_all: bool = False

  # -- Optimisation --
  learning_rate: float = 3.0e-4
  weight_decay: float = 0.0
  action_loss_weight: float = 1.0
  kl_loss_weight: float = 1.0e-3
  batch_size: int = 16_384
  buffer_capacity: int = 262_144
  updates_per_iteration: int = 4
  teacher_action_prob: float = 0.0
  deterministic_rollout: bool = True
  max_grad_norm: float = 1.0

  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",)}
  )
