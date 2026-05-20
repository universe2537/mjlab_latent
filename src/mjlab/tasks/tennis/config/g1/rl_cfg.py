"""Unitree G1 PPO configuration for tennis latent-control tasks."""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg

DEFAULT_CROSS_RESUME_CHECKPOINT = (
  "logs/rsl_rl/g1_tennis_latent_hit/"
  "tennis_cloud_tennis_B_curr_quarter_2026-05-18_22-02-07/model_29999.pt"
)


def unitree_g1_tennis_latent_ppo_runner_cfg(
  *,
  experiment_name: str = "g1_tennis_latent_hit",
  run_name: str = "",
  resume: bool = False,
  load_checkpoint_file: str | None = None,
) -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis high-level latent policy."""
  return TennisLatentOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    run_name=run_name,
    resume=resume,
    load_checkpoint_file=load_checkpoint_file,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30000,
    clip_actions=4.0,
    require_decoder_checkpoint=True,
  )


def unitree_g1_tennis_latent_cross_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis cross-court task."""
  return unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_latent_cross",
    run_name="tennis_cross_from_hit",
    resume=True,
    load_checkpoint_file=DEFAULT_CROSS_RESUME_CHECKPOINT,
  )
