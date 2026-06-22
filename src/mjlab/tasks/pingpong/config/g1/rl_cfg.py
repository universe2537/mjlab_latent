"""Unitree G1 PPO configuration for table-tennis latent-control tasks."""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg

DEFAULT_RETURN_RESUME_CHECKPOINT = ""


def unitree_g1_pingpong_latent_ppo_runner_cfg(
  *,
  experiment_name: str = "g1_pingpong_latent_hit",
  run_name: str = "pingpong_hit_scratch",
  resume: bool = False,
  load_checkpoint_file: str | None = None,
  reset_resume_progress: bool = False,
) -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for G1 table-tennis latent policies."""
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
    reset_resume_progress=reset_resume_progress,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30000,
    clip_actions=4.0,
    require_decoder_checkpoint=True,
  )


def unitree_g1_pingpong_return_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 table-tennis legal-return task."""
  cfg = unitree_g1_pingpong_latent_ppo_runner_cfg(
    experiment_name="g1_pingpong_latent_return",
    run_name="pingpong_return_from_hit",
    resume=False,
    load_checkpoint_file=DEFAULT_RETURN_RESUME_CHECKPOINT or None,
  )
  cfg.algorithm.entropy_coef = 0.003
  cfg.max_iterations = 40000
  return cfg


__all__ = [
  "DEFAULT_RETURN_RESUME_CHECKPOINT",
  "unitree_g1_pingpong_latent_ppo_runner_cfg",
  "unitree_g1_pingpong_return_ppo_runner_cfg",
]
