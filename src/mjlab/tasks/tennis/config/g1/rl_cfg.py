"""Unitree G1 PPO configuration for tennis latent-control tasks."""

from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunnerCfg

DEFAULT_CROSS_RESUME_CHECKPOINT = (
  "logs/rsl_rl/g1_tennis_latent_hit/"
  "tennis_cloud_tennis_B_curr_quarter_2026-05-18_22-02-07/model_29999.pt"
)
DEFAULT_CROSS_LAB_RESUME_CHECKPOINT = (
  "logs/rsl_rl/g1_tennis_latent_cross/"
  "tennis_cross_from_hit_2026-05-21_15-21-12/model_59998.pt"
)
DEFAULT_CROSS_WRIST_LAB_RESUME_CHECKPOINT = (
  "logs/rsl_rl/g1_tennis_latent_cross/"
  "tennis_cross_from_hit_2026-05-21_15-21-12/model_59998.pt"
)
DEFAULT_CONTINUOUS_RESUME_CHECKPOINT = (
  "logs/rsl_rl/g1_tennis_latent_cross/"
  "tennis_cross_from_hit_2026-05-21_15-21-12/model_59998.pt"
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
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_latent_cross",
    run_name="tennis_cross_from_hit",
    resume=True,
    load_checkpoint_file=DEFAULT_CROSS_RESUME_CHECKPOINT,
  )
  cfg.algorithm.entropy_coef = 0.003
  return cfg


def unitree_g1_tennis_hit_lab_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis Hit-LAB task."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_latent_hit_lab",
    run_name="tennis_hit_lab_scratch",
  )
  cfg.algorithm.entropy_coef = 0.003
  return cfg


def unitree_g1_tennis_cross_lab_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis Cross-LAB task."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_latent_cross_lab",
    run_name="tennis_cross_lab_from_cross",
    resume=True,
    load_checkpoint_file=DEFAULT_CROSS_LAB_RESUME_CHECKPOINT,
  )
  cfg.algorithm.entropy_coef = 0.001
  cfg.max_iterations = 30000
  return cfg


def unitree_g1_tennis_cross_wrist_lab_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis Cross-Wrist-LAB task."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_latent_cross_wrist_lab",
    run_name="tennis_cross_wrist_lab_from_cross",
    resume=True,
    load_checkpoint_file=DEFAULT_CROSS_WRIST_LAB_RESUME_CHECKPOINT,
  )
  cfg.algorithm.entropy_coef = 0.001
  cfg.max_iterations = 30000
  return cfg


def unitree_g1_tennis_continuous_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis continuous-rally task."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_latent_continuous",
    run_name="tennis_continuous_from_cross",
    resume=True,
    load_checkpoint_file=DEFAULT_CONTINUOUS_RESUME_CHECKPOINT,
  )
  cfg.algorithm.entropy_coef = 0.003
  cfg.max_iterations = 40000
  return cfg


def unitree_g1_tennis_sonic_hit_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis Hit-SONIC task."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_sonic_hit",
    run_name="tennis_hit_sonic_token",
  )
  cfg.algorithm.entropy_coef = 0.003
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.2,
    "std_type": "scalar",
  }
  cfg.clip_actions = 1.0
  cfg.require_decoder_checkpoint = False
  return cfg


def unitree_g1_tennis_sonic_cross_ppo_runner_cfg() -> TennisLatentOnPolicyRunnerCfg:
  """Create PPO config for the G1 tennis Cross-SONIC task."""
  cfg = unitree_g1_tennis_latent_ppo_runner_cfg(
    experiment_name="g1_tennis_sonic_cross",
    run_name="tennis_cross_sonic_scratch",
  )
  cfg.algorithm.entropy_coef = 0.001
  cfg.actor.distribution_cfg = {
    "class_name": "GaussianDistribution",
    "init_std": 0.2,
    "std_type": "scalar",
  }
  cfg.clip_actions = 1.0
  cfg.require_decoder_checkpoint = False
  cfg.max_iterations = 40000
  return cfg
