"""Unitree G1 latent distillation runner configuration."""

from mjlab.tasks.distillation.rl.runner import DistillationRunnerCfg


def unitree_g1_distillation_runner_cfg() -> DistillationRunnerCfg:
  """Create the online distillation runner config for G1 LaFAN primitives."""
  return DistillationRunnerCfg(
    experiment_name="g1_distillation",
    run_name="lafan_walk_run_sprint",
    teacher_task_id="Mjlab-Tracking-Flat-Unitree-G1",
    # 默认指向当前可用的 G1+racket tracking PPO 检查点。
    # 也可在 CLI 通过 --agent.teacher-checkpoint 覆盖。
    teacher_checkpoint=(
      "./logs/rsl_rl/g1_tracking/2026-04-30_21-55-08/model_19500.pt"
    ),
    latent_dim=16,
    encoder_hidden_dims=(512, 256),
    prior_hidden_dims=(512, 256),
    decoder_hidden_dims=(512, 256),
    state_terms=(
      "motion_anchor_pos_b",
      "motion_anchor_ori_b",
      "base_lin_vel",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ),
    target_terms=("command",),
    action_loss_weight=1.0,
    kl_loss_weight=1.0e-3,
    learning_rate=8.0e-4,
    buffer_capacity=262_144,
    batch_size=16_384,
    updates_per_iteration=4,
    num_steps_per_env=16,
    max_iterations=30000,
    save_interval=250,
    upload_model=False,
    wandb_project="mjlab",
  )
