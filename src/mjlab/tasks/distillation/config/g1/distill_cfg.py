"""Unitree G1 latent distillation runner configuration."""

from mjlab.tasks.distillation.runner import DistillationRunnerCfg


def unitree_g1_distillation_runner_cfg() -> DistillationRunnerCfg:
  """Create the online distillation runner config for G1 LaFAN primitives."""
  return DistillationRunnerCfg(
    experiment_name="g1_distillation",
    run_name="lafan_walk_run_sprint",
    teacher_task_id="Mjlab-Tracking-Flat-Unitree-G1",
    teacher_checkpoint="",
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
    learning_rate=3.0e-4,
    buffer_capacity=262_144,
    batch_size=16_384,
    updates_per_iteration=4,
    num_steps_per_env=16,
    max_iterations=10_000,
    save_interval=250,
    logger="tensorboard",
    upload_model=False,
  )
