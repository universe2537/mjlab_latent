"""Unitree G1 的 latent distillation runner 配置。

这里给出的是一个可直接运行的实验实例：

1. 教师来自 G1 tracking 任务。
2. 学生使用 PHC/LATENT 风格 VAE。
3. ``state_terms`` / ``target_terms`` 明确了训练时的输入切分。
"""

from mjlab.tasks.distillation.rl.runner import DistillationRunnerCfg


def unitree_g1_distillation_runner_cfg() -> DistillationRunnerCfg:
  """构造 G1 在线蒸馏配置。

  关键参数说明:
    teacher_checkpoint: 预训练 tracking teacher 的权重路径。
    latent_dim: latent 空间维度。
    state_terms: prior / decoder 在 rollout 和部署时可见的状态量。
    target_terms: posterior 在训练时额外可见的目标量。
    action_loss_weight: 行为克隆损失权重。
    kl_loss_weight: posterior 到 prior 的 KL 正则权重。
    num_steps_per_env: 每轮先 rollout 多少步，再做更新。
    updates_per_iteration: 每轮 rollout 后做多少次梯度更新。
  """
  return DistillationRunnerCfg(
    experiment_name="g1_distillation",
    teacher_task_id="Mjlab-Tracking-Flat-Unitree-G1",
    # 默认指向当前可用的 G1 tracking teacher 检查点。
    # 若实验中需要替换教师，可在 CLI 中覆盖该字段。
    teacher_checkpoint=("./logs/rsl_rl/g1_tracking/tennis/model_29999.pt"),
    latent_dim=16,
    posterior_feature_multiplier=3,
    encoder_hidden_dims=(512, 256),
    prior_hidden_dims=(512, 256),
    decoder_hidden_dims=(512, 256),
    # ``state_terms`` 是部署时可获得的信息，供 prior/decoder 使用。
    state_terms=(
      "base_lin_vel",
      "base_ang_vel",
      "joint_pos",
      "joint_vel",
      "actions",
    ),
    # ``target_terms`` 是训练时才可见的目标项，供 posterior 使用。
    target_terms=(
      "motion_anchor_pos_b",
      "motion_anchor_ori_b",
      "command",
    ),
    action_loss_weight=1.0,
    kl_loss_weight=1.0e-3,
    kl_loss_weight_end=5.0e-3,
    kl_loss_anneal_start=2500,
    kl_loss_anneal_end=10000,
    learning_rate=8.0e-4,
    # 扩大 buffer，让每批 DAgger 数据在被覆盖前能被 SGD 重复利用约 2 次。
    buffer_capacity=1048576,
    batch_size=32768,
    updates_per_iteration=16,
    
    teacher_action_prob=1.0,
    teacher_action_prob_end=0.2,
    teacher_action_prob_anneal_iters=15000,
    num_steps_per_env=16,
    max_iterations=30000,
    save_interval=250,
    upload_model=False,
  )
