"""在线 latent 蒸馏的配置定义。

该配置类集中描述 distillation 训练所需的超参数，包含：

1. 教师策略的来源。
2. 观测如何切分为 ``state`` 和 ``target``。
3. VAE 编码器 / 先验 / 解码器的结构。
4. 优化器、损失权重和 rollout 行为。

之所以单独定义为 dataclass，而不是散落在 runner 中，是为了让 CLI
覆盖、配置导出和实验复现都更直接。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  """在线 latent action distillation 的配置。

  结构上尽量保持与 ``mjlab.rl.config.RslRlOnPolicyRunnerCfg`` 一致，
  这样训练脚本、配置导出和命令行覆盖都可以沿用现有工具链。
  """

  # -- Teacher ------------------------------------------------------------
  # 教师策略对应的任务 ID。runner 会通过 registry 加载它的 runner 类型、
  # agent 配置和推理策略。
  teacher_task_id: str = "Mjlab-Tracking-Flat-Unitree-G1"
  # 教师 checkpoint 路径。通常是 tracking 任务已经训练好的 ``model_x.pt``。
  teacher_checkpoint: str = ""
  # 加载教师权重时是否严格匹配 state_dict 键名。
  # 设为 True 更安全；当教师网络结构发生小改动时，可临时关闭排查兼容性。
  teacher_strict_load: bool = True

  # -- Observation routing ------------------------------------------------
  # 从哪个 observation group 读取学生输入。当前 distillation 默认复用 actor。
  obs_group: str = "actor"
  # ``state_terms`` 表示先验 ``P(z|s)`` 和解码器 ``D(a|s,z)`` 可见的状态项。
  # 留空时，runner 会自动取 ``obs_group`` 中除 ``target_terms`` 外的全部项。
  state_terms: tuple[str, ...] = ()
  # ``target_terms`` 表示仅训练时后验 ``E(z|s,s_tilde)`` 可见的目标项，
  # 例如 motion command / 未来参考等。部署时这些项通常不可得。
  target_terms: tuple[str, ...] = ("command",)

  # -- VAE topology -------------------------------------------------------
  # 潜变量维度。越大表示 latent 空间容量越高，但 KL 约束通常也更难稳定。
  latent_dim: int = 16
  # 后验编码器 ``E(z|s,s_tilde)`` 的 MLP 隐层宽度。
  encoder_hidden_dims: tuple[int, ...] = (512, 256)
  # 先验网络 ``P(z|s)`` 的 MLP 隐层宽度。
  prior_hidden_dims: tuple[int, ...] = (512, 256)
  # 解码器 ``D(a|s,z)`` 或 ``D(a|z)`` 的 MLP 隐层宽度。
  decoder_hidden_dims: tuple[int, ...] = (512, 256)
  # MLP 使用的激活函数名称，需与 ``models._activation_cls`` 支持的名称一致。
  activation: str = "elu"
  # 对高斯分布 ``log_std`` 的下界裁剪，防止方差过小导致数值不稳定。
  min_log_std: float = -5.0
  # 对高斯分布 ``log_std`` 的上界裁剪，防止方差过大导致采样过于发散。
  max_log_std: float = 2.0
  # 后验支路在 ``mu`` / ``log_std`` 头之前的特征放大倍数。
  # PHC / PULSE 的默认做法是 ``latent_dim * 5``，即更宽的 posterior bottleneck。
  posterior_feature_multiplier: int = 5
  # 先验支路在输出头之前的特征放大倍数。通常保持为 1，
  # 即先验头直接挂在 trunk 最后一层上。
  prior_feature_multiplier: int = 1
  # 若为 True，解码器仅使用 ``z`` 生成动作；否则使用 ``[state, z]``。
  # 这对应 PHC 中的 ``z_all`` 选项。
  z_all: bool = False

  # -- Optimisation -------------------------------------------------------
  # AdamW 学习率。
  learning_rate: float = 3.0e-4
  # AdamW 的权重衰减系数。
  weight_decay: float = 0.0
  # 动作回归损失的权重，即 ``MSE(student_action, teacher_action)`` 的系数。
  action_loss_weight: float = 1.0
  # KL 正则项初始权重，即 ``KL(q(z|s,s_tilde) || p(z|s))`` 的系数。
  kl_loss_weight: float = 1.0e-2
  # KL 正则项的目标权重。若为 None，则训练中保持常数 ``kl_loss_weight``。
  kl_loss_weight_end: float | None = None
  # KL 退火起始迭代。小于等于 0 表示从训练一开始就按 ``kl_loss_weight`` 生效。
  kl_loss_anneal_start: int = 0
  # KL 退火结束迭代。若不大于 ``kl_loss_anneal_start``，则不启用退火。
  kl_loss_anneal_end: int = 0
  # 每次参数更新时从 replay buffer 中采样的 batch 大小。
  batch_size: int = 16_384
  # replay buffer 容量，存放 ``(actor_obs, teacher_action)`` 样本。
  buffer_capacity: int = 262_144
  # 每轮 rollout 之后进行多少次梯度更新。
  updates_per_iteration: int = 4
  # rollout 时初始以多大概率直接使用教师动作与环境交互。
  # 若 ``teacher_action_prob_end`` 为 None，则训练中保持常数。
  teacher_action_prob: float = 0.0
  # rollout 时教师动作概率的目标值。配合 ``teacher_action_prob_anneal_iters``
  # 可实现从 teacher-forcing 到纯 student rollout 的线性退火。
  teacher_action_prob_end: float | None = None
  # 教师动作概率退火结束迭代。若不大于 0，则不启用退火。
  teacher_action_prob_anneal_iters: int = 0
  # rollout 时是否使用确定性 posterior mean；默认从 posterior 采样。
  deterministic_rollout: bool = False
  # 梯度裁剪阈值，防止少数 batch 导致训练发散。
  max_grad_norm: float = 1.0

  # distillation 只需要 actor 观测组；teacher 不是通过 env 的 teacher group
  # 直接前向，而是加载一个完整 tracking runner 做推理。
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",)}
  )
