"""在线 DAgger 风格的 latent 蒸馏 runner。

Implements Algorithm 1 from LATENT §3.2.2: collect rollouts from the
student, query a frozen tracking teacher for actions, and train the
student VAE with action-imitation + KL-to-prior loss.

Auxiliary concerns are split into sibling modules:

* :mod:`config`      -- ``DistillationRunnerCfg`` dataclass.
* :mod:`models`      -- PHC-style ``LatentStudentModel`` + KL helper.
* :mod:`buffer`      -- on-device circular replay buffer.
* :mod:`obs_slicer`  -- state / target index splitter.
* :mod:`logger`      -- tensorboard / wandb logging shim.
* :mod:`onnx`        -- prior-only student 的 ONNX 导出。

本文件只负责“训练流程编排”：

1. 加载冻结的 tracking teacher。
2. 用 posterior student / teacher 在环境中 rollout 收集样本。
3. 从 replay buffer 采样并更新 VAE student。
4. 保存 checkpoint 和导出部署用 ONNX。

模型结构、观测切分、buffer、日志等逻辑都拆到单独模块，便于维护。
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed.distributed_c10d import (
  ReduceOp,
  all_reduce,
  barrier,
  broadcast,
  init_process_group,
  is_initialized,
)

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.distillation.rl.buffer import ReplayBuffer
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg  # noqa: F401
from mjlab.tasks.distillation.rl.logger import DistillationLogger
from mjlab.tasks.distillation.rl.models import (
  LatentStudentModel,
  diagonal_gaussian_kl,
)
from mjlab.tasks.distillation.rl.obs_slicer import ObservationSlicer
from mjlab.tasks.distillation.rl.onnx import export_student_to_onnx
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls


class OnlineDistillationRunner:
  """在线训练 latent student，使其模仿冻结的 tracking teacher。"""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ) -> None:
    del kwargs
    # ``cfg`` 在训练脚本中已经被 ``asdict`` 转成字典，这里直接按字典访问。
    self.env = env
    self.cfg = train_cfg
    self.device = torch.device(device)
    self._configure_multi_gpu()
    requested_log_dir = Path(log_dir) if log_dir is not None else None
    self.log_dir = requested_log_dir if self.gpu_global_rank == 0 else None
    self.current_learning_iteration = 0

    # 将 actor observation 按配置切成：
    # 1. state  : prior / decoder 可见
    # 2. target : 仅 posterior 训练可见
    self.slicer = ObservationSlicer(
      env,
      group_name=self.cfg["obs_group"],
      state_terms=tuple(self.cfg["state_terms"]),
      target_terms=tuple(self.cfg["target_terms"]),
    )
    self.slicer.to(self.device)

    self.model = LatentStudentModel(
      state_dim=self.slicer.state_dim,
      target_dim=self.slicer.target_dim,
      action_dim=self.env.num_actions,
      latent_dim=self.cfg["latent_dim"],
      encoder_hidden_dims=tuple(self.cfg["encoder_hidden_dims"]),
      prior_hidden_dims=tuple(self.cfg["prior_hidden_dims"]),
      decoder_hidden_dims=tuple(self.cfg["decoder_hidden_dims"]),
      activation=self.cfg["activation"],
      min_log_std=self.cfg["min_log_std"],
      max_log_std=self.cfg["max_log_std"],
      posterior_feature_multiplier=int(self.cfg.get("posterior_feature_multiplier", 5)),
      prior_feature_multiplier=int(self.cfg.get("prior_feature_multiplier", 1)),
      z_all=bool(self.cfg.get("z_all", False)),
    ).to(self.device)
    # student 训练只需要优化 VAE 参数，因此这里直接使用 AdamW。
    self.optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=self.cfg["learning_rate"],
      weight_decay=self.cfg["weight_decay"],
    )
    # replay buffer 中保存的是 actor_obs 和 teacher_action，
    # 不保存 student_action，因为后者每次都可由当前模型重新前向得到。
    self.buffer = ReplayBuffer(
      capacity=self.cfg["buffer_capacity"],
      obs_dim=self.slicer.obs_dim,
      action_dim=self.env.num_actions,
      device=self.device,
    )
    self.teacher_policy = None
    self.logger = DistillationLogger(self.log_dir, self.cfg, self.env)

  def _configure_multi_gpu(self) -> None:
    """Initialize process-group state used by synchronous data parallelism."""
    self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    self.is_distributed = self.gpu_world_size > 1
    if not self.is_distributed:
      self.gpu_local_rank = 0
      self.gpu_global_rank = 0
      self.cfg["multi_gpu"] = None
      return

    self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
    self.gpu_global_rank = int(os.getenv("RANK", "0"))
    expected_device = f"cuda:{self.gpu_local_rank}"
    if str(self.device) != expected_device:
      raise ValueError(
        f"Device {str(self.device)!r} does not match local rank device "
        f"{expected_device!r}."
      )
    self.cfg["multi_gpu"] = {
      "global_rank": self.gpu_global_rank,
      "local_rank": self.gpu_local_rank,
      "world_size": self.gpu_world_size,
    }
    if not is_initialized():
      init_process_group(
        backend="nccl",
        rank=self.gpu_global_rank,
        world_size=self.gpu_world_size,
      )
    torch.cuda.set_device(self.gpu_local_rank)

  def _broadcast_model_parameters(self) -> None:
    """Make every rank start from rank 0's model parameters and buffers."""
    if not self.is_distributed:
      return
    for tensor in (*self.model.parameters(), *self.model.buffers()):
      broadcast(tensor.data, src=0)

  def _reduce_gradients(self) -> None:
    """Average model gradients across all distillation workers."""
    if not self.is_distributed:
      return
    params_and_grads = [
      (param, param.grad) for param in self.model.parameters() if param.grad is not None
    ]
    if not params_and_grads:
      return
    flat_grads = torch.cat([grad.view(-1) for _, grad in params_and_grads])
    all_reduce(flat_grads, op=ReduceOp.SUM)
    flat_grads /= self.gpu_world_size
    offset = 0
    for param, grad in params_and_grads:
      numel = param.numel()
      grad.copy_(flat_grads[offset : offset + numel].view_as(grad))
      offset += numel

  def _reduce_stats(self, stats: dict[str, float]) -> dict[str, float]:
    """Average scalar training diagnostics across workers for rank-0 logging."""
    if not self.is_distributed:
      return stats
    keys = tuple(stats)
    values = torch.tensor(
      [stats[key] for key in keys], device=self.device, dtype=torch.float64
    )
    all_reduce(values, op=ReduceOp.SUM)
    values /= self.gpu_world_size
    return {key: float(value) for key, value in zip(keys, values, strict=True)}

  @staticmethod
  def _linear_schedule(
    start_value: float,
    end_value: float | None,
    iteration: int,
    start_iteration: int,
    end_iteration: int,
  ) -> float:
    """按 iteration 做线性插值；未配置区间时退化为常数。"""
    if end_value is None or end_iteration <= start_iteration:
      return float(start_value)
    if iteration <= start_iteration:
      return float(start_value)
    if iteration >= end_iteration:
      return float(end_value)
    alpha = (iteration - start_iteration) / max(end_iteration - start_iteration, 1)
    return float((1.0 - alpha) * start_value + alpha * end_value)

  def _teacher_action_prob(self, iteration: int) -> float:
    """返回当前迭代使用的 teacher-forcing 概率。"""
    return self._linear_schedule(
      start_value=float(self.cfg["teacher_action_prob"]),
      end_value=self.cfg.get("teacher_action_prob_end"),
      iteration=iteration,
      start_iteration=0,
      end_iteration=int(self.cfg.get("teacher_action_prob_anneal_iters", 0)),
    )

  def _kl_loss_weight(self, iteration: int) -> float:
    """返回当前迭代使用的 KL 权重。"""
    return self._linear_schedule(
      start_value=float(self.cfg["kl_loss_weight"]),
      end_value=self.cfg.get("kl_loss_weight_end"),
      iteration=iteration,
      start_iteration=int(self.cfg.get("kl_loss_anneal_start", 0)),
      end_iteration=int(self.cfg.get("kl_loss_anneal_end", 0)),
    )

  # -- teacher / inference helpers -----------------------------------------

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    """让 logger 在训练开始时额外记录某个仓库的 git diff。"""
    self.logger.git_status_repos.append(repo_file_path)

  def _load_teacher_policy(self):
    """加载冻结 teacher，并返回仅推理使用的 policy callable。

    teacher 本身仍然沿用 tracking 任务原有的 runner / actor 结构；
    distillation runner 只把它当作动作标签提供者，不参与梯度更新。
    """
    checkpoint = Path(os.path.expandvars(self.cfg["teacher_checkpoint"])).expanduser()
    if not checkpoint.exists():
      raise FileNotFoundError(
        "Distillation requires a pretrained tracker checkpoint. "
        "Set `--agent.teacher-checkpoint /path/to/model.pt`."
      )
    teacher_cfg = load_rl_cfg(self.cfg["teacher_task_id"])
    teacher_runner_cls = (
      load_runner_cls(self.cfg["teacher_task_id"]) or MjlabOnPolicyRunner
    )
    # The frozen teacher is inference-only. Prevent its RSL runner from trying
    # to initialize the already-active student process group a second time.
    worker_world_size = os.environ.get("WORLD_SIZE")
    if self.is_distributed:
      os.environ["WORLD_SIZE"] = "1"
    try:
      teacher_runner = teacher_runner_cls(
        self.env,
        asdict(teacher_cfg),
        device=str(self.device),
      )
    finally:
      if worker_world_size is None:
        os.environ.pop("WORLD_SIZE", None)
      else:
        os.environ["WORLD_SIZE"] = worker_world_size
    teacher_runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=self.cfg["teacher_strict_load"],
      map_location=str(self.device),
    )
    return teacher_runner.get_inference_policy(device=str(self.device))

  def _posterior_action(
    self, actor_obs: torch.Tensor, *, deterministic: bool
  ) -> torch.Tensor:
    """Generate a reference-conditioned action from the posterior."""
    state, target = self.slicer.split(actor_obs)
    return self.model.act(
      state,
      target,
      deterministic=deterministic,
      source="posterior",
    )

  def _prior_action(
    self, actor_obs: torch.Tensor, *, deterministic: bool
  ) -> torch.Tensor:
    """Generate a state-only action from the deployment prior."""
    state, _ = self.slicer.split(actor_obs)
    return self.model.act(
      state,
      deterministic=deterministic,
      source="prior",
    )

  # -- training loop -------------------------------------------------------

  def _rollout(self, obs, teacher_prob: float) -> Any:
    """执行一段环境 rollout，并把 ``(actor_obs, teacher_action)`` 存入 buffer。

    这里使用的是 DAgger 风格数据收集：

    1. 环境主要由 student action 驱动。
    2. teacher 仅提供监督标签。
    3. 若 ``teacher_action_prob > 0``，则部分环境步会直接执行教师动作，
       用于缓解早期 student rollout 偏离分布过快的问题。
    """
    if self.teacher_policy is None:
      self.teacher_policy = self._load_teacher_policy()
    self.model.eval()
    for _ in range(self.cfg["num_steps_per_env"]):
      actor_obs = obs[self.cfg["obs_group"]].to(self.device)
      with torch.no_grad():
        teacher_action = self.teacher_policy(obs).to(self.device)
        student_action = self._posterior_action(
          actor_obs,
          deterministic=bool(self.cfg["deterministic_rollout"]),
        )
        rollout_action = student_action
        if teacher_prob > 0:
          mask = (
            torch.rand(student_action.shape[0], 1, device=self.device) < teacher_prob
          )
          rollout_action = torch.where(mask, teacher_action, student_action)
      self.buffer.add(actor_obs.detach(), teacher_action.detach())
      obs, _, _, _ = self.env.step(rollout_action)
    return obs

  def _update(self, kl_loss_weight: float) -> dict[str, float]:
    """从 replay buffer 采样并执行若干次梯度更新。"""
    self.model.train()
    stats = {
      "loss": 0.0,
      "action_loss": 0.0,
      "prior_action_loss": 0.0,
      "kl_loss": 0.0,
    }
    for _ in range(self.cfg["updates_per_iteration"]):
      obs, teacher_action = self.buffer.sample(self.cfg["batch_size"])
      state, target = self.slicer.split(obs)
      # ``forward_train`` 返回 posterior / prior，便于外部显式计算 KL。
      pred_action, posterior, prior = self.model.forward_train(state, target)
      action_loss = F.mse_loss(pred_action, teacher_action)
      with torch.no_grad():
        prior_action = self.model.decode(state, prior.mean)
        prior_action_loss = F.mse_loss(prior_action, teacher_action)
      kl_loss = diagonal_gaussian_kl(posterior, prior).mean()
      loss = self.cfg["action_loss_weight"] * action_loss + kl_loss_weight * kl_loss
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      self._reduce_gradients()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
      self.optimizer.step()
      stats["loss"] += float(loss.detach())
      stats["action_loss"] += float(action_loss.detach())
      stats["prior_action_loss"] += float(prior_action_loss.detach())
      stats["kl_loss"] += float(kl_loss.detach())
    for key in stats:
      stats[key] /= self.cfg["updates_per_iteration"]
    return stats

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """主训练循环。

    参数:
      num_learning_iterations: 总迭代数。
      init_at_random_ep_len: 为兼容 on-policy runner 接口而保留；
        distillation 当前未使用该选项。
    """
    del init_at_random_ep_len
    self._broadcast_model_parameters()
    self.logger.init()
    obs = self.env.get_observations()
    start_iter = self.current_learning_iteration
    end_iter = start_iter + num_learning_iterations
    tot_iter = end_iter - start_iter
    start_time = time.time()

    for iteration in range(start_iter, end_iter):
      iter_start = time.time()
      teacher_prob = self._teacher_action_prob(iteration)
      kl_loss_weight = self._kl_loss_weight(iteration)
      obs = self._rollout(obs, teacher_prob=teacher_prob)
      stats = self._update(kl_loss_weight=kl_loss_weight)
      stats = self._reduce_stats(stats)
      iter_time = time.time() - iter_start

      self.logger.add_scalar("distillation/loss", stats["loss"], iteration)
      self.logger.add_scalar(
        "distillation/action_loss", stats["action_loss"], iteration
      )
      self.logger.add_scalar(
        "distillation/prior_action_loss", stats["prior_action_loss"], iteration
      )
      self.logger.add_scalar("distillation/kl_loss", stats["kl_loss"], iteration)
      self.logger.add_scalar(
        "distillation/teacher_action_prob", teacher_prob, iteration
      )
      self.logger.add_scalar("distillation/kl_loss_weight", kl_loss_weight, iteration)
      self.logger.add_scalar("distillation/buffer_size", self.buffer.size, iteration)

      if iteration % self.cfg["save_interval"] == 0 and self.log_dir is not None:
        self.save(str(self.log_dir / f"model_{iteration}.pt"))

      if iteration % 10 == 0 and self.gpu_global_rank == 0:
        elapsed = time.time() - start_time
        done_frac = max((iteration - start_iter + 1) / tot_iter, 1e-6)
        eta = elapsed / done_frac * (1.0 - done_frac)
        print(
          f"[distillation] iter: {iteration}/{end_iter - 1} "
          f"loss: {stats['loss']:.5f}  "
          f"action: {stats['action_loss']:.5f}  "
          f"prior_action: {stats['prior_action_loss']:.5f}  "
          f"kl: {stats['kl_loss']:.5f}  "
          f"teacher_prob: {teacher_prob:.3f}  "
          f"kl_w: {kl_loss_weight:.4g}  "
          f"buf: {self.buffer.size}  "
          f"iter_time: {iter_time:.2f}s  "
          f"eta: {eta / 60:.1f}min"
        )

      self.current_learning_iteration = iteration + 1

    if self.is_distributed:
      barrier()
    if self.log_dir is not None:
      self.save(str(self.log_dir / f"model_{self.current_learning_iteration}.pt"))
    if self.is_distributed:
      barrier()
    self.logger.stop()

  # -- checkpoint & export -------------------------------------------------

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def save(self, path: str, infos=None) -> None:
    """保存 student/optimizer/obs_slicer，并导出 prior-only ONNX。"""
    del infos
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "cfg": self.cfg,
        "obs_slicer": {
          "state_indices": self.slicer.state_indices.cpu(),
          "target_indices": self.slicer.target_indices.cpu(),
        },
      },
      path,
    )
    if self.cfg.get("upload_model"):
      self.logger.save_model(path, self.current_learning_iteration)
    export_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(export_dir), filename)
      run_name: str = "local"
      try:
        import wandb

        if self.logger.logger_type == "wandb" and wandb.run:
          run_name = wandb.run.name  # type: ignore[assignment]
      except ImportError:
        pass
      metadata = get_base_metadata(self.env.unwrapped, run_name)
      metadata.update(
        {
          "latent_dim": self.cfg["latent_dim"],
          "state_terms": list(self.cfg["state_terms"]),
          "target_terms": list(self.cfg["target_terms"]),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if self.logger.logger_type == "wandb" and self.cfg.get("upload_model"):
        try:
          import wandb as _wandb

          if _wandb.run:
            _wandb.save(str(onnx_path), base_path=str(export_dir))
        except ImportError:
          pass
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the deployment-time prior-only student to ONNX."""
    export_student_to_onnx(
      model=self.model,
      state_indices=self.slicer.state_indices,
      obs_dim=self.slicer.obs_dim,
      path=path,
      filename=filename,
      verbose=verbose,
    )

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """从 checkpoint 恢复 student 权重和优化器状态。"""
    del load_cfg
    checkpoint = torch.load(
      path,
      map_location=map_location or str(self.device),
      weights_only=False,
    )
    self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if "optimizer_state_dict" in checkpoint:
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    self.current_learning_iteration = int(checkpoint.get("iter", 0))
    return {}

  def get_inference_policy(self, device: str | None = None):
    """返回使用 posterior mean 的 reference-tracking 推理函数。"""
    if device is not None:
      self.model.to(device)
      self.device = torch.device(device)
      self.slicer.to(self.device)
    self.model.eval()

    def policy(obs) -> torch.Tensor:
      actor_obs = obs[self.cfg["obs_group"]].to(self.device)
      with torch.no_grad():
        return self._posterior_action(actor_obs, deterministic=True)

    return policy
