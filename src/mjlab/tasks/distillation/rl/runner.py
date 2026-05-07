"""Online DAgger-style latent distillation runner."""

from __future__ import annotations

import os
import pathlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import git
import rsl_rl
import torch
import torch.nn.functional as F
from torch import nn

from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.distillation.rl.models import (
  LatentStudentModel,
  diagonal_gaussian_kl,
)
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration for online latent action distillation."""

  teacher_task_id: str = "Mjlab-Tracking-Flat-Unitree-G1"
  teacher_checkpoint: str = ""
  teacher_strict_load: bool = True
  obs_group: str = "actor"
  state_terms: tuple[str, ...] = ()
  target_terms: tuple[str, ...] = ("command",)
  latent_dim: int = 16
  encoder_hidden_dims: tuple[int, ...] = (512, 256)
  prior_hidden_dims: tuple[int, ...] = (512, 256)
  decoder_hidden_dims: tuple[int, ...] = (512, 256)
  activation: str = "elu"
  min_log_std: float = -5.0
  max_log_std: float = 2.0
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


class _DistillationLogger:
  """Thin logging wrapper matching rsl_rl style for non-PPO runners.

  Supports tensorboard and wandb backends based on ``cfg["logger"]``.
  Call :meth:`init` before the training loop and :meth:`stop` after.
  """

  def __init__(
    self,
    log_dir: Path | None,
    cfg: dict[str, Any],
    env: RslRlVecEnvWrapper,
  ) -> None:
    self.log_dir = log_dir
    self.cfg = cfg
    self.env = env
    self.logger_type = cfg.get("logger", "tensorboard").lower()
    self.git_status_repos: list[str] = [rsl_rl.__file__]
    self.writer = None

  def init(self) -> None:
    """Create the writer and persist git diffs / configs."""
    if self.log_dir is None:
      return
    if self.logger_type == "wandb":
      from rsl_rl.utils.wandb_utils import WandbSummaryWriter

      self.writer = WandbSummaryWriter(
        log_dir=str(self.log_dir), flush_secs=10, cfg=self.cfg
      )
      self.writer.store_config(self.env.unwrapped.cfg, self.cfg)
    elif self.logger_type == "tensorboard":
      from torch.utils.tensorboard import SummaryWriter

      self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)
    else:
      raise ValueError(
        f"Unknown logger type {self.logger_type!r}. Choose 'tensorboard' or 'wandb'."
      )

    files_to_upload = self._store_code_state()
    if self.logger_type == "wandb":
      for path in files_to_upload:
        self.writer.save_file(path)  # type: ignore[union-attr]

  def _store_code_state(self) -> list[str]:
    """Write git diff files to ``<log_dir>/git/`` (mirrors rsl_rl Logger)."""
    files_to_upload: list[str] = []
    if self.log_dir is None:
      return files_to_upload
    git_log_dir = self.log_dir / "git"
    git_log_dir.mkdir(parents=True, exist_ok=True)
    for repo_file in self.git_status_repos:
      try:
        repo = git.Repo(repo_file, search_parent_directories=True)
        commit_hash = repo.head.commit.hexsha
        t = repo.head.commit.tree
      except Exception:
        print(f"[WARN] Could not find git repository in {repo_file}. Skipping.")
        continue
      repo_name = pathlib.Path(repo.working_dir).name
      diff_path = git_log_dir / f"{repo_name}.diff"
      if diff_path.exists():
        continue
      print(f"Storing git diff for '{repo_name}' in: {diff_path}")
      diff_path.write_text(
        f"--- git commit ---\n{commit_hash}\n\n\n"
        f"--- git status ---\n{repo.git.status()}\n\n\n"
        f"--- git diff ---\n{repo.git.diff(t)}",
        encoding="utf-8",
      )
      files_to_upload.append(str(diff_path))
    return files_to_upload

  def add_scalar(self, tag: str, value: float, step: int) -> None:
    if self.writer is not None:
      self.writer.add_scalar(tag, value, step)

  def save_model(self, path: str, it: int) -> None:
    if self.writer is not None and self.logger_type in ("wandb", "neptune"):
      self.writer.save_model(path, it)  # type: ignore[union-attr]

  def stop(self) -> None:
    if self.writer is not None and self.logger_type in ("wandb", "neptune"):
      self.writer.stop()  # type: ignore[union-attr]


class ReplayBuffer:
  def __init__(
    self,
    *,
    capacity: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
  ) -> None:
    self.capacity = capacity
    self.device = device
    self.obs = torch.empty(capacity, obs_dim, device=device)
    self.teacher_actions = torch.empty(capacity, action_dim, device=device)
    self.size = 0
    self.pos = 0

  def add(self, obs: torch.Tensor, teacher_actions: torch.Tensor) -> None:
    num_items = obs.shape[0]
    if num_items >= self.capacity:
      self.obs[:] = obs[-self.capacity :]
      self.teacher_actions[:] = teacher_actions[-self.capacity :]
      self.size = self.capacity
      self.pos = 0
      return

    end = self.pos + num_items
    if end <= self.capacity:
      self.obs[self.pos : end] = obs
      self.teacher_actions[self.pos : end] = teacher_actions
    else:
      first = self.capacity - self.pos
      self.obs[self.pos :] = obs[:first]
      self.teacher_actions[self.pos :] = teacher_actions[:first]
      self.obs[: end - self.capacity] = obs[first:]
      self.teacher_actions[: end - self.capacity] = teacher_actions[first:]

    self.pos = end % self.capacity
    self.size = min(self.capacity, self.size + num_items)

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if self.size == 0:
      raise RuntimeError("Cannot sample from an empty replay buffer.")
    idx = torch.randint(self.size, (min(batch_size, self.size),), device=self.device)
    return self.obs[idx], self.teacher_actions[idx]


class ObservationSlicer:
  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    *,
    group_name: str,
    state_terms: tuple[str, ...],
    target_terms: tuple[str, ...],
  ) -> None:
    obs_manager = env.unwrapped.observation_manager
    if not obs_manager.group_obs_concatenate[group_name]:
      raise ValueError(
        f"Distillation expects observation group {group_name!r} to be concatenated."
      )

    term_names = obs_manager.active_terms[group_name]
    term_dims = obs_manager.group_obs_term_dim[group_name]
    if len(state_terms) == 0:
      state_terms = tuple(name for name in term_names if name not in target_terms)

    self.state_indices = self._indices_for_terms(term_names, term_dims, state_terms)
    self.target_indices = self._indices_for_terms(term_names, term_dims, target_terms)
    self.obs_dim = sum(int(torch.tensor(dim).prod().item()) for dim in term_dims)

  @staticmethod
  def _indices_for_terms(
    term_names: list[str],
    term_dims: list[tuple[int, ...]],
    selected_terms: tuple[str, ...],
  ) -> torch.Tensor:
    missing = sorted(set(selected_terms).difference(term_names))
    if missing:
      raise ValueError(f"Unknown observation term(s): {missing}")

    indices: list[int] = []
    offset = 0
    for term_name, term_dim in zip(term_names, term_dims, strict=True):
      length = int(torch.tensor(term_dim).prod().item())
      if term_name in selected_terms:
        indices.extend(range(offset, offset + length))
      offset += length
    return torch.tensor(indices, dtype=torch.long)

  @property
  def state_dim(self) -> int:
    return int(self.state_indices.numel())

  @property
  def target_dim(self) -> int:
    return int(self.target_indices.numel())

  def to(self, device: torch.device) -> None:
    self.state_indices = self.state_indices.to(device)
    self.target_indices = self.target_indices.to(device)

  def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return obs[:, self.state_indices], obs[:, self.target_indices]


class _OnnxStudentModel(nn.Module):
  """ONNX-exportable wrapper for the prior-only student inference path.

  At deployment only the prior P(z|s) is available (no target s̃_{t+1}).
  This wrapper slices the full actor observation down to the state subset,
  runs the prior, and decodes the action deterministically.
  """

  def __init__(self, model: LatentStudentModel, state_indices: torch.Tensor) -> None:
    super().__init__()
    self.model = model
    self.register_buffer("state_indices", state_indices.cpu())

  def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
    state = actor_obs[:, self.state_indices]  # type: ignore[index]
    prior = self.model.prior_distribution(state)
    return self.model.decode(state, prior.mean)


class OnlineDistillationRunner:
  """Train a latent student online against a frozen tracking teacher.

  Implements Algorithm 1 from LATENT §3.2.2: DAgger rollout with teacher
  supervision, variational bottleneck (CVAE encoder/prior/decoder), and
  joint action-imitation + KL loss.
  """

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
    self.env = env
    self.cfg = train_cfg
    self.log_dir = Path(log_dir) if log_dir is not None else None
    self.device = torch.device(device)
    self.current_learning_iteration = 0

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
    ).to(self.device)
    self.optimizer = torch.optim.AdamW(
      self.model.parameters(),
      lr=self.cfg["learning_rate"],
      weight_decay=self.cfg["weight_decay"],
    )
    self.buffer = ReplayBuffer(
      capacity=self.cfg["buffer_capacity"],
      obs_dim=self.slicer.obs_dim,
      action_dim=self.env.num_actions,
      device=self.device,
    )
    self.teacher_policy = None
    self.logger = _DistillationLogger(self.log_dir, self.cfg, self.env)

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    """Register a repo path so its git diff is saved at the start of training."""
    self.logger.git_status_repos.append(repo_file_path)

  def _load_teacher_policy(self):
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
    teacher_runner = teacher_runner_cls(
      self.env,
      asdict(teacher_cfg),
      device=str(self.device),
    )
    teacher_runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=self.cfg["teacher_strict_load"],
      map_location=str(self.device),
    )
    return teacher_runner.get_inference_policy(device=str(self.device))

  def _student_action(self, actor_obs: torch.Tensor) -> torch.Tensor:
    state, _ = self.slicer.split(actor_obs)
    return self.model.act(state, deterministic=self.cfg["deterministic_rollout"])

  def _rollout(self, obs) -> Any:
    if self.teacher_policy is None:
      self.teacher_policy = self._load_teacher_policy()
    self.model.eval()
    for _ in range(self.cfg["num_steps_per_env"]):
      actor_obs = obs[self.cfg["obs_group"]].to(self.device)
      with torch.no_grad():
        teacher_action = self.teacher_policy(obs).to(self.device)
        student_action = self._student_action(actor_obs)
        rollout_action = student_action
        teacher_prob = self.cfg["teacher_action_prob"]
        if teacher_prob > 0:
          mask = (
            torch.rand(student_action.shape[0], 1, device=self.device) < teacher_prob
          )
          rollout_action = torch.where(mask, teacher_action, student_action)
      self.buffer.add(actor_obs.detach(), teacher_action.detach())
      obs, _, _, _ = self.env.step(rollout_action)
    return obs

  def _update(self) -> dict[str, float]:
    self.model.train()
    stats = {"loss": 0.0, "action_loss": 0.0, "kl_loss": 0.0}
    for _ in range(self.cfg["updates_per_iteration"]):
      obs, teacher_action = self.buffer.sample(self.cfg["batch_size"])
      state, target = self.slicer.split(obs)
      pred_action, posterior, prior = self.model.forward_train(state, target)
      action_loss = F.mse_loss(pred_action, teacher_action)
      kl_loss = diagonal_gaussian_kl(posterior, prior).mean()
      loss = (
        self.cfg["action_loss_weight"] * action_loss
        + self.cfg["kl_loss_weight"] * kl_loss
      )
      self.optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
      self.optimizer.step()
      stats["loss"] += float(loss.detach())
      stats["action_loss"] += float(action_loss.detach())
      stats["kl_loss"] += float(kl_loss.detach())
    for key in stats:
      stats[key] /= self.cfg["updates_per_iteration"]
    return stats

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    del init_at_random_ep_len
    self.logger.init()
    obs = self.env.get_observations()
    start_iter = self.current_learning_iteration
    end_iter = start_iter + num_learning_iterations
    tot_iter = end_iter - start_iter
    start_time = time.time()

    for iteration in range(start_iter, end_iter):
      iter_start = time.time()
      obs = self._rollout(obs)
      stats = self._update()
      iter_time = time.time() - iter_start

      self.logger.add_scalar("distillation/loss", stats["loss"], iteration)
      self.logger.add_scalar(
        "distillation/action_loss", stats["action_loss"], iteration
      )
      self.logger.add_scalar("distillation/kl_loss", stats["kl_loss"], iteration)
      self.logger.add_scalar("distillation/buffer_size", self.buffer.size, iteration)

      if iteration % self.cfg["save_interval"] == 0 and self.log_dir is not None:
        self.save(str(self.log_dir / f"model_{iteration}.pt"))

      if iteration % 10 == 0:
        elapsed = time.time() - start_time
        done_frac = max((iteration - start_iter + 1) / tot_iter, 1e-6)
        eta = elapsed / done_frac * (1.0 - done_frac)
        print(
          f"[distillation] iter: {iteration}/{end_iter - 1} "
          f"loss: {stats['loss']:.5f}  "
          f"action: {stats['action_loss']:.5f}  "
          f"kl: {stats['kl_loss']:.5f}  "
          f"buf: {self.buffer.size}  "
          f"iter_time: {iter_time:.2f}s  "
          f"eta: {eta / 60:.1f}min"
        )

      self.current_learning_iteration = iteration + 1

    if self.log_dir is not None:
      self.save(str(self.log_dir / f"model_{self.current_learning_iteration}.pt"))
    self.logger.stop()

  def save(self, path: str, infos=None) -> None:
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
    """Export the prior-only student policy to ONNX.

    The exported model takes the full actor observation and returns the
    deterministic action using the prior mean (no target required).
    """
    import copy

    os.makedirs(path, exist_ok=True)
    # deepcopy to avoid moving the live training model to CPU
    onnx_model = _OnnxStudentModel(copy.deepcopy(self.model), self.slicer.state_indices)
    onnx_model.to("cpu")
    onnx_model.eval()
    obs_dim = self.slicer.obs_dim
    dummy_obs = torch.zeros(1, obs_dim)
    torch.onnx.export(
      onnx_model,
      (dummy_obs,),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["actor_obs"],
      output_names=["actions"],
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
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
    if device is not None:
      self.model.to(device)
      self.device = torch.device(device)
      self.slicer.to(self.device)
    self.model.eval()

    def policy(obs) -> torch.Tensor:
      actor_obs = obs[self.cfg["obs_group"]].to(self.device)
      with torch.no_grad():
        return self._student_action(actor_obs)

    return policy
