"""Online DAgger-style latent distillation runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.distillation.models import (
  LatentStudentModel,
  diagonal_gaussian_kl,
)
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Configuration for online latent action distillation."""

  class_name: str = "OnlineDistillationRunner"
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
  save_final: bool = True
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",)}
  )


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


class OnlineDistillationRunner:
  """Train a latent student online against a frozen tracking teacher."""

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
    self.writer = SummaryWriter(str(self.log_dir)) if self.log_dir is not None else None

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    del repo_file_path

  def _load_teacher_policy(self):
    checkpoint = Path(os.path.expandvars(self.cfg["teacher_checkpoint"])).expanduser()
    if not checkpoint.exists():
      raise FileNotFoundError(
        "Distillation requires a pretrained tracker checkpoint. Set "
        "`--agent.teacher-checkpoint /path/to/model.pt`."
      )

    teacher_cfg = load_rl_cfg(self.cfg["teacher_task_id"])
    teacher_runner_cls = (
      load_runner_cls(self.cfg["teacher_task_id"]) or MjlabOnPolicyRunner
    )
    teacher_runner = teacher_runner_cls(
      self.env,
      self._strip_distillation_only_cfg(teacher_cfg),
      device=str(self.device),
    )
    teacher_runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=self.cfg["teacher_strict_load"],
      map_location=str(self.device),
    )
    policy = teacher_runner.get_inference_policy(device=str(self.device))
    return policy

  @staticmethod
  def _strip_distillation_only_cfg(cfg) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(cfg)

  def _student_action(self, actor_obs: torch.Tensor) -> torch.Tensor:
    state, _ = self.slicer.split(actor_obs)
    return self.model.act(
      state,
      deterministic=self.cfg["deterministic_rollout"],
    )

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
    obs = self.env.get_observations()
    for iteration in range(
      self.current_learning_iteration,
      self.current_learning_iteration + num_learning_iterations,
    ):
      obs = self._rollout(obs)
      stats = self._update()

      if self.writer is not None:
        for key, value in stats.items():
          self.writer.add_scalar(f"distillation/{key}", value, iteration)
        self.writer.add_scalar("distillation/buffer_size", self.buffer.size, iteration)

      if iteration % 10 == 0:
        print(
          "[INFO] distill iter="
          f"{iteration} loss={stats['loss']:.5f} "
          f"action={stats['action_loss']:.5f} kl={stats['kl_loss']:.5f} "
          f"buffer={self.buffer.size}"
        )

      if self.log_dir is not None and iteration % self.cfg["save_interval"] == 0:
        self.save(str(self.log_dir / f"model_{iteration}.pt"))

      self.current_learning_iteration = iteration + 1

    if self.cfg["save_final"] and self.log_dir is not None:
      self.save(str(self.log_dir / f"model_{self.current_learning_iteration}.pt"))
    if self.writer is not None:
      self.writer.flush()

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
