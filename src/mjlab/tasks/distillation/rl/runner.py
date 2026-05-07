"""Online DAgger-style latent distillation runner.

Implements Algorithm 1 from LATENT §3.2.2: collect rollouts from the
student, query a frozen tracking teacher for actions, and train the
student VAE with action-imitation + KL-to-prior loss.

Auxiliary concerns are split into sibling modules:

* :mod:`config`      -- ``DistillationRunnerCfg`` dataclass.
* :mod:`models`      -- PHC-style ``LatentStudentModel`` + KL helper.
* :mod:`buffer`      -- on-device circular replay buffer.
* :mod:`obs_slicer`  -- state / target index splitter.
* :mod:`logger`      -- tensorboard / wandb logging shim.
* :mod:`onnx`        -- ONNX export of the prior-only student.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
      posterior_feature_multiplier=int(self.cfg.get("posterior_feature_multiplier", 5)),
      prior_feature_multiplier=int(self.cfg.get("prior_feature_multiplier", 1)),
      z_all=bool(self.cfg.get("z_all", False)),
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
    self.logger = DistillationLogger(self.log_dir, self.cfg, self.env)

  # -- teacher / inference helpers -----------------------------------------

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
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
    return self.model.act(
      state,
      deterministic=self.cfg["deterministic_rollout"],
      source="prior",
    )

  # -- training loop -------------------------------------------------------

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

  # -- checkpoint & export -------------------------------------------------

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"
    return export_dir, filename, export_dir / filename

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
