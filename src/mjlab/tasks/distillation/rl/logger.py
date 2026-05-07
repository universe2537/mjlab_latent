"""Lightweight tensorboard / wandb logger matching rsl_rl semantics.

Used by the distillation runner because it does not subclass
``rsl_rl.runners.OnPolicyRunner`` (no PPO algorithm) and therefore cannot
reuse rsl_rl's built-in logger directly.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any

import git
import rsl_rl

from mjlab.rl import RslRlVecEnvWrapper


class DistillationLogger:
  """Thin tensorboard / wandb wrapper for non-PPO runners.

  Mirrors the subset of ``rsl_rl.utils.Logger`` actually used by the
  distillation training loop: writer init, scalar logging, checkpoint
  upload, git diff snapshotting.
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
