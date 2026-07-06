"""Play a distilled student with posterior encoder latents.

This script is intentionally self-contained: it defaults to the distillation
task, loads a distillation checkpoint, runs the encoder to produce the expected
latent ``z`` (posterior mean by default), and decodes actions for playback.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.distillation.rl import OnlineDistillationRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

DEFAULT_TASK_ID = "Mjlab-Distill-Flat-Unitree-G1"


@dataclass(frozen=True)
class PlayDistillConfig:
  """Configuration for posterior-guided distillation playback."""

  task_id: str = DEFAULT_TASK_ID
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  motion_files: str | tuple[str, ...] | None = None
  num_envs: int | None = None
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  log_root: str = "logs/rsl_rl"
  latent_source: Literal["posterior", "prior"] = "posterior"
  use_expected_z: bool = True
  video: bool = False
  video_folder: str | None = None
  video_length: int = 600
  video_height: int | None = None
  video_width: int | None = None
  video_name_prefix: str = "play-distill"


def _motion_file_refs(motion_files: str | tuple[str, ...] | None) -> tuple[str, ...]:
  if motion_files is None:
    return ()
  if isinstance(motion_files, str):
    return (motion_files,)
  return tuple(motion_files)


def _validate_local_files(
  paths: str | tuple[str, ...] | None,
  *,
  label: str,
) -> tuple[str, ...]:
  refs = _motion_file_refs(paths)
  if not refs:
    return ()
  missing = [ref for ref in refs if not Path(ref).exists()]
  if missing:
    raise FileNotFoundError(f"Missing {label}: {missing}")
  return refs


def _existing_local_files(paths: str | tuple[str, ...] | None) -> tuple[str, ...]:
  refs = _motion_file_refs(paths)
  return refs if refs and all(Path(ref).exists() for ref in refs) else ()


def _checkpoint_step(path: Path) -> int:
  try:
    return int(path.stem.split("_")[1])
  except (IndexError, ValueError):
    return -1


def _find_latest_local_checkpoint(log_root_path: Path) -> Path:
  if not log_root_path.exists():
    raise FileNotFoundError(f"Local log root not found: {log_root_path}")
  run_dirs = sorted(path for path in log_root_path.iterdir() if path.is_dir())
  for run_dir in reversed(run_dirs):
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_step)
    if checkpoints:
      return checkpoints[-1]
  raise FileNotFoundError(f"No checkpoints found under local log root: {log_root_path}")


def _resolve_checkpoint(
  cfg: PlayDistillConfig,
  *,
  experiment_name: str,
) -> Path:
  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file).expanduser()
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    return checkpoint_path

  log_root_path = (Path(cfg.log_root) / experiment_name).resolve()
  if cfg.wandb_run_path is not None:
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root_path,
      Path(cfg.wandb_run_path),
      cfg.wandb_checkpoint_name,
    )
    cache_state = "cached" if was_cached else "downloaded"
    print(
      "[INFO] Loading checkpoint: "
      f"{checkpoint_path.name} ({cache_state}, run={checkpoint_path.parent.name})"
    )
    return checkpoint_path

  checkpoint_path = _find_latest_local_checkpoint(log_root_path)
  print(f"[INFO] Loading latest local checkpoint: {checkpoint_path}")
  return checkpoint_path


def _load_checkpoint(path: Path, *, device: str) -> dict[str, Any]:
  checkpoint = torch.load(path, map_location=device, weights_only=False)
  if not isinstance(checkpoint, dict):
    raise ValueError(f"Unexpected checkpoint format in {path}")
  return checkpoint


def _runner_cfg_from_checkpoint(
  checkpoint: dict[str, Any],
  fallback_cfg: Any,
) -> dict[str, Any]:
  saved_cfg = checkpoint.get("cfg")
  if not isinstance(saved_cfg, dict):
    return asdict(fallback_cfg)
  return dict(saved_cfg)


def _restore_obs_slicer(
  runner: OnlineDistillationRunner,
  checkpoint: dict[str, Any],
) -> None:
  saved_slicer = checkpoint.get("obs_slicer")
  if not isinstance(saved_slicer, dict):
    return

  state_indices = saved_slicer.get("state_indices")
  target_indices = saved_slicer.get("target_indices")

  if isinstance(state_indices, torch.Tensor):
    if state_indices.numel() != runner.model.state_dim:
      raise ValueError(
        "Checkpoint state_indices length does not match model state_dim: "
        f"{state_indices.numel()} vs {runner.model.state_dim}."
      )
    if (
      state_indices.numel() and int(state_indices.max().item()) >= runner.slicer.obs_dim
    ):
      raise ValueError("Checkpoint state_indices exceed current observation dimension.")
    runner.slicer.state_indices = state_indices.to(runner.device)

  if isinstance(target_indices, torch.Tensor):
    if target_indices.numel() != runner.model.target_dim:
      raise ValueError(
        "Checkpoint target_indices length does not match model target_dim: "
        f"{target_indices.numel()} vs {runner.model.target_dim}."
      )
    if (
      target_indices.numel()
      and int(target_indices.max().item()) >= runner.slicer.obs_dim
    ):
      raise ValueError(
        "Checkpoint target_indices exceed current observation dimension."
      )
    runner.slicer.target_indices = target_indices.to(runner.device)


def _restore_runner_from_checkpoint(
  runner: OnlineDistillationRunner,
  checkpoint: dict[str, Any],
  *,
  strict: bool = True,
) -> None:
  runner.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
  if "optimizer_state_dict" in checkpoint:
    runner.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  runner.current_learning_iteration = int(checkpoint.get("iter", 0))
  _restore_obs_slicer(runner, checkpoint)


def _configure_motion_command(cfg: PlayDistillConfig, env_cfg) -> None:
  motion_cmd = env_cfg.commands.get("motion")
  if motion_cmd is None or not hasattr(motion_cmd, "motion_files"):
    return

  cli_motion_files = _validate_local_files(cfg.motion_files, label="motion_files")
  if cli_motion_files:
    motion_cmd.motion_files = cli_motion_files
    print(f"[INFO] Using {len(cli_motion_files)} local motion file(s) from CLI")
    return

  configured_local_files = _existing_local_files(motion_cmd.motion_files)
  if configured_local_files:
    motion_cmd.motion_files = configured_local_files
    print(f"[INFO] Using {len(configured_local_files)} configured local motion file(s)")
    return

  if cfg.wandb_run_path is None:
    raise FileNotFoundError(
      "No local motion files found. Provide --motion-files or --wandb-run-path."
    )

  import wandb

  api = wandb.Api()
  wandb_run = api.run(str(cfg.wandb_run_path))
  artifact = next(
    (item for item in wandb_run.used_artifacts() if item.type == "motions"), None
  )
  if artifact is None:
    raise RuntimeError("No motion artifact found in the W&B run.")
  motion_path = str(Path(artifact.download()) / "motion.npz")
  motion_cmd.motion_files = motion_path
  print(f"[INFO] Using motion artifact: {motion_path}")


class DistillationPlayPolicy:
  """Playback policy that reconstructs actions from prior/posterior latents."""

  def __init__(
    self,
    runner: OnlineDistillationRunner,
    *,
    latent_source: Literal["posterior", "prior"],
    use_expected_z: bool,
  ) -> None:
    self.runner = runner
    self.latent_source = latent_source
    self.use_expected_z = use_expected_z
    self._printed_once = False

  def __call__(self, obs) -> torch.Tensor:
    actor_obs = obs[self.runner.cfg["obs_group"]].to(self.runner.device)
    state, target = self.runner.slicer.split(actor_obs)

    if self.latent_source == "posterior":
      dist = self.runner.model.posterior_distribution(state, target)
    else:
      dist = self.runner.model.prior_distribution(state)

    latent = dist.mean if self.use_expected_z else dist.sample()
    if not self._printed_once:
      latent_mode = (
        "expected z (distribution mean)" if self.use_expected_z else "sampled z"
      )
      print(
        "[INFO] Playback latent source: "
        f"{self.latent_source}, mode: {latent_mode}, latent_dim={latent.shape[-1]}"
      )
      self._printed_once = True
    return self.runner.model.decode(state, latent)


def run_play_distill(cfg: PlayDistillConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  runner_cls = load_runner_cls(cfg.task_id)
  if runner_cls is not OnlineDistillationRunner:
    raise ValueError(
      f"Task {cfg.task_id!r} is not registered with OnlineDistillationRunner."
    )

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO] Terminations disabled")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  if cfg.video_length <= 0:
    raise ValueError(f"video_length must be positive, got {cfg.video_length}.")
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  _configure_motion_command(cfg, env_cfg)
  checkpoint_path = _resolve_checkpoint(cfg, experiment_name=agent_cfg.experiment_name)
  checkpoint = _load_checkpoint(checkpoint_path, device=device)
  runner_cfg = _runner_cfg_from_checkpoint(checkpoint, agent_cfg)
  clip_actions = runner_cfg.get("clip_actions", agent_cfg.clip_actions)

  env = ManagerBasedRlEnv(
    cfg=env_cfg,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )
  video_folder = (
    Path(cfg.video_folder)
    if cfg.video_folder is not None
    else checkpoint_path.parent / "videos" / "play_distill"
  )
  if cfg.video:
    print(f"[INFO] Recording video to: {video_folder}")
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      name_prefix=cfg.video_name_prefix,
      disable_logger=True,
    )
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

  try:
    runner = runner_cls(wrapped_env, runner_cfg, device=device)
    _restore_runner_from_checkpoint(runner, checkpoint, strict=False)

    checkpoint_state_terms = tuple(runner_cfg.get("state_terms", ()))
    checkpoint_target_terms = tuple(runner_cfg.get("target_terms", ()))
    print(
      "[INFO] Loaded checkpoint split: "
      f"state_terms={checkpoint_state_terms}, target_terms={checkpoint_target_terms}"
    )

    if cfg.latent_source == "posterior" and runner.slicer.target_dim == 0:
      raise ValueError(
        "Posterior playback requires non-empty target terms, but target_dim is 0."
      )

    runner.model.eval()
    policy = DistillationPlayPolicy(
      runner,
      latent_source=cfg.latent_source,
      use_expected_z=cfg.use_expected_z,
    )

    if cfg.viewer == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved_viewer = "native" if has_display else "viser"
    else:
      resolved_viewer = cfg.viewer

    print(
      "[INFO] Running playback for "
      f"{cfg.task_id} with state_dim={runner.slicer.state_dim}, "
      f"target_dim={runner.slicer.target_dim}, latent_dim={runner.model.latent_dim}"
    )

    if cfg.video:
      obs = wrapped_env.get_observations()
      for _ in range(cfg.video_length):
        with torch.no_grad():
          actions = policy(obs)
        obs, _, _, _ = wrapped_env.step(actions)
      print(f"[INFO] Saved video(s) under: {video_folder}")
      return

    if resolved_viewer == "native":
      NativeMujocoViewer(wrapped_env, policy).run()
    elif resolved_viewer == "viser":
      ViserPlayViewer(wrapped_env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    wrapped_env.close()


def main() -> None:
  import mjlab
  import mjlab.tasks  # noqa: F401

  args = tyro.cli(
    PlayDistillConfig,
    default=PlayDistillConfig(),
    config=mjlab.TYRO_FLAGS,
  )
  run_play_distill(args)


if __name__ == "__main__":
  main()
