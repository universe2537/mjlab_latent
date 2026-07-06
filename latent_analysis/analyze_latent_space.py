#!/usr/bin/env python3
"""Offline diagnostics for the table-tennis distillation latent checkpoint."""

from __future__ import annotations

import argparse
import csv
import html
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

import mjlab.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.distillation.rl.runner import OnlineDistillationRunner  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.tasks.tracking.mdp.commands import (  # noqa: E402
  MotionCommand,
  MotionCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

DEFAULT_TABLE_TENNIS_CHECKPOINT = Path(
  "/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/"
  "g1_distillation_table_tennis/"
  "table_tennis_distill_v1_46080env_from_tracking18000_2026-07-03_10-14-26/"
  "model_30000.pt"
)
DEFAULT_TASK_ID = "Mjlab-Distill-TableTennis-Unitree-G1"
DEFAULT_MOTION_PATTERN = "*/motion.npz"
ACTIVE_DIM_STD_THRESHOLD = 0.03
PLOT_SAMPLE_LIMIT = 6000


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Analyze a latent distillation checkpoint without modifying training "
      "code or decoder defaults."
    )
  )
  parser.add_argument(
    "--checkpoint", type=Path, default=DEFAULT_TABLE_TENNIS_CHECKPOINT
  )
  parser.add_argument("--task-id", type=str, default=DEFAULT_TASK_ID)
  parser.add_argument("--analysis-label", type=str, default="table-tennis")
  parser.add_argument(
    "--motion-root", type=Path, default=Path("artifacts/table_tennis")
  )
  parser.add_argument("--motion-pattern", type=str, default=DEFAULT_MOTION_PATTERN)
  parser.add_argument(
    "--output-dir", type=Path, default=Path("latent_analysis/outputs")
  )
  parser.add_argument("--event-dir", type=Path, default=None)
  parser.add_argument("--samples-per-motion", type=int, default=1024)
  parser.add_argument("--num-envs", type=int, default=256)
  parser.add_argument("--steps-per-reset", type=int, default=8)
  parser.add_argument("--warmup-steps", type=int, default=1)
  parser.add_argument("--max-batches", type=int, default=128)
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--tsne", action="store_true")
  parser.add_argument("--umap", action="store_true")
  parser.add_argument(
    "--keep-disturbances",
    action="store_true",
    help="Keep reset perturbations, pushes, corruption, and wrist encoder bias.",
  )
  return parser.parse_args()


def json_default(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, torch.Tensor):
    return value.detach().cpu().tolist()
  return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
    json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n"
  )


def motion_split(name: str) -> str:
  if name == "test_001":
    return "heldout"
  if "badend" in name:
    return "diagnostic"
  return "train"


def motion_family(name: str) -> str:
  if name.startswith("fanshou"):
    return "fanshou"
  if name.startswith("zhengshou"):
    return "zhengshou"
  if name.startswith("mix"):
    return "mix"
  if name.startswith("test"):
    return "test"
  if name.startswith("tennis_random"):
    return "tennis_random"
  if name.startswith("tennis"):
    return "tennis"
  return "other"


def discover_motion_files(motion_root: Path, motion_pattern: str) -> list[Path]:
  root = motion_root.expanduser()
  if not root.is_absolute():
    root = REPO_ROOT / root
  files = sorted(root.glob(motion_pattern), key=lambda path: path.parent.name)
  if not files:
    raise FileNotFoundError(
      f"No motion files found under {root} with pattern {motion_pattern!r}"
    )
  return [path.resolve() for path in files]


def resolve_teacher_checkpoint(raw_path: str, student_checkpoint: Path) -> Path:
  checkpoint = Path(os.path.expandvars(raw_path)).expanduser()
  candidates: list[Path] = []
  if checkpoint.is_absolute():
    candidates.append(checkpoint)
  else:
    candidates.append((REPO_ROOT / checkpoint).resolve())
    for parent in student_checkpoint.resolve().parents:
      if parent.name == REPO_ROOT.name:
        candidates.append((parent / checkpoint).resolve())
        break
  for candidate in candidates:
    if candidate.exists():
      return candidate
  searched = ", ".join(str(path) for path in candidates)
  raise FileNotFoundError(
    f"Teacher checkpoint from student cfg was not found: {raw_path}. "
    f"Searched: {searched}"
  )


def summarize_tensorboard(event_dir: Path) -> tuple[dict[str, Any], list[str]]:
  notes: list[str] = []
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except Exception as exc:  # pragma: no cover - depends on optional runtime package.
    return {
      "available": False,
      "reason": f"tensorboard import failed: {exc}",
    }, [f"TensorBoard metrics skipped: {exc}"]

  if event_dir.is_file():
    event_dir = event_dir.parent
  event_files = sorted(event_dir.glob("events.out.tfevents*"))
  if not event_files:
    return {
      "available": False,
      "event_dir": str(event_dir),
      "reason": "no event files found",
    }, [f"No TensorBoard event files found under {event_dir}"]

  accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  tags = set(accumulator.Tags().get("scalars", []))
  requested = {
    "loss": ("distillation/loss", True),
    "action_loss": ("distillation/action_loss", True),
    "prior_action_loss": ("distillation/prior_action_loss", True),
    "kl_loss": ("distillation/kl_loss", True),
    "teacher_action_prob": ("distillation/teacher_action_prob", False),
    "buffer_size": ("distillation/buffer_size", False),
  }

  metrics: dict[str, Any] = {
    "available": True,
    "event_dir": str(event_dir),
    "event_files": [str(path) for path in event_files],
    "scalars": {},
  }
  for name, (tag, lower_is_better) in requested.items():
    if tag not in tags:
      notes.append(f"TensorBoard tag missing: {tag}")
      metrics["scalars"][name] = {"available": False, "tag": tag}
      continue
    events = accumulator.Scalars(tag)
    values = np.array([event.value for event in events], dtype=np.float64)
    steps = np.array([event.step for event in events], dtype=np.int64)
    if values.size == 0:
      metrics["scalars"][name] = {"available": False, "tag": tag}
      continue
    best_index = int(np.argmin(values) if lower_is_better else np.argmax(values))
    window = values[-100:]
    metrics["scalars"][name] = {
      "available": True,
      "tag": tag,
      "count": int(values.size),
      "final": float(values[-1]),
      "final_step": int(steps[-1]),
      "best": float(values[best_index]),
      "best_step": int(steps[best_index]),
      "best_direction": "min" if lower_is_better else "max",
      "last100_mean": float(window.mean()),
      "last100_min": float(window.min()),
      "last100_max": float(window.max()),
    }
  return metrics, notes


def write_metrics_markdown(path: Path, metrics: dict[str, Any]) -> None:
  lines = ["# Training Metrics Summary", ""]
  if not metrics.get("available"):
    lines.append(f"TensorBoard metrics unavailable: {metrics.get('reason', 'unknown')}")
    path.write_text("\n".join(lines) + "\n")
    return
  lines.append(f"- Event dir: `{metrics['event_dir']}`")
  lines.append("")
  lines.append("| Metric | Final | Best | Best step | Last100 mean |")
  lines.append("| --- | ---: | ---: | ---: | ---: |")
  for name, scalar in metrics["scalars"].items():
    if not scalar.get("available"):
      lines.append(f"| {name} | n/a | n/a | n/a | n/a |")
      continue
    lines.append(
      "| "
      + " | ".join(
        [
          name,
          f"{scalar['final']:.8g}",
          f"{scalar['best']:.8g}",
          str(scalar["best_step"]),
          f"{scalar['last100_mean']:.8g}",
        ]
      )
      + " |"
    )
  path.write_text("\n".join(lines) + "\n")


def configure_analysis_env(
  motion_files: list[Path],
  *,
  task_id: str,
  num_envs: int,
  seed: int,
  keep_disturbances: bool,
) -> Any:
  env_cfg = load_env_cfg(task_id)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = int(num_envs)

  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_source = "local"
  motion_cmd.motion_files = tuple(str(path) for path in motion_files)
  motion_cmd.motion_sample_probs = ()
  motion_cmd.sampling_mode = "uniform"

  if not keep_disturbances:
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events.pop("push_robot", None)
    env_cfg.events.pop("wrist_encoder_bias", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
  return env_cfg


@torch.no_grad()
def seed_balanced_reference_state(
  vec_env: RslRlVecEnvWrapper,
  motion_cmd: Any,
  *,
  batch_index: int,
) -> None:
  device = motion_cmd.device
  num_envs = motion_cmd.num_envs
  num_motions = motion_cmd.motion.num_motions
  env_ids = torch.arange(num_envs, device=device)
  motion_ids = (env_ids + batch_index * num_envs) % num_motions
  lengths = motion_cmd.motion.motion_lengths[motion_ids]
  random_steps = torch.floor(
    torch.rand(num_envs, device=device) * lengths.to(dtype=torch.float32)
  ).to(dtype=torch.long)
  motion_cmd.motion_ids[env_ids] = motion_ids
  motion_cmd.time_steps[env_ids] = torch.clamp(random_steps, max=lengths - 1)

  motion_cmd._write_reference_state_to_sim(  # noqa: SLF001
    env_ids,
    motion_cmd.body_pos_w[env_ids, 0],
    motion_cmd.body_quat_w[env_ids, 0],
    motion_cmd.body_lin_vel_w[env_ids, 0],
    motion_cmd.body_ang_vel_w[env_ids, 0],
    motion_cmd.joint_pos[env_ids],
    motion_cmd.joint_vel[env_ids],
  )
  motion_cmd.update_relative_body_poses()
  vec_env.unwrapped.episode_length_buf[env_ids] = 0


def load_runner(
  checkpoint: Path,
  env: RslRlVecEnvWrapper,
  *,
  device: str,
) -> tuple[OnlineDistillationRunner, dict[str, Any], Path, dict[str, Any]]:
  checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  cfg_dict = dict(checkpoint_data["cfg"])
  teacher_checkpoint = resolve_teacher_checkpoint(
    str(cfg_dict["teacher_checkpoint"]),
    checkpoint,
  )
  cfg_dict["teacher_checkpoint"] = str(teacher_checkpoint)

  runner = OnlineDistillationRunner(env, cfg_dict, log_dir=None, device=device)
  runner.load(str(checkpoint), strict=True, map_location=device)
  runner.model.eval()
  return runner, cfg_dict, teacher_checkpoint, checkpoint_data


def append_tensor(
  storage: dict[str, list[np.ndarray]],
  key: str,
  tensor: torch.Tensor,
) -> None:
  storage[key].append(tensor.detach().cpu().numpy())


@torch.no_grad()
def collect_samples(
  checkpoint: Path,
  motion_files: list[Path],
  args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  configure_torch_backends()

  env_cfg = configure_analysis_env(
    motion_files,
    task_id=args.task_id,
    num_envs=max(args.num_envs, len(motion_files) * 4),
    seed=args.seed,
    keep_disturbances=args.keep_disturbances,
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  vec_env = RslRlVecEnvWrapper(env)
  notes: list[str] = []

  try:
    runner, cfg_dict, teacher_checkpoint, checkpoint_data = load_runner(
      checkpoint,
      vec_env,
      device=args.device,
    )
    teacher_policy = runner._load_teacher_policy()  # noqa: SLF001
    motion_cmd = cast(
      MotionCommand,
      vec_env.unwrapped.command_manager.get_term("motion"),
    )
    motion_names = [Path(path).parent.name for path in motion_cmd.motion.motion_files]
    joint_names = list(vec_env.unwrapped.scene["robot"].joint_names)
    if len(joint_names) != vec_env.num_actions:
      joint_names = [f"joint_{idx:02d}" for idx in range(vec_env.num_actions)]

    storage: dict[str, list[np.ndarray]] = {
      "q_mean": [],
      "q_std": [],
      "p_mean": [],
      "p_std": [],
      "posterior_action": [],
      "posterior_sample_action": [],
      "prior_action": [],
      "teacher_action": [],
      "kl": [],
      "prior_posterior_l2": [],
    }
    sample_meta: dict[str, list[Any]] = {
      "motion_name": [],
      "split": [],
      "family": [],
      "phase": [],
      "motion_id": [],
      "time_step": [],
    }
    counts = {name: 0 for name in motion_names}
    obs_group = str(cfg_dict["obs_group"])

    for batch_index in range(args.max_batches):
      seed_balanced_reference_state(vec_env, motion_cmd, batch_index=batch_index)
      obs = vec_env.get_observations()
      for step_index in range(args.steps_per_reset):
        teacher_action = teacher_policy(obs).to(args.device)
        if step_index >= args.warmup_steps:
          ids = motion_cmd.motion_ids.detach().cpu().numpy()
          steps = motion_cmd.time_steps.detach().cpu().numpy()
          lengths = motion_cmd.motion.motion_lengths.detach().cpu().numpy()
          selected: list[int] = []
          selected_names: list[str] = []
          selected_phases: list[float] = []
          selected_steps: list[int] = []
          selected_ids: list[int] = []
          for env_idx, motion_id in enumerate(ids.tolist()):
            name = motion_names[motion_id]
            if counts[name] >= args.samples_per_motion:
              continue
            counts[name] += 1
            selected.append(env_idx)
            selected_names.append(name)
            selected_ids.append(int(motion_id))
            selected_steps.append(int(steps[env_idx]))
            denom = max(int(lengths[motion_id]) - 1, 1)
            selected_phases.append(float(steps[env_idx]) / float(denom))

          if selected:
            selected_idx = torch.tensor(selected, dtype=torch.long, device=args.device)
            actor_obs = obs[obs_group].to(args.device)[selected_idx]
            state, target = runner.slicer.split(actor_obs)
            q_dist = runner.model.posterior_distribution(state, target)
            p_dist = runner.model.prior_distribution(state)
            posterior_action = runner.model.decode(state, q_dist.mean)
            posterior_sample = runner.model.decode(state, q_dist.rsample())
            prior_action = runner.model.decode(state, p_dist.mean)
            teacher_selected = teacher_action[selected_idx]
            kl = (
              p_dist.log_std
              - q_dist.log_std
              + (q_dist.std.square() + (q_dist.mean - p_dist.mean).square())
              / (2.0 * p_dist.std.square())
              - 0.5
            ).sum(dim=-1)
            mean_l2 = torch.linalg.norm(q_dist.mean - p_dist.mean, dim=-1)

            append_tensor(storage, "q_mean", q_dist.mean)
            append_tensor(storage, "q_std", q_dist.std)
            append_tensor(storage, "p_mean", p_dist.mean)
            append_tensor(storage, "p_std", p_dist.std)
            append_tensor(storage, "posterior_action", posterior_action)
            append_tensor(storage, "posterior_sample_action", posterior_sample)
            append_tensor(storage, "prior_action", prior_action)
            append_tensor(storage, "teacher_action", teacher_selected)
            append_tensor(storage, "kl", kl)
            append_tensor(storage, "prior_posterior_l2", mean_l2)

            sample_meta["motion_name"].extend(selected_names)
            sample_meta["split"].extend(motion_split(name) for name in selected_names)
            sample_meta["family"].extend(motion_family(name) for name in selected_names)
            sample_meta["phase"].extend(selected_phases)
            sample_meta["motion_id"].extend(selected_ids)
            sample_meta["time_step"].extend(selected_steps)

        obs, _, _, _ = vec_env.step(teacher_action)

      if all(count >= args.samples_per_motion for count in counts.values()):
        break

    missing = {
      name: args.samples_per_motion - count
      for name, count in counts.items()
      if count < args.samples_per_motion
    }
    if missing:
      notes.append(f"Sample collection stopped before target counts: {missing}")

    if not storage["q_mean"]:
      raise RuntimeError("No latent samples were collected.")

    arrays = {key: np.concatenate(value, axis=0) for key, value in storage.items()}
    arrays["kl"] = arrays["kl"].reshape(-1)
    arrays["prior_posterior_l2"] = arrays["prior_posterior_l2"].reshape(-1)
    for key, value in sample_meta.items():
      arrays[key] = np.array(value)

    collection_meta = {
      "checkpoint": str(checkpoint),
      "checkpoint_iteration": int(checkpoint_data.get("iter", -1)),
      "teacher_checkpoint": str(teacher_checkpoint),
      "task_id": args.task_id,
      "analysis_label": args.analysis_label,
      "device": args.device,
      "num_envs": int(vec_env.num_envs),
      "samples_per_motion_target": int(args.samples_per_motion),
      "steps_per_reset": int(args.steps_per_reset),
      "warmup_steps": int(args.warmup_steps),
      "max_batches": int(args.max_batches),
      "collected_samples": int(arrays["q_mean"].shape[0]),
      "counts_by_motion": counts,
      "motion_files": [str(path) for path in motion_files],
      "joint_names": joint_names,
      "latent_dim": int(runner.model.latent_dim),
      "state_dim": int(runner.slicer.state_dim),
      "target_dim": int(runner.slicer.target_dim),
      "action_dim": int(runner.model.action_dim),
      "state_terms": list(cfg_dict["state_terms"]),
      "target_terms": list(cfg_dict["target_terms"]),
      "clean_env": not args.keep_disturbances,
      "collection_mode": "teacher-forced reference-state env probe",
    }
    return arrays, collection_meta, notes
  finally:
    vec_env.close()


def covariance_spectrum(values: np.ndarray) -> dict[str, Any]:
  if values.shape[0] < 2:
    return {"eigenvalues": [], "explained_ratio": [], "effective_rank": 0.0}
  centered = values - values.mean(axis=0, keepdims=True)
  cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
  eigenvalues = np.linalg.eigvalsh(cov)[::-1]
  total = float(np.clip(eigenvalues.sum(), 1e-12, None))
  ratio = np.clip(eigenvalues / total, 0.0, None)
  entropy = -float(np.sum(ratio * np.log(ratio + 1e-12)))
  return {
    "eigenvalues": eigenvalues.tolist(),
    "explained_ratio": ratio.tolist(),
    "effective_rank": float(math.exp(entropy)),
  }


def compute_latent_stats(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
  q_mean = arrays["q_mean"]
  p_mean = arrays["p_mean"]
  q_std = arrays["q_std"]
  p_std = arrays["p_std"]
  q_mean_std = q_mean.std(axis=0)
  p_mean_std = p_mean.std(axis=0)
  active_mask = (q_mean_std > ACTIVE_DIM_STD_THRESHOLD) | (
    p_mean_std > ACTIVE_DIM_STD_THRESHOLD
  )
  per_dim = []
  for idx in range(q_mean.shape[1]):
    per_dim.append(
      {
        "dim": idx,
        "posterior_mean_mean": float(q_mean[:, idx].mean()),
        "posterior_mean_std": float(q_mean_std[idx]),
        "posterior_mean_min": float(q_mean[:, idx].min()),
        "posterior_mean_max": float(q_mean[:, idx].max()),
        "posterior_std_mean": float(q_std[:, idx].mean()),
        "prior_mean_mean": float(p_mean[:, idx].mean()),
        "prior_mean_std": float(p_mean_std[idx]),
        "prior_mean_min": float(p_mean[:, idx].min()),
        "prior_mean_max": float(p_mean[:, idx].max()),
        "prior_std_mean": float(p_std[:, idx].mean()),
        "active": bool(active_mask[idx]),
      }
    )
  return {
    "num_samples": int(q_mean.shape[0]),
    "latent_dim": int(q_mean.shape[1]),
    "active_dim_std_threshold": ACTIVE_DIM_STD_THRESHOLD,
    "active_dim_count": int(active_mask.sum()),
    "active_dims": np.where(active_mask)[0].astype(int).tolist(),
    "posterior_prior_kl": {
      "mean": float(arrays["kl"].mean()),
      "std": float(arrays["kl"].std()),
      "min": float(arrays["kl"].min()),
      "max": float(arrays["kl"].max()),
    },
    "prior_posterior_mean_l2": {
      "mean": float(arrays["prior_posterior_l2"].mean()),
      "std": float(arrays["prior_posterior_l2"].std()),
      "min": float(arrays["prior_posterior_l2"].min()),
      "max": float(arrays["prior_posterior_l2"].max()),
    },
    "per_dim": per_dim,
    "posterior_covariance_spectrum": covariance_spectrum(q_mean),
    "prior_covariance_spectrum": covariance_spectrum(p_mean),
  }


def error_stats(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
  err = pred - target
  return {
    "mse": float(np.mean(np.square(err))),
    "mae": float(np.mean(np.abs(err))),
  }


def grouped_error_stats(
  pred: np.ndarray,
  target: np.ndarray,
  labels: np.ndarray,
) -> dict[str, dict[str, float]]:
  result: dict[str, dict[str, float]] = {}
  for label in sorted(set(labels.tolist())):
    mask = labels == label
    stats = error_stats(pred[mask], target[mask])
    stats["samples"] = int(mask.sum())
    result[str(label)] = stats
  return result


def compute_reconstruction_metrics(
  arrays: dict[str, np.ndarray],
  joint_names: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  teacher = arrays["teacher_action"]
  posterior = arrays["posterior_action"]
  posterior_sample = arrays["posterior_sample_action"]
  prior = arrays["prior_action"]
  split = arrays["split"]
  motion_names = arrays["motion_name"]

  metrics: dict[str, Any] = {
    "global": {
      "posterior": error_stats(posterior, teacher),
      "posterior_sample": error_stats(posterior_sample, teacher),
      "prior": error_stats(prior, teacher),
    },
    "by_split": {
      "posterior": grouped_error_stats(posterior, teacher, split),
      "posterior_sample": grouped_error_stats(posterior_sample, teacher, split),
      "prior": grouped_error_stats(prior, teacher, split),
    },
    "by_motion": {
      "posterior": grouped_error_stats(posterior, teacher, motion_names),
      "posterior_sample": grouped_error_stats(posterior_sample, teacher, motion_names),
      "prior": grouped_error_stats(prior, teacher, motion_names),
    },
  }

  prior_joint_err = prior - teacher
  posterior_joint_err = posterior - teacher
  per_joint = []
  for idx, name in enumerate(joint_names):
    per_joint.append(
      {
        "joint": name,
        "index": idx,
        "posterior_mse": float(np.mean(np.square(posterior_joint_err[:, idx]))),
        "posterior_mae": float(np.mean(np.abs(posterior_joint_err[:, idx]))),
        "prior_mse": float(np.mean(np.square(prior_joint_err[:, idx]))),
        "prior_mae": float(np.mean(np.abs(prior_joint_err[:, idx]))),
      }
    )
  metrics["per_joint"] = per_joint

  per_motion_rows: list[dict[str, Any]] = []
  for motion in sorted(set(motion_names.tolist())):
    mask = motion_names == motion
    row = {
      "motion": motion,
      "split": motion_split(motion),
      "family": motion_family(motion),
      "samples": int(mask.sum()),
      "posterior_mse": float(np.mean(np.square(posterior[mask] - teacher[mask]))),
      "posterior_mae": float(np.mean(np.abs(posterior[mask] - teacher[mask]))),
      "prior_mse": float(np.mean(np.square(prior[mask] - teacher[mask]))),
      "prior_mae": float(np.mean(np.abs(prior[mask] - teacher[mask]))),
      "kl_mean": float(arrays["kl"][mask].mean()),
      "prior_posterior_l2_mean": float(arrays["prior_posterior_l2"][mask].mean()),
    }
    per_motion_rows.append(row)
  return metrics, per_motion_rows


def write_per_motion_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = [
    "motion",
    "split",
    "family",
    "samples",
    "posterior_mse",
    "posterior_mae",
    "prior_mse",
    "prior_mae",
    "kl_mean",
    "prior_posterior_l2_mean",
  ]
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def pca_2d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  centered = values - values.mean(axis=0, keepdims=True)
  _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
  coords = centered @ vt[:2].T
  variance = singular_values**2
  explained = variance[:2] / max(float(variance.sum()), 1e-12)
  return coords, explained


def plot_scatter_by_category(
  coords: np.ndarray,
  labels: np.ndarray,
  title: str,
  output_path: Path,
  explained: np.ndarray,
) -> None:
  fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
  unique = sorted(set(labels.tolist()))
  cmap = plt.get_cmap("tab20", max(len(unique), 1))
  for idx, label in enumerate(unique):
    mask = labels == label
    ax.scatter(
      coords[mask, 0],
      coords[mask, 1],
      s=8,
      alpha=0.62,
      label=str(label),
      color=cmap(idx),
      linewidths=0,
    )
  ax.set_title(title)
  ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
  ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
  ax.legend(markerscale=2.0, fontsize=7, ncol=2, frameon=False)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def plot_phase(coords: np.ndarray, phases: np.ndarray, output_path: Path) -> None:
  fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
  sc = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=phases,
    s=8,
    alpha=0.65,
    cmap="viridis",
    linewidths=0,
  )
  ax.set_title("Posterior latent PCA by motion phase")
  ax.set_xlabel("PC1")
  ax.set_ylabel("PC2")
  fig.colorbar(sc, ax=ax, label="normalized motion phase")
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def plot_latent_activity(latent_stats: dict[str, Any], output_path: Path) -> None:
  per_dim = latent_stats["per_dim"]
  dims = [item["dim"] for item in per_dim]
  q_std = [item["posterior_mean_std"] for item in per_dim]
  p_std = [item["prior_mean_std"] for item in per_dim]
  width = 0.38
  x = np.arange(len(dims))
  fig, ax = plt.subplots(figsize=(9, 4), dpi=160)
  ax.bar(x - width / 2, q_std, width, label="posterior mean std")
  ax.bar(x + width / 2, p_std, width, label="prior mean std")
  ax.axhline(ACTIVE_DIM_STD_THRESHOLD, color="black", linestyle="--", linewidth=1)
  ax.set_xticks(x)
  ax.set_xticklabels([str(dim) for dim in dims])
  ax.set_xlabel("latent dimension")
  ax.set_ylabel("std across samples")
  ax.set_title("Latent dimension activity")
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def plot_alignment(arrays: dict[str, np.ndarray], output_path: Path) -> None:
  fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
  axes[0].hist(arrays["prior_posterior_l2"], bins=48, color="#4c78a8", alpha=0.82)
  axes[0].set_title("Prior/posterior mean L2")
  axes[0].set_xlabel("L2 distance")
  axes[0].set_ylabel("count")
  axes[1].hist(arrays["kl"], bins=48, color="#f58518", alpha=0.82)
  axes[1].set_title("KL(q || p)")
  axes[1].set_xlabel("KL")
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def plot_per_motion_error(
  rows: list[dict[str, Any]],
  output_path: Path,
) -> None:
  names = [row["motion"] for row in rows]
  prior = [row["prior_mse"] for row in rows]
  posterior = [row["posterior_mse"] for row in rows]
  x = np.arange(len(names))
  width = 0.38
  fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
  ax.bar(x - width / 2, posterior, width, label="posterior")
  ax.bar(x + width / 2, prior, width, label="prior")
  ax.set_xticks(x)
  ax.set_xticklabels(names, rotation=40, ha="right")
  ax.set_ylabel("action MSE")
  ax.set_title("Reconstruction error by motion")
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def plot_per_joint_error(metrics: dict[str, Any], output_path: Path) -> None:
  rows = metrics["per_joint"]
  names = [row["joint"] for row in rows]
  prior = [row["prior_mse"] for row in rows]
  posterior = [row["posterior_mse"] for row in rows]
  x = np.arange(len(names))
  width = 0.38
  fig, ax = plt.subplots(figsize=(13, 5), dpi=160)
  ax.bar(x - width / 2, posterior, width, label="posterior")
  ax.bar(x + width / 2, prior, width, label="prior")
  ax.set_xticks(x)
  ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
  ax.set_ylabel("action MSE")
  ax.set_title("Reconstruction error by joint")
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(output_path)
  plt.close(fig)


def maybe_subsample_indices(count: int, seed: int) -> np.ndarray:
  if count <= PLOT_SAMPLE_LIMIT:
    return np.arange(count)
  rng = np.random.default_rng(seed)
  return np.sort(rng.choice(count, size=PLOT_SAMPLE_LIMIT, replace=False))


def generate_plots(
  arrays: dict[str, np.ndarray],
  latent_stats: dict[str, Any],
  reconstruction_metrics: dict[str, Any],
  per_motion_rows: list[dict[str, Any]],
  output_dir: Path,
  *,
  seed: int,
  enable_tsne: bool,
  enable_umap: bool,
) -> list[str]:
  notes: list[str] = []
  idx = maybe_subsample_indices(arrays["q_mean"].shape[0], seed)
  q_mean = arrays["q_mean"][idx]
  coords, explained = pca_2d(q_mean)

  plot_scatter_by_category(
    coords,
    arrays["motion_name"][idx],
    "Posterior latent PCA by motion",
    output_dir / "latent_pca_by_motion.png",
    explained,
  )
  plot_scatter_by_category(
    coords,
    arrays["family"][idx],
    "Posterior latent PCA by family",
    output_dir / "latent_pca_by_family.png",
    explained,
  )
  plot_phase(
    coords, arrays["phase"][idx].astype(float), output_dir / "latent_pca_by_phase.png"
  )
  plot_latent_activity(latent_stats, output_dir / "latent_dim_activity.png")
  plot_alignment(arrays, output_dir / "prior_posterior_alignment.png")
  plot_per_motion_error(
    per_motion_rows, output_dir / "per_motion_reconstruction_error.png"
  )
  plot_per_joint_error(
    reconstruction_metrics, output_dir / "per_joint_reconstruction_error.png"
  )

  if enable_tsne:
    try:
      TSNE = importlib.import_module("sklearn.manifold").TSNE
    except Exception as exc:  # pragma: no cover - optional dependency.
      notes.append(f"t-SNE skipped because sklearn is unavailable: {exc}")
    else:
      tsne_coords = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=min(30.0, max(5.0, q_mean.shape[0] / 100.0)),
        random_state=seed,
      ).fit_transform(q_mean)
      plot_scatter_by_category(
        tsne_coords,
        arrays["family"][idx],
        "Posterior latent t-SNE by family",
        output_dir / "latent_tsne_by_family.png",
        np.array([0.0, 0.0]),
      )

  if enable_umap:
    try:
      umap_module = importlib.import_module("umap")
    except Exception as exc:  # pragma: no cover - optional dependency.
      notes.append(f"UMAP skipped because umap-learn is unavailable: {exc}")
    else:
      umap_coords = umap_module.UMAP(random_state=seed).fit_transform(q_mean)
      plot_scatter_by_category(
        umap_coords,
        arrays["family"][idx],
        "Posterior latent UMAP by family",
        output_dir / "latent_umap_by_family.png",
        np.array([0.0, 0.0]),
      )
  return notes


def judge_verdict(
  latent_stats: dict[str, Any],
  reconstruction_metrics: dict[str, Any],
) -> tuple[str, list[str]]:
  reasons: list[str] = []
  active_dims = int(latent_stats["active_dim_count"])
  prior_global = float(reconstruction_metrics["global"]["prior"]["mse"])
  posterior_global = float(reconstruction_metrics["global"]["posterior"]["mse"])
  by_split = reconstruction_metrics["by_split"]["prior"]
  train = by_split.get("train", {})
  heldout = by_split.get("heldout", {})
  train_mse = float(train.get("mse", math.nan))
  heldout_mse = float(heldout.get("mse", math.nan))
  finite_values = [
    float(active_dims),
    prior_global,
    posterior_global,
    latent_stats["posterior_prior_kl"]["mean"],
  ]
  if math.isfinite(train_mse):
    finite_values.append(train_mse)
  if math.isfinite(heldout_mse):
    finite_values.append(heldout_mse)
  finite = np.isfinite(finite_values).all()

  reasons.append(f"active latent dims: {active_dims}/{latent_stats['latent_dim']}")
  reasons.append(f"global prior action MSE: {prior_global:.6g}")
  reasons.append(f"global posterior action MSE: {posterior_global:.6g}")
  if math.isfinite(train_mse):
    reasons.append(f"train prior action MSE: {train_mse:.6g}")
  if math.isfinite(heldout_mse):
    reasons.append(f"held-out test_001 prior action MSE: {heldout_mse:.6g}")

  if not finite or active_dims < 3 or prior_global > 0.08:
    return "not_ready", reasons
  if math.isfinite(heldout_mse) and math.isfinite(train_mse):
    heldout_limit = max(train_mse * 1.8, train_mse + 0.015, 0.035)
    if active_dims >= 6 and prior_global <= 0.035 and heldout_mse <= heldout_limit:
      return "usable", reasons
  if not math.isfinite(heldout_mse) and active_dims >= 6 and prior_global <= 0.035:
    reasons.append("no held-out split was detected; keeping verdict cautious")
    return "usable_with_caution", reasons
  if active_dims >= 4 and prior_global <= 0.06:
    return "usable_with_caution", reasons
  return "not_ready", reasons


def write_report(
  output_dir: Path,
  *,
  verdict: str,
  reasons: list[str],
  notes: list[str],
  collection_meta: dict[str, Any],
  metrics_summary: dict[str, Any],
  latent_stats: dict[str, Any],
  reconstruction_metrics: dict[str, Any],
  per_motion_rows: list[dict[str, Any]],
) -> None:
  lines = [
    f"# {collection_meta['analysis_label']} Latent Analysis Report",
    "",
    f"Verdict: **{verdict}**",
    "",
    "## Basis",
  ]
  lines.extend(f"- {reason}" for reason in reasons)
  lines.extend(
    [
      "",
      "## Scope",
      (
        "- This is an offline, teacher-forced reference-state probe. It checks "
        "checkpoint loading, latent usage, prior/posterior alignment, and "
        "teacher-action reconstruction on motion references."
      ),
      (
        "- It does not modify training code, the checkpoint, or decoder defaults."
      ),
      (
        "- Closed-loop downstream-task validation is still required before "
        "using the decoder for high-level RL."
      ),
      "",
      "## Key Numbers",
      f"- Samples: {collection_meta['collected_samples']}",
      f"- Task id: `{collection_meta['task_id']}`",
      (
        f"- Latent/state/target/action dims: {collection_meta['latent_dim']} / "
        f"{collection_meta['state_dim']} / {collection_meta['target_dim']} / "
        f"{collection_meta['action_dim']}"
      ),
      (f"- Prior/posterior KL mean: {latent_stats['posterior_prior_kl']['mean']:.6g}"),
      (
        f"- Prior/posterior mean L2: "
        f"{latent_stats['prior_posterior_mean_l2']['mean']:.6g}"
      ),
      (
        f"- Global posterior/prior action MSE: "
        f"{reconstruction_metrics['global']['posterior']['mse']:.6g} / "
        f"{reconstruction_metrics['global']['prior']['mse']:.6g}"
      ),
      "",
      "## Per Motion Prior MSE",
      "| Motion | Split | Samples | Prior MSE | Posterior MSE | KL mean |",
      "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
  )
  for row in per_motion_rows:
    lines.append(
      f"| {row['motion']} | {row['split']} | {row['samples']} | "
      f"{row['prior_mse']:.6g} | {row['posterior_mse']:.6g} | "
      f"{row['kl_mean']:.6g} |"
    )
  if notes:
    lines.extend(["", "## Notes"])
    lines.extend(f"- {note}" for note in notes)

  if metrics_summary.get("available"):
    loss = metrics_summary["scalars"].get("loss", {})
    prior_loss = metrics_summary["scalars"].get("prior_action_loss", {})
    if loss.get("available"):
      lines.extend(
        [
          "",
          "## Training Event Snapshot",
          f"- Final loss: {loss['final']:.6g} at step {loss['final_step']}",
        ]
      )
    if prior_loss.get("available"):
      lines.append(
        f"- Final prior action loss: {prior_loss['final']:.6g}; "
        f"last100 mean: {prior_loss['last100_mean']:.6g}"
      )

  report_md = output_dir / "report.md"
  report_md.write_text("\n".join(lines) + "\n")

  image_names = [
    "latent_pca_by_motion.png",
    "latent_pca_by_family.png",
    "latent_pca_by_phase.png",
    "latent_dim_activity.png",
    "prior_posterior_alignment.png",
    "per_motion_reconstruction_error.png",
    "per_joint_reconstruction_error.png",
  ]
  html_lines = [
    "<!doctype html>",
    "<html><head><meta charset='utf-8'>",
    f"<title>{html.escape(collection_meta['analysis_label'])} Latent Analysis</title>",
    "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;line-height:1.5}"
    "img{max-width:100%;border:1px solid #ddd;margin:12px 0}code{background:#f2f4f7;padding:2px 4px}</style>",
    "</head><body>",
    f"<h1>{html.escape(collection_meta['analysis_label'])} Latent Analysis</h1>",
    f"<p><strong>Verdict:</strong> {html.escape(verdict)}</p>",
    "<h2>Basis</h2><ul>",
  ]
  html_lines.extend(f"<li>{html.escape(reason)}</li>" for reason in reasons)
  html_lines.extend(["</ul>", "<h2>Figures</h2>"])
  for image_name in image_names:
    if (output_dir / image_name).exists():
      html_lines.append(f"<h3>{html.escape(image_name)}</h3>")
      html_lines.append(f"<img src='{html.escape(image_name)}'>")
  html_lines.append("</body></html>")
  (output_dir / "report.html").write_text("\n".join(html_lines) + "\n")


def main() -> None:
  args = parse_args()
  checkpoint = args.checkpoint.expanduser()
  if not checkpoint.is_absolute():
    checkpoint = (REPO_ROOT / checkpoint).resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  output_dir = args.output_dir.expanduser()
  if not output_dir.is_absolute():
    output_dir = (REPO_ROOT / output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  event_dir = args.event_dir
  if event_dir is None:
    event_dir = checkpoint.parent
  elif not event_dir.is_absolute():
    event_dir = (REPO_ROOT / event_dir).resolve()

  motion_files = discover_motion_files(args.motion_root, args.motion_pattern)
  metrics_summary, metric_notes = summarize_tensorboard(event_dir)
  write_json(output_dir / "metrics_summary.json", metrics_summary)
  write_metrics_markdown(output_dir / "metrics_summary.md", metrics_summary)

  arrays, collection_meta, collection_notes = collect_samples(
    checkpoint,
    motion_files,
    args,
  )
  latent_stats = compute_latent_stats(arrays)
  reconstruction_metrics, per_motion_rows = compute_reconstruction_metrics(
    arrays,
    collection_meta["joint_names"],
  )

  write_json(output_dir / "latent_stats.json", latent_stats)
  write_json(output_dir / "reconstruction_metrics.json", reconstruction_metrics)
  write_json(output_dir / "collection_meta.json", collection_meta)
  write_per_motion_csv(output_dir / "per_motion_metrics.csv", per_motion_rows)

  plot_notes = generate_plots(
    arrays,
    latent_stats,
    reconstruction_metrics,
    per_motion_rows,
    output_dir,
    seed=args.seed,
    enable_tsne=args.tsne,
    enable_umap=args.umap,
  )
  verdict, reasons = judge_verdict(latent_stats, reconstruction_metrics)
  notes = metric_notes + collection_notes + plot_notes
  write_report(
    output_dir,
    verdict=verdict,
    reasons=reasons,
    notes=notes,
    collection_meta=collection_meta,
    metrics_summary=metrics_summary,
    latent_stats=latent_stats,
    reconstruction_metrics=reconstruction_metrics,
    per_motion_rows=per_motion_rows,
  )

  print(f"[latent-analysis] wrote outputs to {output_dir}")
  print(f"[latent-analysis] verdict: {verdict}")
  for reason in reasons:
    print(f"[latent-analysis] {reason}")


if __name__ == "__main__":
  main()
