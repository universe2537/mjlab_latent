#!/usr/bin/env python3
"""Validate the default tennis distillation decoder on four tennis motions."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.distillation.rl import OnlineDistillationRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

DEFAULT_TASK_ID = "Mjlab-Distill-Flat-Unitree-G1"
DEFAULT_CHECKPOINT = Path(
  "/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_distillation/"
  "distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"
)
DEFAULT_TEACHER_CHECKPOINT = Path(
  "/data0/universe/home_moved/mjlab_latent/logs/rsl_rl/g1_tracking/tennis/"
  "model_29999.pt"
)
DEFAULT_MOTION_FILES = (
  REPO_ROOT / "artifacts/tennis_random_001/motion.npz",
  REPO_ROOT / "artifacts/tennis_random_002/motion.npz",
  REPO_ROOT / "artifacts/tennis_random_003/motion.npz",
  REPO_ROOT / "artifacts/tennis_random_004/motion.npz",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "latent_analysis/outputs_tennis_verify_4x4096"
POLICY_KINDS = ("teacher", "posterior", "prior")


class DistillationEvalPolicy:
  """Callable policy for teacher, posterior student, or prior student."""

  def __init__(
    self,
    runner: OnlineDistillationRunner,
    *,
    policy_kind: Literal["teacher", "posterior", "prior"],
  ) -> None:
    self.runner = runner
    self.policy_kind = policy_kind
    self._teacher_policy = None

  def __call__(self, obs: Any) -> torch.Tensor:
    if self.policy_kind == "teacher":
      if self._teacher_policy is None:
        self._teacher_policy = self.runner._load_teacher_policy()  # noqa: SLF001
      return self._teacher_policy(obs).to(self.runner.device)

    actor_obs = obs[self.runner.cfg["obs_group"]].to(self.runner.device)
    state, target = self.runner.slicer.split(actor_obs)
    if self.policy_kind == "posterior":
      dist = self.runner.model.posterior_distribution(state, target)
    else:
      dist = self.runner.model.prior_distribution(state)
    return self.runner.model.decode(state, dist.mean)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Validate the tennis distillation decoder on tennis_random_001-004."
  )
  parser.add_argument("--task-id", type=str, default=DEFAULT_TASK_ID)
  parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
  parser.add_argument(
    "--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT
  )
  parser.add_argument(
    "--motion-files",
    type=Path,
    nargs="*",
    default=list(DEFAULT_MOTION_FILES),
  )
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--device", type=str, default="cpu")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--samples-per-motion", type=int, default=4096)
  parser.add_argument("--rollout-num-envs", type=int, default=128)
  parser.add_argument("--max-rollout-steps", type=int, default=None)
  parser.add_argument("--skip-offline", action="store_true")
  parser.add_argument("--skip-rollout", action="store_true")
  parser.add_argument("--run-videos", action="store_true")
  parser.add_argument(
    "--video-only",
    action="store_true",
    help="Record videos and attach them to an existing validation_results.json.",
  )
  parser.add_argument("--video-length", type=int, default=600)
  parser.add_argument("--video-height", type=int, default=720)
  parser.add_argument("--video-width", type=int, default=1280)
  return parser.parse_args()


def path_json(path: Path) -> str:
  return str(path.expanduser().resolve())


def read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_paths(args: argparse.Namespace) -> None:
  for label, path in (
    ("checkpoint", args.checkpoint),
    ("teacher checkpoint", args.teacher_checkpoint),
  ):
    if not path.expanduser().exists():
      raise FileNotFoundError(f"Missing {label}: {path}")
  missing = [str(path) for path in args.motion_files if not path.expanduser().exists()]
  if missing:
    raise FileNotFoundError(f"Missing motion files: {missing}")


def run_offline_probe(args: argparse.Namespace) -> None:
  command = [
    sys.executable,
    str(REPO_ROOT / "latent_analysis/analyze_latent_space.py"),
    "--analysis-label",
    "tennis-verify-4x4096",
    "--task-id",
    args.task_id,
    "--checkpoint",
    path_json(args.checkpoint),
    "--teacher-checkpoint",
    path_json(args.teacher_checkpoint),
    "--motion-root",
    str(REPO_ROOT / "artifacts"),
    "--motion-pattern",
    "tennis_random_00[1-4]/motion.npz",
    "--output-dir",
    path_json(args.output_dir),
    "--samples-per-motion",
    str(args.samples_per_motion),
    "--device",
    args.device,
    "--seed",
    str(args.seed),
  ]
  print("[verify] running offline latent probe")
  subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_checkpoint_cfg(
  checkpoint_path: Path,
  *,
  teacher_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  if not isinstance(checkpoint, dict):
    raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
  cfg = dict(checkpoint["cfg"])
  cfg["teacher_checkpoint"] = path_json(teacher_checkpoint)
  return cfg, checkpoint


def restore_obs_slicer(
  runner: OnlineDistillationRunner,
  checkpoint: dict[str, Any],
) -> None:
  saved_slicer = checkpoint.get("obs_slicer")
  if not isinstance(saved_slicer, dict):
    return
  state_indices = saved_slicer.get("state_indices")
  target_indices = saved_slicer.get("target_indices")
  if isinstance(state_indices, torch.Tensor):
    runner.slicer.state_indices = state_indices.to(runner.device)
  if isinstance(target_indices, torch.Tensor):
    runner.slicer.target_indices = target_indices.to(runner.device)


def configure_eval_env(
  args: argparse.Namespace,
  motion_file: Path,
  *,
  num_envs: int,
  render_mode: str | None,
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, dict[str, Any], MotionCommand]:
  env_cfg = load_env_cfg(args.task_id, play=False)
  env_cfg.seed = int(args.seed)
  env_cfg.scene.num_envs = int(num_envs)
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events = {}
  env_cfg.episode_length_s = 1.0e9
  env_cfg.viewer.height = int(args.video_height)
  env_cfg.viewer.width = int(args.video_width)

  motion_cmd = env_cfg.commands["motion"]
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {args.task_id!r} does not expose a motion command.")
  motion_cmd.motion_source = "local"
  motion_cmd.motion_files = (path_json(motion_file),)
  motion_cmd.motion_sample_probs = ()
  motion_cmd.sampling_mode = "start"
  motion_cmd.pose_range = {}
  motion_cmd.velocity_range = {}
  motion_cmd.joint_position_range = (0.0, 0.0)
  motion_cmd.debug_vis = False

  agent_cfg = load_rl_cfg(args.task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=render_mode)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  command = cast(MotionCommand, vec_env.unwrapped.command_manager.get_term("motion"))
  runner_cfg = asdict(agent_cfg)
  return env, vec_env, runner_cfg, command


def load_runner_for_env(
  vec_env: RslRlVecEnvWrapper,
  args: argparse.Namespace,
) -> OnlineDistillationRunner:
  runner_cls = load_runner_cls(args.task_id)
  if runner_cls is not OnlineDistillationRunner:
    raise ValueError(f"Task {args.task_id!r} is not a distillation task.")
  cfg, checkpoint = load_checkpoint_cfg(
    args.checkpoint.expanduser(),
    teacher_checkpoint=args.teacher_checkpoint.expanduser(),
  )
  runner = runner_cls(vec_env, cfg, log_dir=None, device=args.device)
  runner.load(str(args.checkpoint.expanduser()), strict=False, map_location=args.device)
  restore_obs_slicer(runner, checkpoint)
  runner.model.eval()
  return runner


def tensor_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
  if not mask.any():
    return math.nan
  return float(values[mask].detach().float().mean().cpu().item())


def summarize_vector(values: list[float]) -> dict[str, float]:
  finite = np.array([value for value in values if math.isfinite(value)], dtype=float)
  if finite.size == 0:
    return {"mean": math.nan, "max": math.nan}
  return {"mean": float(finite.mean()), "max": float(finite.max())}


def compute_rollout_metrics(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
  metrics = {
    "mpkpe": compute_mpkpe(command),
    "r_mpkpe": compute_root_relative_mpkpe(command),
    "joint_vel_error_eval": compute_joint_velocity_error(command),
    "ee_pos_error": compute_ee_position_error(command, ee_body_names),
    "ee_ori_error": compute_ee_orientation_error(command, ee_body_names),
  }
  for key in (
    "error_anchor_pos",
    "error_anchor_rot",
    "error_body_pos",
    "error_body_rot",
    "error_joint_pos",
    "error_joint_vel",
  ):
    metrics[key] = command.metrics[key]
  return metrics


def evaluate_policy_rollout(
  args: argparse.Namespace,
  motion_file: Path,
  policy_kind: Literal["teacher", "posterior", "prior"],
  *,
  num_envs: int,
  record_video: bool,
) -> dict[str, Any]:
  motion_name = motion_file.parent.name
  video_folder = args.output_dir / "videos" / motion_name / policy_kind
  render_mode = "rgb_array" if record_video else None
  env, vec_env, _runner_cfg, command = configure_eval_env(
    args,
    motion_file,
    num_envs=num_envs,
    render_mode=render_mode,
  )
  if record_video:
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=int(args.video_length),
      name_prefix=f"{motion_name}-{policy_kind}",
      disable_logger=False,
    )
    vec_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task_id).clip_actions)
    command = cast(MotionCommand, vec_env.unwrapped.command_manager.get_term("motion"))

  closed = False
  try:
    runner = load_runner_for_env(vec_env, args)
    policy = DistillationEvalPolicy(runner, policy_kind=policy_kind)
    ee_body_names = tuple(
      vec_env.unwrapped.cfg.terminations["ee_body_pos"].params["body_names"]
    )
    motion_length = int(command.motion.motion_lengths[0].item())
    eval_steps = max(motion_length - 1, 1)
    if args.max_rollout_steps is not None:
      eval_steps = min(eval_steps, int(args.max_rollout_steps))
    if record_video:
      eval_steps = min(eval_steps, int(args.video_length))

    done_envs = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=args.device)
    active_steps = torch.zeros(num_envs, dtype=torch.float32, device=args.device)
    metric_sums: dict[str, torch.Tensor] = {}
    termination_counts = {
      name: 0 for name in vec_env.unwrapped.termination_manager.active_terms
    }

    obs = vec_env.get_observations()
    for _step in range(eval_steps):
      active_before = ~done_envs
      with torch.no_grad():
        actions = policy(obs)
      obs, _, dones, _ = vec_env.step(actions)
      dones_bool = dones.bool()
      newly_done = active_before & dones_bool
      terminated = vec_env.unwrapped.termination_manager.terminated
      truncated = vec_env.unwrapped.termination_manager.time_outs

      if newly_done.any():
        success |= newly_done & truncated & ~terminated
        for term_name in termination_counts:
          term_value = vec_env.unwrapped.termination_manager.get_term(term_name)
          termination_counts[term_name] += int(
            torch.count_nonzero(term_value & newly_done).item()
          )
        done_envs |= newly_done

      metric_mask = active_before & ~newly_done
      if metric_mask.any():
        metrics = compute_rollout_metrics(command, ee_body_names)
        if not metric_sums:
          metric_sums = {
            key: torch.zeros(num_envs, dtype=torch.float32, device=args.device)
            for key in metrics
          }
        for key, value in metrics.items():
          metric_sums[key][metric_mask] += value[metric_mask].detach().float()
        active_steps[metric_mask] += 1.0

      if bool(done_envs.all()):
        break

    survived = ~done_envs
    success |= survived
    metric_means = {}
    valid_metric_envs = active_steps > 0
    for key, total in metric_sums.items():
      per_env = total / torch.clamp(active_steps, min=1.0)
      metric_means[key] = tensor_mean(per_env, valid_metric_envs)

    if record_video:
      vec_env.close()
      closed = True

    video_paths = []
    if record_video:
      video_paths = sorted(str(path) for path in video_folder.glob("*.mp4"))

    result = {
      "motion_name": motion_name,
      "motion_file": path_json(motion_file),
      "policy_kind": policy_kind,
      "num_envs": int(num_envs),
      "motion_length": int(motion_length),
      "eval_steps": int(eval_steps),
      "success_rate": float(success.float().mean().cpu().item()),
      "failure_rate": float((~success).float().mean().cpu().item()),
      "terminated_envs": int(torch.count_nonzero(done_envs & ~success).item()),
      "survived_envs": int(torch.count_nonzero(survived).item()),
      "termination_counts": termination_counts,
      "metrics": metric_means,
      "video_paths": video_paths,
    }
    print(
      "[verify] rollout "
      f"{motion_name}/{policy_kind}: success={result['success_rate']:.3f}"
    )
    return result
  finally:
    if not closed:
      vec_env.close()


def run_rollouts(args: argparse.Namespace) -> list[dict[str, Any]]:
  results = []
  for motion_file in args.motion_files:
    for policy_kind in POLICY_KINDS:
      results.append(
        evaluate_policy_rollout(
          args,
          motion_file.expanduser(),
          cast(Literal["teacher", "posterior", "prior"], policy_kind),
          num_envs=int(args.rollout_num_envs),
          record_video=False,
        )
      )
      if args.run_videos:
        try:
          video_result = evaluate_policy_rollout(
            args,
            motion_file.expanduser(),
            cast(Literal["teacher", "posterior", "prior"], policy_kind),
            num_envs=1,
            record_video=True,
          )
          results[-1]["video_paths"] = video_result["video_paths"]
        except Exception as exc:
          results[-1]["video_error"] = str(exc)
          print(f"[verify] video failed for {motion_file.parent.name}/{policy_kind}: {exc}")
    if args.run_videos:
      make_side_by_side_video(args.output_dir, motion_file.parent.name)
  return results


def run_video_rollouts(
  args: argparse.Namespace,
) -> dict[tuple[str, str], dict[str, Any]]:
  records = {}
  for motion_file in args.motion_files:
    motion_name = motion_file.parent.name
    for policy_kind in POLICY_KINDS:
      try:
        video_result = evaluate_policy_rollout(
          args,
          motion_file.expanduser(),
          cast(Literal["teacher", "posterior", "prior"], policy_kind),
          num_envs=1,
          record_video=True,
        )
        records[(motion_name, policy_kind)] = {
          "video_paths": video_result["video_paths"],
        }
      except Exception as exc:
        records[(motion_name, policy_kind)] = {"video_error": str(exc)}
        print(f"[verify] video failed for {motion_name}/{policy_kind}: {exc}")
    make_side_by_side_video(args.output_dir, motion_name)
  return records


def attach_video_records(
  rollouts: list[dict[str, Any]],
  records: dict[tuple[str, str], dict[str, Any]],
) -> None:
  for row in rollouts:
    record = records.get((row["motion_name"], row["policy_kind"]))
    if not record:
      continue
    if "video_paths" in record:
      row["video_paths"] = record["video_paths"]
    if "video_error" in record:
      row["video_error"] = record["video_error"]


def make_side_by_side_video(output_dir: Path, motion_name: str) -> None:
  try:
    import mediapy as media
  except Exception as exc:
    print(f"[verify] side-by-side skipped for {motion_name}: {exc}")
    return

  paths = []
  for policy_kind in POLICY_KINDS:
    folder = output_dir / "videos" / motion_name / policy_kind
    candidates = sorted(folder.glob("*.mp4"))
    if not candidates:
      return
    paths.append(candidates[0])
  try:
    videos = [media.read_video(str(path)) for path in paths]
    frame_count = min(video.shape[0] for video in videos)
    if frame_count == 0:
      return
    videos = [video[:frame_count] for video in videos]
    combined = np.concatenate(videos, axis=2)
    out_path = output_dir / "videos" / motion_name / f"{motion_name}-compare.mp4"
    media.write_video(str(out_path), combined, fps=30)
  except Exception as exc:
    print(f"[verify] side-by-side failed for {motion_name}: {exc}")


def offline_summary(output_dir: Path) -> dict[str, Any]:
  report_path = output_dir / "report.md"
  if not (output_dir / "reconstruction_metrics.json").exists():
    return {"available": False, "reason": "offline outputs not found"}
  reconstruction = read_json(output_dir / "reconstruction_metrics.json")
  latent_stats = read_json(output_dir / "latent_stats.json")
  per_motion = []
  csv_path = output_dir / "per_motion_metrics.csv"
  if csv_path.exists():
    import csv

    with csv_path.open() as f:
      per_motion = list(csv.DictReader(f))
  per_joint = sorted(
    reconstruction.get("per_joint", []),
    key=lambda row: float(row.get("prior_mse", 0.0)),
    reverse=True,
  )[:12]
  return {
    "available": True,
    "report_path": str(report_path),
    "global": reconstruction["global"],
    "per_motion": per_motion,
    "per_joint_top": per_joint,
    "active_dim_count": latent_stats["active_dim_count"],
    "active_dims": latent_stats["active_dims"],
    "posterior_prior_kl": latent_stats["posterior_prior_kl"],
    "prior_posterior_mean_l2": latent_stats["prior_posterior_mean_l2"],
  }


def judge_offline(offline: dict[str, Any]) -> tuple[str, list[str]]:
  if not offline.get("available"):
    return "not_ready", [str(offline.get("reason", "offline unavailable"))]
  global_metrics = offline["global"]
  posterior_mse = float(global_metrics["posterior"]["mse"])
  prior_mse = float(global_metrics["prior"]["mse"])
  reasons = [
    f"offline posterior MSE={posterior_mse:.6g}",
    f"offline prior MSE={prior_mse:.6g}",
    f"active latent dims={offline['active_dim_count']}/16",
  ]
  verdict = "usable"
  if posterior_mse > 0.08 or prior_mse > 0.08:
    verdict = "not_ready"
  elif posterior_mse > 0.06 or prior_mse > 0.06:
    verdict = "usable_with_caution"

  if offline["per_motion"]:
    global_prior = max(prior_mse, 1.0e-12)
    bad_motions = [
      row["motion"]
      for row in offline["per_motion"]
      if float(row["prior_mse"]) > global_prior * 1.35
    ]
    if bad_motions:
      verdict = "not_ready"
      reasons.append(f"per-motion prior MSE outliers={bad_motions}")
  return verdict, reasons


def group_rollouts(
  rollouts: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
  return {
    (row["motion_name"], row["policy_kind"]): row
    for row in rollouts
  }


def judge_rollouts(rollouts: list[dict[str, Any]]) -> tuple[str, list[str]]:
  if not rollouts:
    return "not_ready", ["closed-loop rollouts were not run"]
  grouped = group_rollouts(rollouts)
  verdict = "usable"
  reasons: list[str] = []
  motion_names = sorted({row["motion_name"] for row in rollouts})
  for motion_name in motion_names:
    teacher = grouped.get((motion_name, "teacher"))
    posterior = grouped.get((motion_name, "posterior"))
    prior = grouped.get((motion_name, "prior"))
    if teacher is None or posterior is None or prior is None:
      verdict = "not_ready"
      reasons.append(f"{motion_name}: missing policy rollout")
      continue

    teacher_success = float(teacher["success_rate"])
    posterior_success = float(posterior["success_rate"])
    prior_success = float(prior["success_rate"])
    reasons.append(
      f"{motion_name}: success teacher/posterior/prior="
      f"{teacher_success:.3f}/{posterior_success:.3f}/{prior_success:.3f}"
    )
    if teacher_success < 0.85:
      verdict = "not_ready"
      reasons.append(f"{motion_name}: teacher baseline success below 0.85")
    if posterior_success < max(0.85, teacher_success * 0.9):
      verdict = "not_ready"
      reasons.append(f"{motion_name}: posterior success below threshold")
    if prior_success < max(0.75, teacher_success * 0.8):
      verdict = "not_ready"
      reasons.append(f"{motion_name}: prior success below threshold")

    for metric_name, posterior_limit, prior_limit in (
      ("error_body_pos", 1.5, 2.0),
      ("ee_pos_error", 1.5, 2.0),
      ("ee_ori_error", 1.5, 2.0),
    ):
      teacher_metric = float(teacher["metrics"].get(metric_name, math.nan))
      posterior_metric = float(posterior["metrics"].get(metric_name, math.nan))
      prior_metric = float(prior["metrics"].get(metric_name, math.nan))
      if not math.isfinite(teacher_metric) or teacher_metric <= 1.0e-8:
        continue
      if posterior_metric > teacher_metric * posterior_limit:
        verdict = "not_ready"
        reasons.append(f"{motion_name}: posterior {metric_name} too high")
      if prior_metric > teacher_metric * prior_limit:
        verdict = "not_ready"
        reasons.append(f"{motion_name}: prior {metric_name} too high")
  return verdict, reasons


def combine_verdicts(
  offline_verdict: str,
  rollout_verdict: str,
) -> str:
  if "not_ready" in (offline_verdict, rollout_verdict):
    return "not_ready"
  if "usable_with_caution" in (offline_verdict, rollout_verdict):
    return "usable_with_caution"
  return "usable"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
  ]
  lines.extend("| " + " | ".join(row) + " |" for row in rows)
  return lines


def write_report(
  output_dir: Path,
  *,
  args: argparse.Namespace,
  offline: dict[str, Any],
  offline_verdict: str,
  offline_reasons: list[str],
  rollouts: list[dict[str, Any]],
  rollout_verdict: str,
  rollout_reasons: list[str],
  final_verdict: str,
) -> None:
  lines = [
    "# Tennis Distillation Validation Report",
    "",
    f"Final verdict: **{final_verdict}**",
    "",
    "## Inputs",
    f"- Student checkpoint: `{path_json(args.checkpoint)}`",
    f"- Teacher checkpoint: `{path_json(args.teacher_checkpoint)}`",
    f"- Output dir: `{path_json(output_dir)}`",
    f"- Device: `{args.device}`",
    "",
    "## Offline Reconstruction",
    f"- Verdict: **{offline_verdict}**",
  ]
  lines.extend(f"- {reason}" for reason in offline_reasons)
  if offline.get("available"):
    global_metrics = offline["global"]
    lines.extend(
      [
        "",
        *markdown_table(
          ["path", "mse", "mae"],
          [
            [
              "posterior",
              f"{global_metrics['posterior']['mse']:.6g}",
              f"{global_metrics['posterior']['mae']:.6g}",
            ],
            [
              "prior",
              f"{global_metrics['prior']['mse']:.6g}",
              f"{global_metrics['prior']['mae']:.6g}",
            ],
          ],
        ),
        "",
        "### Per Motion Offline",
      ]
    )
    lines.extend(
      markdown_table(
        ["motion", "posterior_mse", "prior_mse", "kl_mean"],
        [
          [
            row["motion"],
            f"{float(row['posterior_mse']):.6g}",
            f"{float(row['prior_mse']):.6g}",
            f"{float(row['kl_mean']):.6g}",
          ]
          for row in offline["per_motion"]
        ],
      )
    )
    lines.extend(["", "### Top Prior Joint Errors"])
    lines.extend(
      markdown_table(
        ["joint", "prior_mse", "posterior_mse"],
        [
          [
            row["joint"],
            f"{float(row['prior_mse']):.6g}",
            f"{float(row['posterior_mse']):.6g}",
          ]
          for row in offline["per_joint_top"][:10]
        ],
      )
    )

  lines.extend(
    [
      "",
      "## Closed-Loop Rollouts",
      f"- Verdict: **{rollout_verdict}**",
    ]
  )
  lines.extend(f"- {reason}" for reason in rollout_reasons)
  if rollouts:
    lines.extend(["", "### Rollout Metrics"])
    lines.extend(
      markdown_table(
        [
          "motion",
          "policy",
          "success",
          "body_pos",
          "ee_pos",
          "ee_ori",
          "terminations",
        ],
        [
          [
            row["motion_name"],
            row["policy_kind"],
            f"{float(row['success_rate']):.3f}",
            f"{float(row['metrics'].get('error_body_pos', math.nan)):.5g}",
            f"{float(row['metrics'].get('ee_pos_error', math.nan)):.5g}",
            f"{float(row['metrics'].get('ee_ori_error', math.nan)):.5g}",
            ", ".join(
              f"{key}:{value}"
              for key, value in row["termination_counts"].items()
              if int(value) > 0
            )
            or "-",
          ]
          for row in rollouts
        ],
      )
    )
    video_rows = [
      [row["motion_name"], row["policy_kind"], "<br>".join(row.get("video_paths", []))]
      for row in rollouts
      if row.get("video_paths")
    ]
    if video_rows:
      lines.extend(["", "### Videos"])
      lines.extend(markdown_table(["motion", "policy", "paths"], video_rows))

  lines.extend(
    [
      "",
      "## Tennis Downstream Check",
      "- Not executed by this script. Use the closed-loop tracking verdict as the "
      "decoder-level gate before spending GPU/EGL time on Tennis Hit/Cross play.",
      "- If this report is not `usable`, downstream Tennis play is expected to be "
      "diagnostic rather than a pass/fail promotion run.",
    ]
  )
  (output_dir / "VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
  args = parse_args()
  args.checkpoint = args.checkpoint.expanduser()
  args.teacher_checkpoint = args.teacher_checkpoint.expanduser()
  args.motion_files = [path.expanduser() for path in args.motion_files]
  args.output_dir = args.output_dir.expanduser()
  if not args.output_dir.is_absolute():
    args.output_dir = (REPO_ROOT / args.output_dir).resolve()
  args.output_dir.mkdir(parents=True, exist_ok=True)
  validate_paths(args)

  os.environ.setdefault("MPLBACKEND", "Agg")
  torch.manual_seed(int(args.seed))
  np.random.seed(int(args.seed))
  configure_torch_backends()

  existing_payload: dict[str, Any] | None = None
  if args.video_only:
    payload_path = args.output_dir / "validation_results.json"
    if not payload_path.exists():
      raise FileNotFoundError(f"Missing existing validation payload: {payload_path}")
    existing_payload = read_json(payload_path)

  if not args.skip_offline and not args.video_only:
    run_offline_probe(args)
  if existing_payload is not None:
    offline = existing_payload.get("offline", offline_summary(args.output_dir))
  else:
    offline = offline_summary(args.output_dir)
  offline_verdict, offline_reasons = judge_offline(offline)

  rollouts: list[dict[str, Any]] = []
  if existing_payload is not None:
    rollouts = list(existing_payload.get("rollouts", []))
    attach_video_records(rollouts, run_video_rollouts(args))
  elif not args.skip_rollout:
    rollouts = run_rollouts(args)
  rollout_verdict, rollout_reasons = judge_rollouts(rollouts)
  final_verdict = combine_verdicts(offline_verdict, rollout_verdict)

  payload = {
    "schema_version": 1,
    "config": {
      "task_id": args.task_id,
      "checkpoint": path_json(args.checkpoint),
      "teacher_checkpoint": path_json(args.teacher_checkpoint),
      "motion_files": [path_json(path) for path in args.motion_files],
      "output_dir": path_json(args.output_dir),
      "device": args.device,
      "samples_per_motion": int(args.samples_per_motion),
      "rollout_num_envs": int(args.rollout_num_envs),
    },
    "offline": offline,
    "offline_verdict": offline_verdict,
    "offline_reasons": offline_reasons,
    "rollouts": rollouts,
    "rollout_verdict": rollout_verdict,
    "rollout_reasons": rollout_reasons,
    "final_verdict": final_verdict,
  }
  if existing_payload is not None:
    payload["config"] = existing_payload.get("config", payload["config"])
    payload["video_config"] = {
      "device": args.device,
      "video_length": int(args.video_length),
      "video_height": int(args.video_height),
      "video_width": int(args.video_width),
    }
  write_json(args.output_dir / "validation_results.json", payload)
  write_report(
    args.output_dir,
    args=args,
    offline=offline,
    offline_verdict=offline_verdict,
    offline_reasons=offline_reasons,
    rollouts=rollouts,
    rollout_verdict=rollout_verdict,
    rollout_reasons=rollout_reasons,
    final_verdict=final_verdict,
  )
  print(f"[verify] wrote {args.output_dir / 'VALIDATION_REPORT.md'}")
  print(f"[verify] final verdict: {final_verdict}")


if __name__ == "__main__":
  main()
