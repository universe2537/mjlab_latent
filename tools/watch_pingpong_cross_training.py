#!/usr/bin/env python3
"""Watch a long Pingpong Cross training run and write evidence reports."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
  _event_accumulator_module = importlib.import_module(
    "tensorboard.backend.event_processing.event_accumulator"
  )
  _EventAccumulator: Any | None = _event_accumulator_module.EventAccumulator
except ImportError:  # pragma: no cover - exercised only on minimal installs.
  _EventAccumulator = None

DECISIONS = {
  "continue",
  "risky_continue",
  "stop",
  "stop_and_reconfigure",
  "insufficient_evidence",
}

SCALAR_SPECS: dict[str, tuple[str, ...]] = {
  "self_table_bounce_count": ("Episode_Metrics/self_table_bounce_count",),
  "paddle_hit_count": ("Episode_Metrics/paddle_hit_count",),
  "crossed_net_count": ("Episode_Metrics/crossed_net_count",),
  "opponent_table_bounce_count": ("Episode_Metrics/opponent_table_bounce_count",),
  "legal_return_count": ("Episode_Metrics/legal_return_count",),
  "fault_count": ("Episode_Metrics/fault_count",),
  "ball_fault": ("Episode_Termination/ball_fault", "Episode_Terminations/ball_fault"),
  "post_hit_x_progress": ("Episode_Reward/post_hit_x_progress",),
  "post_hit_ball_velocity_direction": (
    "Episode_Reward/post_hit_ball_velocity_direction",
  ),
  "robot_ball_contact_count": ("Episode_Metrics/robot_ball_contact_count",),
  "hit/post_vx": ("Episode_Metrics/hit/post_vx",),
  "hit/post_vy": ("Episode_Metrics/hit/post_vy",),
  "hit/post_vz": ("Episode_Metrics/hit/post_vz",),
  "hit/post_speed": ("Episode_Metrics/hit/post_speed",),
  "hit/post_vx_toward_opponent_ratio": (
    "Episode_Metrics/hit/post_vx_toward_opponent_ratio",
  ),
  "hit/pred_net_clearance": ("Episode_Metrics/hit/pred_net_clearance",),
  "hit/pred_net_clearance_positive": (
    "Episode_Metrics/hit/pred_net_clearance_positive",
  ),
  "hit/pred_landing_inside_opponent_table": (
    "Episode_Metrics/hit/pred_landing_inside_opponent_table",
  ),
  "hit/paddle_speed": ("Episode_Metrics/hit/paddle_speed",),
  "hit/paddle_normal_alignment": ("Episode_Metrics/hit/paddle_normal_alignment",),
  "hit/paddle_velocity_along_normal": (
    "Episode_Metrics/hit/paddle_velocity_along_normal",
  ),
  "fault_reason/body_ball": ("Episode_Metrics/fault_reason/body_ball",),
  "fault_reason/low_net": ("Episode_Metrics/fault_reason/low_net",),
  "fault_reason/net_contact": ("Episode_Metrics/fault_reason/net_contact",),
  "fault_reason/return_out": ("Episode_Metrics/fault_reason/return_out",),
  "fault_reason/failed_bounce": ("Episode_Metrics/fault_reason/failed_bounce",),
  "fault_reason/double_paddle": ("Episode_Metrics/fault_reason/double_paddle",),
  "fault_reason/early_hit": ("Episode_Metrics/fault_reason/early_hit",),
  "root_height": (
    "Episode_Termination/root_height",
    "Episode_Terminations/root_height",
  ),
  "bad_orientation": (
    "Episode_Termination/bad_orientation",
    "Episode_Terminations/bad_orientation",
  ),
  "fall_penalty": ("Episode_Reward/fall_penalty",),
  "Mean action std": (
    "Policy/mean_std",
    "Policy/mean_action_std",
    "Train/mean_action_std",
    "Mean action std",
  ),
}

CORE_METRICS = (
  "self_table_bounce_count",
  "paddle_hit_count",
  "crossed_net_count",
  "opponent_table_bounce_count",
  "legal_return_count",
  "fault_count",
)
FAULT_METRICS = tuple(k for k in SCALAR_SPECS if k.startswith("fault_reason/"))
STABILITY_METRICS = ("ball_fault", "root_height", "bad_orientation", "fall_penalty")
TREND_METRICS = (
  "paddle_hit_count",
  "crossed_net_count",
  "opponent_table_bounce_count",
  "legal_return_count",
  "post_hit_x_progress",
  "post_hit_ball_velocity_direction",
  "hit/post_vx_toward_opponent_ratio",
  "hit/pred_net_clearance",
  "hit/pred_landing_inside_opponent_table",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Watch Pingpong Cross training and emit evidence reports.",
  )
  parser.add_argument("--logdir", required=True, type=Path)
  parser.add_argument("--pid", type=int)
  parser.add_argument("--eta-hours", type=float, default=14.0)
  parser.add_argument("--observe-interval-minutes", type=float, default=84.0)
  parser.add_argument("--min-fraction-before-stop", type=float, default=0.4)
  parser.add_argument("--kill-on-stop", action="store_true")
  parser.add_argument("--report-dir", type=Path)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--once", action="store_true")
  parser.add_argument(
    "--scalar-source",
    type=Path,
    help="TensorBoard event file or directory. Defaults to recursive logdir search.",
  )
  return parser.parse_args()


def process_alive(pid: int | None) -> bool | None:
  if pid is None:
    return None
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def discover_event_files(logdir: Path, scalar_source: Path | None) -> list[Path]:
  source = scalar_source or logdir
  if source.is_file():
    return [source]
  if not source.exists():
    return []
  return sorted(source.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)


def read_event_scalars(event_files: list[Path]) -> tuple[dict[str, Any], list[str]]:
  if _EventAccumulator is None:
    return {}, ["tensorboard is not importable"]

  by_tag: dict[str, list[dict[str, float]]] = {}
  errors: list[str] = []
  for event_file in event_files:
    try:
      acc = _EventAccumulator(str(event_file), size_guidance={"scalars": 0})
      acc.Reload()
      for tag in acc.Tags().get("scalars", []):
        values = by_tag.setdefault(tag, [])
        for event in acc.Scalars(tag):
          values.append(
            {
              "step": int(event.step),
              "wall_time": float(event.wall_time),
              "value": float(event.value),
              "tag": tag,
            }
          )
    except Exception as exc:  # noqa: BLE001 - watcher must not crash on bad events.
      errors.append(f"{event_file}: {exc}")

  metrics: dict[str, Any] = {}
  for name, candidates in SCALAR_SPECS.items():
    best: dict[str, float] | None = None
    for tag in candidates:
      history = by_tag.get(tag, [])
      if not history:
        continue
      history = sorted(history, key=lambda item: (item["step"], item["wall_time"]))
      latest = history[-1]
      if best is None or latest["wall_time"] > best["wall_time"]:
        best = latest
        metrics[name] = {
          **latest,
          "history_tail": history[-10:],
        }
    if best is None:
      metrics.pop(name, None)
  all_wall_times = [
    item["wall_time"] for history in by_tag.values() for item in history
  ]
  if all_wall_times:
    metrics["_event_time_bounds"] = {
      "value": 0.0,
      "step": 0,
      "wall_time": max(all_wall_times),
      "tag": "event_time_bounds",
      "history_tail": [
        {"wall_time": min(all_wall_times)},
        {"wall_time": max(all_wall_times)},
      ],
    }
  return metrics, errors


def iter_log_text_files(logdir: Path) -> list[Path]:
  if not logdir.exists():
    return []
  patterns = ("*.log", "*.out", "*.txt", "*stdout*", "*stderr*")
  files: set[Path] = set()
  for pattern in patterns:
    files.update(logdir.rglob(pattern))
  return sorted(files, key=lambda p: p.stat().st_mtime)[-12:]


def read_text_fallback(logdir: Path) -> dict[str, Any]:
  metrics: dict[str, Any] = {}
  action_std_pattern = re.compile(r"Mean action std\s*[:=]\s*([-+0-9.eE]+)")
  iteration_pattern = re.compile(r"(?:Learning iteration|iteration)\s+(\d+)")
  for path in iter_log_text_files(logdir):
    try:
      text = path.read_text(errors="ignore")[-200_000:]
    except OSError:
      continue
    for match in action_std_pattern.finditer(text):
      metrics["Mean action std"] = {
        "value": float(match.group(1)),
        "step": -1,
        "wall_time": path.stat().st_mtime,
        "tag": f"text:{path.name}",
        "history_tail": [],
      }
    for match in iteration_pattern.finditer(text):
      metrics["_text_iteration"] = {
        "value": float(match.group(1)),
        "step": int(match.group(1)),
        "wall_time": path.stat().st_mtime,
        "tag": f"text:{path.name}",
        "history_tail": [],
      }
  return metrics


def latest_checkpoint(logdir: Path) -> dict[str, Any] | None:
  checkpoints = []
  for path in logdir.glob("model_*.pt"):
    match = re.search(r"model_(\d+)\.pt$", path.name)
    if match:
      checkpoints.append((int(match.group(1)), path))
  if not checkpoints:
    return None
  iteration, path = max(checkpoints)
  return {"iteration": iteration, "path": str(path)}


def load_previous_report(report_dir: Path) -> dict[str, Any] | None:
  reports = sorted(report_dir.glob("*.json"))
  if not reports:
    return None
  try:
    return json.loads(reports[-1].read_text())
  except (OSError, json.JSONDecodeError):
    return None


def metric_value(metrics: dict[str, Any], name: str, default: float = 0.0) -> float:
  item = metrics.get(name)
  if item is None:
    return default
  value = float(item["value"])
  if math.isnan(value) or math.isinf(value):
    return value
  return value


def metric_trends(
  metrics: dict[str, Any],
  previous: dict[str, Any] | None,
) -> dict[str, str]:
  previous_metrics = (previous or {}).get("metrics", {})
  trends: dict[str, str] = {}
  for name in TREND_METRICS:
    if name not in metrics or name not in previous_metrics:
      trends[name] = "missing"
      continue
    delta = metric_value(metrics, name) - metric_value(previous_metrics, name)
    if abs(delta) < 1.0e-4:
      trends[name] = "flat"
    elif delta > 0.0:
      trends[name] = "up"
    else:
      trends[name] = "down"
  return trends


def runtime_summary(
  metrics: dict[str, Any],
  logdir: Path,
  eta_hours: float,
  alive: bool | None,
) -> dict[str, Any]:
  wall_times = [
    float(item["wall_time"])
    for item in metrics.values()
    if isinstance(item, dict) and float(item.get("wall_time", 0.0)) > 0.0
  ]
  for item in metrics.values():
    if not isinstance(item, dict):
      continue
    for history_item in item.get("history_tail", []):
      wall_time = float(history_item.get("wall_time", 0.0))
      if wall_time > 0.0:
        wall_times.append(wall_time)
  now = time.time()
  if wall_times:
    start = min(wall_times)
    end = now if alive else max(wall_times)
  elif logdir.exists():
    start = logdir.stat().st_mtime
    end = now
  else:
    start = now
    end = now
  runtime_seconds = max(0.0, end - start)
  eta_seconds = max(eta_hours * 3600.0, 1.0)
  return {
    "runtime_seconds": runtime_seconds,
    "eta_hours": eta_hours,
    "progress_fraction": min(runtime_seconds / eta_seconds, 1.0),
  }


def dominant_fault(metrics: dict[str, Any]) -> tuple[str | None, float]:
  values = [(name, metric_value(metrics, name)) for name in FAULT_METRICS]
  values = [(name, value) for name, value in values if value > 0.0]
  if not values:
    return None, 0.0
  return max(values, key=lambda item: item[1])


def decide(
  metrics: dict[str, Any],
  missing: list[str],
  trends: dict[str, str],
  runtime: dict[str, Any],
  alive: bool | None,
  min_fraction_before_stop: float,
) -> tuple[str, list[str], str | None, str]:
  reasons: list[str] = []
  stop_rule: str | None = None
  next_focus = "watch hit-to-return funnel and new strike-quality metrics"

  if alive is False and not metrics:
    return (
      "insufficient_evidence",
      ["process is not alive and no metrics were read"],
      None,
      ("verify logdir/pid and restart watcher with the correct paths"),
    )
  if not any(name in metrics for name in CORE_METRICS):
    return (
      "insufficient_evidence",
      ["core funnel metrics are missing"],
      None,
      ("confirm TensorBoard events or stdout/stderr paths"),
    )
  if any(math.isnan(metric_value(metrics, name)) for name in metrics):
    return "stop", ["NaN metric detected"], "nan_metric", "inspect crash/loss logs"

  progress = float(runtime["progress_fraction"])
  crossed = metric_value(metrics, "crossed_net_count")
  opponent = metric_value(metrics, "opponent_table_bounce_count")
  legal = metric_value(metrics, "legal_return_count")
  paddle = metric_value(metrics, "paddle_hit_count")
  post_x = metric_value(metrics, "post_hit_x_progress")
  post_dir = metric_value(metrics, "post_hit_ball_velocity_direction")
  vx_ratio = metric_value(metrics, "hit/post_vx_toward_opponent_ratio")
  net_clearance = metric_value(metrics, "hit/pred_net_clearance")
  landing_inside = metric_value(metrics, "hit/pred_landing_inside_opponent_table")
  action_std = metric_value(metrics, "Mean action std")
  root_height = metric_value(metrics, "root_height")
  bad_orientation = metric_value(metrics, "bad_orientation")
  body_ball = metric_value(metrics, "fault_reason/body_ball")
  double_paddle = metric_value(metrics, "fault_reason/double_paddle")
  dominant_fault_name, dominant_fault_value = dominant_fault(metrics)

  sparse_return_started = crossed > 0.0 or opponent > 0.0 or legal > 0.0
  improving = any(trends.get(name) == "up" for name in TREND_METRICS)
  unstable = root_height > 0.1 or bad_orientation > 0.1

  if sparse_return_started and not unstable:
    reasons.append("sparse return funnel has nonzero crossed/opponent/legal metric")
    return (
      "continue",
      reasons,
      None,
      "confirm opponent-table bounce and landing quality",
    )
  if sparse_return_started and unstable:
    reasons.append("sparse return appears but stability is degrading")
    return (
      "risky_continue",
      reasons,
      None,
      "watch fall/root/orientation before next window",
    )
  if progress < 0.3 and not unstable and action_std < 50.0:
    reasons.append("run is before 30% ETA and no severe instability is visible")
    return "continue", reasons, None, next_focus
  if improving and not unstable:
    reasons.append("at least one intermediate hit-quality metric is improving")
    return "continue", reasons, None, next_focus
  if improving and unstable:
    reasons.append("intermediate metrics improve but stability worsens")
    return "risky_continue", reasons, None, "watch stability and action std"
  if action_std > 50.0 and crossed <= 0.0:
    reasons.append("action std is high without crossed-net progress")
    return "risky_continue", reasons, None, "watch for instability or sparse return"
  if paddle > 0.2 and (body_ball > 0.2 or double_paddle > 0.2):
    reasons.append(
      "paddle hit is reachable but body-ball/double-paddle faults are high"
    )
    return "risky_continue", reasons, None, "inspect fault_reason breakdown"

  can_stop = progress >= min_fraction_before_stop
  if can_stop and paddle < 0.2:
    reasons.append(
      "paddle_hit_count remains low after the minimum observation fraction"
    )
    stop_rule = "low_paddle_hit_after_min_fraction"
    return (
      "stop_and_reconfigure",
      reasons,
      stop_rule,
      ("return to timing/proximity/paddle-hit reward before post-hit tuning"),
    )
  if can_stop and crossed <= 0.0 and opponent <= 0.0 and legal <= 0.0:
    stagnant_post_hit = post_x < 0.2 and post_dir < 0.05 and vx_ratio < 0.1
    if stagnant_post_hit:
      reasons.append("funnel is stuck after paddle hit with weak outgoing velocity")
      stop_rule = "no_crossed_net_and_stagnant_post_hit"
      return (
        "stop_and_reconfigure",
        reasons,
        stop_rule,
        ("increase outgoing velocity, net clearance, or hit-window energy"),
      )
  if can_stop and dominant_fault_name is not None and dominant_fault_value > 0.7:
    reasons.append(f"dominant fault reason is {dominant_fault_name}")
    stop_rule = "dominant_fault_reason"
    return (
      "stop_and_reconfigure",
      reasons,
      stop_rule,
      ("reconfigure against the dominant fault before continuing"),
    )
  if progress >= 0.7 and crossed <= 0.0 and vx_ratio < 0.1 and net_clearance <= 0.0:
    reasons.append("past 70% ETA and funnel still lacks crossed-net quality")
    stop_rule = "late_no_hit_quality"
    return (
      "stop_and_reconfigure",
      reasons,
      stop_rule,
      ("change strike-quality reward or energy relaxation"),
    )
  if missing:
    reasons.append("some requested metrics are missing; need another instrumented run")
    return "insufficient_evidence", reasons, None, "launch diagnostics-only run"
  if landing_inside <= 0.0 and crossed > 0.0:
    reasons.append("crossed-net appears but predicted landing is outside")
    return "risky_continue", reasons, None, "watch landing reward/target region"

  reasons.append("no hard stop rule triggered")
  return "continue", reasons, None, next_focus


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
  compact: dict[str, Any] = {}
  for name, item in sorted(metrics.items()):
    compact[name] = {
      "value": item["value"],
      "step": item["step"],
      "wall_time": item["wall_time"],
      "tag": item["tag"],
    }
  return compact


def write_reports(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
  report_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  json_path = report_dir / f"{timestamp}_pingpong_cross_watch.json"
  md_path = report_dir / f"{timestamp}_pingpong_cross_watch.md"
  json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
  md_path.write_text(render_markdown(report))
  return json_path, md_path


def render_markdown(report: dict[str, Any]) -> str:
  lines = [
    "# Pingpong Cross Watch Report",
    "",
    f"- run: `{report['run_name']}`",
    f"- logdir: `{report['logdir']}`",
    f"- pid: `{report['pid']}`",
    f"- process_alive: `{report['process_alive']}`",
    f"- observed_at: `{report['observed_at']}`",
    f"- runtime_hours: `{report['runtime']['runtime_seconds'] / 3600.0:.2f}`",
    f"- eta_hours: `{report['runtime']['eta_hours']:.2f}`",
    f"- progress_fraction: `{report['runtime']['progress_fraction']:.3f}`",
    f"- latest_checkpoint: `{report.get('latest_checkpoint')}`",
    f"- decision: `{report['decision']}`",
    f"- stop_rule: `{report.get('stop_rule')}`",
    "",
    "## Reasons",
  ]
  lines.extend(f"- {reason}" for reason in report["decision_reasons"])
  lines.append("")
  lines.append("## Funnel")
  for name in CORE_METRICS + (
    "hit/post_vx_toward_opponent_ratio",
    "hit/pred_net_clearance",
    "hit/pred_net_clearance_positive",
    "hit/pred_landing_inside_opponent_table",
  ):
    value = report["metrics"].get(name, {}).get("value", "missing")
    lines.append(f"- {name}: `{value}`")
  lines.append("")
  lines.append("## Fault Breakdown")
  for name in FAULT_METRICS:
    value = report["metrics"].get(name, {}).get("value", "missing")
    lines.append(f"- {name}: `{value}`")
  lines.append("")
  lines.append("## Stability And Action Std")
  for name in STABILITY_METRICS + ("Mean action std",):
    value = report["metrics"].get(name, {}).get("value", "missing")
    lines.append(f"- {name}: `{value}`")
  lines.append("")
  lines.append("## Missing Metrics")
  if report["missing_metrics"]:
    lines.extend(f"- {name}" for name in report["missing_metrics"])
  else:
    lines.append("- none")
  lines.append("")
  lines.append("## Next Focus")
  lines.append(report["next_focus"])
  lines.append("")
  return "\n".join(lines)


def maybe_stop_process(report: dict[str, Any], args: argparse.Namespace) -> None:
  if args.dry_run or not args.kill_on_stop or args.pid is None:
    return
  if report["decision"] not in {"stop", "stop_and_reconfigure"}:
    return
  os.kill(args.pid, signal.SIGINT)
  report["stop_signal_sent"] = "SIGINT"


def observe(args: argparse.Namespace) -> dict[str, Any]:
  logdir = args.logdir.resolve()
  event_files = discover_event_files(logdir, args.scalar_source)
  metrics, event_errors = read_event_scalars(event_files)
  fallback_metrics = read_text_fallback(logdir)
  metrics.update({k: v for k, v in fallback_metrics.items() if k not in metrics})
  missing = [name for name in SCALAR_SPECS if name not in metrics]
  report_dir = (args.report_dir or (logdir / "watch_reports")).resolve()
  previous = load_previous_report(report_dir)
  alive = process_alive(args.pid)
  runtime = runtime_summary(metrics, logdir, args.eta_hours, alive)
  trends = metric_trends(metrics, previous)
  decision, reasons, stop_rule, next_focus = decide(
    metrics,
    missing,
    trends,
    runtime,
    alive,
    args.min_fraction_before_stop,
  )
  checkpoint = latest_checkpoint(logdir)
  iteration = checkpoint["iteration"] if checkpoint else None
  if iteration is None and "_text_iteration" in metrics:
    iteration = int(metrics["_text_iteration"]["step"])

  report = {
    "run_name": logdir.name,
    "logdir": str(logdir),
    "pid": args.pid,
    "process_alive": alive,
    "observed_at": datetime.now().isoformat(timespec="seconds"),
    "runtime": runtime,
    "observe_interval_minutes": args.observe_interval_minutes,
    "current_iteration": iteration,
    "latest_checkpoint": checkpoint,
    "event_files": [str(path) for path in event_files],
    "event_errors": event_errors,
    "metrics": compact_metrics(
      {k: v for k, v in metrics.items() if not k.startswith("_")}
    ),
    "missing_metrics": missing,
    "trends": trends,
    "decision": decision,
    "decision_reasons": reasons,
    "stop_rule": stop_rule,
    "next_focus": next_focus,
    "dry_run": args.dry_run,
  }
  assert report["decision"] in DECISIONS
  maybe_stop_process(report, args)
  json_path, md_path = write_reports(report, report_dir)
  report["json_report"] = str(json_path)
  report["markdown_report"] = str(md_path)
  return report


def main() -> int:
  args = parse_args()
  while True:
    report = observe(args)
    print(
      f"decision={report['decision']} json={report['json_report']} "
      f"markdown={report['markdown_report']}",
      flush=True,
    )
    if args.once:
      return 0
    time.sleep(args.observe_interval_minutes * 60.0)


if __name__ == "__main__":
  sys.exit(main())
