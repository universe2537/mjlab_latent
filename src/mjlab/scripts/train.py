"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _load_env_file(path: Path = Path(".env"), override: bool = False) -> None:
  """Load simple KEY=VALUE or export KEY=VALUE entries from a .env file."""
  if not path.exists():
    return

  for line in path.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("export "):
      line = line[len("export ") :].strip()
    if "=" not in line:
      continue

    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip("'\"")
    if not key or (not override and key in os.environ):
      continue
    os.environ[key] = value


def _first_env(*names: str) -> str | None:
  for name in names:
    value = os.environ.get(name)
    if value:
      return value
  return None


def _apply_env_defaults(cfg: TrainConfig) -> None:
  """Apply .env defaults that are consumed by rsl_rl/W&B."""
  wandb_project = _first_env("WANDB_PROJECT")
  if wandb_project:
    cfg.agent.wandb_project = wandb_project

  wandb_entity = _first_env("WANDB_ENTITY")
  if wandb_entity and "WANDB_USERNAME" not in os.environ:
    # rsl_rl's WandbSummaryWriter reads WANDB_USERNAME for the entity.
    os.environ["WANDB_USERNAME"] = wandb_entity


def _requires_opengl_backend(cfg: TrainConfig) -> bool:
  """Return whether training needs an OpenGL backend in child processes.

  Multi-GPU launches start fresh Python interpreters after environment variables
  are exported. Setting ``MUJOCO_GL=egl`` unconditionally forces those fresh
  interpreters to import MuJoCo's EGL path even for tasks that never render,
  which can fail on machines without EGL runtime libraries installed.

  We therefore only request EGL when a run will actually render frames: either
  training video is enabled or the scene contains camera sensors.
  """
  if cfg.video:
    return True

  sensors = getattr(cfg.env.scene, "sensors", ()) or ()
  return any(sensor.__class__.__name__ == "CameraSensorCfg" for sensor in sensors)


def _normalize_wandb_motion_ref(ref: str, alias: str | None = None) -> str:
  entity = _first_env("WANDB_ENTITY")
  if "/" not in ref:
    if entity is None:
      raise ValueError(
        f"W&B motion artifact {ref!r} is missing an entity. "
        "Set WANDB_ENTITY in .env or use a full artifact path."
      )
    ref = f"{entity}/motions/{ref}"
  elif ref.count("/") == 1:
    if entity is None:
      raise ValueError(
        f"W&B motion artifact {ref!r} is missing an entity. "
        "Set WANDB_ENTITY in .env or use a full artifact path."
      )
    ref = f"{entity}/{ref}"

  if ":" not in ref.rsplit("/", 1)[-1]:
    ref = f"{ref}:{alias or 'latest'}"
  return ref


def _motion_file_refs(motion_files: str | tuple[str, ...]) -> tuple[str, ...]:
  if isinstance(motion_files, str):
    return (motion_files,) if motion_files else ()
  return tuple(motion_files)


def _configured_motion_paths(motion_file: str) -> list[Path]:
  path = Path(os.path.expandvars(motion_file)).expanduser()
  if path.is_absolute():
    return [path]

  paths = []
  gli_path = _first_env("GLI_PATH")
  if gli_path:
    paths.append(Path(os.path.expandvars(gli_path)).expanduser() / path)
  paths.append(Path.cwd() / path)

  unique_paths = []
  seen = set()
  for candidate in paths:
    absolute = candidate.absolute()
    if absolute in seen:
      continue
    seen.add(absolute)
    unique_paths.append(candidate)
  return unique_paths


def _wandb_ref_from_configured_motion(
  cfg: TrainConfig, motion_cmd: Any
) -> tuple[str, ...]:
  alias = _first_env("MJLAB_MOTION_ALIAS", "WANDB_ARTIFACT_ALIAS") or "latest"
  motion_refs = (
    (cfg.registry_name,)
    if cfg.registry_name
    else _motion_file_refs(motion_cmd.motion_files)
  )
  return tuple(_normalize_wandb_motion_ref(ref, alias) for ref in motion_refs)


def _download_motion_from_registry(registry_name: str) -> Path:
  import wandb

  api = wandb.Api()
  artifact = api.artifact(registry_name)
  download_dir = Path(artifact.download())
  source_motion_file = download_dir / "motion.npz"

  if not source_motion_file.exists():
    npz_files = sorted(download_dir.rglob("*.npz"))
    if not npz_files:
      raise FileNotFoundError(f"No .npz file found in W&B artifact {registry_name}.")
    source_motion_file = npz_files[0]

  return source_motion_file


def _resolve_tracking_motion(
  cfg: TrainConfig, motion_cmd: Any
) -> tuple[tuple[Path, ...], str | None]:
  if motion_cmd.motion_source == "local":
    motion_refs = _motion_file_refs(motion_cmd.motion_files)
    if not motion_refs:
      raise ValueError(
        "MotionCommandCfg.motion_source is 'local', but motion_files is empty."
      )
    resolved_paths = []
    for motion_ref in motion_refs:
      configured_paths = _configured_motion_paths(motion_ref)
      local_motion = next((path for path in configured_paths if path.exists()), None)
      if local_motion is None:
        raise ValueError(
          "Configured local motion file was not found. "
          f"motion_files entry={motion_ref!r}; searched: "
          + ", ".join(str(path) for path in configured_paths)
        )
      resolved_paths.append(local_motion)
    print(f"[INFO] Using {len(resolved_paths)} configured local motion file(s).")
    return tuple(resolved_paths), None

  if motion_cmd.motion_source == "wandb":
    registry_names = _wandb_ref_from_configured_motion(cfg, motion_cmd)
    if not registry_names:
      raise ValueError(
        "MotionCommandCfg.motion_source is 'wandb', but motion_files is empty."
      )
    print(
      f"[INFO] Downloading {len(registry_names)} configured W&B motion artifact(s)."
    )
    motion_paths = tuple(
      _download_motion_from_registry(registry_name) for registry_name in registry_names
    )
    return motion_paths, registry_names[0] if len(registry_names) == 1 else None

  raise ValueError(f"Unknown motion_source: {motion_cmd.motion_source!r}")


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  _load_env_file()
  _apply_env_defaults(cfg)

  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  is_tracking_task = "motion" in cfg.env.commands and hasattr(
    cfg.env.commands["motion"], "motion_files"
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    motion_paths, registry_name = _resolve_tracking_motion(cfg, motion_cmd)
    motion_cmd.motion_files = tuple(str(path) for path in motion_paths)  # type: ignore[union-attr]

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  _load_env_file()
  args = args or TrainConfig.from_task(task_id)
  _apply_env_defaults(args)

  # Create log directory once before launching workers.
  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name = f"{args.agent.run_name}_{log_dir_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if _requires_opengl_backend(args):
    os.environ["MUJOCO_GL"] = "egl"
  else:
    os.environ.pop("MUJOCO_GL", None)
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
      + ("MUJOCO*", "WANDB*", "GLI_PATH", "MJLAB*"),
    ).run(run_train, task_id, args, log_dir)


def main():
  _load_env_file()
  maybe_print_top_level_help("train")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
