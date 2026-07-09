"""Reserve GPU memory and CPU cores for placeholder occupancy.

Example:
  uv run python scripts/tools/resource_placeholder.py --gpu-id 3
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import time
from typing import Any


def _parse_cpu_workers(value: str) -> int:
  if value.lower() == "all":
    return max(1, os.cpu_count() or 1)
  workers = int(value)
  if workers < 0:
    raise argparse.ArgumentTypeError("cpu workers must be >= 0 or 'all'")
  return workers


def _cpu_burn_worker(stop_event: Any, worker_id: int) -> None:
  # Keep the loop pure-Python so each worker reliably occupies one CPU process.
  value = 0x9E3779B97F4A7C15 ^ worker_id
  while not stop_event.is_set():
    for _ in range(200_000):
      value ^= (value << 13) & 0xFFFFFFFFFFFFFFFF
      value ^= value >> 7
      value ^= (value << 17) & 0xFFFFFFFFFFFFFFFF


def _reserve_gpu_memory(gpu_id: int, memory_gb: float, chunk_gb: float) -> list[Any]:
  if memory_gb <= 0:
    return []
  if chunk_gb <= 0:
    raise ValueError("--chunk-gb must be > 0")

  # Interpret --gpu-id as the physical CUDA id, then allocate on visible cuda:0.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

  import torch

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in this process.")

  device = torch.device("cuda:0")
  torch.cuda.set_device(device)

  total_bytes = int(memory_gb * 1024**3)
  chunk_bytes = max(1, int(chunk_gb * 1024**3))
  tensors: list[Any] = []
  allocated = 0

  while allocated < total_bytes:
    this_chunk = min(chunk_bytes, total_bytes - allocated)
    tensor = torch.empty(this_chunk, dtype=torch.uint8, device=device)
    tensor.fill_(1)
    tensors.append(tensor)
    allocated += this_chunk

  torch.cuda.synchronize(device)
  return tensors


def _build_arg_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Reserve approximately fixed GPU memory and all CPU cores.",
  )
  parser.add_argument(
    "--gpu-id",
    type=int,
    default=0,
    help="Physical CUDA GPU id to reserve, e.g. 3 for GPU 3.",
  )
  parser.add_argument(
    "--gpu-memory-gb",
    type=float,
    default=16.0,
    help="Approximate GPU memory to reserve in GiB.",
  )
  parser.add_argument(
    "--chunk-gb",
    type=float,
    default=1.0,
    help="GPU allocation chunk size in GiB.",
  )
  parser.add_argument(
    "--cpu-workers",
    type=_parse_cpu_workers,
    default=_parse_cpu_workers("all"),
    help="Number of busy CPU worker processes, or 'all'. Use 0 to disable.",
  )
  return parser


def main() -> None:
  args = _build_arg_parser().parse_args()
  stop_event = mp.Event()

  def _handle_stop(_signum: int, _frame: Any) -> None:
    stop_event.set()

  signal.signal(signal.SIGTERM, _handle_stop)
  signal.signal(signal.SIGINT, _handle_stop)

  gpu_tensors = _reserve_gpu_memory(
    gpu_id=args.gpu_id,
    memory_gb=args.gpu_memory_gb,
    chunk_gb=args.chunk_gb,
  )
  workers = [
    mp.Process(target=_cpu_burn_worker, args=(stop_event, i), daemon=True)
    for i in range(args.cpu_workers)
  ]
  for worker in workers:
    worker.start()

  reserved_gib = sum(t.numel() * t.element_size() for t in gpu_tensors) / 1024**3
  print(
    f"Reserved ~{reserved_gib:.2f} GiB on physical GPU {args.gpu_id}; "
    f"started {len(workers)} CPU workers. Press Ctrl-C to stop.",
    flush=True,
  )

  try:
    while not stop_event.is_set():
      time.sleep(1.0)
  finally:
    stop_event.set()
    for worker in workers:
      worker.join(timeout=2.0)


if __name__ == "__main__":
  main()
