"""Runner config for PACE-style pingpong tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class PingpongPaceOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO runner config with a PACE-style auxiliary ball predictor."""

  class_name: str = "PingpongPaceOnPolicyRunner"

  predictor: dict[str, Any] = field(
    default_factory=lambda: {
      "history_len": 5,
      "traj_max_len": 128,
      "hidden_sizes": [64, 64],
      "lr": 5.0e-4,
      "epochs_per_update": 1,
      "batch_size": 1024,
      "train_until_iters": 20,
    }
  )
