"""RL components for the latent distillation task.

Public surface:

* :class:`DistillationRunnerCfg`   -- runner configuration dataclass.
* :class:`OnlineDistillationRunner` -- DAgger training loop.
* :class:`LatentStudentModel`       -- PHC-style VAE student.
* :func:`diagonal_gaussian_kl`      -- KL helper for the loss.
* :class:`ReplayBuffer`             -- on-device circular buffer.
* :class:`ObservationSlicer`        -- state / target index splitter.
"""

from mjlab.tasks.distillation.rl.buffer import ReplayBuffer
from mjlab.tasks.distillation.rl.config import DistillationRunnerCfg
from mjlab.tasks.distillation.rl.models import (
  DiagGaussian,
  LatentStudentModel,
  diagonal_gaussian_kl,
)
from mjlab.tasks.distillation.rl.obs_slicer import ObservationSlicer
from mjlab.tasks.distillation.rl.runner import OnlineDistillationRunner

__all__ = [
  "DiagGaussian",
  "DistillationRunnerCfg",
  "LatentStudentModel",
  "ObservationSlicer",
  "OnlineDistillationRunner",
  "ReplayBuffer",
  "diagonal_gaussian_kl",
]
