"""latent distillation 任务对外暴露的 RL 组件。

Public surface:

* :class:`DistillationRunnerCfg`   -- runner configuration dataclass.
* :class:`OnlineDistillationRunner` -- DAgger training loop.
* :class:`LatentStudentModel`       -- PHC-style VAE student.
* :func:`diagonal_gaussian_kl`      -- KL helper for the loss.
* :class:`ReplayBuffer`             -- 设备端环形缓存。
* :class:`ObservationSlicer`        -- state / target 索引切分器。

本模块的作用只是整理公共导出，便于外部通过
``from mjlab.tasks.distillation.rl import ...`` 统一导入。
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
