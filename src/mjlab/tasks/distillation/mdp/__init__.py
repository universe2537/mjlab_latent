# pyright: reportWildcardImportFromLibrary=false
"""Distillation 任务使用的 MDP 接口导出。

由于 distillation 与 tracking 在环境动力学、奖励、终止和命令定义上保持一致，
这里直接复用 tracking 的 MDP 层，而不是再复制一份实现。

这样可以确保：

1. teacher 与 student 使用相同的命令和观测定义。
2. tracking 修复自动同步到 distillation。
3. distillation 模块只关心蒸馏差异，而不是环境细节。

The wildcard re-export is intentional: callers use ``mdp.<symbol>`` to mirror
the tracking-task style in ``distill_env_cfg.py`` and ``config/g1/env_cfgs.py``.
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403
from mjlab.tasks.tracking.mdp import (  # noqa: F401  # explicit re-exports for IDE
  MotionCommand,
  MotionCommandCfg,
)
