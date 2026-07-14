"""Latent action distillation 的基础环境配置。

Distillation reuses the tracking MDP / environment unchanged. Robot-specific
configs may add training disturbances without changing the shared tracking
observation and action contract.

1. teacher 与 student 在尽可能一致的环境中交互。
2. tracking 侧任何修复都能自动同步到 distillation。
3. distillation 只保留训练所需的最小差异。
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def make_distillation_env_cfg() -> ManagerBasedRlEnvCfg:
  """创建与 tracking 共享环境契约的 distillation 基础配置。"""
  return make_tracking_env_cfg()
