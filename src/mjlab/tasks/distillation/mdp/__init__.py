# pyright: reportWildcardImportFromLibrary=false
"""Distillation MDP terms.

Distillation reuses the entire tracking MDP layer (commands, observations,
rewards, terminations, metrics) since the student rolls out in the same
environment as the teacher. Re-exporting keeps a single source of truth.

The wildcard re-export is intentional: callers use ``mdp.<symbol>`` to mirror
the tracking-task style in ``distill_env_cfg.py`` and ``config/g1/env_cfgs.py``.
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403
from mjlab.tasks.tracking.mdp import (  # noqa: F401  # explicit re-exports for IDE
  MotionCommand,
  MotionCommandCfg,
)
