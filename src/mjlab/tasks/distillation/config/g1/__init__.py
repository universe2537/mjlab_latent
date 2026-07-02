from mjlab.tasks.distillation.rl import OnlineDistillationRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_flat_distillation_env_cfg,
  unitree_g1_table_tennis_distillation_env_cfg,
)
from .rl_cfg import (
  unitree_g1_distillation_runner_cfg,
  unitree_g1_table_tennis_distillation_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Distill-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_distillation_env_cfg(),
  play_env_cfg=unitree_g1_flat_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_distillation_runner_cfg(),
  runner_cls=OnlineDistillationRunner,
)

register_mjlab_task(
  task_id="Mjlab-Distill-TableTennis-Unitree-G1",
  env_cfg=unitree_g1_table_tennis_distillation_env_cfg(),
  play_env_cfg=unitree_g1_table_tennis_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_table_tennis_distillation_runner_cfg(),
  runner_cls=OnlineDistillationRunner,
)
