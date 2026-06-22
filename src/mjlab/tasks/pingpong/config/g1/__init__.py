from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunner

from .env_cfgs import (
  unitree_g1_pingpong_latent_hit_env_cfg,
  unitree_g1_pingpong_latent_return_env_cfg,
)
from .rl_cfg import (
  unitree_g1_pingpong_latent_ppo_runner_cfg,
  unitree_g1_pingpong_return_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Hit-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_hit_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_hit_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_latent_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Return-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_return_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_return_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_return_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)
