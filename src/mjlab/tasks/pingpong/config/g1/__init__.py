from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunner

from .env_cfgs import (
  unitree_g1_pingpong_latent_cross_diag_env_cfg,
  unitree_g1_pingpong_latent_cross_env_cfg,
  unitree_g1_pingpong_latent_cross_impact_env_cfg,
  unitree_g1_pingpong_latent_cross_strike_quality_energy_relax_env_cfg,
  unitree_g1_pingpong_latent_cross_strike_quality_env_cfg,
  unitree_g1_pingpong_latent_hit_env_cfg,
  unitree_g1_pingpong_latent_return_env_cfg,
)
from .rl_cfg import (
  unitree_g1_pingpong_cross_diag_ppo_runner_cfg,
  unitree_g1_pingpong_cross_impact_ppo_runner_cfg,
  unitree_g1_pingpong_cross_ppo_runner_cfg,
  unitree_g1_pingpong_cross_strike_quality_energy_relax_ppo_runner_cfg,
  unitree_g1_pingpong_cross_strike_quality_ppo_runner_cfg,
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
  task_id="Mjlab-Pingpong-Cross-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_cross_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_cross_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_cross_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Cross-Diag-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_cross_diag_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_cross_diag_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_cross_diag_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Cross-StrikeQuality-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_cross_strike_quality_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_cross_strike_quality_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_cross_strike_quality_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Cross-Impact-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_cross_impact_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_cross_impact_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_cross_impact_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Cross-StrikeQualityEnergyRelax-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_cross_strike_quality_energy_relax_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_cross_strike_quality_energy_relax_env_cfg(
    play=True
  ),
  rl_cfg=unitree_g1_pingpong_cross_strike_quality_energy_relax_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Pingpong-Return-Unitree-G1",
  env_cfg=unitree_g1_pingpong_latent_return_env_cfg(),
  play_env_cfg=unitree_g1_pingpong_latent_return_env_cfg(play=True),
  rl_cfg=unitree_g1_pingpong_return_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)
