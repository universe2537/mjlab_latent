from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tennis.rl import TennisLatentOnPolicyRunner
from mjlab.tasks.tennis.tennis_env_cfg import DEFAULT_COURT_SIZE

from .env_cfgs import unitree_g1_tennis_latent_hit_env_cfg
from .rl_cfg import unitree_g1_tennis_latent_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tennis-Hit-Unitree-G1",
  env_cfg=unitree_g1_tennis_latent_hit_env_cfg(court_size=DEFAULT_COURT_SIZE),
  play_env_cfg=unitree_g1_tennis_latent_hit_env_cfg(
    play=True, court_size=DEFAULT_COURT_SIZE
  ),
  rl_cfg=unitree_g1_tennis_latent_ppo_runner_cfg(),
  runner_cls=TennisLatentOnPolicyRunner,
)
