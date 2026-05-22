source .venv/bin/activate

uv run play Mjlab-Tracking-Flat-Unitree-G1 --checkpoint-file ./logs/rsl_rl/g1_tracking/tennis/model_29999.pt --motion-files ./artifacts/tennis_random_001/motion.npz --viewer viser

MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_41000.pt --video True --video-length 2000

MUJOCO_GL=egl uv run play Mjlab-Tennis-Hit-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_curr_quarter_2026-05-18_22-02-07/model_29999.pt --video True --video-length 2000

tensorboard --logdir logs/rsl_rl/g1_tennis_latent_hit/

uv run train Mjlab-Tennis-Hit-Unitree-G1 --agent.run_name tennis_cloud_tennis_B_curr_mini --env.scene.num_envs 4096 --env.court_size mini --gpu-ids ['6,7']

uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_40500.pt --viewer viser 

uv run play Mjlab-Tennis-Hit-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_2026-05-13_14-41-26/model_4200.pt --viewer viser

uv run play Mjlab-Velocity-Stairs-Unitree-Go1 --checkpoint-file logs/rsl_rl/go1_velocity/stairs_go1_2026-05-19_15-35-35/model_3400.pt --viewer viser 

uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_40500.pt --viewer viser