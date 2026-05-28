source .venv/bin/activate

uv run play Mjlab-Tracking-Flat-Unitree-G1 --checkpoint-file ./logs/rsl_rl/g1_tracking/tennis/model_29999.pt --motion-files ./artifacts/tennis_random_001/motion.npz --viewer viser

MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_41000.pt --video True --video-length 2000

MUJOCO_GL=egl uv run play Mjlab-Tennis-Hit-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_curr_quarter_2026-05-18_22-02-07/model_29999.pt --video True --video-length 2000

tensorboard --logdir logs/rsl_rl/g1_tennis_latent_hit/

uv run play Mjlab-Tennis-Hit-LAB-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit_lab/tennis_hit_lab_2026-05-24_01-49-58/model_29999.pt --viewer viser

MUJOCO_GL=egl uv run play Mjlab-Tennis-Hit-LAB-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit_lab/tennis_hit_lab_2026-05-24_01-49-58/model_29999.pt --video True --video-length 2000

uv run train Mjlab-Tennis-Hit-Unitree-G1 --agent.run_name tennis_cloud_tennis_B_curr_mini --env.scene.num_envs 4096 --env.court_size mini --gpu-ids ['6,7']

uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_40500.pt --viewer viser 

uv run play Mjlab-Tennis-Hit-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_2026-05-13_14-41-26/model_4200.pt --viewer viser

uv run play Mjlab-Velocity-Stairs-Unitree-Go1 --checkpoint-file logs/rsl_rl/go1_velocity/stairs_go1_2026-05-19_15-35-35/model_3400.pt --viewer viser 

# Normal
uv run play Mjlab-Tennis-Cross-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_59998.pt --viewer viser

# LAB
uv run train Mjlab-Tennis-Hit-LAB-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]
uv run train Mjlab-Tennis-Cross-LAB-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]
uv run train Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]

uv run play Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross_wrist_lab/tennis_cross_wrist_lab_from_cross_2026-05-26_15-07-20/model_73500.pt --viewer viser
uv run play Mjlab-Tennis-Cross-LAB-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross_lab/old_tennis_cross_lab_2026-05-24_01-49-21/model_109996.pt --viewer viser
MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-LAB-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross_lab/tennis_cross_lab_2026-05-25_10-10-42/model_42000.pt --viewer viser --video True --video-length 20000

# Continuous
uv run train Mjlab-Tennis-Hit-Continuous-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [4,5,6]
uv run play Mjlab-Tennis-Continuous-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_continuous/tennis_continuous_from_cross_2026-05-27_14-37-48/model_68500.pt --viewer viser
MUJOCO_GL=egl uv run play Mjlab-Tennis-Continuous-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_continuous/tennis_continuous_from_cross_2026-05-27_14-37-48/model_68500.pt --video True --video-length 2000

# SONIC
uv run train Mjlab-Tennis-Hit-SONIC-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]
uv run train Mjlab-Tennis-Cross-SONIC-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]
uv run train Mjlab-Tennis-Hit-SONIC-Encoder-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]
uv run train Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1 --env.scene.num_envs 4096 --gpu-ids [0,1]

uv run play Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_sonic_encoder_hit/tennis_hit_sonic_encoder_prior_2026-05-27_01-51-16/model_5000.pt --viewer viser
MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-SONIC-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_sonic_hit/tennis_hit_sonic_token_2026-05-26_01-50-16/model_12000.pt --viewer viser
# Saved wo-encoder baseline video:
# logs/rsl_rl/g1_tennis_sonic_hit/tennis_sonic_cross_wo_encoder_video_2026-05-27/videos/play/sonic-wo-encoder-step-0.mp4

uv run play-distill.py --checkpoint-file logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt --viewer viser
