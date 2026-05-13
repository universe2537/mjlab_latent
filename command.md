source .venv/bin/activate

uv run play Mjlab-Tracking-Flat-Unitree-G1 --checkpoint-file ./logs/rsl_rl/g1_tracking/tennis/model_29999.pt --motion-files ./artifacts/tennis_random_001/motion.npz --viewer viser

uv run play-distill.py --checkpoint-file logs/rsl_rl/g1_tracking/lafan/model_29999.pt  --motion-file artifacts/tennis_random_001/motion.npz --viewer viser

uv run play Mjlab-Tennis-Hit-Unitree-G1 --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_curr_2026-05-13_15-35-28/model_200.pt --viewer viser


uv run train Mjlab-Tennis-Hit-Unitree-G1 --agent.run_name tennis_cloud_tennis_Return --env.scene.num-envs 4096 --gpu-ids [6,7]