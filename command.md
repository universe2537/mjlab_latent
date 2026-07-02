# Command Notes

## Setup

```sh
source .venv/bin/activate
```

## Pingpong

### Train

Current active V3 collision continuation run, `512*20` envs, resumed from the
latest clean V3 checkpoint:

```sh
tmux new-session -d -s pingpang_v3_collision \
  -c /home/universe/workspace/mjlab_latent \
  'WANDB_MODE=offline uv run train Mjlab-Pingpong-Hit-Unitree-G1 \
  --env.scene.num-envs 10240 \
  --gpu-ids "[1,2]" \
  --agent.max-iterations 30000 \
  --agent.run-name v3_collision \
  --agent.resume True \
  --agent.load-checkpoint-file logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_v3_clean_contact_scratch_10240env_gpu1_2_2026-06-24_11-35-28/model_3500.pt'
```

### Play

MuJoCo offscreen video, no viser:

```sh
MUJOCO_GL=egl uv run play Mjlab-Pingpong-Hit-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_v3_collision_10240env_gpu1_2_from15500_2026-06-23_17-00-15/model_5500.pt \
  --video True \
  --video-length 2000 \
  --video-height 1080 \
  --video-width 1920 \
  --num-envs 1 \
  --device cuda:1 \
  --viewer none
```

Interactive viewer:

```sh
uv run play Mjlab-Pingpong-Cross-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_pingpong_latent_cross/pingpong_cross_dense_retune_from_hit_v3_collision4500_16384env_gpu4_6_2026-06-25_14-26-38/model_10500.pt \
  --viewer viser

uv run play Mjlab-Pingpong-Cross-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_pingpong_latent_cross_strike_quality/pingpong_cross_strike_quality_from_hit_16384env_gpu0_1_20260629_224812_2026-06-29_22-48-24/model_16500.pt --viewer viser

MUJOCO_GL=egl uv run play Mjlab-Pingpong-Cross-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_pingpong_latent_cross/pingpong_cross_dense_retune_from_hit_v3_collision4500_16384env_gpu4_6_2026-06-25_14-26-38/model_10500.pt \
  --viewer none --video True --video-length 500 --video-height 1080 --video-width 1920 --num-envs 1 --device cuda:1
```

### Logs

```sh
tensorboard --logdir logs/rsl_rl/g1_pingpong_latent_hit/
```

Useful runs:

```text
V3 collision active continuation:
logs/rsl_rl/g1_pingpong_latent_hit/v3_collision_2026-06-24_15-53-52

Clean V3 stopped checkpoint source:
logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_v3_clean_contact_scratch_10240env_gpu1_2_2026-06-24_11-35-28

Legacy V2-family scratch/checkpoint-comparison runs:
logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_v4_contact_scratch_10240env_gpu1_2_2026-06-24_00-35-24

logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_v4_contact_scratch_8192env_gpu1_2_2026-06-24_00-11-43

logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_v3_collision_10240env_gpu1_2_from15500_2026-06-23_17-00-15

V1 chase-and-hit milestone:
logs/rsl_rl/g1_pingpong_latent_hit/pingpong_hit_8192env_gpu2_4_2026-06-21_23-07-11
```

## Tracking

```sh
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint-file ./logs/rsl_rl/g1_tracking/tennis/model_29999.pt \
  --motion-files ./artifacts/tennis_random_001/motion.npz \
  --viewer viser
```

## Tennis

### Hit

```sh
uv run train Mjlab-Tennis-Hit-Unitree-G1 \
  --agent.run_name tennis_cloud_tennis_B_curr_mini \
  --env.scene.num_envs 4096 \
  --env.court_size mini \
  --gpu-ids ['6,7']
```

```sh
uv run play Mjlab-Tennis-Hit-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_2026-05-13_14-41-26/model_4200.pt \
  --viewer viser
```

```sh
MUJOCO_GL=egl uv run play Mjlab-Tennis-Hit-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit/tennis_cloud_tennis_B_curr_quarter_2026-05-18_22-02-07/model_29999.pt \
  --video True \
  --video-length 2000
```

```sh
tensorboard --logdir logs/rsl_rl/g1_tennis_latent_hit/
```

### Cross

```sh
uv run play Mjlab-Tennis-Cross-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_59998.pt \
  --viewer viser
```

```sh
uv run play Mjlab-Tennis-Cross-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_40500.pt \
  --viewer viser
```

```sh
MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross/tennis_cross_from_hit_2026-05-21_15-21-12/model_41000.pt \
  --video True \
  --video-length 2000
```

### LAB

```sh
uv run train Mjlab-Tennis-Hit-LAB-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run train Mjlab-Tennis-Cross-LAB-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run train Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run play Mjlab-Tennis-Hit-LAB-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit_lab/tennis_hit_lab_2026-05-24_01-49-58/model_29999.pt \
  --viewer viser
```

```sh
uv run play Mjlab-Tennis-Cross-LAB-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross_lab/old_tennis_cross_lab_2026-05-24_01-49-21/model_109996.pt \
  --viewer viser
```

```sh
uv run play Mjlab-Tennis-Cross-Wrist-LAB-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross_wrist_lab/tennis_cross_wrist_lab_from_cross_2026-05-26_15-07-20/model_73500.pt \
  --viewer viser
```

```sh
MUJOCO_GL=egl uv run play Mjlab-Tennis-Hit-LAB-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_hit_lab/tennis_hit_lab_2026-05-24_01-49-58/model_29999.pt \
  --video True \
  --video-length 2000
```

```sh
MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-LAB-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_cross_lab/tennis_cross_lab_2026-05-25_10-10-42/model_42000.pt \
  --viewer viser \
  --video True \
  --video-length 20000
```

### Continuous

```sh
uv run train Mjlab-Tennis-Continuous-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [4,5,6]
```

```sh
uv run play Mjlab-Tennis-Continuous-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_continuous/tennis_continuous_from_cross_2026-05-28_16-47-00/model_39999.pt \
  --viewer viser
```

```sh
MUJOCO_GL=egl uv run play Mjlab-Tennis-Continuous-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_latent_continuous/tennis_continuous_from_cross_2026-05-27_14-37-48/model_68500.pt \
  --video True \
  --video-length 2000
```

### SONIC

```sh
uv run train Mjlab-Tennis-Hit-SONIC-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run train Mjlab-Tennis-Cross-SONIC-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run train Mjlab-Tennis-Hit-SONIC-Encoder-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run train Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1 \
  --env.scene.num_envs 4096 \
  --gpu-ids [0,1]
```

```sh
uv run play Mjlab-Tennis-Cross-SONIC-Encoder-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_sonic_encoder_hit/tennis_hit_sonic_encoder_prior_2026-05-27_01-51-16/model_5000.pt \
  --viewer viser
```

```sh
MUJOCO_GL=egl uv run play Mjlab-Tennis-Cross-SONIC-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_tennis_sonic_hit/tennis_hit_sonic_token_2026-05-26_01-50-16/model_12000.pt \
  --viewer viser
```

Saved no-encoder baseline video:

```text
logs/rsl_rl/g1_tennis_sonic_hit/tennis_sonic_cross_wo_encoder_video_2026-05-27/videos/play/sonic-wo-encoder-step-0.mp4
```

## Velocity

```sh
uv run play Mjlab-Velocity-Stairs-Unitree-Go1 \
  --checkpoint-file logs/rsl_rl/go1_velocity/stairs_go1_2026-05-19_15-35-35/model_3400.pt \
  --viewer viser
```

## Distillation

```sh
uv run play-distill.py \
  --checkpoint-file logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt \
  --viewer viser
```


腿部前四个 p90 max_torque 120 max_vel 29 rad/s
