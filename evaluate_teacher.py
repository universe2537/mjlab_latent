from dataclasses import asdict
from pathlib import Path
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

teacher_task = "Mjlab-Tracking-Flat-Unitree-G1"
distill_task = "Mjlab-Distill-Flat-Unitree-G1"
checkpoint = Path("./logs/rsl_rl/g1_tracking/2026-05-09_18-20-24/model_29999.pt")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_envs = 64
num_steps = 200

disable_push_robot = False
disable_obs_corruption = False
disable_wrist_bias = True

env_cfg = load_env_cfg(distill_task, play=False)
env_cfg.scene.num_envs = num_envs

if disable_push_robot:
  env_cfg.events.pop("push_robot", None)
if disable_obs_corruption:
  env_cfg.observations["actor"].enable_corruption = False
if disable_wrist_bias:
  env_cfg.events.pop("wrist_encoder_bias", None)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
vec = RslRlVecEnvWrapper(env, clip_actions=None)

agent_cfg = load_rl_cfg(teacher_task)
runner_cls = load_runner_cls(teacher_task)
runner = runner_cls(vec, asdict(agent_cfg), device=device)
runner.load(
  str(checkpoint),
  load_cfg={"actor": True},
  strict=True,
  map_location=device,
)
policy = runner.get_inference_policy(device=device)

obs = vec.get_observations()
done_total = 0
terminated_total = 0
timeout_total = 0
reward_sum = 0.0

for step in range(num_steps):
  with torch.no_grad():
    action = policy(obs)
  obs, rew, dones, infos = vec.step(action)

  reward_sum += rew.mean().item()
  done_total += int(dones.sum().item())
  terminated_total += int(vec.unwrapped.termination_manager.terminated.sum().item())
  timeout_total += int(vec.unwrapped.termination_manager.time_outs.sum().item())

  if step % 50 == 0:
    print(
      f"step={step} "
      f"reward_mean={rew.mean().item():.4f} "
      f"dones={int(dones.sum().item())}"
    )

print()
print("summary")
print("avg_reward =", reward_sum / num_steps)
print("done_total =", done_total)
print("terminated_total =", terminated_total)
print("timeout_total =", timeout_total)
