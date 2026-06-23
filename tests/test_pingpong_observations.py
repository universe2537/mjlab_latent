from types import SimpleNamespace
from typing import Any

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.pingpong.mdp.ball_providers import (
  resolve_pingpong_ball_sport_geometry,
)
from mjlab.tasks.pingpong.mdp.observations import ball_predicted_edge_hit_point_b
from mjlab.tasks.pingpong.scene import get_pingpong_ball_cfg, get_pingpong_table_cfg


def _robot_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="robot_body")
  body.add_freejoint(name="robot_freejoint")
  body.add_geom(
    name="robot_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.05, 0.05, 0.05),
    mass=1.0,
  )
  return spec


def _make_env() -> Any:
  scene_cfg = SceneCfg(
    num_envs=1,
    env_spacing=4.0,
    entities={
      "robot": EntityCfg(spec_fn=_robot_spec),
      "ball": get_pingpong_ball_cfg(),
      "table": get_pingpong_table_cfg(),
    },
    extent=3.0,
  )
  scene = Scene(scene_cfg, device="cpu")
  model = scene.compile()
  sim = Simulation(
    num_envs=1,
    cfg=SimulationCfg(nconmax=16, njmax=32),
    model=model,
    device="cpu",
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return SimpleNamespace(num_envs=1, device="cpu", scene=scene, sim=sim)


def _write_ball(env: Any, pos: torch.Tensor, vel: torch.Tensor) -> None:
  ball = env.scene["ball"]
  pose = torch.zeros(1, 7)
  pose[:, :3] = pos
  pose[:, 3] = 1.0
  twist = torch.zeros(1, 6)
  twist[:, :3] = vel
  ball.write_root_link_pose_to_sim(pose)
  ball.write_root_link_velocity_to_sim(twist)
  env.sim.forward()


def test_pingpong_edge_hit_point_targets_scene_derived_end_line() -> None:
  env = _make_env()
  geometry = resolve_pingpong_ball_sport_geometry(env)
  edge_x = geometry.self_baseline_x[0]
  bounce_z = geometry.bounce_z[0]
  start_x = torch.tensor(0.65)
  t_edge = torch.tensor(0.5)
  vx = (edge_x - start_x) / t_edge
  target_edge_z = bounce_z + 0.05
  vz = (target_edge_z - bounce_z + 0.5 * 9.81 * t_edge * t_edge) / t_edge
  _write_ball(
    env,
    torch.tensor([[start_x, 0.0, bounce_z]], dtype=torch.float32),
    torch.tensor([[vx, 0.0, vz]], dtype=torch.float32),
  )

  obs = ball_predicted_edge_hit_point_b(env)

  torch.testing.assert_close(obs[0, 0], edge_x, atol=1.0e-5, rtol=0.0)
  torch.testing.assert_close(obs[0, 2], target_edge_z, atol=1.0e-5, rtol=0.0)
  torch.testing.assert_close(obs[0, 3], t_edge, atol=1.0e-5, rtol=0.0)


def test_pingpong_edge_hit_point_fallback_is_finite() -> None:
  env = _make_env()
  _write_ball(
    env,
    torch.tensor([[0.2, 0.0, 1.2]], dtype=torch.float32),
    torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32),
  )

  obs = ball_predicted_edge_hit_point_b(env)

  assert obs.shape == (1, 4)
  assert torch.isfinite(obs).all()
