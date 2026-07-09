from types import SimpleNamespace
from typing import Any

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.pingpong.bounce import (
  PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE,
  PINGPONG_POST_BOUNCE_VERTICAL_SCALE,
)
from mjlab.tasks.pingpong.mdp.ball_providers import (
  resolve_pingpong_ball_sport_geometry,
)
from mjlab.tasks.pingpong.mdp.observations import ball_predicted_edge_hit_point_b
from mjlab.tasks.pingpong.scene import (
  BALL_CENTER_TABLE_Z,
  get_pingpong_ball_cfg,
  get_pingpong_table_cfg,
)


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


def _measure_first_table_bounce(
  model: mujoco.MjModel,
  *,
  pos: tuple[float, float, float],
  vel: tuple[float, float, float],
) -> tuple[float, float]:
  data = mujoco.MjData(model)
  joint_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "ball/pingpong_ball_freejoint",
  )
  assert joint_id >= 0
  qposadr = int(model.jnt_qposadr[joint_id])
  dofadr = int(model.jnt_dofadr[joint_id])
  data.qpos[qposadr : qposadr + 3] = pos
  data.qpos[qposadr + 3 : qposadr + 7] = (1.0, 0.0, 0.0, 0.0)
  data.qvel[dofadr : dofadr + 3] = vel
  mujoco.mj_forward(model, data)

  gravity = -float(model.opt.gravity[2])
  pre_h = (vel[0] * vel[0] + vel[1] * vel[1]) ** 0.5
  impact_vz = -(
    vel[2] * vel[2] + 2.0 * gravity * (pos[2] - BALL_CENTER_TABLE_Z)
  ) ** 0.5
  prev_vz = vel[2]
  seen_rebound = False
  max_post_vz = 0.0
  post_h_at_peak = 0.0
  for _ in range(1500):
    mujoco.mj_step(model, data)
    cur_vz = float(data.qvel[dofadr + 2])
    cur_z = float(data.qpos[qposadr + 2])
    if prev_vz < 0.0 and cur_vz > 0.0 and abs(cur_z - BALL_CENTER_TABLE_Z) < 0.08:
      seen_rebound = True
    if seen_rebound and cur_vz > max_post_vz:
      max_post_vz = cur_vz
      post_h_at_peak = (
        float(data.qvel[dofadr]) ** 2 + float(data.qvel[dofadr + 1]) ** 2
      ) ** 0.5
    if seen_rebound and cur_z > BALL_CENTER_TABLE_Z + 0.08:
      return post_h_at_peak / pre_h, max_post_vz / (-impact_vz)
    prev_vz = cur_vz
  raise AssertionError("Did not observe a table bounce.")


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


def test_pingpong_table_bounce_profile_matches_shared_prediction_constants() -> None:
  env = _make_env()
  ratios = torch.tensor(
    [
      _measure_first_table_bounce(
        env.sim.mj_model,
        pos=(0.45, -0.20, 1.18),
        vel=(2.6, 0.20, -1.00),
      ),
      _measure_first_table_bounce(
        env.sim.mj_model,
        pos=(0.30, 0.15, 1.10),
        vel=(3.2, -0.10, -0.60),
      ),
      _measure_first_table_bounce(
        env.sim.mj_model,
        pos=(0.10, 0.00, 1.24),
        vel=(2.2, 0.30, -1.20),
      ),
    ],
    dtype=torch.float32,
  )
  mean = ratios.mean(dim=0)
  std = ratios.std(dim=0)

  assert torch.all(std < torch.tensor([0.06, 0.06]))
  torch.testing.assert_close(
    mean[0],
    torch.tensor(PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE),
    atol=0.05,
    rtol=0.0,
  )
  torch.testing.assert_close(
    mean[1],
    torch.tensor(PINGPONG_POST_BOUNCE_VERTICAL_SCALE),
    atol=0.05,
    rtol=0.0,
  )
