from types import SimpleNamespace
from typing import Any

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.pingpong.mdp.ball_providers import (
  TableTennisFeeder,
  TableTennisFeederCfg,
  TableTennisSample,
  TrajectoryCheckCfg,
  check_candidate_trajectory,
  resolve_pingpong_ball_sport_geometry,
)
from mjlab.tasks.pingpong.scene import (
  BALL_RADIUS,
  NET_TOP_Z,
  TABLE_HALF_LENGTH,
  TABLE_HALF_WIDTH,
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


def make_pingpong_provider_env(num_envs: int = 4) -> Any:
  scene_cfg = SceneCfg(
    num_envs=num_envs,
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
    num_envs=num_envs,
    cfg=SimulationCfg(nconmax=16, njmax=32),
    model=model,
    device="cpu",
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene, sim=sim)


def test_pingpong_ball_sport_geometry_reads_scene_model() -> None:
  env = make_pingpong_provider_env(num_envs=2)

  geometry = resolve_pingpong_ball_sport_geometry(env)

  torch.testing.assert_close(
    geometry.self_bounds.x_max,
    torch.full((2,), TABLE_HALF_LENGTH),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.self_bounds.y_max,
    torch.full((2,), TABLE_HALF_WIDTH),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.opponent_bounds.x_min,
    torch.full((2,), -TABLE_HALF_LENGTH),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.net_top_z,
    torch.full((2,), NET_TOP_Z),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.ball_radius,
    torch.full((2,), BALL_RADIUS),
    atol=1.0e-6,
    rtol=0.0,
  )
  assert geometry.surface_friction.shape == (2, 3)
  assert geometry.surface_solref.shape == (2, 2)
  assert geometry.surface_solimp.shape[0] == 2
  assert torch.isfinite(geometry.ball_solref).all()
  assert torch.isfinite(geometry.net_friction).all()


def test_pingpong_provider_samples_scene_derived_first_bounce_feeds() -> None:
  env = make_pingpong_provider_env(num_envs=64)
  cfg = TableTennisFeederCfg(
    target_x_range=(0.30, 0.12),
    target_y_range=(-0.13, 0.13),
    check=TrajectoryCheckCfg(
      require_edge_crossing=True,
      require_second_bounce_outside_self_half=True,
      flight_time_range=(0.32, 0.75),
    ),
  )
  provider = TableTennisFeeder(cfg, env)
  env_ids = torch.arange(env.num_envs)

  sample = provider._sample_candidate(env_ids)
  geometry = resolve_pingpong_ball_sport_geometry(env).select(env_ids)

  assert sample.valid.float().mean() > 0.65
  assert torch.all(sample.px < geometry.net_x)
  assert torch.all(sample.tx > geometry.net_x)
  assert torch.all(sample.tx <= geometry.self_bounds.x_max)
  assert torch.all(sample.py >= geometry.opponent_bounds.y_min)
  assert torch.all(sample.py <= geometry.opponent_bounds.y_max)
  assert torch.all(sample.ty >= geometry.self_bounds.y_min)
  assert torch.all(sample.ty <= geometry.self_bounds.y_max)

  t_net = (geometry.net_x - sample.px) / sample.vx
  z_net = sample.pz + sample.vz * t_net - 0.5 * cfg.gravity * t_net * t_net
  assert torch.all(z_net[sample.valid] >= geometry.net_top_z[sample.valid] + 0.06)

  pred_tx = sample.px + sample.vx * sample.flight_t
  pred_ty = sample.py + sample.vy * sample.flight_t
  torch.testing.assert_close(pred_tx, sample.tx, atol=1.0e-5, rtol=1.0e-5)
  torch.testing.assert_close(pred_ty, sample.ty, atol=1.0e-5, rtol=1.0e-5)


def test_pingpong_trajectory_checker_rejects_invalid_long_ball_cases() -> None:
  env = make_pingpong_provider_env(num_envs=3)
  geometry = resolve_pingpong_ball_sport_geometry(env)
  cfg = TableTennisFeederCfg(
    check=TrajectoryCheckCfg(
      require_edge_crossing=True,
      require_second_bounce_outside_self_half=True,
      flight_time_range=(0.32, 0.75),
      vx_range=(2.0, 8.0),
    )
  )
  bounce_z = geometry.bounce_z
  sample = TableTennisSample(
    px=torch.tensor([-0.7, -0.7, -0.7]),
    py=torch.zeros(3),
    pz=bounce_z + 0.002,
    vx=torch.tensor([4.0, 4.0, 1.0]),
    vy=torch.zeros(3),
    vz=torch.tensor([0.5, 1.9, 1.9]),
    tx=torch.tensor([1.05, 0.30, 1.05]),
    ty=torch.zeros(3),
    flight_t=torch.tensor([0.44, 0.44, 0.44]),
    valid=torch.ones(3, dtype=torch.bool),
  )

  valid = check_candidate_trajectory(sample, geometry, cfg)

  assert not valid[0]  # below the net
  assert not valid[1]  # second bounce remains on the table / misses edge height
  assert not valid[2]  # horizontal speed too slow
