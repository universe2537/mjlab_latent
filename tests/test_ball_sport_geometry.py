from types import SimpleNamespace
from typing import Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from mjlab.tasks.ball_sports import BallSportGeometryCfg, resolve_ball_sport_geometry
from mjlab.tasks.tennis.scene import (
  BALL_RADIUS,
  COURT_HALF_LENGTH,
  COURT_HALF_WIDTH,
  NET_CENTER_HEIGHT,
  get_tennis_ball_cfg,
  get_tennis_court_cfg,
)


def _make_tennis_env(num_envs: int = 2, scale: float = 0.5) -> Any:
  scene_cfg = SceneCfg(
    num_envs=num_envs,
    env_spacing=8.0,
    entities={
      "ball": get_tennis_ball_cfg(),
      "court": get_tennis_court_cfg(scale=scale),
    },
    extent=4.0,
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


def test_tennis_scene_resolves_ball_sport_geometry_without_table_top() -> None:
  scale = 0.5
  env = _make_tennis_env(num_envs=2, scale=scale)

  geometry = resolve_ball_sport_geometry(
    env,
    BallSportGeometryCfg(
      ball_geom_cfg=SceneEntityCfg("ball", geom_names="tennis_ball"),
      play_area_cfg=SceneEntityCfg("court", geom_names="court_visual"),
      net_cfg=SceneEntityCfg("court", geom_names="tennis_net_collision"),
      landing_z_override=0.0,
    ),
  )

  torch.testing.assert_close(geometry.net_x, torch.zeros(2), atol=1.0e-6, rtol=0.0)
  torch.testing.assert_close(
    geometry.self_bounds.x_min,
    torch.zeros(2),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.self_bounds.x_max,
    torch.full((2,), COURT_HALF_LENGTH * scale),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.opponent_bounds.x_min,
    torch.full((2,), -COURT_HALF_LENGTH * scale),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.self_bounds.y_max,
    torch.full((2,), COURT_HALF_WIDTH * scale),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.net_top_z,
    torch.full((2,), NET_CENTER_HEIGHT),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.ball_radius,
    torch.full((2,), BALL_RADIUS),
    atol=1.0e-6,
    rtol=0.0,
  )
  torch.testing.assert_close(
    geometry.landing_z,
    torch.zeros(2),
    atol=1.0e-6,
    rtol=0.0,
  )
  assert geometry.surface_friction.shape == (2, 3)
  assert geometry.net_solref.shape == (2, 2)
