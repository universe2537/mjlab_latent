"""Programmatic table-tennis scene assets."""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
TABLE_HALF_LENGTH = TABLE_LENGTH * 0.5
TABLE_HALF_WIDTH = TABLE_WIDTH * 0.5
TABLE_HEIGHT = 0.76
TABLE_TOP_HALF_THICKNESS = 0.025
NET_X = 0.0
NET_HEIGHT = 0.1525
NET_TOP_Z = TABLE_HEIGHT + NET_HEIGHT
NET_HALF_WIDTH = TABLE_HALF_WIDTH + 0.04
NET_THICKNESS_HALF = 0.006

BALL_RADIUS = 0.02
BALL_MASS = 0.0027
BALL_CENTER_TABLE_Z = TABLE_HEIGHT + BALL_RADIUS
BALL_INIT_POS = (0.0, 0.0, BALL_CENTER_TABLE_Z)
BALL_INIT_LIN_VEL = (0.0, 0.0, 0.0)

_LINE_HALF_W = 0.01
_LINE_Z = TABLE_HEIGHT + 0.002


def get_pingpong_ball_spec() -> mujoco.MjSpec:
  """Return a free-flying 40 mm table-tennis ball."""
  spec = mujoco.MjSpec()
  spec.add_material(name="pingpong_ball_mat", rgba=(1.0, 0.58, 0.18, 1.0))

  body = spec.worldbody.add_body(name="pingpong_ball_body")
  body.add_freejoint(name="pingpong_ball_freejoint")
  geom = body.add_geom(
    name="pingpong_ball",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS,),
    mass=BALL_MASS,
    condim=3,
  )
  geom.material = "pingpong_ball_mat"
  geom.friction[:] = (0.04, 0.002, 0.0001)
  geom.solref[:] = (0.002, 0.50)
  geom.solimp[:3] = (0.93, 0.98, 0.001)

  body.add_site(
    name="pingpong_ball_center",
    size=(0.006,),
    rgba=(1.0, 0.58, 0.18, 1.0),
  )
  return spec


def get_pingpong_ball_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_pingpong_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=BALL_INIT_POS,
      lin_vel=BALL_INIT_LIN_VEL,
    ),
  )


def get_pingpong_table_spec() -> mujoco.MjSpec:
  """Build a fixed-base table, net, legs, and visual court lines."""
  spec = mujoco.MjSpec()
  spec.add_material(name="pingpong_table_blue", rgba=(0.04, 0.20, 0.43, 1.0))
  spec.add_material(name="pingpong_line_white", rgba=(0.95, 0.95, 0.9, 1.0))
  spec.add_material(name="pingpong_net_mat", rgba=(0.05, 0.05, 0.05, 0.65))
  spec.add_material(name="pingpong_net_band_mat", rgba=(0.96, 0.96, 0.9, 1.0))
  spec.add_material(name="pingpong_leg_mat", rgba=(0.15, 0.15, 0.16, 1.0))
  spec.add_material(name="pingpong_target_mat", rgba=(0.9, 0.3, 0.08, 0.25))

  body = spec.worldbody.add_body(name="pingpong_table_body")

  def add_box(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    material: str,
    *,
    collidable: bool = False,
  ) -> mujoco.MjsGeom:
    geom = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
    )
    geom.material = material
    geom.contype = 1 if collidable else 0
    geom.conaffinity = 1 if collidable else 0
    return geom

  top = add_box(
    "pingpong_table_top_collision",
    pos=(0.0, 0.0, TABLE_HEIGHT - TABLE_TOP_HALF_THICKNESS),
    size=(TABLE_HALF_LENGTH, TABLE_HALF_WIDTH, TABLE_TOP_HALF_THICKNESS),
    material="pingpong_table_blue",
    collidable=True,
  )
  top.friction[:] = (0.04, 0.002, 0.0001)
  top.solref[:] = (0.002, 0.50)
  top.solimp[:3] = (0.93, 0.98, 0.001)

  # White outline and center line, visual only.
  add_box(
    "pingpong_baseline_self",
    (TABLE_HALF_LENGTH, 0.0, _LINE_Z),
    (_LINE_HALF_W, TABLE_HALF_WIDTH, 0.002),
    "pingpong_line_white",
  )
  add_box(
    "pingpong_baseline_opp",
    (-TABLE_HALF_LENGTH, 0.0, _LINE_Z),
    (_LINE_HALF_W, TABLE_HALF_WIDTH, 0.002),
    "pingpong_line_white",
  )
  add_box(
    "pingpong_sideline_left",
    (0.0, TABLE_HALF_WIDTH, _LINE_Z),
    (TABLE_HALF_LENGTH, _LINE_HALF_W, 0.002),
    "pingpong_line_white",
  )
  add_box(
    "pingpong_sideline_right",
    (0.0, -TABLE_HALF_WIDTH, _LINE_Z),
    (TABLE_HALF_LENGTH, _LINE_HALF_W, 0.002),
    "pingpong_line_white",
  )
  add_box(
    "pingpong_center_line",
    (0.0, 0.0, _LINE_Z + 0.001),
    (TABLE_HALF_LENGTH, 0.004, 0.0015),
    "pingpong_line_white",
  )

  net = add_box(
    "pingpong_net_collision",
    pos=(NET_X, 0.0, TABLE_HEIGHT + NET_HEIGHT * 0.5),
    size=(NET_THICKNESS_HALF, NET_HALF_WIDTH, NET_HEIGHT * 0.5),
    material="pingpong_net_mat",
    collidable=True,
  )
  net.friction[:] = (0.8, 0.005, 0.0001)
  net.solref[:] = (0.002, 0.8)
  net.solimp[:3] = (0.9, 0.95, 0.001)
  add_box(
    "pingpong_net_top_band",
    pos=(NET_X, 0.0, NET_TOP_Z),
    size=(NET_THICKNESS_HALF * 1.5, NET_HALF_WIDTH + 0.01, 0.008),
    material="pingpong_net_band_mat",
  )

  for x in (-TABLE_HALF_LENGTH * 0.82, TABLE_HALF_LENGTH * 0.82):
    for y in (-TABLE_HALF_WIDTH * 0.82, TABLE_HALF_WIDTH * 0.82):
      leg = body.add_geom(
        name=f"pingpong_leg_{'self' if x > 0 else 'opp'}_{'left' if y > 0 else 'right'}",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=(x, y, TABLE_HEIGHT * 0.5),
        size=(0.018, TABLE_HEIGHT * 0.5, 0.0),
      )
      leg.material = "pingpong_leg_mat"
      leg.contype = 0
      leg.conaffinity = 0

  target = body.add_geom(
    name="pingpong_target_landing_region",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=(-TABLE_HALF_LENGTH * 0.45, 0.0, TABLE_HEIGHT + 0.004),
    size=(0.22, 0.003, 0.0),
  )
  target.material = "pingpong_target_mat"
  target.contype = 0
  target.conaffinity = 0

  def add_marker(name: str, pos: tuple[float, float, float]) -> None:
    body.add_site(name=name, pos=pos, size=(0.012,), rgba=(1.0, 1.0, 1.0, 0.7))

  add_marker("pingpong_net_top_center", (NET_X, 0.0, NET_TOP_Z))
  add_marker("pingpong_baseline_self", (TABLE_HALF_LENGTH, 0.0, TABLE_HEIGHT))
  add_marker("pingpong_baseline_opp", (-TABLE_HALF_LENGTH, 0.0, TABLE_HEIGHT))
  add_marker("pingpong_sideline_left", (0.0, TABLE_HALF_WIDTH, TABLE_HEIGHT))
  add_marker("pingpong_sideline_right", (0.0, -TABLE_HALF_WIDTH, TABLE_HEIGHT))
  return spec


def get_pingpong_table_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=get_pingpong_table_spec)


def get_pingpong_terrain_cfg() -> TerrainEntityCfg:
  """Return a simple neutral floor under the table."""
  return TerrainEntityCfg(
    terrain_type="plane",
    textures=(
      spec_cfg.TextureCfg(
        name="pingpong_floor_texture",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=(0.20, 0.22, 0.23),
        rgb2=(0.16, 0.17, 0.18),
        markrgb=(0.35, 0.35, 0.35),
        width=300,
        height=300,
      ),
    ),
    materials=(
      spec_cfg.MaterialCfg(
        name="pingpong_floor",
        texuniform=True,
        texrepeat=(2.0, 2.0),
        reflectance=0.0,
        texture="pingpong_floor_texture",
        geom_names_expr=("terrain$",),
      ),
    ),
  )


__all__ = [
  "BALL_CENTER_TABLE_Z",
  "BALL_MASS",
  "BALL_RADIUS",
  "NET_HEIGHT",
  "NET_TOP_Z",
  "NET_X",
  "TABLE_HALF_LENGTH",
  "TABLE_HALF_WIDTH",
  "TABLE_HEIGHT",
  "get_pingpong_ball_cfg",
  "get_pingpong_ball_spec",
  "get_pingpong_table_cfg",
  "get_pingpong_table_spec",
  "get_pingpong_terrain_cfg",
]
