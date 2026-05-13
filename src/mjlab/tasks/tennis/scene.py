"""Scene helpers for the G1 tennis task.

Court layout
------------
The G1 robot is roughly 1.0 m tall (about 0.57x a 1.75 m human player), so the
court is scaled down from the singles regulation (23.77 m x 8.23 m, net center
0.914 m) by the same factor:

  * Total court length : 14.0 m   (each side: 7.0 m from net to baseline)
  * Singles court width: 4.8 m    (sidelines at y = +/-2.4 m)
  * Net center height  : 0.52 m
  * Net post height    : 0.61 m
  * Service line       : 3.65 m from the net (each side)

Robot side ("self") is x in (0, 7),   opponent side ("opp") is x in (-7, 0).
The net plane sits at x = 0; positive y is the deuce side.
"""

from __future__ import annotations

import mujoco

from mjlab.entity import EntityCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils import spec_config as spec_cfg

# ---------------------------------------------------------------------------
# Court geometry constants (re-exported for use by env config / curriculum).
# ---------------------------------------------------------------------------

# Per-side dimensions.
COURT_HALF_LENGTH = 7.0  # x extent of one side (net -> baseline)
COURT_HALF_WIDTH = 2.4  # y extent (singles half-width)
SERVICE_LINE_FROM_NET = 3.65  # x distance from net to service line

# Net.
NET_CENTER_HEIGHT = 0.52
NET_POST_HEIGHT = 0.61
NET_HALF_WIDTH = 2.55  # extends slightly beyond singles sidelines
NET_THICKNESS_HALF = 0.012

# Ball.
BALL_RADIUS = 0.0335
BALL_MASS = 0.057
BALL_INIT_POS = (0.0, 0.0, 0.0)
BALL_INIT_LIN_VEL = (0.0, 0.0, 0.0)

# Court line visual half-thickness.
_LINE_HALF_W = 0.025  # along-court direction (visual)
_LINE_HALF_H = 0.003  # vertical

# X positions of the baselines (each side).
BASELINE_SELF_X = COURT_HALF_LENGTH  # +7.0
BASELINE_OPP_X = -COURT_HALF_LENGTH  # -7.0


def get_tennis_ball_spec() -> mujoco.MjSpec:
  """Return a free tennis ball spec."""
  spec = mujoco.MjSpec()
  spec.add_material(name="tennis_ball_mat", rgba=(0.85, 1.0, 0.05, 1.0))

  body = spec.worldbody.add_body(name="tennis_ball_body")
  body.add_freejoint(name="tennis_ball_freejoint")
  geom = body.add_geom(
    name="tennis_ball",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(BALL_RADIUS,),
    mass=BALL_MASS,
    condim=3,
  )
  geom.material = "tennis_ball_mat"
  geom.friction[:] = (0.8, 0.005, 0.0001)
  geom.solref[:] = (0.004, 1.0)
  geom.solimp[:3] = (0.9, 0.95, 0.001)

  body.add_site(
    name="tennis_ball_center",
    size=(0.01,),
    rgba=(0.85, 1.0, 0.05, 1.0),
  )
  return spec


def get_tennis_ball_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_tennis_ball_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=BALL_INIT_POS,
      lin_vel=BALL_INIT_LIN_VEL,
    ),
  )


def get_tennis_court_spec() -> mujoco.MjSpec:
  """Build the court (surface, painted lines, net) as a fixed-base entity.

  The court is centered on the world origin so the net plane is at x = 0.
  Per-side regions are symmetric: x in (0, 7) on the robot side and
  x in (-7, 0) on the opponent side.
  """
  spec = mujoco.MjSpec()
  spec.add_material(name="court_green", rgba=(0.13, 0.42, 0.22, 1.0))
  spec.add_material(name="court_line_mat", rgba=(0.95, 0.95, 0.95, 1.0))
  spec.add_material(name="tennis_net_mat", rgba=(0.05, 0.05, 0.05, 0.6))
  spec.add_material(name="tennis_net_band_mat", rgba=(0.95, 0.95, 0.95, 1.0))
  spec.add_material(name="tennis_net_post_mat", rgba=(0.2, 0.2, 0.2, 1.0))
  spec.add_material(name="tennis_target_mat", rgba=(0.2, 0.45, 1.0, 0.30))

  body = spec.worldbody.add_body(name="tennis_court_body")

  def add_box(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    material: str,
    *,
    collidable: bool = False,
  ) -> None:
    geom = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
    )
    geom.material = material
    geom.contype = 1 if collidable else 0
    geom.conaffinity = 1 if collidable else 0

  # --- Court surface (visual only; ground is provided by the terrain plane) -
  add_box(
    "court_visual",
    pos=(0.0, 0.0, 0.002),
    size=(COURT_HALF_LENGTH, COURT_HALF_WIDTH, 0.002),
    material="court_green",
  )

  # --- Painted lines ------------------------------------------------------
  # Baselines (perpendicular to the long axis, x = +/-7).
  add_box(
    "court_baseline_self",
    pos=(BASELINE_SELF_X, 0.0, 0.006),
    size=(_LINE_HALF_W, COURT_HALF_WIDTH, _LINE_HALF_H),
    material="court_line_mat",
  )
  add_box(
    "court_baseline_opp",
    pos=(BASELINE_OPP_X, 0.0, 0.006),
    size=(_LINE_HALF_W, COURT_HALF_WIDTH, _LINE_HALF_H),
    material="court_line_mat",
  )
  # Sidelines (parallel to long axis, y = +/-2.4).
  add_box(
    "court_sideline_left",
    pos=(0.0, COURT_HALF_WIDTH, 0.006),
    size=(COURT_HALF_LENGTH, _LINE_HALF_W, _LINE_HALF_H),
    material="court_line_mat",
  )
  add_box(
    "court_sideline_right",
    pos=(0.0, -COURT_HALF_WIDTH, 0.006),
    size=(COURT_HALF_LENGTH, _LINE_HALF_W, _LINE_HALF_H),
    material="court_line_mat",
  )
  # Service lines (x = +/-3.65).
  add_box(
    "court_service_self",
    pos=(SERVICE_LINE_FROM_NET, 0.0, 0.006),
    size=(_LINE_HALF_W, COURT_HALF_WIDTH, _LINE_HALF_H),
    material="court_line_mat",
  )
  add_box(
    "court_service_opp",
    pos=(-SERVICE_LINE_FROM_NET, 0.0, 0.006),
    size=(_LINE_HALF_W, COURT_HALF_WIDTH, _LINE_HALF_H),
    material="court_line_mat",
  )
  # Centre service line (between the two service lines).
  add_box(
    "court_centre_service",
    pos=(0.0, 0.0, 0.006),
    size=(SERVICE_LINE_FROM_NET, _LINE_HALF_W, _LINE_HALF_H),
    material="court_line_mat",
  )

  # --- Net ---------------------------------------------------------------
  # Collidable net mesh (single thin box approximating the cloth volume).
  add_box(
    "tennis_net_collision",
    pos=(0.0, 0.0, NET_CENTER_HEIGHT * 0.5),
    size=(NET_THICKNESS_HALF, NET_HALF_WIDTH, NET_CENTER_HEIGHT * 0.5),
    material="tennis_net_mat",
    collidable=True,
  )
  # White top band (visual).
  add_box(
    "tennis_net_top_band",
    pos=(0.0, 0.0, NET_CENTER_HEIGHT),
    size=(NET_THICKNESS_HALF * 1.5, NET_HALF_WIDTH + 0.02, 0.018),
    material="tennis_net_band_mat",
  )
  # Net posts (visual cylinders just outside the singles sideline).
  for tag, y_post in (("left", NET_HALF_WIDTH), ("right", -NET_HALF_WIDTH)):
    post = body.add_geom(
      name=f"tennis_net_post_{tag}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=(0.0, y_post, NET_POST_HEIGHT * 0.5),
      size=(0.03, NET_POST_HEIGHT * 0.5, 0.0),
    )
    post.material = "tennis_net_post_mat"
    post.contype = 0
    post.conaffinity = 0

  # --- Optional landing-target hint (visual only) ------------------------
  target = body.add_geom(
    name="target_landing_region",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    pos=(-SERVICE_LINE_FROM_NET, 0.0, 0.012),
    size=(1.2, 0.004, 0.0),
  )
  target.material = "tennis_target_mat"
  target.contype = 0
  target.conaffinity = 0

  # --- Semantic reference sites (consumed by event detector / curricula) -
  def add_marker(name: str, pos: tuple[float, float, float]) -> None:
    body.add_site(name=name, pos=pos, size=(0.025,), rgba=(1.0, 1.0, 1.0, 0.7))

  add_marker("net_top_center", (0.0, 0.0, NET_CENTER_HEIGHT))
  add_marker("baseline_self", (BASELINE_SELF_X, 0.0, 0.02))
  add_marker("baseline_opp", (BASELINE_OPP_X, 0.0, 0.02))
  add_marker("sideline_left", (0.0, COURT_HALF_WIDTH, 0.02))
  add_marker("sideline_right", (0.0, -COURT_HALF_WIDTH, 0.02))
  add_marker(
    "service_self_deuce", (SERVICE_LINE_FROM_NET * 0.5, COURT_HALF_WIDTH * 0.5, 0.02)
  )
  add_marker(
    "service_self_ad", (SERVICE_LINE_FROM_NET * 0.5, -COURT_HALF_WIDTH * 0.5, 0.02)
  )
  add_marker(
    "service_opp_deuce",
    (-SERVICE_LINE_FROM_NET * 0.5, -COURT_HALF_WIDTH * 0.5, 0.02),
  )
  add_marker(
    "service_opp_ad", (-SERVICE_LINE_FROM_NET * 0.5, COURT_HALF_WIDTH * 0.5, 0.02)
  )
  return spec


def get_tennis_court_cfg() -> EntityCfg:
  """Return a fixed court entity that can be reset to each env origin."""
  return EntityCfg(spec_fn=get_tennis_court_spec)


def get_tennis_terrain_cfg() -> TerrainEntityCfg:
  """Return a green checker plane matching the standalone MJCF scene."""
  return TerrainEntityCfg(
    terrain_type="plane",
    textures=(
      spec_cfg.TextureCfg(
        name="tennis_court_texture",
        type="2d",
        builtin="checker",
        mark="edge",
        rgb1=(0.13, 0.42, 0.22),
        rgb2=(0.10, 0.32, 0.18),
        markrgb=(0.9, 0.9, 0.9),
        width=300,
        height=300,
      ),
    ),
    materials=(
      spec_cfg.MaterialCfg(
        name="tennis_court",
        texuniform=True,
        texrepeat=(2.0, 2.0),
        reflectance=0.0,
        texture="tennis_court_texture",
        geom_names_expr=("terrain$",),
      ),
    ),
  )
