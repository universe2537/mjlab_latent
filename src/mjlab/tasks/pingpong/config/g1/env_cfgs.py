"""Unitree G1 configuration for table-tennis latent-control tasks."""

from __future__ import annotations

import mujoco

from mjlab.asset_zoo.robots import G1_W_RACKET_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  FULL_COLLISION,
  G1_ARTICULATION,
  KNEES_BENT_KEYFRAME,
)
from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  get_spec as get_g1_w_racket_spec,
)
from mjlab.entity import EntityCfg
from mjlab.tasks.pingpong.pingpong_env_cfg import (
  make_pingpong_latent_env_cfg,
  make_pingpong_latent_return_env_cfg,
)
from mjlab.tasks.pingpong.scene import get_pingpong_ball_cfg, get_pingpong_table_cfg
from mjlab.tasks.tennis.mdp import FrozenDecoderLatentJointPositionActionCfg

DEFAULT_DECODER_CHECKPOINT = "logs/rsl_rl/g1_distillation/distill_cloud_unitree_racket_tennis_2026-05-12_09-35-14/model_30000.pt"
PINGPONG_PADDLE_RADIUS = 0.065
PINGPONG_PADDLE_SCALE = PINGPONG_PADDLE_RADIUS / 0.12
_TENNIS_RACKET_COLLISION_POS = (0.1025, -0.004, 0.4)


def _iter_body_tree(body: mujoco.MjsBody):
  yield body
  for child in body.bodies:
    yield from _iter_body_tree(child)


def _find_geom(spec: mujoco.MjSpec, name: str) -> mujoco.MjsGeom:
  for body in _iter_body_tree(spec.worldbody):
    for geom in body.geoms:
      if geom.name == name:
        return geom
  raise ValueError(f"Could not find geom {name!r} in G1 racket spec.")


def _find_geoms_by_mesh(spec: mujoco.MjSpec, meshname: str) -> list[mujoco.MjsGeom]:
  geoms = []
  for body in _iter_body_tree(spec.worldbody):
    for geom in body.geoms:
      if geom.meshname == meshname:
        geoms.append(geom)
  if not geoms:
    raise ValueError(f"Could not find geom using mesh {meshname!r}.")
  return geoms


def _find_mesh(spec: mujoco.MjSpec, name: str) -> mujoco.MjsMesh:
  for mesh in spec.meshes:
    if mesh.name == name:
      return mesh
  raise ValueError(f"Could not find mesh {name!r} in G1 racket spec.")


def _find_site(spec: mujoco.MjSpec, name: str) -> mujoco.MjsSite:
  for body in _iter_body_tree(spec.worldbody):
    for site in body.sites:
      if site.name == name:
        return site
  raise ValueError(f"Could not find site {name!r} in G1 racket spec.")


def _scale_from_anchor(
  pos: tuple[float, float, float],
  anchor: tuple[float, float, float],
  scale: float,
) -> tuple[float, float, float]:
  return (
    anchor[0] + (pos[0] - anchor[0]) * scale,
    anchor[1] + (pos[1] - anchor[1]) * scale,
    anchor[2] + (pos[2] - anchor[2]) * scale,
  )


def get_g1_w_pingpong_paddle_spec() -> mujoco.MjSpec:
  """Reuse the held-racket G1 XML with a smaller pingpong paddle."""
  spec = get_g1_w_racket_spec()

  visual_mesh = _find_mesh(spec, "tennis_racket")
  visual_mesh.name = "pingpong_paddle_visual"
  visual_mesh.scale[:] = tuple(v * PINGPONG_PADDLE_SCALE for v in visual_mesh.scale)

  visual_geoms = _find_geoms_by_mesh(spec, "tennis_racket")
  visual_anchor = (
    float(visual_geoms[0].pos[0]),
    float(visual_geoms[0].pos[1]),
    float(visual_geoms[0].pos[2]),
  )
  for idx, geom in enumerate(visual_geoms):
    geom.meshname = "pingpong_paddle_visual"
    geom.name = "pingpong_paddle_visual" if idx == 0 else f"pingpong_paddle_visual_{idx}"

  paddle_center = _scale_from_anchor(
    _TENNIS_RACKET_COLLISION_POS,
    visual_anchor,
    PINGPONG_PADDLE_SCALE,
  )
  paddle = _find_geom(spec, "tennis_racket_collision")
  paddle.name = "pingpong_paddle_collision"
  paddle.size[0] = PINGPONG_PADDLE_RADIUS
  paddle.size[1] = 0.004
  paddle.pos[:] = paddle_center
  paddle.rgba[:] = (0.85, 0.12, 0.06, 0.35)

  center = _find_site(spec, "tennis_racket_center")
  center.name = "pingpong_paddle_center"
  center.pos[:] = paddle_center
  center.size[0] = 0.01
  return spec


def get_g1_w_pingpong_paddle_robot_cfg() -> EntityCfg:
  """Return a G1 robot cfg with a table-tennis paddle collision proxy."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_g1_w_pingpong_paddle_spec,
    articulation=G1_ARTICULATION,
  )


def _apply_g1_pingpong_common(cfg, play: bool):
  cfg.scene.entities = {
    "robot": get_g1_w_pingpong_paddle_robot_cfg(),
    "ball": get_pingpong_ball_cfg(),
    "table": get_pingpong_table_cfg(),
  }
  cfg.viewer.body_name = "torso_link"
  cfg.viewer.elevation = -16.0
  cfg.viewer.azimuth = 135.0

  action = cfg.actions["latent_joint_pos"]
  assert isinstance(action, FrozenDecoderLatentJointPositionActionCfg)
  action.scale = G1_W_RACKET_ACTION_SCALE
  action.decoder_checkpoint = DEFAULT_DECODER_CHECKPOINT

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg


def unitree_g1_pingpong_latent_hit_env_cfg(play: bool = False):
  """Create G1 table-tennis legal-hit task."""
  cfg = make_pingpong_latent_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


def unitree_g1_pingpong_latent_return_env_cfg(play: bool = False):
  """Create G1 table-tennis legal-return task."""
  cfg = make_pingpong_latent_return_env_cfg()
  return _apply_g1_pingpong_common(cfg, play=play)


__all__ = [
  "DEFAULT_DECODER_CHECKPOINT",
  "PINGPONG_PADDLE_RADIUS",
  "PINGPONG_PADDLE_SCALE",
  "get_g1_w_pingpong_paddle_robot_cfg",
  "get_g1_w_pingpong_paddle_spec",
  "unitree_g1_pingpong_latent_hit_env_cfg",
  "unitree_g1_pingpong_latent_return_env_cfg",
]
