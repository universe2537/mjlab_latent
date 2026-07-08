"""Unitree G1 robot helper with a scaled held-racket pingpong proxy."""

from __future__ import annotations

import math
from typing import Any

import mujoco

from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  FULL_COLLISION,
  G1_ARTICULATION,
  KNEES_BENT_KEYFRAME,
)
from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  get_spec as get_g1_w_racket_spec,
)
from mjlab.entity import EntityCfg

PINGPONG_PADDLE_RADIUS = 0.065
PINGPONG_PADDLE_SCALE = PINGPONG_PADDLE_RADIUS / 0.12
PINGPONG_PADDLE_HAND_CLEARANCE = 0.001
_TENNIS_RACKET_COLLISION_POS = (0.1025, -0.004, 0.4)
_RIGHT_HAND_COLLISION_FORWARD_END_X = 0.13
_RIGHT_HAND_COLLISION_RADIUS = 0.05
_PADDLE_CENTER_Y = -0.004
_PADDLE_HAND_CLEARANCE_RADIUS = (
  _RIGHT_HAND_COLLISION_RADIUS
  + PINGPONG_PADDLE_RADIUS
  + PINGPONG_PADDLE_HAND_CLEARANCE
)
PINGPONG_PADDLE_CENTER_POS = (
  _RIGHT_HAND_COLLISION_FORWARD_END_X
  + math.sqrt(_PADDLE_HAND_CLEARANCE_RADIUS**2 - _PADDLE_CENTER_Y**2),
  _PADDLE_CENTER_Y,
  0.0,
)
_PADDLE_END_EFFECTOR_FORWARD_ROLL_QUAT = (0.5, 0.5, 0.5, 0.5)


def _iter_body_tree(body: mujoco.MjsBody):
  yield body
  for child in body.bodies:
    yield from _iter_body_tree(child)


def _find_geom_with_body(
  spec: mujoco.MjSpec, name: str
) -> tuple[mujoco.MjsBody, mujoco.MjsGeom]:
  for body in _iter_body_tree(spec.worldbody):
    for geom in body.geoms:
      if geom.name == name:
        return body, geom
  raise ValueError(f"Could not find geom {name!r} in G1 racket spec.")


def _find_mesh(spec: mujoco.MjSpec, name: str) -> mujoco.MjsMesh:
  for mesh in spec.meshes:
    if mesh.name == name:
      return mesh
  raise ValueError(f"Could not find mesh {name!r} in G1 racket spec.")


def _find_geoms_by_mesh(spec: mujoco.MjSpec, meshname: str) -> list[mujoco.MjsGeom]:
  geoms = []
  for body in _iter_body_tree(spec.worldbody):
    for geom in body.geoms:
      if geom.meshname == meshname:
        geoms.append(geom)
  if not geoms:
    raise ValueError(f"Could not find geom using mesh {meshname!r}.")
  return geoms


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


def _quat_to_matrix(q: Any) -> tuple[tuple[float, float, float], ...]:
  w, x, y, z = (float(v) for v in q)
  return (
    (
      1.0 - 2.0 * (y * y + z * z),
      2.0 * (x * y - z * w),
      2.0 * (x * z + y * w),
    ),
    (
      2.0 * (x * y + z * w),
      1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - x * w),
    ),
    (
      2.0 * (x * z - y * w),
      2.0 * (y * z + x * w),
      1.0 - 2.0 * (x * x + y * y),
    ),
  )


def _quat_mul(lhs: Any, rhs: Any) -> tuple[float, float, float, float]:
  lw, lx, ly, lz = (float(v) for v in lhs)
  rw, rx, ry, rz = (float(v) for v in rhs)
  return (
    lw * rw - lx * rx - ly * ry - lz * rz,
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
  )


def _vec_add(
  lhs: tuple[float, float, float], rhs: tuple[float, float, float]
) -> tuple[float, float, float]:
  return (lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2])


def _mat_vec_mul(
  mat: tuple[tuple[float, float, float], ...],
  vec: tuple[float, float, float],
) -> tuple[float, float, float]:
  return (
    mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2],
    mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2],
    mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2],
  )


def _vec_sub(
  lhs: tuple[float, float, float], rhs: tuple[float, float, float]
) -> tuple[float, float, float]:
  return (lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2])


def _rotate_pose_about_center(
  pos: Any,
  quat: Any,
  *,
  old_center: tuple[float, float, float],
  new_center: tuple[float, float, float],
  rot_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
  rot = _quat_to_matrix(rot_quat)
  old_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
  rel = _vec_sub(old_pos, old_center)
  new_pos = _vec_add(new_center, _mat_vec_mul(rot, rel))
  return new_pos, _quat_mul(rot_quat, quat)


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
  old_paddle_center = _scale_from_anchor(
    _TENNIS_RACKET_COLLISION_POS,
    visual_anchor,
    PINGPONG_PADDLE_SCALE,
  )
  for idx, geom in enumerate(visual_geoms):
    new_pos, new_quat = _rotate_pose_about_center(
      geom.pos,
      geom.quat,
      old_center=old_paddle_center,
      new_center=PINGPONG_PADDLE_CENTER_POS,
      rot_quat=_PADDLE_END_EFFECTOR_FORWARD_ROLL_QUAT,
    )
    geom.meshname = "pingpong_paddle_visual"
    geom.name = (
      "pingpong_paddle_visual"
      if idx == 0
      else f"pingpong_paddle_visual_{idx}"
    )
    geom.pos[:] = new_pos
    geom.quat[:] = new_quat

  _, paddle = _find_geom_with_body(spec, "tennis_racket_collision")
  paddle.name = "pingpong_paddle_collision"
  paddle.size[0] = PINGPONG_PADDLE_RADIUS
  paddle.size[1] = 0.004
  paddle.pos[:] = PINGPONG_PADDLE_CENTER_POS
  paddle.quat[:] = _quat_mul(_PADDLE_END_EFFECTOR_FORWARD_ROLL_QUAT, paddle.quat)
  paddle.rgba[:] = (0.85, 0.12, 0.06, 0.35)

  center = _find_site(spec, "tennis_racket_center")
  center.name = "pingpong_paddle_center"
  center.pos[:] = PINGPONG_PADDLE_CENTER_POS
  center.quat[:] = _quat_mul(_PADDLE_END_EFFECTOR_FORWARD_ROLL_QUAT, center.quat)
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


__all__ = [
  "PINGPONG_PADDLE_HAND_CLEARANCE",
  "PINGPONG_PADDLE_RADIUS",
  "PINGPONG_PADDLE_SCALE",
  "PINGPONG_PADDLE_CENTER_POS",
  "get_g1_w_pingpong_paddle_robot_cfg",
  "get_g1_w_pingpong_paddle_spec",
]
