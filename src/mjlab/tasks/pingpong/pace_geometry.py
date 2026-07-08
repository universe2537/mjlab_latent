"""G1 geometry defaults for the Pingpong PACE baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class G1PaceGeometryCfg:
  """Robot-specific geometry values used by the direct-joint PACE task."""

  target_base_offset_xy: tuple[float, float]
  target_root_height: float
  target_base_vel_gain: float
  target_base_vel_max: float
  forehand_paddle_offset: tuple[float, float, float]
  forehand_paddle_offset_std: tuple[float, float, float]
  forehand_elbow_target_ratio: float
  strike_direction_pelvis: tuple[float, float, float]
  strike_upward_angle: float
  foot_geom_names: tuple[str, ...]
  bad_orientation_limit: float
  root_height_minimum: float


G1_PACE_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

_STRIKE_UPWARD_ANGLE = math.radians(15.0)
_STRIKE_DIRECTION_PELVIS = (
  math.cos(_STRIKE_UPWARD_ANGLE),
  0.0,
  math.sin(_STRIKE_UPWARD_ANGLE),
)

G1_PACE_GEOMETRY = G1PaceGeometryCfg(
  target_base_offset_xy=(-0.3112, 0.4510),
  target_root_height=0.760,
  target_base_vel_gain=4.0,
  target_base_vel_max=7.0,
  forehand_paddle_offset=(0.3112, -0.4510, 0.0290),
  forehand_paddle_offset_std=(0.15, 0.14, 0.08),
  forehand_elbow_target_ratio=0.97,
  strike_direction_pelvis=_STRIKE_DIRECTION_PELVIS,
  strike_upward_angle=_STRIKE_UPWARD_ANGLE,
  foot_geom_names=G1_PACE_FOOT_GEOM_NAMES,
  bad_orientation_limit=math.radians(40.0),
  root_height_minimum=0.68,
)


__all__ = [
  "G1_PACE_FOOT_GEOM_NAMES",
  "G1_PACE_GEOMETRY",
  "G1PaceGeometryCfg",
]
