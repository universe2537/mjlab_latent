"""G1 geometry defaults for the Pingpong PACE baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class G1PaceGeometryCfg:
  """Robot-specific geometry values used by the direct-joint PACE task."""

  target_base_offset_xy: tuple[float, float]
  natural_hit_x: float
  target_root_height: float
  target_base_vel_gain: float
  target_base_vel_max: float
  robot_reset_yaw: float
  robot_reset_x_center: float
  forehand_paddle_offset: tuple[float, float, float]
  forehand_paddle_offset_table_xy: tuple[float, float]
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

_ROBOT_RESET_YAW = math.pi
_ROBOT_RESET_X_CENTER = 1.84
_FOREHAND_PADDLE_OFFSET = (0.2541, -0.6239, 0.0442)


def rotate_xy_by_yaw(
  xy: tuple[float, float],
  yaw: float,
) -> tuple[float, float]:
  """Rotate a 2D offset by a table/world yaw angle."""
  cos_yaw = math.cos(yaw)
  sin_yaw = math.sin(yaw)
  x, y = xy
  return (cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y)


def target_base_offset_xy_from_paddle_offset(
  paddle_offset_xy: tuple[float, float],
  yaw: float,
) -> tuple[float, float]:
  """Return table-frame base offset that puts the paddle at the target point."""
  paddle_table_xy = rotate_xy_by_yaw(paddle_offset_xy, yaw)
  return (-paddle_table_xy[0], -paddle_table_xy[1])


_FOREHAND_PADDLE_OFFSET_TABLE_XY = rotate_xy_by_yaw(
  _FOREHAND_PADDLE_OFFSET[:2],
  _ROBOT_RESET_YAW,
)
_TARGET_BASE_OFFSET_XY = target_base_offset_xy_from_paddle_offset(
  _FOREHAND_PADDLE_OFFSET[:2],
  _ROBOT_RESET_YAW,
)
_NATURAL_HIT_X = _ROBOT_RESET_X_CENTER + _FOREHAND_PADDLE_OFFSET_TABLE_XY[0]

_STRIKE_UPWARD_ANGLE = math.radians(15.0)
_STRIKE_DIRECTION_PELVIS = (
  math.cos(_STRIKE_UPWARD_ANGLE),
  0.0,
  math.sin(_STRIKE_UPWARD_ANGLE),
)

G1_PACE_GEOMETRY = G1PaceGeometryCfg(
  target_base_offset_xy=_TARGET_BASE_OFFSET_XY,
  natural_hit_x=_NATURAL_HIT_X,
  target_root_height=0.760,
  target_base_vel_gain=2.0,
  target_base_vel_max=2.5,
  robot_reset_yaw=_ROBOT_RESET_YAW,
  robot_reset_x_center=_ROBOT_RESET_X_CENTER,
  forehand_paddle_offset=_FOREHAND_PADDLE_OFFSET,
  forehand_paddle_offset_table_xy=_FOREHAND_PADDLE_OFFSET_TABLE_XY,
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
  "rotate_xy_by_yaw",
  "target_base_offset_xy_from_paddle_offset",
]
