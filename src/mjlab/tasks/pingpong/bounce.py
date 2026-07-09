"""Shared table-tennis bounce constants.

The ball/table profile is intentionally shared by scene construction, feeder
trajectory checks, predicted-hit observations, and PACE future-target logic.
This keeps the sampled serve model and the controller's prediction model from
quietly drifting apart.
"""

from __future__ import annotations

PINGPONG_BALL_RADIUS = 0.02
PINGPONG_BALL_MASS = 0.0034

# MuJoCo profile measured on the local table scene. The resulting bounce ratios
# are close to the PACE-style targets h ~= 0.94 and v ~= 0.90.
PINGPONG_BOUNCE_FRICTION = (0.02, 0.001, 0.0001)
PINGPONG_BOUNCE_SOLREF = (-5000.0, -5.0)
PINGPONG_BOUNCE_SOLIMP = (0.93, 0.98, 0.001, 0.5, 2.0)

PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE = 0.94
PINGPONG_POST_BOUNCE_VERTICAL_SCALE = 0.90


__all__ = [
  "PINGPONG_BALL_MASS",
  "PINGPONG_BALL_RADIUS",
  "PINGPONG_BOUNCE_FRICTION",
  "PINGPONG_BOUNCE_SOLIMP",
  "PINGPONG_BOUNCE_SOLREF",
  "PINGPONG_POST_BOUNCE_HORIZONTAL_SCALE",
  "PINGPONG_POST_BOUNCE_VERTICAL_SCALE",
]
