"""Observation terms specific to table-tennis geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.ball_sports import BallSportGeometryCfg, resolve_ball_sport_geometry
from mjlab.tasks.tennis.mdp.observations import ball_predicted_hit_point_b
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_BALL_CFG = SceneEntityCfg("ball")
_BALL_GEOM_CFG = SceneEntityCfg("ball", geom_names="pingpong_ball")
_PLAY_AREA_CFG = SceneEntityCfg("table", geom_names="pingpong_table_top_collision")
_NET_CFG = SceneEntityCfg("table", geom_names="pingpong_net_collision")


def _time_to_bounce_plane(
  z: torch.Tensor,
  vz: torch.Tensor,
  *,
  bounce_z: float | torch.Tensor,
  gravity: float,
  min_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  a = 0.5 * gravity
  b = -vz
  c = -(z - bounce_z)
  disc = b * b - 4.0 * a * c
  has_real_root = disc >= 0.0
  sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
  denom = 2.0 * a
  t0 = (-b - sqrt_disc) / denom
  t1 = (-b + sqrt_disc) / denom
  candidates = torch.stack((t0, t1), dim=-1)
  inf = torch.full_like(candidates, float("inf"))
  candidates = torch.where(candidates > min_time, candidates, inf)
  t_hit = candidates.amin(dim=-1)
  valid = has_real_root & torch.isfinite(t_hit)
  return t_hit, valid


def _fallback_hit_point_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg,
  robot_cfg: SceneEntityCfg,
  hit_height_offset: float,
  gravity: float,
  max_horizon: float,
) -> torch.Tensor:
  return ball_predicted_hit_point_b(
    env,
    ball_cfg=ball_cfg,
    robot_cfg=robot_cfg,
    hit_height_offset=hit_height_offset,
    gravity=gravity,
    max_horizon=max_horizon,
  )


def ball_predicted_edge_hit_point_b(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  ball_geom_cfg: SceneEntityCfg = _BALL_GEOM_CFG,
  play_area_cfg: SceneEntityCfg = _PLAY_AREA_CFG,
  bounce_surface_cfg: SceneEntityCfg | None = _PLAY_AREA_CFG,
  net_cfg: SceneEntityCfg = _NET_CFG,
  hit_height_offset: float = 0.02,
  gravity: float = 9.81,
  max_horizon: float = 1.5,
  bounce_z: float | None = None,
  net_x: float | None = None,
  edge_x: float | None = None,
  post_bounce_horizontal_scale: float = 0.94,
  post_bounce_vertical_scale: float = 0.90,
  edge_clearance: float = 0.02,
  min_time: float = 1.0e-3,
) -> torch.Tensor:
  """Predict where the incoming ball trajectory reaches the robot-side end line.

  The output shape and frame match the tennis ``predicted_hit_point`` term:
  ``(x, y, z, time)`` in robot-base coordinates. The preferred target is the
  post-self-bounce trajectory at ``edge_x``. If that cannot be predicted safely,
  the term falls back to the old fixed-height hit-point prediction.
  """
  robot: Entity = env.scene[robot_cfg.name]
  ball: Entity = env.scene[ball_cfg.name]
  geometry = resolve_ball_sport_geometry(
    env,
    BallSportGeometryCfg(
      ball_geom_cfg=ball_geom_cfg,
      play_area_cfg=play_area_cfg,
      net_cfg=net_cfg,
      bounce_surface_cfg=bounce_surface_cfg,
      self_side="positive_x",
    ),
  )
  origins = env.scene.env_origins
  pos_w = ball.data.root_link_pos_w
  pos = pos_w - origins
  vel = ball.data.root_link_lin_vel_w
  bounce_z_t = (
    geometry.bounce_z if bounce_z is None else torch.full_like(pos[:, 0], bounce_z)
  )
  net_x_t = geometry.net_x if net_x is None else torch.full_like(pos[:, 0], net_x)
  edge_x_t = (
    geometry.self_baseline_x
    if edge_x is None
    else torch.full_like(pos[:, 0], edge_x)
  )

  direct_t = (edge_x_t - pos[:, 0]) / vel[:, 0].clamp_min(1.0e-6)
  direct_z = pos[:, 2] + vel[:, 2] * direct_t - 0.5 * gravity * direct_t * direct_t
  direct_y = pos[:, 1] + vel[:, 1] * direct_t
  direct_valid = (
    (pos[:, 0] >= net_x_t)
    & (vel[:, 0] > 0.0)
    & (vel[:, 2] >= -0.05)
    & (direct_t > min_time)
    & (direct_t <= max_horizon)
    & (direct_z >= bounce_z_t + edge_clearance)
    & (direct_y >= geometry.self_bounds.y_min)
    & (direct_y <= geometry.self_bounds.y_max)
  )

  bounce_t, bounce_valid_t = _time_to_bounce_plane(
    pos[:, 2],
    vel[:, 2],
    bounce_z=bounce_z_t,
    gravity=gravity,
    min_time=min_time,
  )
  bounce_pos = pos + vel * bounce_t.unsqueeze(-1)
  bounce_pos[:, 2] = bounce_z_t
  impact_vz = vel[:, 2] - gravity * bounce_t
  post_vx = vel[:, 0] * post_bounce_horizontal_scale
  post_vy = vel[:, 1] * post_bounce_horizontal_scale
  post_vz = -impact_vz * post_bounce_vertical_scale
  edge_t_after_bounce = (edge_x_t - bounce_pos[:, 0]) / post_vx.clamp_min(1.0e-6)
  edge_z = (
    bounce_z_t
    + post_vz * edge_t_after_bounce
    - 0.5 * gravity * edge_t_after_bounce * edge_t_after_bounce
  )
  edge_y = bounce_pos[:, 1] + post_vy * edge_t_after_bounce
  bounce_valid = (
    bounce_valid_t
    & (bounce_t <= max_horizon)
    & (bounce_pos[:, 0] >= net_x_t)
    & (bounce_pos[:, 0] <= edge_x_t)
    & (bounce_pos[:, 1] >= geometry.self_bounds.y_min)
    & (bounce_pos[:, 1] <= geometry.self_bounds.y_max)
    & (impact_vz < 0.0)
    & (post_vx > 0.0)
    & (edge_t_after_bounce > min_time)
    & ((bounce_t + edge_t_after_bounce) <= max_horizon)
    & (edge_z >= bounce_z_t + edge_clearance)
    & (edge_y >= geometry.self_bounds.y_min)
    & (edge_y <= geometry.self_bounds.y_max)
  )

  hit_w = pos.clone()
  total_t = torch.zeros_like(pos[:, 0])
  hit_w[:, 0] = edge_x_t
  hit_w[:, 1] = torch.where(direct_valid, direct_y, edge_y)
  hit_w[:, 2] = torch.where(direct_valid, direct_z, edge_z)
  total_t = torch.where(direct_valid, direct_t, bounce_t + edge_t_after_bounce)
  valid = direct_valid | bounce_valid

  hit_w = hit_w + origins
  delta_w = hit_w - robot.data.root_link_pos_w
  hit_b = quat_apply_inverse(robot.data.root_link_quat_w, delta_w)
  edge_obs = torch.cat((hit_b, total_t.unsqueeze(-1)), dim=-1)
  fallback = _fallback_hit_point_b(
    env,
    ball_cfg,
    robot_cfg,
    hit_height_offset,
    gravity,
    max_horizon,
  )
  return torch.where(valid.unsqueeze(-1), edge_obs, fallback)


__all__ = ["ball_predicted_edge_hit_point_b"]
