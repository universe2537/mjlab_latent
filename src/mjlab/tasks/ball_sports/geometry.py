"""Scene-driven geometry helpers shared by ball-sport tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(frozen=True)
class BallSportBounds:
  """Axis-aligned 2D bounds for one side of a ball-sport field."""

  x_min: torch.Tensor
  x_max: torch.Tensor
  y_min: torch.Tensor
  y_max: torch.Tensor

  def select(self, env_ids: torch.Tensor) -> "BallSportBounds":
    return BallSportBounds(
      x_min=self.x_min[env_ids],
      x_max=self.x_max[env_ids],
      y_min=self.y_min[env_ids],
      y_max=self.y_max[env_ids],
    )

  @property
  def center_y(self) -> torch.Tensor:
    return 0.5 * (self.y_min + self.y_max)

  @property
  def half_width(self) -> torch.Tensor:
    return 0.5 * (self.y_max - self.y_min)


@dataclass(frozen=True)
class BallSportGeometry:
  """Scene-derived, sport-agnostic field geometry."""

  opponent_bounds: BallSportBounds
  self_bounds: BallSportBounds
  self_side: str
  play_area_center: torch.Tensor
  play_area_half_size: torch.Tensor
  net_center: torch.Tensor
  net_half_size: torch.Tensor
  ball_radius: torch.Tensor
  bounce_z: torch.Tensor
  landing_z: torch.Tensor
  surface_friction: torch.Tensor
  surface_solref: torch.Tensor
  surface_solimp: torch.Tensor
  ball_friction: torch.Tensor
  ball_solref: torch.Tensor
  ball_solimp: torch.Tensor
  net_friction: torch.Tensor
  net_solref: torch.Tensor
  net_solimp: torch.Tensor

  def select(self, env_ids: torch.Tensor) -> "BallSportGeometry":
    return BallSportGeometry(
      opponent_bounds=self.opponent_bounds.select(env_ids),
      self_bounds=self.self_bounds.select(env_ids),
      self_side=self.self_side,
      play_area_center=self.play_area_center[env_ids],
      play_area_half_size=self.play_area_half_size[env_ids],
      net_center=self.net_center[env_ids],
      net_half_size=self.net_half_size[env_ids],
      ball_radius=self.ball_radius[env_ids],
      bounce_z=self.bounce_z[env_ids],
      landing_z=self.landing_z[env_ids],
      surface_friction=self.surface_friction[env_ids],
      surface_solref=self.surface_solref[env_ids],
      surface_solimp=self.surface_solimp[env_ids],
      ball_friction=self.ball_friction[env_ids],
      ball_solref=self.ball_solref[env_ids],
      ball_solimp=self.ball_solimp[env_ids],
      net_friction=self.net_friction[env_ids],
      net_solref=self.net_solref[env_ids],
      net_solimp=self.net_solimp[env_ids],
    )

  @property
  def net_x(self) -> torch.Tensor:
    return self.net_center[:, 0]

  @property
  def net_top_z(self) -> torch.Tensor:
    return self.net_center[:, 2] + self.net_half_size[:, 2]

  @property
  def net_y_min(self) -> torch.Tensor:
    return self.net_center[:, 1] - self.net_half_size[:, 1]

  @property
  def net_y_max(self) -> torch.Tensor:
    return self.net_center[:, 1] + self.net_half_size[:, 1]

  @property
  def self_baseline_x(self) -> torch.Tensor:
    if self.self_side == "negative_x":
      return self.self_bounds.x_min
    return self.self_bounds.x_max

  @property
  def opponent_baseline_x(self) -> torch.Tensor:
    if self.self_side == "negative_x":
      return self.opponent_bounds.x_max
    return self.opponent_bounds.x_min


@dataclass(kw_only=True)
class BallSportGeometryCfg:
  """Configuration for resolving ball-sport geometry from the active scene."""

  ball_geom_cfg: SceneEntityCfg
  play_area_cfg: SceneEntityCfg
  net_cfg: SceneEntityCfg
  bounce_surface_cfg: SceneEntityCfg | None = None
  landing_z_override: float | None = None
  self_side: str = "positive_x"


def _as_tensor(value: Any, device: str | torch.device) -> torch.Tensor:
  if isinstance(value, torch.Tensor):
    return value.to(device=device)
  return torch.as_tensor(value, device=device)


def _expand_first_dim(value: torch.Tensor, num_envs: int) -> torch.Tensor:
  return value.unsqueeze(0).expand(num_envs, *value.shape)


def _model_geom_field(
  env: "ManagerBasedRlEnv", field_name: str, geom_id: int
) -> torch.Tensor:
  value = _as_tensor(getattr(env.sim.model, field_name), env.device)
  if (
    value.ndim >= 2
    and value.shape[0] == env.num_envs
    and value.shape[1] == env.sim.mj_model.ngeom
  ):
    return value[:, geom_id]
  return _expand_first_dim(value[geom_id], env.num_envs)


def _data_geom_field(
  env: "ManagerBasedRlEnv", field_name: str, geom_id: int
) -> torch.Tensor:
  value = _as_tensor(getattr(env.sim.data, field_name), env.device)
  if value.ndim >= 2 and value.shape[0] == env.num_envs:
    return value[:, geom_id]
  return _expand_first_dim(value[geom_id], env.num_envs)


def _first_local_geom_id(entity: Entity, cfg: SceneEntityCfg, label: str) -> int:
  if cfg.geom_names is not None:
    ids, _ = entity.find_geoms(cfg.geom_names, preserve_order=True)
  elif isinstance(cfg.geom_ids, list):
    ids = cfg.geom_ids
  elif entity.num_geoms == 1:
    ids = [0]
  else:
    raise ValueError(
      f"{label} must specify one geom name or id for entity {cfg.name!r}."
    )
  if len(ids) != 1:
    raise ValueError(f"{label} must resolve to one geom, got {ids!r}.")
  return int(ids[0])


def _global_geom_id(
  env: "ManagerBasedRlEnv", cfg: SceneEntityCfg, label: str
) -> int:
  entity: Entity = env.scene[cfg.name]
  local_id = _first_local_geom_id(entity, cfg, label)
  return int(entity.indexing.geom_ids[local_id].item())


def _bounds_from_center_size(
  center: torch.Tensor, half_size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  return (
    center[:, 0] - half_size[:, 0],
    center[:, 0] + half_size[:, 0],
    center[:, 1] - half_size[:, 1],
    center[:, 1] + half_size[:, 1],
  )


def _split_halves(
  *,
  x_min: torch.Tensor,
  x_max: torch.Tensor,
  y_min: torch.Tensor,
  y_max: torch.Tensor,
  net_x: torch.Tensor,
  self_side: str,
) -> tuple[BallSportBounds, BallSportBounds]:
  if self_side == "positive_x":
    opponent = BallSportBounds(x_min=x_min, x_max=net_x, y_min=y_min, y_max=y_max)
    self_bounds = BallSportBounds(x_min=net_x, x_max=x_max, y_min=y_min, y_max=y_max)
  elif self_side == "negative_x":
    opponent = BallSportBounds(x_min=net_x, x_max=x_max, y_min=y_min, y_max=y_max)
    self_bounds = BallSportBounds(x_min=x_min, x_max=net_x, y_min=y_min, y_max=y_max)
  else:
    raise ValueError(f"Unknown ball-sport self_side {self_side!r}.")
  return opponent, self_bounds


def resolve_ball_sport_geometry(
  env: "ManagerBasedRlEnv",
  cfg: BallSportGeometryCfg,
) -> BallSportGeometry:
  """Resolve sport-agnostic field geometry from scene geoms."""
  play_gid = _global_geom_id(env, cfg.play_area_cfg, "play_area_cfg")
  ball_gid = _global_geom_id(env, cfg.ball_geom_cfg, "ball_geom_cfg")
  net_gid = _global_geom_id(env, cfg.net_cfg, "net_cfg")
  surface_gid = (
    _global_geom_id(env, cfg.bounce_surface_cfg, "bounce_surface_cfg")
    if cfg.bounce_surface_cfg is not None
    else play_gid
  )

  origins = env.scene.env_origins
  play_center = _data_geom_field(env, "geom_xpos", play_gid) - origins
  play_half_size = _model_geom_field(env, "geom_size", play_gid)
  net_center = _data_geom_field(env, "geom_xpos", net_gid) - origins
  net_half_size = _model_geom_field(env, "geom_size", net_gid)
  surface_center = _data_geom_field(env, "geom_xpos", surface_gid) - origins
  surface_half_size = _model_geom_field(env, "geom_size", surface_gid)
  ball_radius = _model_geom_field(env, "geom_size", ball_gid)[:, 0]

  x_min, x_max, y_min, y_max = _bounds_from_center_size(
    play_center, play_half_size
  )
  opponent_bounds, self_bounds = _split_halves(
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    net_x=net_center[:, 0],
    self_side=cfg.self_side,
  )
  bounce_z = surface_center[:, 2] + surface_half_size[:, 2] + ball_radius
  if cfg.landing_z_override is None:
    landing_z = bounce_z
  else:
    landing_z = torch.full_like(bounce_z, cfg.landing_z_override)

  return BallSportGeometry(
    opponent_bounds=opponent_bounds,
    self_bounds=self_bounds,
    self_side=cfg.self_side,
    play_area_center=play_center,
    play_area_half_size=play_half_size,
    net_center=net_center,
    net_half_size=net_half_size,
    ball_radius=ball_radius,
    bounce_z=bounce_z,
    landing_z=landing_z,
    surface_friction=_model_geom_field(env, "geom_friction", surface_gid),
    surface_solref=_model_geom_field(env, "geom_solref", surface_gid),
    surface_solimp=_model_geom_field(env, "geom_solimp", surface_gid),
    ball_friction=_model_geom_field(env, "geom_friction", ball_gid),
    ball_solref=_model_geom_field(env, "geom_solref", ball_gid),
    ball_solimp=_model_geom_field(env, "geom_solimp", ball_gid),
    net_friction=_model_geom_field(env, "geom_friction", net_gid),
    net_solref=_model_geom_field(env, "geom_solref", net_gid),
    net_solimp=_model_geom_field(env, "geom_solimp", net_gid),
  )


__all__ = [
  "BallSportBounds",
  "BallSportGeometry",
  "BallSportGeometryCfg",
  "resolve_ball_sport_geometry",
]
