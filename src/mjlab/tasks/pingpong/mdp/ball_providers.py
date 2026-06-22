"""Ball feeders for table-tennis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from mjlab.entity.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.scene import BALL_CENTER_TABLE_Z, NET_TOP_Z, NET_X

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _write_ball_state(
  env: "ManagerBasedRlEnv",
  ball: Entity,
  env_ids: torch.Tensor,
  pos_l: torch.Tensor,
  quat: torch.Tensor,
  lin_vel: torch.Tensor,
  ang_vel: torch.Tensor,
) -> None:
  origins = env.scene.env_origins[env_ids]
  pos_w = pos_l + origins
  pose = torch.cat([pos_w, quat], dim=-1)
  twist = torch.cat([lin_vel, ang_vel], dim=-1)
  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(twist, env_ids=env_ids)


def _uniform(
  env_ids: torch.Tensor, lo: float, hi: float, device: str | torch.device
) -> torch.Tensor:
  return torch.empty(env_ids.numel(), device=device).uniform_(lo, hi)


@dataclass(kw_only=True)
class TableTennisFeederCfg:
  """Generate opponent feeds that bounce once on the robot-side table.

  The feeder samples a start point on the opponent side and a target first
  bounce on the robot side. It solves the ballistic velocity needed to reach
  the bounce point at ``BALL_CENTER_TABLE_Z`` while clearing the net.
  """

  ball_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("ball"))

  spawn_x_range: tuple[float, float] = (-1.20, -0.25)
  spawn_y_range: tuple[float, float] = (-0.45, 0.45)
  spawn_z_range: tuple[float, float] = (1.05, 1.35)

  target_x_range: tuple[float, float] = (0.55, 0.75)
  target_y_range: tuple[float, float] = (-0.12, 0.12)

  flight_time_range: tuple[float, float] = (0.55, 0.85)
  net_x: float = NET_X
  net_top_z: float = NET_TOP_Z
  net_clearance: float = 0.06
  target_z: float = BALL_CENTER_TABLE_Z
  gravity: float = 9.81
  max_resample_attempts: int = 64

  def build(self, env: "ManagerBasedRlEnv") -> "TableTennisFeeder":
    return TableTennisFeeder(self, env)


class TableTennisFeeder:
  """Runtime feeder with rejection sampling for net clearance."""

  def __init__(self, cfg: TableTennisFeederCfg, env: "ManagerBasedRlEnv") -> None:
    self.cfg = cfg
    self._env = env
    self._ball: Entity = env.scene[cfg.ball_cfg.name]

  @property
  def device(self) -> str | torch.device:
    return self._env.device

  def spawn(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    cfg = self.cfg
    dev = self.device
    n = env_ids.numel()
    px = torch.empty(n, device=dev)
    py = torch.empty(n, device=dev)
    pz = torch.empty(n, device=dev)
    vx = torch.empty(n, device=dev)
    vy = torch.empty(n, device=dev)
    vz = torch.empty(n, device=dev)

    remaining = torch.arange(n, device=dev)
    for _ in range(max(1, int(cfg.max_resample_attempts))):
      if remaining.numel() == 0:
        break
      sample = self._sample_candidate(env_ids[remaining])
      valid = sample[-1]
      if torch.any(valid):
        write_ids = remaining[valid]
        px[write_ids] = sample[0][valid]
        py[write_ids] = sample[1][valid]
        pz[write_ids] = sample[2][valid]
        vx[write_ids] = sample[3][valid]
        vy[write_ids] = sample[4][valid]
        vz[write_ids] = sample[5][valid]
      remaining = remaining[~valid]

    if remaining.numel() > 0:
      fallback = self._fallback_candidate(env_ids[remaining])
      px[remaining] = fallback[0]
      py[remaining] = fallback[1]
      pz[remaining] = fallback[2]
      vx[remaining] = fallback[3]
      vy[remaining] = fallback[4]
      vz[remaining] = fallback[5]

    pos = torch.stack((px, py, pz), dim=-1)
    lin = torch.stack((vx, vy, vz), dim=-1)
    quat = torch.zeros(n, 4, device=dev)
    quat[:, 0] = 1.0
    ang = torch.zeros(n, 3, device=dev)
    _write_ball_state(self._env, self._ball, env_ids, pos, quat, lin, ang)

  def _sample_candidate(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cfg = self.cfg
    dev = self.device
    px = _uniform(env_ids, cfg.spawn_x_range[0], cfg.spawn_x_range[1], dev)
    py = _uniform(env_ids, cfg.spawn_y_range[0], cfg.spawn_y_range[1], dev)
    pz = _uniform(env_ids, cfg.spawn_z_range[0], cfg.spawn_z_range[1], dev)
    tx = _uniform(env_ids, cfg.target_x_range[0], cfg.target_x_range[1], dev)
    ty = _uniform(env_ids, cfg.target_y_range[0], cfg.target_y_range[1], dev)
    flight_t = _uniform(
      env_ids, cfg.flight_time_range[0], cfg.flight_time_range[1], dev
    )

    vx = (tx - px) / flight_t
    vy = (ty - py) / flight_t
    vz = (cfg.target_z - pz + 0.5 * cfg.gravity * flight_t * flight_t) / flight_t

    t_net = (cfg.net_x - px) / vx.clamp_min(1.0e-6)
    z_net = pz + vz * t_net - 0.5 * cfg.gravity * t_net * t_net
    valid = (
      (vx > 0.0)
      & (t_net > 0.0)
      & (t_net < flight_t)
      & (z_net >= cfg.net_top_z + cfg.net_clearance)
    )
    return px, py, pz, vx, vy, vz, valid

  def _fallback_candidate(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    cfg = self.cfg
    dev = self.device
    n = env_ids.numel()
    px = torch.full((n,), 0.5 * sum(cfg.spawn_x_range), device=dev)
    py = torch.full((n,), 0.5 * sum(cfg.spawn_y_range), device=dev)
    pz = torch.full((n,), max(cfg.spawn_z_range[1], cfg.net_top_z + 0.35), device=dev)
    tx = torch.full((n,), 0.5 * sum(cfg.target_x_range), device=dev)
    ty = torch.full((n,), 0.5 * sum(cfg.target_y_range), device=dev)
    flight_t = torch.full((n,), max(0.8, cfg.flight_time_range[1]), device=dev)
    vx = (tx - px) / flight_t
    vy = (ty - py) / flight_t
    vz = (cfg.target_z - pz + 0.5 * cfg.gravity * flight_t * flight_t) / flight_t
    return px, py, pz, vx, vy, vz


def spawn_ball_from_provider(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  provider_cfg: Any,
) -> None:
  """Event-manager wrapper for table-tennis feeders."""
  cache_key = f"_pingpong_ball_provider_{id(provider_cfg)}"
  provider: TableTennisFeeder | None = getattr(env, cache_key, None)
  if provider is None:
    provider = provider_cfg.build(env)
    setattr(env, cache_key, provider)
  provider.spawn(env_ids)


__all__ = [
  "TableTennisFeeder",
  "TableTennisFeederCfg",
  "spawn_ball_from_provider",
]
