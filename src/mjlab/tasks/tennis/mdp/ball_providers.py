"""Pluggable ball providers (P0 Fixed / P1 RandomFeeder / P2 BallisticOpponent).

A *ball provider* is a strategy object owned by :class:`RallyCommand` that is
responsible for two things:

1. **Spawning** the ball at the start of each point (``spawn``).
2. Optionally **responding** during a point — e.g., an opponent agent that
   returns the ball after the player hits it (``respond``).

The abstract base class makes both methods hookable so that high-level tasks
can be composed by swapping providers without touching reward/termination
logic.

Difficulty knobs
----------------
``BallProvider.bump_difficulty(key)`` is invoked by curriculum terms to
nudge sampling ranges (e.g., wider speed range) over the course of training.
Concrete providers decide how to react.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.tennis.mdp.events import EventCode  # noqa: F401


# ---------------------------------------------------------------------------
# Spawn helper.
# ---------------------------------------------------------------------------


def _write_ball_state(
  env: "ManagerBasedRlEnv",
  ball: Entity,
  env_ids: torch.Tensor,
  pos_l: torch.Tensor,
  quat: torch.Tensor,
  lin_vel: torch.Tensor,
  ang_vel: torch.Tensor,
) -> None:
  """Write a ball pose+twist (env-local frame) into the simulator."""
  origins = env.scene.env_origins[env_ids]
  pos_w = pos_l + origins
  pose = torch.cat([pos_w, quat], dim=-1)  # (k, 7)
  twist = torch.cat([lin_vel, ang_vel], dim=-1)  # (k, 6)
  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(twist, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Abstract interface.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BallProviderCfg(abc.ABC):
  """Abstract base config for a ball provider."""

  ball_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("ball"))

  @abc.abstractmethod
  def build(self, env: "ManagerBasedRlEnv") -> "BallProvider":
    """Instantiate the runtime provider object."""


class BallProvider(abc.ABC):
  """Strategy object that spawns the ball (and optionally responds)."""

  cfg: BallProviderCfg

  def __init__(self, cfg: BallProviderCfg, env: "ManagerBasedRlEnv") -> None:
    self.cfg = cfg
    self._env = env
    self._ball: Entity = env.scene[cfg.ball_cfg.name]
    self._difficulty = 0.0

  # --- Lifecycle hooks --------------------------------------------------

  @abc.abstractmethod
  def spawn(self, env_ids: torch.Tensor) -> None:
    """Place the ball at the start of a new point for ``env_ids``."""

  def respond(self, env_ids: torch.Tensor) -> None:  # noqa: B027
    """Optional in-rally response (e.g., opponent return). Default no-op."""

  def reset(self, env_ids: torch.Tensor) -> None:  # noqa: B027
    """Optional internal-state reset (default no-op)."""

  # --- Curriculum hooks -------------------------------------------------

  def bump_difficulty(self, key: str, delta: float = 0.05) -> None:  # noqa: ARG002
    """Increase difficulty. Subclasses override to widen sampling ranges."""
    self._difficulty = min(1.0, self._difficulty + delta)

  @property
  def difficulty(self) -> float:
    return self._difficulty

  # --- Convenience ------------------------------------------------------

  @property
  def device(self) -> str | torch.device:
    return self._env.device

  @property
  def num_envs(self) -> int:
    return self._env.num_envs


# ---------------------------------------------------------------------------
# P0: Fixed spawn — single deterministic state.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class FixedSpawnerCfg(BallProviderCfg):
  """Spawn the ball at a fixed pose and velocity every point.

  Useful for unit tests, debugging reward shaping, and evaluation.
  """

  pos: tuple[float, float, float] = (1.5, 0.0, 1.0)
  lin_vel: tuple[float, float, float] = (-1.5, 0.0, 0.0)

  def build(self, env: "ManagerBasedRlEnv") -> "FixedSpawner":
    return FixedSpawner(self, env)


class FixedSpawner(BallProvider):
  cfg: FixedSpawnerCfg

  def spawn(self, env_ids: torch.Tensor) -> None:
    k = env_ids.numel()
    pos = torch.tensor(self.cfg.pos, device=self.device).expand(k, 3)
    quat = torch.zeros(k, 4, device=self.device)
    quat[:, 0] = 1.0
    lin = torch.tensor(self.cfg.lin_vel, device=self.device).expand(k, 3)
    ang = torch.zeros(k, 3, device=self.device)
    _write_ball_state(self._env, self._ball, env_ids, pos, quat, lin, ang)


# ---------------------------------------------------------------------------
# P1: Random feeder — ballistic trajectory from spawn region to target zone.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RandomFeederCfg(BallProviderCfg):
  """Spawn the ball in a configurable region and send it toward a random landing target.

  The ball is placed at a random position within ``[spawn_x_range, spawn_y_range,
  spawn_z_range]`` (default: above the net). A 2-D landing target is sampled
  uniformly within ``[target_x_range, target_y_range]`` on the ground (z = 0).
  The vertical launch speed ``vz0`` is sampled from ``lin_vel_z_range``; the
  horizontal components (vx, vy) are then solved analytically so that the ball
  reaches the target at z = 0 under constant gravity.

  Solving for flight time ``t`` from the z-equation::

      z0 + vz0 * t - 0.5 * g * t^2 = 0
      t = (vz0 + sqrt(vz0^2 + 2 * g * z0)) / g   (positive root)

  Then::

      vx = (target_x - spawn_x) / t
      vy = (target_y - spawn_y) / t

  Defaults place the spawn region directly above the net (x ≈ 0) and target
  the robot's side of the court (x ∈ (0.5, 2.5)), so the ball always flies
  toward the player.

  Curriculum knobs
  ----------------
  ``bump_difficulty("ball_speed")`` tightens flight time by widening
  ``lin_vel_z_range`` (larger vz0 → higher arc → more time to react declines
  as arc flattens with further bumps).
  ``bump_difficulty("ball_lateral")`` widens ``target_y_range``.
  """

  # Spawn region (ball starting position) -- default: above the net.
  spawn_x_range: tuple[float, float] = (-0.4, 0.4)
  spawn_y_range: tuple[float, float] = (-2.0, 2.0)
  spawn_z_range: tuple[float, float] = (1.0, 1.6)

  # Target landing region on the ground (z = 0) -- default: robot side service box.
  target_x_range: tuple[float, float] = (1.0, 4.0)
  target_y_range: tuple[float, float] = (-2.0, 2.0)

  # Vertical launch speed.  Positive = upward arc; must be large enough
  # so the ball actually reaches the landing zone (not hit the ground immediately).
  lin_vel_z_range: tuple[float, float] = (1.5, 3.5)

  # Physical constant; override for custom gravity environments.
  gravity: float = 9.81

  def build(self, env: "ManagerBasedRlEnv") -> "RandomFeeder":
    return RandomFeeder(self, env)


def _uniform(
  env_ids: torch.Tensor, lo: float, hi: float, device: str | torch.device
) -> torch.Tensor:
  return torch.empty(env_ids.numel(), device=device).uniform_(lo, hi)


class RandomFeeder(BallProvider):
  cfg: RandomFeederCfg

  def spawn(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    dev = self.device

    # --- Spawn position ---------------------------------------------------
    px = _uniform(env_ids, *cfg.spawn_x_range, dev)
    py = _uniform(env_ids, *cfg.spawn_y_range, dev)
    pz = _uniform(env_ids, *cfg.spawn_z_range, dev)

    # --- Target landing point (z = 0) -------------------------------------
    tx = _uniform(env_ids, *cfg.target_x_range, dev)
    ty = _uniform(env_ids, *cfg.target_y_range, dev)

    # --- Vertical speed ---------------------------------------------------
    vz = _uniform(env_ids, *cfg.lin_vel_z_range, dev)

    # --- Solve flight time from z-equation --------------------------------
    # pz + vz*t - 0.5*g*t^2 = 0  =>  t = (vz + sqrt(vz^2 + 2*g*pz)) / g
    g = cfg.gravity
    disc = torch.clamp(vz * vz + 2.0 * g * pz, min=1e-6)
    flight_t = (vz + torch.sqrt(disc)) / g  # always positive since pz > 0

    # --- Horizontal speeds (kinematic inverse) ----------------------------
    vx = (tx - px) / flight_t
    vy = (ty - py) / flight_t

    pos = torch.stack([px, py, pz], dim=-1)
    lin = torch.stack([vx, vy, vz], dim=-1)
    quat = torch.zeros(env_ids.numel(), 4, device=dev)
    quat[:, 0] = 1.0
    ang = torch.zeros(env_ids.numel(), 3, device=dev)
    _write_ball_state(self._env, self._ball, env_ids, pos, quat, lin, ang)

  def bump_difficulty(self, key: str, delta: float = 0.05) -> None:
    """Adjust difficulty by modifying target or speed ranges."""
    super().bump_difficulty(key, delta)
    if key == "ball_speed":
      # Flatten the arc by reducing vz upper bound → shorter flight time.
      lo, hi = self.cfg.lin_vel_z_range
      self.cfg.lin_vel_z_range = (
        max(0.5, lo - delta * 0.5),
        max(lo + 0.1, hi - delta * 0.5),
      )
    elif key == "ball_lateral":
      lo, hi = self.cfg.target_y_range
      self.cfg.target_y_range = (lo - delta * 0.3, hi + delta * 0.3)


# ---------------------------------------------------------------------------
# P2: Ballistic opponent — one-shot serve from opponent's side.
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BallisticOpponentCfg(BallProviderCfg):
  """Compute initial velocity to land ball at a sampled point on self side.

  Solves the projectile motion equations under gravity ``g`` so that a ball
  launched from ``launch_pos`` reaches ``target_pos`` after ``flight_time``
  seconds. ``target_pos`` is sampled per-env per-spawn from the configured
  bounding box.

  Notes
  -----
  This is *not* a learned opponent; it's a deterministic ball-feeder that
  produces realistic incoming arcs. A true opponent agent would replace
  this provider with one that queries an external policy.
  """

  launch_pos: tuple[float, float, float] = (-3.0, 0.0, 1.5)
  target_x_range: tuple[float, float] = (1.0, 2.5)
  target_y_range: tuple[float, float] = (-1.5, 1.5)
  target_z: float = 0.06
  flight_time_range: tuple[float, float] = (0.7, 1.0)
  gravity: float = 9.81
  jitter_launch_y: float = 0.5

  def build(self, env: "ManagerBasedRlEnv") -> "BallisticOpponent":
    return BallisticOpponent(self, env)


class BallisticOpponent(BallProvider):
  cfg: BallisticOpponentCfg

  def spawn(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    dev = self.device
    k = env_ids.numel()

    # Sample a target landing point and a flight time.
    target_x = _uniform(env_ids, *cfg.target_x_range, dev)
    target_y = _uniform(env_ids, *cfg.target_y_range, dev)
    target_z = torch.full((k,), cfg.target_z, device=dev)
    flight_t = _uniform(env_ids, *cfg.flight_time_range, dev)

    # Launch position with mild lateral jitter so the trajectory direction
    # varies even at fixed flight time.
    lx = torch.full((k,), cfg.launch_pos[0], device=dev)
    ly = torch.full((k,), cfg.launch_pos[1], device=dev) + _uniform(
      env_ids, -cfg.jitter_launch_y, cfg.jitter_launch_y, dev
    )
    lz = torch.full((k,), cfg.launch_pos[2], device=dev)

    # Solve v0 from kinematics: target = launch + v0 * t + 0.5 * a * t^2.
    dx = target_x - lx
    dy = target_y - ly
    dz = target_z - lz
    vx0 = dx / flight_t
    vy0 = dy / flight_t
    vz0 = dz / flight_t + 0.5 * cfg.gravity * flight_t  # gravity along -z

    pos = torch.stack([lx, ly, lz], dim=-1)
    lin = torch.stack([vx0, vy0, vz0], dim=-1)
    quat = torch.zeros(k, 4, device=dev)
    quat[:, 0] = 1.0
    ang = torch.zeros(k, 3, device=dev)
    _write_ball_state(self._env, self._ball, env_ids, pos, quat, lin, ang)

  def bump_difficulty(self, key: str, delta: float = 0.05) -> None:
    super().bump_difficulty(key, delta)
    if key == "opponent_level":
      # Tighter flight time → faster, harder-to-reach shots.
      lo, hi = self.cfg.flight_time_range
      self.cfg.flight_time_range = (max(0.3, lo - delta * 0.1), hi)
    elif key == "ball_lateral":
      lo, hi = self.cfg.target_y_range
      self.cfg.target_y_range = (lo - delta * 0.3, hi + delta * 0.3)


# ---------------------------------------------------------------------------
# Aliases for re-export convenience.
# ---------------------------------------------------------------------------

__all__ = [
  "BallProvider",
  "BallProviderCfg",
  "FixedSpawner",
  "FixedSpawnerCfg",
  "RandomFeeder",
  "RandomFeederCfg",
  "BallisticOpponent",
  "BallisticOpponentCfg",
]

# Silence unused-import warning when TYPE_CHECKING is off.
_ = math
