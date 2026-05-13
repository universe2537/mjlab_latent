"""Tennis event detection.

This module is the single, vectorised event-detection layer for the tennis
task. It converts low-level simulator state (ball position, ball velocity,
contact sensors) into a compact stream of *event flags* that the high-level
:class:`RallyCommand` finite-state machine can consume.

Design notes
------------
- Events are returned as an ``IntFlag``-encoded ``int64`` tensor of shape
  ``(num_envs,)``. Each bit corresponds to one event class (see
  :class:`EventCode`). Multiple events can fire in the same step.
- Edge detection (e.g., the bounce moment, the racket-hit moment) is *not*
  done by this module — callers maintain whatever rolling state they need
  and call the helpers each step. The helpers are pure functions of the
  current step's state and a small ``prev_state`` dict supplied by the
  caller.
- Court-polygon checks here are deliberately simple axis-aligned bounds.
  They can be replaced by polygon membership tests later without changing
  the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING

import torch

from mjlab.entity.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.tasks.tennis.mdp.ball_providers import BallProviderCfg

# ---------------------------------------------------------------------------
# Event codes.
# ---------------------------------------------------------------------------


class EventCode(IntFlag):
  """Bit-mask of tennis events that can fire on a single physics step."""

  NONE = 0
  RACKET_HIT = 1 << 0
  BOUNCE = 1 << 1
  BOUNCE_IN_SELF = 1 << 2
  BOUNCE_IN_OPP = 1 << 3
  BOUNCE_OUT = 1 << 4
  NET_TOUCH = 1 << 5
  CROSSED_NET_TO_OPP = 1 << 6
  CROSSED_NET_TO_SELF = 1 << 7
  BALL_OUT_OF_PLAY = 1 << 8


# ---------------------------------------------------------------------------
# Court bounds (singles, stylized half-scale court — see ``scene.py``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CourtBounds:
  """Axis-aligned bounds describing the in-play region of each side.

  Defaults match the G1-scaled tennis court built in
  :mod:`mjlab.tasks.tennis.scene` (each side is 7 m x 4.8 m, net at x = 0).
  """

  self_x: tuple[float, float] = (0.0, 7.0)
  opp_x: tuple[float, float] = (-7.0, 0.0)
  y_range: tuple[float, float] = (-2.4, 2.4)
  ball_z_floor: float = 0.06  # ball-radius (~0.034) + small tolerance
  net_x: float = 0.0


# ---------------------------------------------------------------------------
# Per-step state used by the detector.
# ---------------------------------------------------------------------------


@dataclass
class EventState:
  """Persistent per-env state that the detector needs across calls.

  All tensors are shape ``(num_envs, ...)``.
  """

  prev_contact: torch.Tensor  # bool, (B,)
  prev_ball_vz: torch.Tensor  # float, (B,)
  prev_ball_x: torch.Tensor  # float, (B,)
  prev_net_contact: torch.Tensor  # bool, (B,)
  last_bounce_pos: torch.Tensor  # float, (B, 3)

  @classmethod
  def zeros(cls, num_envs: int, device: str | torch.device) -> "EventState":
    return cls(
      prev_contact=torch.zeros(num_envs, dtype=torch.bool, device=device),
      prev_ball_vz=torch.zeros(num_envs, device=device),
      prev_ball_x=torch.zeros(num_envs, device=device),
      prev_net_contact=torch.zeros(num_envs, dtype=torch.bool, device=device),
      last_bounce_pos=torch.zeros(num_envs, 3, device=device),
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.prev_contact[env_ids] = False
    self.prev_ball_vz[env_ids] = 0.0
    self.prev_ball_x[env_ids] = 0.0
    self.prev_net_contact[env_ids] = False
    self.last_bounce_pos[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Detector helpers.
# ---------------------------------------------------------------------------


def _sensor_active(
  env: "ManagerBasedRlEnv", name: str, threshold: float
) -> torch.Tensor:
  """Return a ``(B,)`` bool mask of envs in which the named sensor fires."""
  sensor: ContactSensor = env.scene[name]
  data = sensor.data
  if data.force is not None:
    mag = torch.linalg.vector_norm(data.force, dim=-1)
    return (mag > threshold).any(dim=1)
  if data.found is not None:
    return (data.found > 0).any(dim=1)
  raise RuntimeError(f"Contact sensor '{name}' must expose 'force' or 'found' fields.")


def detect_events(
  env: "ManagerBasedRlEnv",
  *,
  state: EventState,
  ball_cfg: SceneEntityCfg,
  bounds: CourtBounds,
  racket_ball_sensor: str,
  ball_net_sensor: str | None = None,
  hit_force_threshold: float = 1.0,
  net_force_threshold: float = 0.5,
  out_of_play_z: float = 2.6,
) -> torch.Tensor:
  """Run all event detectors and return per-env flag tensor.

  Returns
  -------
  flags : torch.Tensor, shape ``(B,)``, dtype ``torch.long``
    Bit-OR of any :class:`EventCode` values that fired this step.
  """
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w - env.scene.env_origins  # (B, 3)
  ball_vel = ball.data.root_link_lin_vel_w  # (B, 3) world frame

  flags = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  # --- Racket hit (rising edge of contact sensor) -------------------------
  contact_now = _sensor_active(env, racket_ball_sensor, hit_force_threshold)
  hit_edge = contact_now & ~state.prev_contact
  flags |= hit_edge.long() * int(EventCode.RACKET_HIT)

  # --- Net touch (rising edge) -------------------------------------------
  if ball_net_sensor is not None:
    net_now = _sensor_active(env, ball_net_sensor, net_force_threshold)
    net_edge = net_now & ~state.prev_net_contact
    flags |= net_edge.long() * int(EventCode.NET_TOUCH)
    state.prev_net_contact = net_now

  # --- Bounce: vz transitions from negative to >=0 while ball is low ----
  vz = ball_vel[:, 2]
  bounce_edge = (
    (state.prev_ball_vz < 0.0)
    & (vz >= 0.0)
    & (ball_pos[:, 2] < bounds.ball_z_floor + 0.05)
  )
  flags |= bounce_edge.long() * int(EventCode.BOUNCE)

  # Cache bounce position for envs where bounce fired.
  if bounce_edge.any():
    idx = bounce_edge.nonzero().flatten()
    state.last_bounce_pos[idx] = ball_pos[idx]

  # In/out classification at bounce moment.
  in_y = (ball_pos[:, 1] >= bounds.y_range[0]) & (ball_pos[:, 1] <= bounds.y_range[1])
  in_self = (
    (ball_pos[:, 0] >= bounds.self_x[0]) & (ball_pos[:, 0] <= bounds.self_x[1]) & in_y
  )
  in_opp = (
    (ball_pos[:, 0] >= bounds.opp_x[0]) & (ball_pos[:, 0] <= bounds.opp_x[1]) & in_y
  )
  flags |= (bounce_edge & in_self).long() * int(EventCode.BOUNCE_IN_SELF)
  flags |= (bounce_edge & in_opp).long() * int(EventCode.BOUNCE_IN_OPP)
  flags |= (bounce_edge & ~(in_self | in_opp)).long() * int(EventCode.BOUNCE_OUT)

  # --- Net-plane crossings (sign change in x) ---------------------------
  cross_to_opp = (state.prev_ball_x > bounds.net_x) & (ball_pos[:, 0] <= bounds.net_x)
  cross_to_self = (state.prev_ball_x < bounds.net_x) & (ball_pos[:, 0] >= bounds.net_x)
  flags |= cross_to_opp.long() * int(EventCode.CROSSED_NET_TO_OPP)
  flags |= cross_to_self.long() * int(EventCode.CROSSED_NET_TO_SELF)

  # --- Out of playable volume (very loose envelope) ---------------------
  out = (
    (ball_pos[:, 0] < bounds.opp_x[0] - 1.0)
    | (ball_pos[:, 0] > bounds.self_x[1] + 1.0)
    | (ball_pos[:, 1].abs() > bounds.y_range[1] + 0.5)
    | (ball_pos[:, 2] > out_of_play_z)
  )
  flags |= out.long() * int(EventCode.BALL_OUT_OF_PLAY)

  # --- Update persistent state ------------------------------------------
  state.prev_contact = contact_now
  state.prev_ball_vz = vz.clone()
  state.prev_ball_x = ball_pos[:, 0].clone()

  return flags


def has_event(flags: torch.Tensor, code: EventCode) -> torch.Tensor:
  """Return bool mask: which envs have ``code`` set in their event flags."""
  return (flags & int(code)) != 0


def spawn_ball_from_provider(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  provider_cfg: "BallProviderCfg",
) -> None:
  """Event-manager-compatible wrapper that spawns the ball via a :class:`BallProvider`.

  This lets the Hit task reuse the same ballistic-trajectory spawning logic
  as the Return task without adopting the full Rally FSM.  The provider is
  instantiated lazily on first call and cached on the env under a private
  attribute keyed by the provider config id.

  Parameters
  ----------
  env :
    The running environment.
  env_ids :
    Indices of environments to reset.
  provider_cfg :
    A :class:`BallProviderCfg` instance (e.g. ``RandomFeederCfg``).
    The same config object is reused across calls; changing its fields
    at runtime will affect subsequent spawns.
  """
  from mjlab.tasks.tennis.mdp.ball_providers import BallProvider

  cache_key = f"_ball_provider_{id(provider_cfg)}"
  provider: BallProvider | None = getattr(env, cache_key, None)
  if provider is None:
    provider = provider_cfg.build(env)
    setattr(env, cache_key, provider)
  provider.spawn(env_ids)
