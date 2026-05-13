"""Reward terms for tennis latent-control tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.hit_state import TennisHitStateTerm
from mjlab.tasks.tennis.mdp.observations import racket_to_ball_b

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT_CFG = SceneEntityCfg("robot")
_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_BALL_CFG = SceneEntityCfg("ball")


def racket_ball_distance_exp(
  env: ManagerBasedRlEnv,
  std: float,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Dense reward for bringing the racket center close to the ball."""
  delta_b = racket_to_ball_b(env, racket_cfg, ball_cfg, robot_cfg)
  error = torch.sum(torch.square(delta_b), dim=-1)
  return torch.exp(-error / std**2)


def termination_term(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  """Return a termination mask as a float reward signal."""
  return env.termination_manager.get_term(term_name).float()


def termination_terms_any(
  env: ManagerBasedRlEnv, term_names: tuple[str, ...]
) -> torch.Tensor:
  """Return 1 if any named termination term fired this step."""
  if len(term_names) == 0:
    return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
  stacked = torch.stack(
    [env.termination_manager.get_term(name) for name in term_names], dim=0
  )
  return stacked.any(dim=0).float()


class approach_ball_pre_hit(TennisHitStateTerm):
  """Dense reward for approaching the ball before the first valid hit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    sensor_name: str,
    force_threshold: float = 1.0,
    valid_leftward_speed: float = 2.0,
    valid_ball_speed: float = 2.5,
    target_line_x: float = -2.2,
    miss_x_offset: float = 0.2,
    miss_x_direction: float = 1.0,
    racket_cfg: SceneEntityCfg = _RACKET_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del sensor_name
    del force_threshold
    del valid_leftward_speed
    del valid_ball_speed
    del target_line_x
    del miss_x_offset
    del miss_x_direction
    state = self.state
    reward = racket_ball_distance_exp(env, std, racket_cfg, ball_cfg, robot_cfg)
    return reward * (~state.has_valid_hit).float()


class closing_ball_pre_hit(TennisHitStateTerm):
  """Reward racket velocity that closes the distance to the incoming ball."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    max_speed: float,
    sensor_name: str,
    force_threshold: float = 1.0,
    valid_leftward_speed: float = 2.0,
    valid_ball_speed: float = 2.5,
    target_line_x: float = -2.2,
    miss_x_offset: float = 0.2,
    miss_x_direction: float = 1.0,
    racket_cfg: SceneEntityCfg = _RACKET_CFG,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    robot_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del sensor_name
    del force_threshold
    del valid_leftward_speed
    del valid_ball_speed
    del target_line_x
    del miss_x_offset
    del miss_x_direction
    state = self.state
    robot: Entity = env.scene[robot_cfg.name]
    ball: Entity = env.scene[ball_cfg.name]
    racket_pos = robot.data.site_pos_w[:, racket_cfg.site_ids].squeeze(1)
    racket_vel = robot.data.site_lin_vel_w[:, racket_cfg.site_ids].squeeze(1)
    delta_w = ball.data.root_link_pos_w - racket_pos
    distance = torch.linalg.vector_norm(delta_w, dim=1).clamp_min(1e-6)
    direction = delta_w / distance.unsqueeze(-1)
    relative_v = racket_vel - ball.data.root_link_lin_vel_w
    closing_speed = torch.clamp(
      torch.sum(relative_v * direction, dim=1), 0.0, max_speed
    )
    return (~state.has_valid_hit).float() * (closing_speed / max_speed)


class first_valid_hit_reward(TennisHitStateTerm):
  """Large sparse reward for the first valid directional hit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    valid_leftward_speed: float = 2.0,
    valid_ball_speed: float = 2.5,
    target_line_x: float = -2.2,
    miss_x_offset: float = 0.2,
    miss_x_direction: float = 1.0,
  ) -> torch.Tensor:
    del env
    del sensor_name
    del force_threshold
    del valid_leftward_speed
    del valid_ball_speed
    del target_line_x
    del miss_x_offset
    del miss_x_direction
    return self.state.first_valid_hit.float()


class post_hit_ball_leftward_speed(TennisHitStateTerm):
  """Reward sustaining post-hit ball velocity toward the target side."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    max_speed: float,
    sensor_name: str,
    force_threshold: float = 1.0,
    valid_leftward_speed: float = 2.0,
    valid_ball_speed: float = 2.5,
    target_line_x: float = -2.2,
    miss_x_offset: float = 0.2,
    miss_x_direction: float = 1.0,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
  ) -> torch.Tensor:
    del sensor_name
    del force_threshold
    del valid_leftward_speed
    del valid_ball_speed
    del target_line_x
    del miss_x_offset
    del miss_x_direction
    state = self.state
    ball: Entity = env.scene[ball_cfg.name]
    leftward_speed = torch.clamp(-ball.data.root_link_lin_vel_w[:, 0], 0.0, max_speed)
    return state.has_valid_hit.float() * (leftward_speed / max_speed)


def low_level_action_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str,
) -> torch.Tensor:
  """Penalize changes in decoded low-level joint actions."""
  term = env.action_manager.get_term(action_name)
  action = getattr(term, "low_level_action", None)
  prev_action = getattr(term, "prev_low_level_action", None)
  if action is None or prev_action is None:
    raise ValueError(
      f"Action term {action_name!r} does not expose low-level action history."
    )
  return torch.sum(torch.square(action - prev_action), dim=1)


# ---------------------------------------------------------------------------
# Rally-command-driven rewards (used by the new return task).
# ---------------------------------------------------------------------------


def rally_point_won(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """+1 on the step the player wins a point, 0 otherwise."""
  from mjlab.tasks.tennis.mdp.commands import RallyCommand

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  return (rally.is_point_end & (rally.point_winner > 0)).float()


def rally_point_lost(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """+1 on the step the opponent wins a point (penalty when used with neg weight)."""
  from mjlab.tasks.tennis.mdp.commands import RallyCommand

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  return (rally.is_point_end & (rally.point_winner < 0)).float()


def rally_valid_hit_event(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """Edge reward: +1 only on the single step where a *valid* hit is first registered.

  Bug fix: previous version combined ``hit_now & has_valid_hit``, which fired
  on every subsequent contact after the first valid hit. Correct logic is to
  check ``valid_hit_now`` — the edge that occurs only when the hit *becomes*
  valid (speed thresholds satisfied at the moment of racket contact).
  """
  from mjlab.tasks.tennis.mdp.commands import RallyCommand
  from mjlab.tasks.tennis.mdp.events import EventCode, has_event

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  # ``valid_hit_now`` is computed inside _step_fsm and stored as the edge that
  # caused has_valid_hit to first become True. We re-derive it here from the
  # event flags (RACKET_HIT) plus per-step velocity state on the ball.
  hit_now = has_event(rally.last_events, EventCode.RACKET_HIT)
  ball = env.scene[rally.cfg.ball_cfg.name]
  ball_lin = ball.data.root_link_lin_vel_w
  ball_speed = torch.linalg.vector_norm(ball_lin, dim=-1)
  leftward = -ball_lin[:, 0]
  valid_hit_now = (
    hit_now
    & (leftward >= rally._rules.valid_hit_min_leftward_speed)
    & (ball_speed >= rally._rules.valid_hit_min_ball_speed)
  )
  return valid_hit_now.float()


def rally_over_net_event(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """Edge reward: +1 when the ball crosses the net toward the opponent.

  Bug fix: gated to RETURN phase so it only fires after a valid hit, not during
  incoming ball flight (which also triggers CROSSED_NET_TO_OPP on approach).
  """
  from mjlab.tasks.tennis.mdp.commands import BallPhase, RallyCommand
  from mjlab.tasks.tennis.mdp.events import EventCode, has_event

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  in_return = rally.phase == int(BallPhase.RETURN)
  return (
    has_event(rally.last_events, EventCode.CROSSED_NET_TO_OPP) & in_return
  ).float()


def rally_approach_ball_pre_hit(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
  std: float = 0.4,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Dense approach reward gated to phases where the ball is incoming (pre-hit).

  Bug fix: plain ``racket_ball_distance_exp`` has no phase mask, so it keeps
  rewarding after the ball has been struck and is flying away, which creates a
  gradient toward chasing the departing ball. Gate to SERVE/IN_FLIGHT/BOUNCED.
  """
  from mjlab.tasks.tennis.mdp.commands import BallPhase, RallyCommand

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  pre_hit = (
    (rally.phase == int(BallPhase.SERVE))
    | (rally.phase == int(BallPhase.IN_FLIGHT))
    | (rally.phase == int(BallPhase.BOUNCED))
  )
  reward = racket_ball_distance_exp(env, std, racket_cfg, ball_cfg, robot_cfg)
  return reward * pre_hit.float()


def rally_hit_ball_speed_bonus(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
  max_speed: float = 8.0,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
) -> torch.Tensor:
  """One-shot speed bonus on the valid-hit edge.

  Bug fix for ``post_hit_ball_leftward_speed``: that term fired every step
  after a valid hit, but episodes typically terminate on the same or next
  step after ``successful_return``, so the dense term accumulated ~0 integral.
  Replacing with an edge-based bonus that fires once makes the shaping signal
  reliable and removes the erroneous post-termination reward.
  """
  from mjlab.tasks.tennis.mdp.commands import RallyCommand
  from mjlab.tasks.tennis.mdp.events import EventCode, has_event

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  hit_now = has_event(rally.last_events, EventCode.RACKET_HIT)
  ball = env.scene[ball_cfg.name]
  leftward = torch.clamp(-ball.data.root_link_lin_vel_w[:, 0], 0.0, max_speed)
  return hit_now.float() * (leftward / max_speed)


# ---------------------------------------------------------------------------
# Refactored Hit-task rewards (built on TennisRallyTracker).
# ---------------------------------------------------------------------------

from mjlab.tasks.tennis.mdp.hit_state import TennisRallyTrackerTerm  # noqa: E402


def racket_to_ball_distance_dense(
  env: ManagerBasedRlEnv,
  std: float,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Always-on dense reward for racket-to-ball proximity.

  Unlike :func:`approach_ball_pre_hit`, this term has no phase mask: it
  rewards proximity throughout the episode, which works for the simplified
  Hit task that ends on the first major ball event anyway.
  """
  return racket_ball_distance_exp(env, std, racket_cfg, ball_cfg, robot_cfg)


class racket_hit_event(TennisRallyTrackerTerm):
  """Sparse one-shot reward for the first racket-ball contact."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
  ) -> torch.Tensor:
    del env, sensor_name, ball_cfg, force_threshold, ground_z, net_x
    t = self.tracker
    # Only reward the very first racket hit (count incremented this step).
    return (t.racket_hit_edge & (t.racket_hit_count == 1)).float()


class crossed_net_event(TennisRallyTrackerTerm):
  """Sparse one-shot reward when the ball first crosses the net after a hit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ball_cfg: SceneEntityCfg = _BALL_CFG,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
  ) -> torch.Tensor:
    del env, sensor_name, ball_cfg, force_threshold, ground_z, net_x
    return self.tracker.crossed_net_after_hit_edge.float()
