"""Termination terms for tennis latent-control tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tennis.mdp.hit_state import TennisHitStateTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_BALL_CFG = SceneEntityCfg("ball")
BALL_MIN_HEIGHT = 0.05


def ball_in_play(
  env: ManagerBasedRlEnv,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  x_limits: tuple[float, float] = (-5.8, 3.6),
  y_limits: tuple[float, float] = (-2.7, 2.7),
  z_limits: tuple[float, float] = (BALL_MIN_HEIGHT, 2.6),
) -> torch.Tensor:
  """Terminate when the ball leaves the playable court workspace."""
  ball: Entity = env.scene[ball_cfg.name]
  pos = ball.data.root_link_pos_w - env.scene.env_origins
  out_x = (pos[:, 0] < x_limits[0]) | (pos[:, 0] > x_limits[1])
  out_y = (pos[:, 1] < y_limits[0]) | (pos[:, 1] > y_limits[1])
  out_z = (pos[:, 2] < z_limits[0]) | (pos[:, 2] > z_limits[1])
  return out_x | out_y | out_z


class miss_ball(TennisHitStateTerm):
  """Terminate when the ball has passed the racket plane without a valid hit."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
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
    return self.state.missed_ball


class second_contact_after_valid_hit(TennisHitStateTerm):
  """Terminate to stop the policy from juggling after a valid hit."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
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
    return self.state.repeat_contact_after_valid_hit


class successful_return(TennisHitStateTerm):
  """Terminate once a valid hit drives the ball through the target line."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
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
    return self.state.target_line_crossed_edge


def point_ended(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """Terminate when the rally command marks the current point as ended."""
  from mjlab.tasks.tennis.mdp.commands import RallyCommand

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  return rally.is_point_end


# ---------------------------------------------------------------------------
# Refactored Hit-task terminations (built on TennisRallyTracker).
# ---------------------------------------------------------------------------

from mjlab.tasks.tennis.mdp.hit_state import TennisRallyTrackerTerm  # noqa: E402


class second_contact(TennisRallyTrackerTerm):
  """End the episode once the ball has registered its second contact.

  A "contact" is either a racket hit or a ground bounce. The first hit
  starts the rally; the second contact (a return-bounce, a juggling
  re-hit, or simply the ball bouncing twice on the floor) ends it.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
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
    return self.tracker.total_contact_count >= 2


class crossed_net_after_hit(TennisRallyTrackerTerm):
  """End the episode the step the ball first crosses the net after a hit."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
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
    return self.tracker.crossed_net_after_hit_edge
