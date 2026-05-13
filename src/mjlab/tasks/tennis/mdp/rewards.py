"""网球潜变量控制任务的奖励项。"""

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
  """将球拍中心带近球的密集奖励。"""
  delta_b = racket_to_ball_b(env, racket_cfg, ball_cfg, robot_cfg)
  error = torch.sum(torch.square(delta_b), dim=-1)
  return torch.exp(-error / std**2)


def termination_term(env: ManagerBasedRlEnv, term_name: str) -> torch.Tensor:
  """将终止掩码作为浮点奖励信号返回。"""
  return env.termination_manager.get_term(term_name).float()


def termination_terms_any(
  env: ManagerBasedRlEnv, term_names: tuple[str, ...]
) -> torch.Tensor:
  """如果本步任何指定终止项激活，则返回 1。"""
  if len(term_names) == 0:
    return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
  stacked = torch.stack(
    [env.termination_manager.get_term(name) for name in term_names], dim=0
  )
  return stacked.any(dim=0).float()


class approach_ball_pre_hit(TennisHitStateTerm):
  """首次有效击球前走近球的密集奖励。"""

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
  """奖励球拍速度向来球靠近的分量。"""

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
  """首次有效定向击球的大稀疏奖励。"""

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
  """奖励击球后球的速度持续指向目标侧。"""

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
  """惩罚解码后低层关节动作的变化。"""
  term = env.action_manager.get_term(action_name)
  action = getattr(term, "low_level_action", None)
  prev_action = getattr(term, "prev_low_level_action", None)
  if action is None or prev_action is None:
    raise ValueError(
      f"Action term {action_name!r} does not expose low-level action history."
    )
  return torch.sum(torch.square(action - prev_action), dim=1)


# ---------------------------------------------------------------------------
# 回球指令驱动的奖励（用于新回球任务）。
# ---------------------------------------------------------------------------


def rally_point_won(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """玩家得分的步骤返回 +1，其他情况返回 0。"""
  from mjlab.tasks.tennis.mdp.commands import RallyCommand

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  return (rally.is_point_end & (rally.point_winner > 0)).float()


def rally_point_lost(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """对手得分的步骤返回 +1（配合负权重用作惩罚）。"""
  from mjlab.tasks.tennis.mdp.commands import RallyCommand

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  return (rally.is_point_end & (rally.point_winner < 0)).float()


def rally_valid_hit_event(
  env: ManagerBasedRlEnv,
  command_name: str = "rally",
) -> torch.Tensor:
  """边缘奖励：仅在首次登记到「有效」击球的单步返回 +1。

  错误修复：之前的版本组合了 ``hit_now & has_valid_hit``，导致首次
  有效击球后的每次后续接触都会触发。正确的逻辑是检查
  ``valid_hit_now``——仅当击球「变为」有效时（球拍接触时速度閘值满足）才会出现的边缘。
  """
  from mjlab.tasks.tennis.mdp.commands import RallyCommand
  from mjlab.tasks.tennis.mdp.events import EventCode, has_event

  rally = env.command_manager.get_term(command_name)
  assert isinstance(rally, RallyCommand)
  # ``valid_hit_now`` 在 _step_fsm 内部计算，存储为首次使 has_valid_hit 变为 True 的边缘。
  # 此处从事件标志（RACKET_HIT）以及球的每步速度状态重新推导。
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
  """边缘奖励：球过网向对手侧时返回 +1。

  错误修复：将触发限制在 RETURN 阶段，从而确保它仅在有效击球后触发，
  而不是在入球飞行时（入球近距也会触发 CROSSED_NET_TO_OPP）。
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
  """仅在来球阶段（击球前）开启的密集接近奖励。

  错误修复：纯 ``racket_ball_distance_exp`` 没有阶段掩码，击球后球飞离时
  仍继续奖励，导致产生追逐离开的球的梯度。将限制至 SERVE/IN_FLIGHT/BOUNCED 阶段。
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
  """有效击球边缘时的一次性速度奖励。

  修复 ``post_hit_ball_leftward_speed`` 的错误：该项在每次有效击球后的每步都会
  触发，但回合通常在 ``successful_return`` 后的同一步或下一步终止，
  因此密集项积分约为 0。改用一次性边缘奖励可使形塑信号可靠，
  并消除错误的终止后奖励。
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
# 重构后的击球任务奖励（基于 TennisRallyTracker）。
# ---------------------------------------------------------------------------

from mjlab.tasks.tennis.mdp.hit_state import TennisRallyTrackerTerm  # noqa: E402


def racket_to_ball_distance_dense(
  env: ManagerBasedRlEnv,
  std: float,
  racket_cfg: SceneEntityCfg = _RACKET_CFG,
  ball_cfg: SceneEntityCfg = _BALL_CFG,
  robot_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """常开的球拍到球距离密集奖励。

  与 :func:`approach_ball_pre_hit` 不同，该项没有阶段掩码：全回合奖励距离，
  适用于简化击球任务（该任务在首个重大球事件时就会结束）。
  """
  return racket_ball_distance_exp(env, std, racket_cfg, ball_cfg, robot_cfg)


class racket_hit_event(TennisRallyTrackerTerm):
  """首次球拍接触的稀疏一次性奖励。"""

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
    # 仅奖励第一次球拍击球（本步计数递增）。
    return (t.racket_hit_edge & (t.racket_hit_count == 1)).float()


class crossed_net_event(TennisRallyTrackerTerm):
  """球在击球后首次过网的稀疏一次性奖励。"""

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
