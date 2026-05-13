"""网球回球指令项。

:class:`RallyCommand` 是*当前得分状态*（FSM 阶段、最新事件、得分结果、
比分、指标）的唯一权威中心。它驱动高层网球规则层：

- 持有一个 :class:`BallProvider`，在得分开始时生成球，并可选地
  在回球过程中响应。
- 每步调用 :func:`detect_events`，并应用小型有限状态机以维护
  ``phase`` 和 ``is_point_end``。
- 暴露简洁的浮点“指令向量”，供观测/奖励项通过
  :func:`mdp.generated_commands` 消费。

奖励/终止项读取 ``rally.is_point_end``、
``rally.point_winner`` 和 ``rally.last_events``，而不是
从传感器重新推导事件。这将规则逻辑集中在一处。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import torch

from mjlab.entity.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tennis.mdp.ball_providers import (
  BallProvider,
  BallProviderCfg,
)
from mjlab.tasks.tennis.mdp.events import (
  CourtBounds,
  EventCode,
  EventState,
  detect_events,
  has_event,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# 阶段枚举。
# ---------------------------------------------------------------------------


class BallPhase(IntEnum):
  """单个回球得分的粗粒度 FSM 阶段。"""

  IDLE = 0
  SERVE = 1  # ball just spawned, no contact yet
  IN_FLIGHT = 2  # ball travelling toward player, not yet bounced on self
  BOUNCED = 3  # ball bounced on self side, agent must hit
  RETURN = 4  # agent has hit the ball, watching for outcome
  POINT_END = 5


_NUM_PHASES = len(BallPhase)


# ---------------------------------------------------------------------------
# 规则 + 指令配置。
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RulesCfg:
  """回球 FSM 使用的静态规则参数。"""

  bounds: CourtBounds = field(default_factory=CourtBounds)
  hit_force_threshold: float = 1.0
  net_force_threshold: float = 0.5
  valid_hit_min_leftward_speed: float = 2.0
  valid_hit_min_ball_speed: float = 2.5
  target_line_x: float = -2.2
  # 如果为 True，就算尚无任何接触，超出可玩球也会结束得分。
  end_on_ball_out_of_play: bool = True


@dataclass(kw_only=True)
class RallyCommandCfg(CommandTermCfg):
  """:class:`RallyCommand` 的配置。

  ``resampling_time_range`` 默认为一个哨兵値，实质上禁用时间驱动的重采样器；
  回球在回合重置时重采样（= 球重新生成），若 ``auto_respawn_on_point_end``
  为 True 则得分结束时也重采样。
  """

  ball_provider: BallProviderCfg | None = None
  """球提供器策略。必须在构建环境前设置。"""
  rules: RulesCfg = field(default_factory=RulesCfg)

  ball_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("ball"))
  racket_ball_sensor: str = "racket_ball_contact"
  ball_net_sensor: str | None = None  # 如果存在球-网接触传感器则设置

  resampling_time_range: tuple[float, float] = (1e9, 1e9)
  episode_granularity: bool = True  # 每回合一个得分（推荐）

  def build(self, env: "ManagerBasedRlEnv") -> "RallyCommand":
    return RallyCommand(self, env)


# ---------------------------------------------------------------------------
# 指令项。
# ---------------------------------------------------------------------------


class RallyCommand(CommandTerm):
  """网球回球有限状态机 + 球提供器驱动器。"""

  cfg: RallyCommandCfg

  def __init__(self, cfg: RallyCommandCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)

    if cfg.ball_provider is None:
      raise ValueError("RallyCommandCfg.ball_provider must be set before building.")
    self._ball: Entity = env.scene[cfg.ball_cfg.name]
    self._provider: BallProvider = cfg.ball_provider.build(env)
    self._rules: RulesCfg = cfg.rules

    dev = env.device
    B = env.num_envs

    # FSM 状态（每环境）。
    self.phase = torch.zeros(B, dtype=torch.long, device=dev)
    self.last_events = torch.zeros(B, dtype=torch.long, device=dev)
    self.is_point_end = torch.zeros(B, dtype=torch.bool, device=dev)
    self.point_winner = torch.zeros(B, dtype=torch.long, device=dev)  # +1 己方, -1 对手
    self.bounce_xy = torch.zeros(B, 2, device=dev)
    self.has_valid_hit = torch.zeros(B, dtype=torch.bool, device=dev)

    # 事件检测器持久状态。
    self._event_state = EventState.zeros(B, dev)

    # rsl-rl 日志用指标（重置时对各环境求均値）。
    self.metrics["points_won"] = torch.zeros(B, device=dev)
    self.metrics["valid_hits"] = torch.zeros(B, device=dev)
    self.metrics["over_net"] = torch.zeros(B, device=dev)
    self.metrics["bounce_in_opp"] = torch.zeros(B, device=dev)

    # 缓存指令张量（阶段独热编码 + 标量）。
    self._command = torch.zeros(B, _NUM_PHASES + 4, device=dev)

  # ---- CommandTerm API -------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self._command

  @property
  def provider(self) -> BallProvider:
    return self._provider

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    # 清除 FSM 状态。
    self.phase[env_ids] = int(BallPhase.SERVE)
    self.last_events[env_ids] = 0
    self.is_point_end[env_ids] = False
    self.point_winner[env_ids] = 0
    self.bounce_xy[env_ids] = 0.0
    self.has_valid_hit[env_ids] = False
    self._event_state.reset(env_ids)
    # 通过提供器生成全新的球。
    self._provider.spawn(env_ids)

  def _update_command(self) -> None:
    self._step_fsm()
    self._refresh_command_tensor()

  def _update_metrics(self) -> None:
    # 指标在 ``_step_fsm`` 中随事件触发累积。
    pass

  # ---- FSM 核心 --------------------------------------------------------

  def _step_fsm(self) -> None:
    flags = detect_events(
      self._env,
      state=self._event_state,
      ball_cfg=self.cfg.ball_cfg,
      bounds=self._rules.bounds,
      racket_ball_sensor=self.cfg.racket_ball_sensor,
      ball_net_sensor=self.cfg.ball_net_sensor,
      hit_force_threshold=self._rules.hit_force_threshold,
      net_force_threshold=self._rules.net_force_threshold,
    )
    self.last_events = flags

    e_hit = has_event(flags, EventCode.RACKET_HIT)
    e_bounce_self = has_event(flags, EventCode.BOUNCE_IN_SELF)
    e_bounce_opp = has_event(flags, EventCode.BOUNCE_IN_OPP)
    e_bounce_out = has_event(flags, EventCode.BOUNCE_OUT)
    e_net = has_event(flags, EventCode.NET_TOUCH)
    e_out = has_event(flags, EventCode.BALL_OUT_OF_PLAY)

    # Determine if this hit qualified as a "valid" return shot.
    ball = self._ball
    ball_lin = ball.data.root_link_lin_vel_w
    ball_speed = torch.linalg.vector_norm(ball_lin, dim=-1)
    leftward = -ball_lin[:, 0]
    valid_hit_now = (
      e_hit
      & (leftward >= self._rules.valid_hit_min_leftward_speed)
      & (ball_speed >= self._rules.valid_hit_min_ball_speed)
    )
    self.has_valid_hit |= valid_hit_now

    # --- 阶段转换（向量化）------------------------------------------
    # SERVE/IN_FLIGHT → BOUNCED：己方侧弹跳。
    pre_bounce_mask = (
      (self.phase == int(BallPhase.SERVE)) | (self.phase == int(BallPhase.IN_FLIGHT))
    ) & e_bounce_self
    self.phase = torch.where(
      pre_bounce_mask,
      torch.full_like(self.phase, int(BallPhase.BOUNCED)),
      self.phase,
    )

    # SERVE → IN_FLIGHT：球过网平面（尽力而为）。
    in_flight_mask = (self.phase == int(BallPhase.SERVE)) & ~pre_bounce_mask
    # 无条件地将 SERVE 在生成后一步转为 IN_FLIGHT。
    self.phase = torch.where(
      in_flight_mask,
      torch.full_like(self.phase, int(BallPhase.IN_FLIGHT)),
      self.phase,
    )

    # BOUNCED → RETURN：有效球拍击球。
    return_mask = (self.phase == int(BallPhase.BOUNCED)) & valid_hit_now
    self.phase = torch.where(
      return_mask,
      torch.full_like(self.phase, int(BallPhase.RETURN)),
      self.phase,
    )

    # 从事件状态缓存弹跳位置，供奖励项使用。
    bounce_mask = e_bounce_self | e_bounce_opp | e_bounce_out
    if bounce_mask.any():
      idx = bounce_mask.nonzero().flatten()
      self.bounce_xy[idx] = self._event_state.last_bounce_pos[idx, :2]

    # --- 得分结束条件 -----------------------------------------------
    # 1. 己方得分：有效回球后球在对方侧弹跳。
    self_win = (self.phase == int(BallPhase.RETURN)) & e_bounce_opp
    # 2. 对手得分：球越界、网打球、有效击球后在己方侧二次弹跳、
    #    球超出可玩空间，或未击球就在己方侧弹跳两次。
    opp_win_out = e_bounce_out
    opp_win_net = e_net
    opp_win_double = (self.phase == int(BallPhase.BOUNCED)) & e_bounce_self
    opp_win_oop = e_out & (self._rules.end_on_ball_out_of_play)
    opp_win = opp_win_out | opp_win_net | opp_win_double | opp_win_oop

    new_end = (~self.is_point_end) & (self_win | opp_win)
    self.is_point_end |= new_end
    self.point_winner = torch.where(
      new_end & self_win, torch.ones_like(self.point_winner), self.point_winner
    )
    self.point_winner = torch.where(
      new_end & opp_win & ~self_win,
      -torch.ones_like(self.point_winner),
      self.point_winner,
    )
    self.phase = torch.where(
      new_end, torch.full_like(self.phase, int(BallPhase.POINT_END)), self.phase
    )

    # --- 指标累积 ---------------------------------------------------
    self.metrics["points_won"] += (new_end & self_win).float()
    self.metrics["valid_hits"] += valid_hit_now.float()
    self.metrics["over_net"] += has_event(flags, EventCode.CROSSED_NET_TO_OPP).float()
    self.metrics["bounce_in_opp"] += e_bounce_opp.float()

  def _refresh_command_tensor(self) -> None:
    one_hot = torch.nn.functional.one_hot(self.phase, _NUM_PHASES).float()
    extras = torch.stack(
      [
        self.has_valid_hit.float(),
        self.is_point_end.float(),
        self.point_winner.float(),
        torch.clamp(self.metrics["valid_hits"] * 0.25, 0.0, 1.0),
      ],
      dim=-1,
    )
    self._command = torch.cat([one_hot, extras], dim=-1)


__all__ = [
  "BallPhase",
  "RallyCommand",
  "RallyCommandCfg",
  "RulesCfg",
]
