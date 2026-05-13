"""网球潜变量控制任务的共享击球阶段状态。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_RACKET_CFG = SceneEntityCfg("robot", site_names=("tennis_racket_center",))
_DEFAULT_BALL_CFG = SceneEntityCfg("ball")
_STATE_ATTR = "_tennis_hit_state"


class TennisHitState:
  """按环境追踪首次接触、有效击球和单次击球进度。

  该状态在奖励项和终止项之间共享。它同时保存当前任务使用的单次字段
  （首次有效击球、首次有效击球后的重复接触），以及一个与每次有效击球
  边沿通过 XOR 切换的回合奇偶校验位。
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str,
    force_threshold: float,
    valid_leftward_speed: float,
    valid_ball_speed: float,
    target_line_x: float,
    miss_x_offset: float,
    miss_x_direction: float,
    robot_cfg: SceneEntityCfg,
    racket_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
  ) -> None:
    self._env = env
    self.sensor_name = sensor_name
    self.force_threshold = force_threshold
    self.valid_leftward_speed = valid_leftward_speed
    self.valid_ball_speed = valid_ball_speed
    self.target_line_x = target_line_x
    self.miss_x_offset = miss_x_offset
    self.miss_x_direction = 1.0 if miss_x_direction >= 0.0 else -1.0
    self.robot_cfg = robot_cfg
    self.racket_cfg = racket_cfg
    self.ball_cfg = ball_cfg
    self._last_common_step = -1

    bool_shape = (env.num_envs,)
    long_shape = (env.num_envs,)
    self.contact = torch.zeros(bool_shape, dtype=torch.bool, device=env.device)
    self.contact_edge = torch.zeros_like(self.contact)
    self.first_contact = torch.zeros_like(self.contact)
    self.valid_hit_edge = torch.zeros_like(self.contact)
    self.first_valid_hit = torch.zeros_like(self.contact)
    self.has_valid_hit = torch.zeros_like(self.contact)
    self.repeat_contact_after_valid_hit = torch.zeros_like(self.contact)
    self.target_line_crossed = torch.zeros_like(self.contact)
    self.target_line_crossed_edge = torch.zeros_like(self.contact)
    self.missed_ball = torch.zeros_like(self.contact)
    self.rally_parity = torch.zeros_like(self.contact)
    self._prev_contact = torch.zeros_like(self.contact)

    self.contact_count = torch.zeros(long_shape, dtype=torch.long, device=env.device)
    self.valid_hit_count = torch.zeros(long_shape, dtype=torch.long, device=env.device)
    self.first_contact_step = torch.full(
      long_shape, -1, dtype=torch.long, device=env.device
    )
    self.first_valid_hit_step = torch.full(
      long_shape, -1, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.contact[env_ids] = False
    self.contact_edge[env_ids] = False
    self.first_contact[env_ids] = False
    self.valid_hit_edge[env_ids] = False
    self.first_valid_hit[env_ids] = False
    self.has_valid_hit[env_ids] = False
    self.repeat_contact_after_valid_hit[env_ids] = False
    self.target_line_crossed[env_ids] = False
    self.target_line_crossed_edge[env_ids] = False
    self.missed_ball[env_ids] = False
    self.rally_parity[env_ids] = False
    self._prev_contact[env_ids] = False
    self.contact_count[env_ids] = 0
    self.valid_hit_count[env_ids] = 0
    self.first_contact_step[env_ids] = -1
    self.first_valid_hit_step[env_ids] = -1
    self._last_common_step = -1

  def update(self) -> None:
    step = int(self._env.common_step_counter)
    if step == self._last_common_step:
      return

    contact_now = self._contact_mask()
    had_valid_hit = self.has_valid_hit.clone()
    contact_edge = contact_now & ~self._prev_contact

    ball = self._ball()
    robot = self._robot()
    ball_pos = ball.data.root_link_pos_w - self._env.scene.env_origins
    racket_pos = (
      robot.data.site_pos_w[:, self.racket_cfg.site_ids].squeeze(1)
      - self._env.scene.env_origins
    )
    ball_vel = ball.data.root_link_lin_vel_w
    ball_speed = torch.linalg.vector_norm(ball_vel, dim=1)
    leftward_speed = -ball_vel[:, 0]

    valid_hit_edge = (
      contact_edge
      & (leftward_speed >= self.valid_leftward_speed)
      & (ball_speed >= self.valid_ball_speed)
    )
    first_contact = contact_edge & (self.contact_count == 0)
    first_valid_hit = valid_hit_edge & ~had_valid_hit
    repeat_contact = contact_edge & had_valid_hit
    target_crossed = self.has_valid_hit | valid_hit_edge
    target_crossed &= ball_pos[:, 0] <= self.target_line_x
    target_crossed_edge = target_crossed & ~self.target_line_crossed
    if self.miss_x_direction > 0.0:
      ball_passed_racket = ball_pos[:, 0] >= racket_pos[:, 0] + self.miss_x_offset
    else:
      ball_passed_racket = ball_pos[:, 0] <= racket_pos[:, 0] - self.miss_x_offset
    missed_ball = (~(self.has_valid_hit | valid_hit_edge)) & ball_passed_racket

    self.contact[:] = contact_now
    self.contact_edge[:] = contact_edge
    self.first_contact[:] = first_contact
    self.valid_hit_edge[:] = valid_hit_edge
    self.first_valid_hit[:] = first_valid_hit
    self.repeat_contact_after_valid_hit[:] = repeat_contact
    self.contact_count += contact_edge.long()
    self.valid_hit_count += valid_hit_edge.long()
    self.has_valid_hit |= valid_hit_edge
    self.rally_parity[:] = torch.logical_xor(self.rally_parity, valid_hit_edge)
    self.target_line_crossed_edge[:] = target_crossed_edge
    self.target_line_crossed |= target_crossed
    self.missed_ball[:] = missed_ball

    current_step = torch.full_like(self.first_contact_step, step)
    self.first_contact_step[:] = torch.where(
      first_contact, current_step, self.first_contact_step
    )
    self.first_valid_hit_step[:] = torch.where(
      first_valid_hit, current_step, self.first_valid_hit_step
    )

    self._prev_contact[:] = contact_now
    self._last_common_step = step

  def _contact_mask(self) -> torch.Tensor:
    sensor: ContactSensor = self._env.scene[self.sensor_name]
    data = sensor.data
    if data.force is not None:
      force_mag = torch.linalg.vector_norm(data.force, dim=-1)
      return (force_mag > self.force_threshold).any(dim=1)
    if data.found is not None:
      return (data.found > 0).any(dim=1)
    raise ValueError(
      f"Contact sensor {self.sensor_name!r} must expose 'force' or 'found'."
    )

  def _robot(self) -> Entity:
    return self._env.scene[self.robot_cfg.name]

  def _ball(self) -> Entity:
    return self._env.scene[self.ball_cfg.name]


class TennisHitStateTerm:
  """依赖 ``TennisHitState`` 的奖励/终止项的混入类。"""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._state = get_tennis_hit_state(
      env,
      sensor_name=cfg.params["sensor_name"],
      force_threshold=float(cfg.params.get("force_threshold", 1.0)),
      valid_leftward_speed=float(cfg.params.get("valid_leftward_speed", 2.0)),
      valid_ball_speed=float(cfg.params.get("valid_ball_speed", 2.5)),
      target_line_x=float(cfg.params.get("target_line_x", -2.2)),
      miss_x_offset=float(cfg.params.get("miss_x_offset", 0.2)),
      miss_x_direction=float(cfg.params.get("miss_x_direction", 1.0)),
      robot_cfg=cfg.params.get("robot_cfg", _DEFAULT_ROBOT_CFG),
      racket_cfg=cfg.params.get("racket_cfg", _DEFAULT_RACKET_CFG),
      ball_cfg=cfg.params.get("ball_cfg", _DEFAULT_BALL_CFG),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._state.reset(env_ids)

  @property
  def state(self) -> TennisHitState:
    self._state.update()
    return self._state


def get_tennis_hit_state(
  env: ManagerBasedRlEnv,
  *,
  sensor_name: str,
  force_threshold: float = 1.0,
  valid_leftward_speed: float = 2.0,
  valid_ball_speed: float = 2.5,
  target_line_x: float = -2.2,
  miss_x_offset: float = 0.2,
  miss_x_direction: float = 1.0,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  racket_cfg: SceneEntityCfg = _DEFAULT_RACKET_CFG,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
) -> TennisHitState:
  state = getattr(env, _STATE_ATTR, None)
  if isinstance(state, TennisHitState):
    return state

  if racket_cfg.site_names is not None and not isinstance(racket_cfg.site_ids, list):
    racket_cfg.resolve(env.scene)

  state = TennisHitState(
    env,
    sensor_name=sensor_name,
    force_threshold=force_threshold,
    valid_leftward_speed=valid_leftward_speed,
    valid_ball_speed=valid_ball_speed,
    target_line_x=target_line_x,
    miss_x_offset=miss_x_offset,
    miss_x_direction=miss_x_direction,
    robot_cfg=robot_cfg,
    racket_cfg=racket_cfg,
    ball_cfg=ball_cfg,
  )
  setattr(env, _STATE_ATTR, state)
  return state



_RALLY_TRACKER_ATTR = "_tennis_rally_tracker"


class TennisRallyTracker:
  """为球拍击球、弹跳和越网事件提供按环境追踪。

  专为重构后的击球任务设计，该任务在首个重大球事件
  （越界 / 第二次接触 / 越网）发生时结束回合。
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    *,
    sensor_name: str,
    ball_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    ground_z: float = 0.06,
    net_x: float = 0.0,
  ) -> None:
    self._env = env
    self.sensor_name = sensor_name
    self.ball_cfg = ball_cfg
    self.force_threshold = force_threshold
    self.ground_z = ground_z
    self.net_x = net_x
    self._last_step = -1

    B = env.num_envs
    dev = env.device

    def z_long():
      return torch.zeros(B, dtype=torch.long, device=dev)

    def z_bool():
      return torch.zeros(B, dtype=torch.bool, device=dev)

    def z_float():
      return torch.zeros(B, device=dev)

    # Counts.
    self.racket_hit_count = z_long()
    self.bounce_count = z_long()

    # Per-step edges.
    self.racket_hit_edge = z_bool()
    self.bounce_edge = z_bool()
    self.crossed_net_after_hit_edge = z_bool()

    # Latched flags.
    self.has_racket_hit = z_bool()
    self.has_crossed_net_after_hit = z_bool()

    # Internal previous-step caches.
    self._prev_contact = z_bool()
    self._prev_vz = z_float()
    self._prev_x = z_float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.racket_hit_count[env_ids] = 0
    self.bounce_count[env_ids] = 0
    self.racket_hit_edge[env_ids] = False
    self.bounce_edge[env_ids] = False
    self.crossed_net_after_hit_edge[env_ids] = False
    self.has_racket_hit[env_ids] = False
    self.has_crossed_net_after_hit[env_ids] = False
    self._prev_contact[env_ids] = False
    self._prev_vz[env_ids] = 0.0
    self._prev_x[env_ids] = 0.0
    self._last_step = -1

  def update(self) -> None:
    step = int(self._env.common_step_counter)
    if step == self._last_step:
      return

    sensor: ContactSensor = self._env.scene[self.sensor_name]
    sdata = sensor.data
    if sdata.force is not None:
      mag = torch.linalg.vector_norm(sdata.force, dim=-1)
      contact_now = (mag > self.force_threshold).any(dim=1)
    elif sdata.found is not None:
      contact_now = (sdata.found > 0).any(dim=1)
    else:
      raise ValueError(
        f"Contact sensor {self.sensor_name!r} must expose 'force' or 'found'."
      )

    ball: Entity = self._env.scene[self.ball_cfg.name]
    ball_pos = ball.data.root_link_pos_w - self._env.scene.env_origins
    ball_vel = ball.data.root_link_lin_vel_w
    bx = ball_pos[:, 0]
    bz = ball_pos[:, 2]
    vz = ball_vel[:, 2]

    # 球拍击球上升边沿。
    racket_edge = contact_now & ~self._prev_contact

    # 弹跳：vz 从负变为非负，且球接近地面。
    bounce_edge = (self._prev_vz < 0.0) & (vz >= 0.0) & (bz < self.ground_z + 0.05)

    # 越网到对手侧，且至少已有一次球拍击球。
    has_hit_now = self.has_racket_hit | racket_edge
    cross_to_opp = (self._prev_x > self.net_x) & (bx <= self.net_x)
    cross_after_hit_edge = cross_to_opp & has_hit_now & ~self.has_crossed_net_after_hit

    # 提交。
    self.racket_hit_edge[:] = racket_edge
    self.bounce_edge[:] = bounce_edge
    self.crossed_net_after_hit_edge[:] = cross_after_hit_edge
    self.racket_hit_count += racket_edge.long()
    self.bounce_count += bounce_edge.long()
    self.has_racket_hit |= racket_edge
    self.has_crossed_net_after_hit |= cross_after_hit_edge

    self._prev_contact[:] = contact_now
    self._prev_vz[:] = vz
    self._prev_x[:] = bx
    self._last_step = step

  # Derived signals -----------------------------------------------------
  @property
  def total_contact_count(self) -> torch.Tensor:
    """Combined racket + ground contacts so far in the episode."""
    return self.racket_hit_count + self.bounce_count


def get_tennis_rally_tracker(
  env: ManagerBasedRlEnv,
  *,
  sensor_name: str,
  ball_cfg: SceneEntityCfg = _DEFAULT_BALL_CFG,
  force_threshold: float = 1.0,
  ground_z: float = 0.06,
  net_x: float = 0.0,
) -> TennisRallyTracker:
  tracker = getattr(env, _RALLY_TRACKER_ATTR, None)
  if isinstance(tracker, TennisRallyTracker):
    return tracker
  tracker = TennisRallyTracker(
    env,
    sensor_name=sensor_name,
    ball_cfg=ball_cfg,
    force_threshold=force_threshold,
    ground_z=ground_z,
    net_x=net_x,
  )
  setattr(env, _RALLY_TRACKER_ATTR, tracker)
  return tracker


class TennisRallyTrackerTerm:
  """依赖 TennisRallyTracker 的奖励/终止项的混入类。"""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._tracker = get_tennis_rally_tracker(
      env,
      sensor_name=cfg.params["sensor_name"],
      ball_cfg=cfg.params.get("ball_cfg", _DEFAULT_BALL_CFG),
      force_threshold=float(cfg.params.get("force_threshold", 1.0)),
      ground_z=float(cfg.params.get("ground_z", 0.06)),
      net_x=float(cfg.params.get("net_x", 0.0)),
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._tracker.reset(env_ids)

  @property
  def tracker(self) -> TennisRallyTracker:
    self._tracker.update()
    return self._tracker
