"""网球事件检测。

本模块是网球任务的单一向量化事件检测层。它将低层仿真器状态（球位置、球速度、
接触传感器）转换为紧凑的*事件标志*流，供高层
:class:`RallyCommand` 有限状态机消费。

设计说明
--------
- 事件以 ``IntFlag`` 编码的 ``int64`` 张量形式返回，形状为
  ``(num_envs,)``。每个比特对应一种事件类（见
  :class:`EventCode`）。同一步可同时触发多个事件。
- 边沿检测（如弹跳时刻、球拍击球时刻）不由本模块负责——调用方
  维护所需的滚动状态，并在每步调用辅助函数时传入小型
  ``prev_state`` 字典。辅助函数是当前步状态与调用方提供的
  ``prev_state`` 的纯函数。
- 本模块的球场多边形检测刻意采用简单的轴对齐边界框。
  日后可替换为多边形成员检测，而不改变公共 API。
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
# 事件编码。
# ---------------------------------------------------------------------------


class EventCode(IntFlag):
  """单个物理步可触发的网球事件位掩码。"""

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
# 球场边界（单打，风格化半尺度球场——见 ``scene.py``）。
# ---------------------------------------------------------------------------

:class:`RallyCommand` 是*当前得分状态*（FSM 阶段、最新事件、得分结果、

@dataclass(frozen=True)
class CourtBounds:
  """描述每侧可玩区域的轴对齐边界。

  默认属性与 :mod:`mjlab.tasks.tennis.scene` 中构建的
  G1 缩放网球场匹配（每侧 7 m x 4.8 m，网在 x = 0）。
  """

  self_x: tuple[float, float] = (0.0, 7.0)
  opp_x: tuple[float, float] = (-7.0, 0.0)
  y_range: tuple[float, float] = (-2.4, 2.4)
  ball_z_floor: float = 0.06  # 球半径（约0.034）加小余量
  net_x: float = 0.0


# ---------------------------------------------------------------------------
# 检测器使用的每步持久状态。
# ---------------------------------------------------------------------------


@dataclass
class EventState:
  """检测器跨调用所需的每环境持久状态。

  所有张量形状均为 ``(num_envs, ...)``。
  """

  prev_contact: torch.Tensor  # bool, (B,)
  prev_ball_vz: torch.Tensor  # float, (B,)
  prev_ball_x: torch.Tensor  # float, (B,)
  prev_net_contact: torch.Tensor  # bool, (B,)
  last_bounce_pos: torch.Tensor  # float, (B, 3)
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
# 检测器辅助函数。
# ---------------------------------------------------------------------------


def _sensor_active(
  env: "ManagerBasedRlEnv", name: str, threshold: float
) -> torch.Tensor:
  """返回命名传感器触发的环境的 ``(B,)`` 布尔掩码。"""
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
  """运行所有事件检测器并返回每环境标志张量。

  Returns
  -------
  flags : torch.Tensor, shape ``(B,)``, dtype ``torch.long``
    本步触发的所有 :class:`EventCode` 值的按位或结果。
  """
  ball: Entity = env.scene[ball_cfg.name]
  ball_pos = ball.data.root_link_pos_w - env.scene.env_origins  # (B, 3)
  ball_vel = ball.data.root_link_lin_vel_w  # (B, 3) world frame

  flags = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  # --- 球拍击球（接触传感器的上升边沿）-----------------------------------
  contact_now = _sensor_active(env, racket_ball_sensor, hit_force_threshold)
  hit_edge = contact_now & ~state.prev_contact
  flags |= hit_edge.long() * int(EventCode.RACKET_HIT)

  # --- 球网接触（上升边沿）------------------------------------------------
  if ball_net_sensor is not None:
    net_now = _sensor_active(env, ball_net_sensor, net_force_threshold)
    net_edge = net_now & ~state.prev_net_contact
    flags |= net_edge.long() * int(EventCode.NET_TOUCH)
    state.prev_net_contact = net_now

  # --- 弹跳：vz 从负变为 >=0 且球接近地面 --------------------------------
  vz = ball_vel[:, 2]
  bounce_edge = (
    (state.prev_ball_vz < 0.0)
    & (vz >= 0.0)
    & (ball_pos[:, 2] < bounds.ball_z_floor + 0.05)
  )
  flags |= bounce_edge.long() * int(EventCode.BOUNCE)

  # 缓存弹跳发生时各环境的球位置。
  if bounce_edge.any():
    idx = bounce_edge.nonzero().flatten()
    state.last_bounce_pos[idx] = ball_pos[idx]

  # 在弹跳时刻进行内/外分类。
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

  # --- 越网（x 符号改变）------------------------------------------------
  cross_to_opp = (state.prev_ball_x > bounds.net_x) & (ball_pos[:, 0] <= bounds.net_x)
  cross_to_self = (state.prev_ball_x < bounds.net_x) & (ball_pos[:, 0] >= bounds.net_x)
  flags |= cross_to_opp.long() * int(EventCode.CROSSED_NET_TO_OPP)
  flags |= cross_to_self.long() * int(EventCode.CROSSED_NET_TO_SELF)

  # --- 超出可玩空间（非常宽松的包围盒）-----------------------------------
  out = (
    (ball_pos[:, 0] < bounds.opp_x[0] - 1.0)
    | (ball_pos[:, 0] > bounds.self_x[1] + 1.0)
    | (ball_pos[:, 1].abs() > bounds.y_range[1] + 0.5)
    | (ball_pos[:, 2] > out_of_play_z)
  )
  flags |= out.long() * int(EventCode.BALL_OUT_OF_PLAY)

  # --- 更新持久状态 -------------------------------------------------------
  state.prev_contact = contact_now
  state.prev_ball_vz = vz.clone()
  state.prev_ball_x = ball_pos[:, 0].clone()

  return flags


def has_event(flags: torch.Tensor, code: EventCode) -> torch.Tensor:
  """返回布尔掩码：哪些环境在事件标志中设置了 ``code``。"""
  return (flags & int(code)) != 0


def spawn_ball_from_provider(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  provider_cfg: "BallProviderCfg",
) -> None:
  """兼容事件管理器的包装函数，通过 :class:`BallProvider` 生成球。

  允许击球任务复用与回球任务相同的弹道轨迹生成逻辑，而无需采用完整的
  Rally FSM。提供器在首次调用时延迟实例化，并以提供器配置 id 为键
  缓存到 env 的私有属性上。

  Parameters
  ----------
  env :
    运行中的环境。
  env_ids :
    需要重置的环境索引。
  provider_cfg :
    :class:`BallProviderCfg` 实例（例如 ``RandomFeederCfg``）。
    同一配置对象在调用间复用；运行时修改其字段将影响后续生成。
  """
  from mjlab.tasks.tennis.mdp.ball_providers import BallProvider

  cache_key = f"_ball_provider_{id(provider_cfg)}"
  provider: BallProvider | None = getattr(env, cache_key, None)
  if provider is None:
    provider = provider_cfg.build(env)
    setattr(env, cache_key, provider)
  provider.spawn(env_ids)
