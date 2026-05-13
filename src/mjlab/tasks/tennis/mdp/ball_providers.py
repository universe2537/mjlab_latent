"""可插拔球提供器（P0 固定 / P1 随机发球器 / P2 弹道对手）。

*球提供器* 是 :class:`RallyCommand` 拥有的策略对象，负责两件事：

1. 在每个回合开始时**生成**球（``spawn``）。
2. 可选地在回合过程中**回应**——例如在玩家击球后返回球的对手智能体（``respond``）。

抽象基类让两个方法均可被钩入，从而高层任务可以通过
更换提供器来组合行为，而无需修改奖励/终止逻辑。

难度旋钮
----------------
``BallProvider.bump_difficulty(key)`` 由课程项调用，
以逐步调整采样范围（例如更宽的速度范围）。
具体的提供器决定如何响应。
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
# 生成辅助函数。
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
  """将球的位姿+速度（环境局部坐标系）写入仿真器。"""
  origins = env.scene.env_origins[env_ids]
  pos_w = pos_l + origins
  pose = torch.cat([pos_w, quat], dim=-1)  # (k, 7)
  twist = torch.cat([lin_vel, ang_vel], dim=-1)  # (k, 6)
  ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  ball.write_root_link_velocity_to_sim(twist, env_ids=env_ids)


# ---------------------------------------------------------------------------
# 抽象接口。
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BallProviderCfg(abc.ABC):
  """球提供器的抽象基础配置。"""

  ball_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("ball"))

  @abc.abstractmethod
  def build(self, env: "ManagerBasedRlEnv") -> "BallProvider":
    """实例化运行时提供器对象。"""


class BallProvider(abc.ABC):
  """负责生成球（可选在回合中响应）的策略对象。"""

  cfg: BallProviderCfg

  def __init__(self, cfg: BallProviderCfg, env: "ManagerBasedRlEnv") -> None:
    self.cfg = cfg
    self._env = env
    self._ball: Entity = env.scene[cfg.ball_cfg.name]
    self._difficulty = 0.0

  # --- 生命周期钩子 --------------------------------------------------

  @abc.abstractmethod
  def spawn(self, env_ids: torch.Tensor) -> None:
    """在新回合开始时为 ``env_ids`` 中的环境放置球。"""

  def respond(self, env_ids: torch.Tensor) -> None:  # noqa: B027
    """可选的回合内响应（例如对手回球）。默认空操作。"""

  def reset(self, env_ids: torch.Tensor) -> None:  # noqa: B027
    """可选的内部状态重置（默认空操作）。"""

  # --- 课程钩子 -------------------------------------------------

  def bump_difficulty(self, key: str, delta: float = 0.05) -> None:  # noqa: ARG002
    """提升难度。子类覆盖此方法以扩宽采样范围。"""
    self._difficulty = min(1.0, self._difficulty + delta)

  @property
  def difficulty(self) -> float:
    return self._difficulty

  # --- 便捷属性 ------------------------------------------------------

  @property
  def device(self) -> str | torch.device:
    return self._env.device

  @property
  def num_envs(self) -> int:
    return self._env.num_envs


# ---------------------------------------------------------------------------
# P0：固定生成——单一确定性状态。
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class FixedSpawnerCfg(BallProviderCfg):
  """每回合在固定位姿和速度下生成球。

  适用于单元测试、奖励塑形调试和评估。
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
# P1：随机发球器——从生成区到目标区的弹道轨迹。
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RandomFeederCfg(BallProviderCfg):
  """在可配置区域生成球，并将其射向随机落点。

  球放置在 ``[spawn_x_range, spawn_y_range, spawn_z_range]`` 内的随机位置
  （默认：网上方）。落点从地面（z = 0）的 ``[target_x_range, target_y_range]``
  内均匀采样。竖直发射速度 ``vz0`` 从 ``lin_vel_z_range`` 采样；
  水平分量 (vx, vy) 通过解析解确定，使球在重力下落到目标点。

  从 z 方程求解飞行时间 ``t``::

      z0 + vz0 * t - 0.5 * g * t^2 = 0
      t = (vz0 + sqrt(vz0^2 + 2 * g * z0)) / g   （正根）

  然后::

      vx = (target_x - spawn_x) / t
      vy = (target_y - spawn_y) / t

  默认将生成区置于网正上方（x ≈ 0），目标在机器人侧球场（x ∈ (0.5, 2.5)），
  确保球始终朝玩家飞来。

  课程旋钮
  ----------------
  ``bump_difficulty("ball_speed")`` 通过扩宽 ``lin_vel_z_range`` 来压缩飞行时间
  （更大的 vz0 → 更高弧线 → 随后进一步收缩时反应时间减少）。
  ``bump_difficulty("ball_lateral")`` 扩宽 ``target_y_range``。
  """

  # 生成区（球的起始位置）——默认：网上方。
  spawn_x_range: tuple[float, float] = (-0.4, 0.4)
  spawn_y_range: tuple[float, float] = (-2.0, 2.0)
  spawn_z_range: tuple[float, float] = (1.0, 1.6)

  # 地面（z = 0）上的目标落点区域——默认：机器人侧发球区。
  target_x_range: tuple[float, float] = (1.0, 4.0)
  target_y_range: tuple[float, float] = (-2.0, 2.0)

  # 竖直发射速度。正值 = 向上弧线；需足够大以使球确实到达落点（不立即落地）。
  lin_vel_z_range: tuple[float, float] = (1.5, 3.5)

  # 物理常数；如需自定义重力环境可覆盖。
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

    # --- 生成位置 ---------------------------------------------------
    px = _uniform(env_ids, *cfg.spawn_x_range, dev)
    py = _uniform(env_ids, *cfg.spawn_y_range, dev)
    pz = _uniform(env_ids, *cfg.spawn_z_range, dev)

    # --- 目标落点（z = 0）------------------------------------
    tx = _uniform(env_ids, *cfg.target_x_range, dev)
    ty = _uniform(env_ids, *cfg.target_y_range, dev)

    # --- 竖直速度 ---------------------------------------------------
    vz = _uniform(env_ids, *cfg.lin_vel_z_range, dev)

    # --- 从 z 方程求解飞行时间 --------------------------------
    # pz + vz*t - 0.5*g*t^2 = 0  =>  t = (vz + sqrt(vz^2 + 2*g*pz)) / g
    g = cfg.gravity
    disc = torch.clamp(vz * vz + 2.0 * g * pz, min=1e-6)
    flight_t = (vz + torch.sqrt(disc)) / g  # pz > 0 时始终为正

    # --- 水平速度（运动学逆解）----------------------------
    vx = (tx - px) / flight_t
    vy = (ty - py) / flight_t

    pos = torch.stack([px, py, pz], dim=-1)
    lin = torch.stack([vx, vy, vz], dim=-1)
    quat = torch.zeros(env_ids.numel(), 4, device=dev)
    quat[:, 0] = 1.0
    ang = torch.zeros(env_ids.numel(), 3, device=dev)
    _write_ball_state(self._env, self._ball, env_ids, pos, quat, lin, ang)

  def bump_difficulty(self, key: str, delta: float = 0.05) -> None:
    """通过修改目标或速度范围来调整难度。"""
    super().bump_difficulty(key, delta)
    if key == "ball_speed":
      # 通过降低 vz 上限来压缩弧线 → 缩短飞行时间。
      lo, hi = self.cfg.lin_vel_z_range
      self.cfg.lin_vel_z_range = (
        max(0.5, lo - delta * 0.5),
        max(lo + 0.1, hi - delta * 0.5),
      )
    elif key == "ball_lateral":
      lo, hi = self.cfg.target_y_range
      self.cfg.target_y_range = (lo - delta * 0.3, hi + delta * 0.3)


# ---------------------------------------------------------------------------
# P2：弹道对手——从对手侧一次性发球。
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class BallisticOpponentCfg(BallProviderCfg):
  """计算初速度，使球落在己方侧的采样点上。

  在重力 ``g`` 下求解抛体运动方程：从 ``launch_pos`` 出发，
  经 ``flight_time`` 秒后到达 ``target_pos``。
  ``target_pos`` 在每次生成时从配置的边界框中按环境独立采样。

  注意
  -----
  这**不是**学习型对手；它是一个确定性发球器，产生逼真的来球弧线。
  真正的对手智能体需要用查询外部策略的提供器替换本类。
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
      # 压缩飞行时间 → 更快、更难接到的来球。
      lo, hi = self.cfg.flight_time_range
      self.cfg.flight_time_range = (max(0.3, lo - delta * 0.1), hi)
    elif key == "ball_lateral":
      lo, hi = self.cfg.target_y_range
      self.cfg.target_y_range = (lo - delta * 0.3, hi + delta * 0.3)


# ---------------------------------------------------------------------------
# 重导出别名（方便使用）。
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

# 消除 TYPE_CHECKING 关闭时的未使用导入警告。
_ = math
