from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


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
# 随机发球器——从生成区到目标区的弹道轨迹。
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RandomFeederCfg:
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

  默认将生成区置于网正上方（x ≈ 0），目标区位于机器人侧的接球工作区，
  具体范围由环境配置给定，从而可以随球场尺寸一起调整。

  课程旋钮
  ----------------
  ``bump_difficulty("ball_speed")`` 通过扩宽 ``lin_vel_z_range`` 来压缩飞行时间
  （更大的 vz0 → 更高弧线 → 随后进一步收缩时反应时间减少）。
  ``bump_difficulty("ball_lateral")`` 扩宽 ``target_y_range``。
  """

  ball_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("ball"))

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


class RandomFeeder:
  """随机发球器运行时对象。"""

  def __init__(self, cfg: RandomFeederCfg, env: "ManagerBasedRlEnv") -> None:
    self.cfg = cfg
    self._env = env
    self._ball: Entity = env.scene[cfg.ball_cfg.name]
    self._difficulty = 0.0

  @property
  def device(self) -> str | torch.device:
    return self._env.device

  @property
  def difficulty(self) -> float:
    return self._difficulty

  def spawn(self, env_ids: torch.Tensor) -> None:
    cfg = self.cfg
    dev = self.device
    spawn_x_lo, spawn_x_hi = cfg.spawn_x_range
    spawn_y_lo, spawn_y_hi = cfg.spawn_y_range
    spawn_z_lo, spawn_z_hi = cfg.spawn_z_range
    target_x_lo, target_x_hi = cfg.target_x_range
    target_y_lo, target_y_hi = cfg.target_y_range
    lin_vel_z_lo, lin_vel_z_hi = cfg.lin_vel_z_range

    # --- 生成位置 ---------------------------------------------------
    px = _uniform(env_ids, spawn_x_lo, spawn_x_hi, dev)
    py = _uniform(env_ids, spawn_y_lo, spawn_y_hi, dev)
    pz = _uniform(env_ids, spawn_z_lo, spawn_z_hi, dev)

    # --- 目标落点（z = 0）------------------------------------
    tx = _uniform(env_ids, target_x_lo, target_x_hi, dev)
    ty = _uniform(env_ids, target_y_lo, target_y_hi, dev)

    # --- 竖直速度 ---------------------------------------------------
    vz = _uniform(env_ids, lin_vel_z_lo, lin_vel_z_hi, dev)

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
    self._difficulty = min(1.0, self._difficulty + delta)
    cfg = self.cfg
    if key == "ball_speed":
      # 通过降低 vz 上限来压缩弧线 → 缩短飞行时间。
      lo, hi = cfg.lin_vel_z_range
      cfg.lin_vel_z_range = (
        max(0.5, lo - delta * 0.5),
        max(lo + 0.1, hi - delta * 0.5),
      )
    elif key == "ball_lateral":
      lo, hi = cfg.target_y_range
      cfg.target_y_range = (lo - delta * 0.3, hi + delta * 0.3)


def spawn_ball_from_provider(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  *,
  provider_cfg: RandomFeederCfg,
) -> None:
  """兼容事件管理器的包装函数，通过随机发球器生成球。"""
  cache_key = f"_ball_provider_{id(provider_cfg)}"
  provider: RandomFeeder | None = getattr(env, cache_key, None)
  if provider is None:
    provider = provider_cfg.build(env)
    setattr(env, cache_key, provider)
  provider.spawn(env_ids)


# ---------------------------------------------------------------------------
# 重导出别名（方便使用）。
# ---------------------------------------------------------------------------

__all__ = [
  "RandomFeeder",
  "RandomFeederCfg",
  "spawn_ball_from_provider",
]

# 消除 TYPE_CHECKING 关闭时的未使用导入警告。
_ = math
