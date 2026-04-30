from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  """计算平均每关键体位置误差（MPKPE）。

  MPKPE 衡量参考与实际在世界坐标系下所有关键体位置的平均欧氏距离。
  """
  pos_error = command.body_pos_relative_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  """计算相对于根的平均每关键体位置误差（R-MPKPE）。

  R-MPKPE 通过计算相对于根/锚点的位置信息来度量不受全局漂移影响的姿态误差。
  """
  # 计算参考位置相对于参考锚点的坐标。
  ref_anchor_pos = command.anchor_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
  ref_rel_pos = command.body_pos_w - ref_anchor_pos  # (num_envs, num_bodies, 3)

  # 计算机器人位置相对于机器人锚点的坐标。
  robot_anchor_pos = command.robot_anchor_pos_w.unsqueeze(1)  # (num_envs, 1, 3)
  robot_rel_pos = (
    command.robot_body_pos_w - robot_anchor_pos
  )  # (num_envs, num_bodies, 3)

  # 计算相对位置之间的误差。
  pos_error = ref_rel_pos - robot_rel_pos
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def compute_joint_velocity_error(command: MotionCommand) -> torch.Tensor:
  """计算每个环境的关节速度误差的 L2 范数。"""
  vel_error = command.joint_vel - command.robot_joint_vel
  return torch.norm(vel_error, dim=-1)  # (num_envs,)


def compute_ee_position_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """计算所选末端执行器身体集合的平均位置误差。"""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_pos = command.body_pos_relative_w[:, ee_indices]
  robot_ee_pos = command.robot_body_pos_w[:, ee_indices]

  pos_error = ref_ee_pos - robot_ee_pos
  per_ee_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def compute_ee_orientation_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """计算所选末端执行器身体集合的平均朝向误差。"""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_quat = command.body_quat_relative_w[:, ee_indices]
  robot_ee_quat = command.robot_body_quat_w[:, ee_indices]

  per_ee_error = quat_error_magnitude(ref_ee_quat, robot_ee_quat)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def _get_body_indices(
  command: MotionCommand,
  body_names: tuple[str, ...],
) -> list[int]:
  """获取命令中指定身体在 body 列表中的索引。

  Args:
    command: MotionCommand 对象。
    body_names: 要查找的身体名称。

  Returns:
    指向 command.cfg.body_names 的索引列表。
  """
  return [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
