"""动作参考命令项，供蒸馏任务使用。

与跟踪任务共享相同的 MotionLoader / MotionCommand / MotionCommandCfg 设计。
distillation 任务的学生策略在与教师一样的环境中 rollout，因此需要完全相同的动作命令基础设施。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _as_motion_files(motion_files: str | tuple[str, ...]) -> tuple[str, ...]:
  """Normalize user config into a non-empty tuple of motion paths."""
  if isinstance(motion_files, str):
    motion_files = (motion_files,) if motion_files else ()
  if len(motion_files) == 0:
    raise ValueError("MotionCommandCfg.motion_files must contain at least one path.")
  return motion_files


class MotionLoader:
  """Load one or more motion files and expose them as one concatenated timeline.

  Multiple trajectories are stored back-to-back in each tensor for efficient
  indexing on device.  ``split_points`` maps a per-motion local time step to the
  global row index in those concatenated tensors.
  """

  def __init__(
    self,
    motion_files: str | tuple[str, ...],
    body_indexes: torch.Tensor,
    device: str = "cpu",
  ) -> None:
    self.motion_files = _as_motion_files(motion_files)
    # 每个列表从每条轨迹中收集相同的字段。将这些列表串联使得运行时采样更简单：帧查找变为一个全局整数索引，而不是 Python 级别的文件切换。
    arrays: dict[str, list[torch.Tensor]] = {
      "joint_pos": [],
      "joint_vel": [],
      "body_pos_w": [],
      "body_quat_w": [],
      "body_lin_vel_w": [],
      "body_ang_vel_w": [],
    }
    lengths = []
    for motion_file in self.motion_files:
      data = np.load(motion_file)
      length = int(data["joint_pos"].shape[0])
      if length <= 0:
        raise ValueError(f"Motion file {motion_file} has no frames.")
      lengths.append(length)
      for key in arrays:
        arrays[key].append(torch.tensor(data[key], dtype=torch.float32, device=device))

    self.joint_pos = torch.cat(arrays["joint_pos"], dim=0)
    self.joint_vel = torch.cat(arrays["joint_vel"], dim=0)
    self._body_pos_w = torch.cat(arrays["body_pos_w"], dim=0)
    self._body_quat_w = torch.cat(arrays["body_quat_w"], dim=0)
    self._body_lin_vel_w = torch.cat(arrays["body_lin_vel_w"], dim=0)
    self._body_ang_vel_w = torch.cat(arrays["body_ang_vel_w"], dim=0)
    self._body_indexes = body_indexes
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]
    self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    self.split_points = torch.zeros(len(lengths) + 1, dtype=torch.long, device=device)
    # split_points[i] 是 motion i 的全局起始帧；最后一个元素表示总帧数，便于长度检查与调试。
    self.split_points[1:] = torch.cumsum(self.motion_lengths, dim=0)
    self.num_motions = len(lengths)

  def frame_ids(
    self, motion_ids: torch.Tensor, local_time_steps: torch.Tensor
  ) -> torch.Tensor:
    """Convert ``(motion_id, local_step)`` pairs to concatenated frame indices."""
    return self.split_points[motion_ids] + local_time_steps


class MotionCommand(CommandTerm):
  """Command term that advances reference motions and resets envs onto them.

  The term is responsible for three coupled jobs:

  1. sample the next reference motion/frame when an episode starts or ends;
  2. write the sampled reference state into MuJoCo for reference-state
     initialization (RSI);
  3. keep cached reference poses aligned to the robot anchor so reward and
     termination functions can compare body poses in a consistent world frame.
  """

  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # 将面向用户的身体名称映射到机器人索引和参考动作索引。机器人可能包含比动作文件跟踪的更多身体，因此所有命令张量限制为 cfg.body_names。
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_files, self.body_indexes, device=self.device
    )
    # 每个环境的轨迹游标。motion_ids 选择轨迹，而 time_steps 是该轨迹的本地帧，通过 MotionLoader.frame_ids 转换为全局索引后用于索引串联张量。
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_sample_probs = self._make_motion_sample_probs()
    # 锚点对齐之后缓存的参考身体位姿。奖励和终止项使用这些而不是原始动作空间位姿，以便将全局 x/y/yaw 漂移相对于当前机器人锚点进行度量。
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # 自适应采样跟踪哪些归一化时间区间更易失败。表以 [motion_id, bin_id] 索引，因此每条轨迹都有自己的课程分布。
    self.bin_count = (
      int(self.motion.motion_lengths.max().item() // (1 / env.step_dt)) + 1
    )
    self.bin_failed_count = torch.zeros(
      self.motion.num_motions, self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.motion.num_motions, self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _make_motion_sample_probs(self) -> torch.Tensor:
    """Build the categorical distribution used to choose trajectories."""
    if len(self.cfg.motion_sample_probs) > 0:
      if len(self.cfg.motion_sample_probs) != self.motion.num_motions:
        raise ValueError(
          "MotionCommandCfg.motion_sample_probs must have the same length as "
          "motion_files."
        )
      probs = torch.tensor(
        self.cfg.motion_sample_probs, dtype=torch.float32, device=self.device
      )
      if torch.any(probs < 0) or torch.sum(probs) <= 0:
        raise ValueError("motion_sample_probs must be non-negative and non-zero.")
      return probs / torch.sum(probs)
    return torch.ones(self.motion.num_motions, device=self.device) / float(
      self.motion.num_motions
    )

  @property
  def frame_ids(self) -> torch.Tensor:
    """Global frame ids for each environment in MotionLoader tensors."""
    return self.motion.frame_ids(self.motion_ids, self.time_steps)

  @property
  def command(self) -> torch.Tensor:
    """Policy command vector containing target joint position and velocity."""
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    """Reference joint positions at the current frame for each env."""
    return self.motion.joint_pos[self.frame_ids]

  @property
  def joint_vel(self) -> torch.Tensor:
    """Reference joint velocities at the current frame for each env."""
    return self.motion.joint_vel[self.frame_ids]

  @property
  def body_pos_w(self) -> torch.Tensor:
    """Reference body positions shifted into each environment origin."""
    return (
      self.motion.body_pos_w[self.frame_ids] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    """Reference body orientations in world frame."""
    return self.motion.body_quat_w[self.frame_ids]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    """Reference body linear velocities in world frame."""
    return self.motion.body_lin_vel_w[self.frame_ids]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    """Reference body angular velocities in world frame."""
    return self.motion.body_ang_vel_w[self.frame_ids]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    """Reference anchor-body position shifted into each environment origin."""
    return (
      self.motion.body_pos_w[self.frame_ids, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    """Reference anchor-body orientation in world frame."""
    return self.motion.body_quat_w[self.frame_ids, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    """Reference anchor-body linear velocity in world frame."""
    return self.motion.body_lin_vel_w[self.frame_ids, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    """Reference anchor-body angular velocity in world frame."""
    return self.motion.body_ang_vel_w[self.frame_ids, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    """Current simulated robot joint positions."""
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    """Current simulated robot joint velocities."""
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    """Current simulated tracked body positions in world frame."""
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    """Current simulated tracked body orientations in world frame."""
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    """Current simulated tracked body linear velocities in world frame."""
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    """Current simulated tracked body angular velocities in world frame."""
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    """Current simulated anchor-body position in world frame."""
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    """Current simulated anchor-body orientation in world frame."""
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    """Current simulated anchor-body linear velocity in world frame."""
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    """Current simulated anchor-body angular velocity in world frame."""
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    """Publish per-env tracking errors for logging and curriculum diagnostics."""
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    """Sample reset frames from bins where previous episodes failed.

    Failures are recorded in normalized time bins per trajectory.  New reset
    frames are drawn from a smoothed distribution over those bins, with a small
    uniform floor so rarely sampled parts of a motion remain reachable.
    """
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      # 将每个失败的重置环境归因到其终止时所在的时间箱。失败计数按 motion_id 记录，以避免不同轨迹相互污染采样分布。
      motion_lengths = self.motion.motion_lengths[self.motion_ids]
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // torch.clamp(motion_lengths, min=1),
        0,
        self.bin_count - 1,
      )
      failed_env_ids = env_ids[episode_failed]
      for motion_id in range(self.motion.num_motions):
        motion_mask = self.motion_ids[failed_env_ids] == motion_id
        if not torch.any(motion_mask):
          continue
        fail_bins = current_bin_index[failed_env_ids][motion_mask]
        self._current_bin_failed[motion_id] = torch.bincount(
          fail_bins, minlength=self.bin_count
        )

    sampled_motion_ids = self._sample_motion_ids(len(env_ids))
    self.motion_ids[env_ids] = sampled_motion_ids

    for motion_id in range(self.motion.num_motions):
      motion_env_ids = env_ids[sampled_motion_ids == motion_id]
      if motion_env_ids.numel() == 0:
        continue

      # 从历史失败计数加上均匀探索下限开始。非因果平滑核也会在失败箱之前提高概率，使策略能练习接近困难状态的过渡。
      sampling_probabilities = self.bin_failed_count[
        motion_id
      ] + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
      sampling_probabilities = torch.nn.functional.pad(
        sampling_probabilities.unsqueeze(0).unsqueeze(0),
        (0, self.cfg.adaptive_kernel_size - 1),  # 非因果核
        mode="replicate",
      )
      sampling_probabilities = torch.nn.functional.conv1d(
        sampling_probabilities, self.kernel.view(1, 1, -1)
      ).view(-1)
      sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

      # 将采样到的归一化箱转换回所选轨迹的本地帧 id。箱内的均匀偏移避免所有环境落在完全相同的离散帧上。
      sampled_bins = torch.multinomial(
        sampling_probabilities, motion_env_ids.numel(), replacement=True
      )
      motion_length = self.motion.motion_lengths[motion_id]
      self.time_steps[motion_env_ids] = (
        (
          sampled_bins
          + sample_uniform(0.0, 1.0, (motion_env_ids.numel(),), device=self.device)
        )
        / self.bin_count
        * (motion_length - 1)
      ).long()

      # 更新指标。
      H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
      H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
      pmax, imax = sampling_probabilities.max(dim=0)
      self.metrics["sampling_entropy"][motion_env_ids] = H_norm
      self.metrics["sampling_top1_prob"][motion_env_ids] = pmax
      self.metrics["sampling_top1_bin"][motion_env_ids] = imax.float() / self.bin_count

  def _sample_motion_ids(self, count: int) -> torch.Tensor:
    """Sample trajectory ids according to ``motion_sample_probs``."""
    return torch.multinomial(self.motion_sample_probs, count, replacement=True)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    """Uniformly sample both trajectory id and local frame for each env."""
    sampled_motion_ids = self._sample_motion_ids(len(env_ids))
    self.motion_ids[env_ids] = sampled_motion_ids
    motion_lengths = self.motion.motion_lengths[sampled_motion_ids]
    self.time_steps[env_ids] = torch.floor(
      torch.rand(len(env_ids), device=self.device) * motion_lengths
    ).long()
    self.metrics["sampling_entropy"][env_ids] = 1.0
    self.metrics["sampling_top1_prob"][env_ids] = torch.max(self.motion_sample_probs)
    self.metrics["sampling_top1_bin"][env_ids] = 0.5

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write a full reference state into simulation.

    The motion files may contain poses slightly outside the robot's soft joint
    limits.  Clipping happens at reset time so policy targets remain unchanged
    while MuJoCo still starts from a valid state.
    """
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    """Choose a new motion frame and initialize envs near that reference state."""
    if self.cfg.sampling_mode == "start":
      self.motion_ids[env_ids] = self._sample_motion_ids(len(env_ids))
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    # 参考状态初始化（RSI）在采样帧周围对根位姿进行随机化。目标命令仍指向精确的参考，但 episode 从附近状态开始以提高鲁棒性。
    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    # Joint RSI 对每个关节使用配置的标量范围，然后 _write_reference_state_to_sim 将其裁剪到每个关节的软限。
    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    The raw reference motion is anchored in its own world frame, while the robot
    may have drifted in global x/y/yaw.  This method aligns the reference anchor
    to the robot anchor in x/y/yaw but preserves the reference anchor height,
    producing a fair body-pose target for rewards and terminations.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    # 保留机器人在 x/y 方向的平移，使目标随机器人在平面上移动；保留参考帧的 z 分量，以便跌倒或跳跃仍被惩罚。
    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    # 仅应用偏航对齐；滚转/俯仰误差仍由锚点朝向的奖励和终止判定追踪。
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self):
    """Advance reference time, recycle finished motions, and update caches."""
    self.time_steps += 1
    motion_lengths = self.motion.motion_lengths[self.motion_ids]
    env_ids = torch.where(self.time_steps >= motion_lengths)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      # 对失败时间箱进行指数移动平均。当前步的临时计数在合并到持久课程后会被清零。
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # 构建仅包含可视几何体的幽灵模型。碰撞几何体（contype/conaffinity 非零）设置 alpha=0，以便查看器的 alpha 过滤器将其排除。
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      # Frame 模式绘制目标与机器人每个被跟踪身体的坐标系帧。相比幽灵网格噪声更大，但能更清晰地显示朝向误差。
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.motion_lengths.max().item()) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        motion_length = int(self.motion.motion_lengths[self.motion_ids[idx]].item())
        self.time_steps[idx] = min(int(scrubber.value), motion_length - 1)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    """Only allow frame scrubbing while the viewer is paused."""
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    """Apply a GUI-requested deterministic reset if controls exist."""
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    motion_lengths = self.motion.motion_lengths[self.motion_ids[env_ids]]
    self.time_steps[env_ids] = torch.clamp(
      torch.full_like(env_ids, frame), max=motion_lengths - 1
    )
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  """Configuration for motion-reference tracking commands.

  ``motion_files`` may contain one or many trajectories.  Optional
  ``motion_sample_probs`` controls how often each trajectory is selected; when
  omitted, trajectories are sampled uniformly.
  """

  motion_files: str | tuple[str, ...]
  """Path or paths to NPZ files containing reference motion tensors."""
  anchor_body_name: str
  """Body used as the root/anchor for alignment and anchor rewards."""
  body_names: tuple[str, ...]
  """Tracked bodies, in the order expected by the motion tensors."""
  entity_name: str
  """Scene entity that should follow the motion."""
  motion_source: Literal["local", "wandb"] = "local"
  """Where higher-level scripts should resolve ``motion_files`` from."""
  motion_sample_probs: tuple[float, ...] = ()
  """Optional trajectory sampling weights matching ``motion_files``."""
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """RSI root pose perturbation ranges for x/y/z/roll/pitch/yaw."""
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """RSI root velocity perturbation ranges for linear and angular axes."""
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  """Uniform RSI perturbation applied independently to every joint."""
  adaptive_kernel_size: int = 1
  """Smoothing width over failed time bins for adaptive sampling."""
  adaptive_lambda: float = 0.8
  """Geometric decay used by the adaptive sampling smoothing kernel."""
  adaptive_uniform_ratio: float = 0.1
  """Uniform probability floor mixed into adaptive frame sampling."""
  adaptive_alpha: float = 0.001
  """EMA coefficient for failed-bin statistics."""
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  """Frame reset strategy used when an episode starts or a motion ends."""

  @dataclass
  class VizCfg:
    """Debug visualization options for the reference motion."""

    mode: Literal["ghost", "frames"] = "ghost"
    """Draw a ghost robot mesh or per-body coordinate frames."""
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    """RGBA color used for ghost visual geoms."""

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    """Instantiate the command term for a concrete environment."""
    return MotionCommand(self, env)
