"""面向 DAgger 蒸馏的设备端环形 replay buffer。

这里的数据非常简单：

1. actor 观测 ``obs``
2. teacher 对应的动作 ``teacher_action``

训练时 student 会基于当前参数重新前向，因此无需把 posterior / prior / latent
中间量一并缓存下来。
"""

from __future__ import annotations

import torch


class ReplayBuffer:
  """固定容量的环形缓存，保存 ``(actor_obs, teacher_action)`` 配对样本。

  All tensors live on ``device`` so sampling involves no host/device copies.

  参数:
    capacity: 最大样本数。
    obs_dim: 每条 observation 的展平维度。
    action_dim: 每条 teacher action 的维度。
    device: 数据存放设备，通常为当前训练 GPU。
  """

  def __init__(
    self,
    *,
    capacity: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
  ) -> None:
    self.capacity = int(capacity)
    self.device = device
    self.obs = torch.empty(self.capacity, obs_dim, device=device)
    self.teacher_actions = torch.empty(self.capacity, action_dim, device=device)
    self.size = 0
    self.pos = 0

  def add(self, obs: torch.Tensor, teacher_actions: torch.Tensor) -> None:
    """写入一批样本；超出容量时覆盖最旧数据。"""
    num_items = obs.shape[0]
    if num_items >= self.capacity:
      self.obs[:] = obs[-self.capacity :]
      self.teacher_actions[:] = teacher_actions[-self.capacity :]
      self.size = self.capacity
      self.pos = 0
      return

    end = self.pos + num_items
    if end <= self.capacity:
      self.obs[self.pos : end] = obs
      self.teacher_actions[self.pos : end] = teacher_actions
    else:
      first = self.capacity - self.pos
      self.obs[self.pos :] = obs[:first]
      self.teacher_actions[self.pos :] = teacher_actions[:first]
      self.obs[: end - self.capacity] = obs[first:]
      self.teacher_actions[: end - self.capacity] = teacher_actions[first:]

    self.pos = end % self.capacity
    self.size = min(self.capacity, self.size + num_items)

  def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """均匀采样一批样本。

    返回:
      obs_batch, teacher_action_batch
    """
    if self.size == 0:
      raise RuntimeError("Cannot sample from an empty replay buffer.")
    idx = torch.randint(self.size, (min(batch_size, self.size),), device=self.device)
    return self.obs[idx], self.teacher_actions[idx]
