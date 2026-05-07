"""On-device circular replay buffer for DAgger-style distillation."""

from __future__ import annotations

import torch


class ReplayBuffer:
  """Fixed-capacity circular buffer storing ``(actor_obs, teacher_action)`` pairs.

  All tensors live on ``device`` so sampling involves no host/device copies.
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
    """Append a batch of transitions, wrapping around when full."""
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
    """Uniformly sample up to ``batch_size`` transitions (without replacement bias)."""
    if self.size == 0:
      raise RuntimeError("Cannot sample from an empty replay buffer.")
    idx = torch.randint(self.size, (min(batch_size, self.size),), device=self.device)
    return self.obs[idx], self.teacher_actions[idx]
