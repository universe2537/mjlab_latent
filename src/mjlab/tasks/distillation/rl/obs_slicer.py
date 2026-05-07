"""Split the actor observation tensor into ``state`` and ``target`` slices.

LATENT §3.2.2 splits the policy input into two semantic groups:

* ``state``  -- proprioceptive / world signals consumed by both the prior
  and the decoder (e.g. ``joint_pos``, ``base_lin_vel``, ``actions``).
* ``target`` -- the future reference the encoder needs (e.g. ``command``).

This module computes the static per-term index slices once at construction
time, so per-step rollouts only do cheap advanced indexing.
"""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper


class ObservationSlicer:
  """Cache index tensors that split a concatenated observation group."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    *,
    group_name: str,
    state_terms: tuple[str, ...],
    target_terms: tuple[str, ...],
  ) -> None:
    obs_manager = env.unwrapped.observation_manager
    if not obs_manager.group_obs_concatenate[group_name]:
      raise ValueError(
        f"Distillation expects observation group {group_name!r} to be concatenated."
      )

    term_names = obs_manager.active_terms[group_name]
    term_dims = obs_manager.group_obs_term_dim[group_name]
    if len(state_terms) == 0:
      state_terms = tuple(name for name in term_names if name not in target_terms)

    self.state_indices = self._indices_for_terms(term_names, term_dims, state_terms)
    self.target_indices = self._indices_for_terms(term_names, term_dims, target_terms)
    self.obs_dim = sum(int(torch.tensor(dim).prod().item()) for dim in term_dims)

  @staticmethod
  def _indices_for_terms(
    term_names: list[str],
    term_dims: list[tuple[int, ...]],
    selected_terms: tuple[str, ...],
  ) -> torch.Tensor:
    missing = sorted(set(selected_terms).difference(term_names))
    if missing:
      raise ValueError(f"Unknown observation term(s): {missing}")

    indices: list[int] = []
    offset = 0
    for term_name, term_dim in zip(term_names, term_dims, strict=True):
      length = int(torch.tensor(term_dim).prod().item())
      if term_name in selected_terms:
        indices.extend(range(offset, offset + length))
      offset += length
    return torch.tensor(indices, dtype=torch.long)

  @property
  def state_dim(self) -> int:
    return int(self.state_indices.numel())

  @property
  def target_dim(self) -> int:
    return int(self.target_indices.numel())

  def to(self, device: torch.device) -> None:
    self.state_indices = self.state_indices.to(device)
    self.target_indices = self.target_indices.to(device)

  def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return obs[:, self.state_indices], obs[:, self.target_indices]
