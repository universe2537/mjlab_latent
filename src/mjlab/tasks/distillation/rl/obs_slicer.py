"""把 actor observation 按语义切成 ``state`` 和 ``target`` 两段。

LATENT §3.2.2 splits the policy input into two semantic groups:

* ``state``  -- proprioceptive / world signals consumed by both the prior
  and the decoder (e.g. ``joint_pos``, ``base_lin_vel``, ``actions``).
* ``target`` -- the future reference the encoder needs (e.g. ``command``).

该模块在初始化时把 observation term 名称映射到索引区间，后续 rollout / update
阶段只需做张量切片，不必反复遍历 term 定义。
"""

from __future__ import annotations

import torch

from mjlab.rl import RslRlVecEnvWrapper


class ObservationSlicer:
  """缓存 observation 切片索引，避免训练时重复计算。"""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    *,
    group_name: str,
    state_terms: tuple[str, ...],
    target_terms: tuple[str, ...],
  ) -> None:
    """构造切片器。

    参数:
      env: 已经包成 ``RslRlVecEnvWrapper`` 的环境。
      group_name: 要读取的 observation group 名称，通常是 ``actor``。
      state_terms: 属于 ``state`` 的 term 名称集合。
      target_terms: 属于 ``target`` 的 term 名称集合。
    """
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
    """根据 term 名称集合生成对应的一维索引张量。"""
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
    """把切片索引移动到指定设备。"""
    self.state_indices = self.state_indices.to(device)
    self.target_indices = self.target_indices.to(device)

  def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把完整 observation 切成 ``(state, target)``。"""
    return obs[:, self.state_indices], obs[:, self.target_indices]
