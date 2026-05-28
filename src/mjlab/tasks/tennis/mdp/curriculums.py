"""网球任务的课程学习项。"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


def _coerce_range(value: object, name: str) -> tuple[float, float]:
  if not isinstance(value, tuple) or len(value) != 2:
    raise TypeError(f"{name} must be a 2-tuple, got {value!r}.")
  pair = cast(tuple[float | int, float | int], value)
  return float(pair[0]), float(pair[1])


def _lerp_range(
  start: tuple[float, float], end: tuple[float, float], alpha: float
) -> tuple[float, float]:
  return (
    start[0] + (end[0] - start[0]) * alpha,
    start[1] + (end[1] - start[1]) * alpha,
  )


class random_feeder_target_curriculum:
  """按成功率逐步扩大发球落点范围。"""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    provider_cfg = cfg.params["provider_cfg"]
    for attr_name in ("target_x_range", "target_y_range"):
      if not hasattr(provider_cfg, attr_name):
        raise TypeError("provider_cfg must expose target_x_range and target_y_range.")

    self._provider_cfg = provider_cfg
    self._success_term_name = str(
      cfg.params.get("success_term_name", "crossed_net_after_hit")
    )
    self._success_threshold = float(cfg.params.get("success_threshold", 0.8))
    self._success_window = max(1, int(cfg.params.get("success_window", 50)))
    self._num_stages = max(1, int(cfg.params.get("num_stages", 6)))
    self._initial_target_x_range = _coerce_range(
      cfg.params["initial_target_x_range"], "initial_target_x_range"
    )
    self._initial_target_y_range = _coerce_range(
      cfg.params["initial_target_y_range"], "initial_target_y_range"
    )
    self._final_target_x_range = _coerce_range(
      cfg.params.get("final_target_x_range", provider_cfg.target_x_range),
      "final_target_x_range",
    )
    self._final_target_y_range = _coerce_range(
      cfg.params.get("final_target_y_range", provider_cfg.target_y_range),
      "final_target_y_range",
    )

    self._stage = 0
    self._success_history: deque[float] = deque(maxlen=self._success_window)
    self._apply_stage()

  def _stage_alpha(self) -> float:
    if self._num_stages <= 1:
      return 1.0
    return self._stage / float(self._num_stages - 1)

  def _apply_stage(self) -> None:
    alpha = self._stage_alpha()
    self._provider_cfg.target_x_range = _lerp_range(
      self._initial_target_x_range, self._final_target_x_range, alpha
    )
    self._provider_cfg.target_y_range = _lerp_range(
      self._initial_target_y_range, self._final_target_y_range, alpha
    )

  def _record_episode_results(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> None:
    done = env.termination_manager.dones
    success = env.termination_manager.get_term(self._success_term_name)
    done_success = success[env_ids][done[env_ids]]
    if done_success.numel() > 0:
      self._success_history.extend(done_success.float().cpu().tolist())

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    provider_cfg: object,
    initial_target_x_range: tuple[float, float],
    initial_target_y_range: tuple[float, float],
    final_target_x_range: tuple[float, float] | None = None,
    final_target_y_range: tuple[float, float] | None = None,
    success_term_name: str = "crossed_net_after_hit",
    success_threshold: float = 0.8,
    success_window: int = 50,
    num_stages: int = 6,
  ) -> dict[str, torch.Tensor]:
    del (
      provider_cfg,
      initial_target_x_range,
      initial_target_y_range,
      final_target_x_range,
      final_target_y_range,
      success_term_name,
      success_threshold,
      success_window,
      num_stages,
    )

    self._record_episode_results(env, env_ids)

    success_rate = 0.0
    if len(self._success_history) > 0:
      success_rate = sum(self._success_history) / len(self._success_history)

    if (
      len(self._success_history) >= self._success_window
      and success_rate >= self._success_threshold
      and self._stage + 1 < self._num_stages
    ):
      self._stage += 1
      self._success_history.clear()
      self._apply_stage()
      success_rate = 0.0

    return {
      "stage": torch.tensor(float(self._stage), device=env.device),
      "success_rate": torch.tensor(success_rate, device=env.device),
      "target_x_min": torch.tensor(
        self._provider_cfg.target_x_range[0], device=env.device
      ),
      "target_x_max": torch.tensor(
        self._provider_cfg.target_x_range[1], device=env.device
      ),
      "target_y_min": torch.tensor(
        self._provider_cfg.target_y_range[0], device=env.device
      ),
      "target_y_max": torch.tensor(
        self._provider_cfg.target_y_range[1], device=env.device
      ),
    }


__all__ = ["random_feeder_target_curriculum"]
