from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


# Stage schemas.


class _RewardCurriculumStageOptional(TypedDict, total=False):
  weight: float
  params: dict[str, Any]


class RewardCurriculumStage(_RewardCurriculumStageOptional):
  step: int


class _TerminationCurriculumStageOptional(TypedDict, total=False):
  params: dict[str, Any]
  time_out: bool


class TerminationCurriculumStage(_TerminationCurriculumStageOptional):
  step: int


RewardWeightStage = dict[str, float]


# Shared engine.  Stage dicts are passed directly from the public TypedDict
# schemas.  Any key that isn't "step" or "params" is treated as a top-level
# field on the target term config (e.g. "weight" on RewardTermCfg).

_RESERVED_KEYS = {"step", "params"}


def _validate_stages(
  term_cfg: Any,
  term_name: str,
  stages: Sequence[Any],
) -> None:
  """Validate stage ordering, field existence, and param keys."""
  for i in range(1, len(stages)):
    if stages[i]["step"] < stages[i - 1]["step"]:
      raise ValueError(
        f"Curriculum stages must be in nondecreasing step order,"
        f" but stage {i} has step"
        f" {stages[i]['step']} < {stages[i - 1]['step']}."
      )
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS and not hasattr(term_cfg, key):
        raise AttributeError(
          f"Field '{key}' does not exist on the resolved term config for '{term_name}'."
        )
  for stage in stages:
    unknown = stage.get("params", {}).keys() - term_cfg.params.keys()
    if unknown:
      raise KeyError(
        f"Stage at step {stage['step']} sets unknown param(s)"
        f" {unknown} on term '{term_name}'. Check for typos."
      )


def _apply_stages(
  term_cfg: Any,
  step_counter: int,
  stages: Sequence[Any],
) -> dict[str, torch.Tensor]:
  """Apply staged updates and return a logging snapshot."""
  for stage in stages:
    if step_counter >= stage["step"]:
      for key, value in stage.items():
        if key not in _RESERVED_KEYS:
          setattr(term_cfg, key, value)
      if "params" in stage:
        term_cfg.params.update(stage["params"])
  # Only log values that stages actually reference.
  logged_fields: set[str] = set()
  logged_params: set[str] = set()
  for stage in stages:
    for key in stage:
      if key not in _RESERVED_KEYS:
        logged_fields.add(key)
    for key in stage.get("params", {}):
      logged_params.add(key)
  result: dict[str, torch.Tensor] = {}
  for key in logged_fields:
    value = getattr(term_cfg, key)
    if isinstance(value, (int, float, bool)):
      result[key] = torch.tensor(value)
    elif isinstance(value, torch.Tensor):
      result[key] = value
  for key in logged_params:
    v = term_cfg.params[key]
    if isinstance(v, (int, float, bool)):
      result[key] = torch.tensor(v)
    elif isinstance(v, torch.Tensor):
      result[key] = v
  return result


# Public wrappers.


class reward_curriculum:
  """Update a reward term's weight and/or params based on training steps.

  Each stage specifies a ``step`` threshold and optionally a ``weight``
  and/or ``params`` dict.  When ``env.common_step_counter`` reaches a
  stage's ``step``, the corresponding values are applied.  Later stages
  take precedence when multiple thresholds are reached.

  Example::

    CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 12000, "weight": -0.1},
          {"step": 24000, "weight": -1.0, "params": {"max_vel": 1.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    reward_name: str = cfg.params["reward_name"]
    stages: list[RewardCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.reward_manager.get_term_cfg(reward_name)
    self._stages = stages
    _validate_stages(self._term_cfg, reward_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    stages: list[RewardCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)


class success_reward_weight_curriculum:
  """Update multiple reward weights after a success-rate gate is reached."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._success_term_name = str(cfg.params["success_term_name"])
    self._success_threshold = float(cfg.params.get("success_threshold", 0.8))
    self._success_window = max(1, int(cfg.params.get("success_window", 50)))
    self._prerequisite_curriculum_name = cfg.params.get(
      "prerequisite_curriculum_name"
    )
    self._prerequisite_stage_key = str(
      cfg.params.get("prerequisite_stage_key", "stage")
    )
    self._prerequisite_min_stage = float(
      cfg.params.get("prerequisite_min_stage", 0.0)
    )
    self._stage_weights = self._normalize_stage_weights(
      cfg.params["stage_weights"]
    )
    self._reward_names = tuple(self._stage_weights[0].keys())
    self._term_cfgs = {
      name: env.reward_manager.get_term_cfg(name) for name in self._reward_names
    }
    self._stage = 0
    self._success_history: deque[float] = deque(maxlen=self._success_window)
    self._apply_stage()

  @staticmethod
  def _normalize_stage_weights(stages: object) -> list[RewardWeightStage]:
    if not isinstance(stages, Sequence) or len(stages) == 0:
      raise ValueError("stage_weights must be a non-empty sequence.")
    normalized: list[RewardWeightStage] = []
    expected_keys: set[str] | None = None
    for index, stage in enumerate(stages):
      if not isinstance(stage, dict) or len(stage) == 0:
        raise TypeError(f"stage_weights[{index}] must be a non-empty dict.")
      stage_dict = cast(dict[object, float | int], stage)
      weights = {str(name): float(weight) for name, weight in stage_dict.items()}
      keys = set(weights)
      if expected_keys is None:
        expected_keys = keys
      elif keys != expected_keys:
        raise ValueError("All stage_weights entries must update the same rewards.")
      normalized.append(weights)
    return normalized

  def _apply_stage(self) -> None:
    for reward_name, weight in self._stage_weights[self._stage].items():
      self._term_cfgs[reward_name].weight = weight

  def _prerequisite_ready(self, env: ManagerBasedRlEnv) -> bool:
    if self._prerequisite_curriculum_name is None:
      return True
    state = env.curriculum_manager.get_term_state(self._prerequisite_curriculum_name)
    if state is None:
      return False
    if not isinstance(state, dict):
      raise TypeError("Prerequisite curriculum state must be a dict.")
    if self._prerequisite_stage_key not in state:
      raise KeyError(
        f"Prerequisite curriculum state has no key"
        f" {self._prerequisite_stage_key!r}."
      )
    value = state[self._prerequisite_stage_key]
    if isinstance(value, torch.Tensor):
      value = value.detach().float().mean().item()
    return float(value) >= self._prerequisite_min_stage

  def _record_episode_results(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice
  ) -> None:
    done = env.termination_manager.dones
    success = env.termination_manager.get_term(self._success_term_name)
    done_success = success[env_ids][done[env_ids]]
    if done_success.numel() > 0:
      self._success_history.extend(done_success.float().cpu().tolist())

  def _logged_weights(self, env: ManagerBasedRlEnv) -> dict[str, torch.Tensor]:
    return {
      f"{name}_weight": torch.tensor(
        self._term_cfgs[name].weight, device=env.device
      )
      for name in self._reward_names
    }

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    success_term_name: str,
    stage_weights: list[RewardWeightStage],
    success_threshold: float = 0.8,
    success_window: int = 50,
    prerequisite_curriculum_name: str | None = None,
    prerequisite_stage_key: str = "stage",
    prerequisite_min_stage: float = 0.0,
  ) -> dict[str, torch.Tensor]:
    del (
      success_term_name,
      stage_weights,
      success_threshold,
      success_window,
      prerequisite_curriculum_name,
      prerequisite_stage_key,
      prerequisite_min_stage,
    )

    waiting_for_prerequisite = not self._prerequisite_ready(env)
    success_rate = 0.0
    if not waiting_for_prerequisite:
      self._record_episode_results(env, env_ids)
      if len(self._success_history) > 0:
        success_rate = sum(self._success_history) / len(self._success_history)

      if (
        len(self._success_history) >= self._success_window
        and success_rate >= self._success_threshold
        and self._stage + 1 < len(self._stage_weights)
      ):
        self._stage += 1
        self._success_history.clear()
        self._apply_stage()
        success_rate = 0.0

    result = {
      "stage": torch.tensor(float(self._stage), device=env.device),
      "success_rate": torch.tensor(success_rate, device=env.device),
      "waiting_for_prerequisite": torch.tensor(
        float(waiting_for_prerequisite), device=env.device
      ),
    }
    result.update(self._logged_weights(env))
    return result


class termination_curriculum:
  """Update a termination term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict.  When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied.  Later stages take precedence.

  Example::

    CurriculumTermCfg(
      func=mdp.termination_curriculum,
      params={
        "termination_name": "energy",
        "stages": [
          {"step": 12000, "params": {"threshold": 1000.0}},
          {"step": 24000, "params": {"threshold": 700.0}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    termination_name: str = cfg.params["termination_name"]
    stages: list[TerminationCurriculumStage] = cfg.params["stages"]
    self._term_cfg = env.termination_manager.get_term_cfg(termination_name)
    self._stages = stages
    _validate_stages(self._term_cfg, termination_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    termination_name: str,
    stages: list[TerminationCurriculumStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, termination_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)
