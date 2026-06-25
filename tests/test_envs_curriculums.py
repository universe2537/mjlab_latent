"""Tests for reward_curriculum and termination_curriculum."""

from unittest.mock import Mock

import pytest
import torch

from mjlab.envs.mdp.curriculums import (
  reward_curriculum,
  success_reward_weight_curriculum,
  termination_curriculum,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg


def _reward_func(env):
  return torch.ones(env.num_envs)


def _termination_func(env):
  return torch.zeros(env.num_envs, dtype=torch.bool)


def _make_reward_cfg(
  weight: float = 1.0,
  params: dict | None = None,
) -> RewardTermCfg:
  return RewardTermCfg(
    func=_reward_func,
    weight=weight,
    params=params if params is not None else {"std": 0.5, "scale": 1.0},
  )


def _make_termination_cfg(
  params: dict | None = None,
) -> TerminationTermCfg:
  return TerminationTermCfg(
    func=_termination_func,
    params=params if params is not None else {"threshold": float("inf")},
  )


def _build_reward(env, reward_name, stages):
  params = {"reward_name": reward_name, "stages": stages}
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  instance = reward_curriculum(cfg, env)
  return instance(env, env_ids=torch.tensor([0, 1]), **params)


def _build_termination(env, termination_name, stages):
  params = {"termination_name": termination_name, "stages": stages}
  cfg = CurriculumTermCfg(func=termination_curriculum, params=params)
  instance = termination_curriculum(cfg, env)
  return instance(env, env_ids=torch.tensor([0, 1]), **params)


def _make_reward_env(step_counter, reward_cfg):
  env = Mock()
  env.common_step_counter = step_counter
  env.reward_manager.get_term_cfg.return_value = reward_cfg
  return env


def _make_termination_env(step_counter, term_cfg):
  env = Mock()
  env.common_step_counter = step_counter
  env.termination_manager.get_term_cfg.return_value = term_cfg
  return env


def _make_success_reward_env(
  reward_cfgs,
  dones,
  successes,
  prerequisite_state=None,
):
  env = Mock()
  env.device = torch.device("cpu")
  env.reward_manager.get_term_cfg.side_effect = lambda name: reward_cfgs[name]
  env.termination_manager.dones = torch.tensor(dones, dtype=torch.bool)
  env.termination_manager.get_term.return_value = torch.tensor(
    successes, dtype=torch.bool
  )
  env.curriculum_manager.get_term_state.return_value = prerequisite_state
  return env


def _build_success_reward(env, stage_weights, **overrides):
  params = {
    "success_term_name": "success",
    "success_threshold": 0.8,
    "success_window": 50,
    "stage_weights": stage_weights,
    **overrides,
  }
  cfg = CurriculumTermCfg(func=success_reward_weight_curriculum, params=params)
  instance = success_reward_weight_curriculum(cfg, env)
  env_ids = torch.arange(env.termination_manager.dones.shape[0])
  return instance(env, env_ids=env_ids, **params)


# Reward: weight


def test_reward_weight_unchanged_before_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 2.0}])
  assert rc.weight == pytest.approx(1.0)


def test_reward_weight_applied_at_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(100, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 2.0}])
  assert rc.weight == pytest.approx(2.0)


def test_reward_weight_later_stage_wins():
  rc = _make_reward_cfg()
  env = _make_reward_env(500, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 0, "weight": 0.5},
      {"step": 100, "weight": 1.5},
      {"step": 400, "weight": 3.0},
    ],
  )
  assert rc.weight == pytest.approx(3.0)


def test_reward_weight_partial_application():
  rc = _make_reward_cfg()
  env = _make_reward_env(150, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 100, "weight": 2.0},
      {"step": 200, "weight": 4.0},
    ],
  )
  assert rc.weight == pytest.approx(2.0)


def test_step_zero_applies_immediately():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 0, "weight": 9.0}])
  assert rc.weight == pytest.approx(9.0)


# Reward: params


def test_reward_params_updated():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert rc.params["std"] == 0.2


def test_reward_params_unchanged_before_threshold():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert rc.params["std"] == 0.5


def test_reward_multiple_params_updated():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2, "scale": 2.0}}])
  assert rc.params["std"] == 0.2
  assert rc.params["scale"] == 2.0


# Reward: combined weight + params


def test_reward_weight_and_params_in_same_stage():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(env, "r", [{"step": 100, "weight": 5.0, "params": {"std": 0.1}}])
  assert rc.weight == pytest.approx(5.0)
  assert rc.params["std"] == 0.1


# Termination: params


def test_termination_params_updated():
  tc = _make_termination_cfg()
  env = _make_termination_env(200, tc)
  _build_termination(env, "energy", [{"step": 100, "params": {"threshold": 500.0}}])
  assert tc.params["threshold"] == 500.0


def test_termination_params_unchanged_before_threshold():
  tc = _make_termination_cfg()
  env = _make_termination_env(0, tc)
  _build_termination(env, "energy", [{"step": 100, "params": {"threshold": 500.0}}])
  assert tc.params["threshold"] == float("inf")


def test_termination_later_stage_wins():
  tc = _make_termination_cfg()
  env = _make_termination_env(500, tc)
  _build_termination(
    env,
    "energy",
    [
      {"step": 0, "params": {"threshold": 1000.0}},
      {"step": 100, "params": {"threshold": 700.0}},
      {"step": 400, "params": {"threshold": 400.0}},
    ],
  )
  assert tc.params["threshold"] == 400.0


# Validation: shared engine


def test_unknown_reward_param_raises():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  params = {"reward_name": "r", "stages": [{"step": 0, "params": {"stdd": 0.2}}]}
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  with pytest.raises(KeyError, match="unknown param"):
    reward_curriculum(cfg, env)


def test_unknown_termination_param_raises():
  tc = _make_termination_cfg()
  env = _make_termination_env(0, tc)
  params = {
    "termination_name": "energy",
    "stages": [{"step": 0, "params": {"thresholddd": 1.0}}],
  }
  cfg = CurriculumTermCfg(func=termination_curriculum, params=params)
  with pytest.raises(KeyError, match="unknown param"):
    termination_curriculum(cfg, env)


def test_unsorted_stages_raise():
  rc = _make_reward_cfg()
  env = _make_reward_env(0, rc)
  params = {
    "reward_name": "r",
    "stages": [
      {"step": 200, "weight": 1.0},
      {"step": 100, "weight": 2.0},
    ],
  }
  cfg = CurriculumTermCfg(func=reward_curriculum, params=params)
  with pytest.raises(ValueError, match="nondecreasing"):
    reward_curriculum(cfg, env)


def test_duplicate_steps_allowed():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  _build_reward(
    env,
    "r",
    [
      {"step": 100, "weight": 2.0},
      {"step": 100, "params": {"std": 0.1}},
    ],
  )
  assert rc.weight == pytest.approx(2.0)
  assert rc.params["std"] == 0.1


# Logging keys


def test_reward_logs_only_staged_keys():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  result = _build_reward(
    env, "r", [{"step": 100, "weight": 5.0, "params": {"std": 0.2}}]
  )
  assert result["weight"].item() == pytest.approx(5.0)
  assert result["std"].item() == pytest.approx(0.2)
  assert "scale" not in result  # Not in any stage.


def test_reward_omits_weight_when_not_staged():
  rc = _make_reward_cfg()
  env = _make_reward_env(200, rc)
  result = _build_reward(env, "r", [{"step": 100, "params": {"std": 0.2}}])
  assert "weight" not in result
  assert "std" in result


def test_termination_log_keys():
  tc = _make_termination_cfg()
  env = _make_termination_env(200, tc)
  result = _build_termination(
    env, "energy", [{"step": 100, "params": {"threshold": 500.0}}]
  )
  assert "threshold" in result
  assert result["threshold"].item() == pytest.approx(500.0)
  assert "weight" not in result  # No weight for termination.


# Success-driven reward weights


def test_success_reward_waits_for_prerequisite():
  reward_cfgs = {"a": _make_reward_cfg(), "b": _make_reward_cfg()}
  env = _make_success_reward_env(
    reward_cfgs,
    dones=[True] * 50,
    successes=[True] * 50,
    prerequisite_state={"stage": torch.tensor(4.0)},
  )
  result = _build_success_reward(
    env,
    [
      {"a": -1.0, "b": -2.0},
      {"a": -3.0, "b": -4.0},
    ],
    prerequisite_curriculum_name="target",
    prerequisite_stage_key="stage",
    prerequisite_min_stage=5.0,
  )

  assert result["waiting_for_prerequisite"].item() == pytest.approx(1.0)
  assert result["stage"].item() == pytest.approx(0.0)
  assert result["success_rate"].item() == pytest.approx(0.0)
  assert reward_cfgs["a"].weight == pytest.approx(-1.0)
  assert reward_cfgs["b"].weight == pytest.approx(-2.0)


def test_success_reward_keeps_stage_before_window_fills():
  reward_cfgs = {"a": _make_reward_cfg()}
  env = _make_success_reward_env(
    reward_cfgs,
    dones=[True] * 10,
    successes=[True] * 10,
  )
  result = _build_success_reward(env, [{"a": -1.0}, {"a": -2.0}])

  assert result["waiting_for_prerequisite"].item() == pytest.approx(0.0)
  assert result["stage"].item() == pytest.approx(0.0)
  assert result["success_rate"].item() == pytest.approx(1.0)
  assert reward_cfgs["a"].weight == pytest.approx(-1.0)


def test_success_reward_advances_after_threshold():
  reward_cfgs = {"a": _make_reward_cfg()}
  env = _make_success_reward_env(
    reward_cfgs,
    dones=[True] * 50,
    successes=[True] * 40 + [False] * 10,
  )
  result = _build_success_reward(env, [{"a": -1.0}, {"a": -2.0}])

  assert result["stage"].item() == pytest.approx(1.0)
  assert result["success_rate"].item() == pytest.approx(0.0)
  assert reward_cfgs["a"].weight == pytest.approx(-2.0)


def test_success_reward_updates_multiple_rewards_together():
  reward_cfgs = {
    "latent": _make_reward_cfg(),
    "torques": _make_reward_cfg(),
    "acc": _make_reward_cfg(),
  }
  env = _make_success_reward_env(
    reward_cfgs,
    dones=[True] * 50,
    successes=[True] * 50,
  )
  result = _build_success_reward(
    env,
    [
      {"latent": -0.005, "torques": -2e-5, "acc": -2e-6},
      {"latent": -0.01, "torques": -5e-5, "acc": -5e-6},
    ],
  )

  assert result["stage"].item() == pytest.approx(1.0)
  assert result["latent_weight"].item() == pytest.approx(-0.01)
  assert result["torques_weight"].item() == pytest.approx(-5e-5)
  assert result["acc_weight"].item() == pytest.approx(-5e-6)


def test_success_reward_final_stage_does_not_overflow():
  reward_cfgs = {"a": _make_reward_cfg()}
  env = _make_success_reward_env(
    reward_cfgs,
    dones=[True] * 50,
    successes=[True] * 50,
  )
  params = {
    "success_term_name": "success",
    "success_threshold": 0.8,
    "success_window": 50,
    "stage_weights": [{"a": -1.0}, {"a": -2.0}],
  }
  cfg = CurriculumTermCfg(func=success_reward_weight_curriculum, params=params)
  instance = success_reward_weight_curriculum(cfg, env)
  env_ids = torch.arange(50)

  first = instance(env, env_ids=env_ids, **params)
  second = instance(env, env_ids=env_ids, **params)

  assert first["stage"].item() == pytest.approx(1.0)
  assert second["stage"].item() == pytest.approx(1.0)
  assert second["success_rate"].item() == pytest.approx(1.0)
  assert reward_cfgs["a"].weight == pytest.approx(-2.0)
