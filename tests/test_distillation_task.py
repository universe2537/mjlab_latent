"""Regression tests for posterior-driven latent distillation."""

from typing import Any
from unittest.mock import Mock

import pytest
import torch

import mjlab.tasks.distillation.rl.runner as runner_module
from mjlab.envs.mdp.events import apply_body_impulse
from mjlab.tasks.distillation.config.g1.env_cfgs import (
  unitree_g1_flat_distillation_env_cfg,
  unitree_g1_table_tennis_distillation_env_cfg,
)
from mjlab.tasks.distillation.config.g1.rl_cfg import (
  unitree_g1_distillation_runner_cfg,
)
from mjlab.tasks.distillation.rl.runner import OnlineDistillationRunner


def _bare_runner() -> Any:
  return object.__new__(OnlineDistillationRunner)


def test_distillation_action_helpers_route_posterior_and_prior() -> None:
  runner = _bare_runner()
  state = torch.randn(2, 5)
  target = torch.randn(2, 3)
  actor_obs = torch.randn(2, 8)
  expected_action = torch.randn(2, 4)
  runner.slicer = Mock()
  runner.slicer.split.return_value = (state, target)
  runner.model = Mock()
  runner.model.act.return_value = expected_action

  posterior_action = runner._posterior_action(actor_obs, deterministic=False)

  assert posterior_action is expected_action
  runner.model.act.assert_called_once_with(
    state,
    target,
    deterministic=False,
    source="posterior",
  )

  runner.model.act.reset_mock()
  prior_action = runner._prior_action(actor_obs, deterministic=True)

  assert prior_action is expected_action
  runner.model.act.assert_called_once_with(
    state,
    deterministic=True,
    source="prior",
  )


def test_distillation_rollout_uses_stochastic_posterior() -> None:
  runner = _bare_runner()
  actor_obs = torch.randn(2, 8)
  teacher_action = torch.randn(2, 4)
  posterior_action = torch.randn(2, 4)
  obs = {"actor": actor_obs}
  next_obs = {"actor": torch.randn(2, 8)}
  runner.cfg = {
    "obs_group": "actor",
    "num_steps_per_env": 1,
    "deterministic_rollout": False,
  }
  runner.device = torch.device("cpu")
  runner.teacher_policy = Mock(return_value=teacher_action)
  runner.model = Mock()
  runner.buffer = Mock()
  runner.env = Mock()
  runner.env.step.return_value = (next_obs, None, None, None)
  runner._posterior_action = Mock(return_value=posterior_action)

  result = runner._rollout(obs, teacher_prob=0.0)

  assert result is next_obs
  runner._posterior_action.assert_called_once_with(
    actor_obs,
    deterministic=False,
  )
  runner.env.step.assert_called_once_with(posterior_action)
  runner.buffer.add.assert_called_once()


def test_distillation_inference_uses_posterior_mean() -> None:
  runner = _bare_runner()
  actor_obs = torch.randn(2, 8)
  expected_action = torch.randn(2, 4)
  runner.cfg = {"obs_group": "actor"}
  runner.device = torch.device("cpu")
  runner.model = Mock()
  runner._posterior_action = Mock(return_value=expected_action)

  policy = runner.get_inference_policy()
  action = policy({"actor": actor_obs})

  assert action is expected_action
  runner._posterior_action.assert_called_once_with(
    actor_obs,
    deterministic=True,
  )


def test_g1_distillation_schedule_and_kl_defaults() -> None:
  cfg = unitree_g1_distillation_runner_cfg()
  runner = _bare_runner()
  runner.cfg = {
    "teacher_action_prob": cfg.teacher_action_prob,
    "teacher_action_prob_end": cfg.teacher_action_prob_end,
    "teacher_action_prob_anneal_iters": cfg.teacher_action_prob_anneal_iters,
  }

  assert runner._teacher_action_prob(0) == pytest.approx(1.0)
  assert runner._teacher_action_prob(1250) == pytest.approx(0.5)
  assert runner._teacher_action_prob(2500) == pytest.approx(0.0)
  assert runner._teacher_action_prob(5000) == pytest.approx(0.0)
  assert cfg.deterministic_rollout is False
  assert cfg.kl_loss_weight == pytest.approx(1.0e-3)
  assert cfg.kl_loss_weight_end == pytest.approx(5.0e-3)
  assert cfg.kl_loss_anneal_start == 2500
  assert cfg.kl_loss_anneal_end == 10000


def test_distillation_configures_multi_gpu_process_group(monkeypatch) -> None:
  runner = _bare_runner()
  runner.device = torch.device("cuda:1")
  runner.cfg = {}
  init_process_group = Mock()
  set_device = Mock()
  monkeypatch.setenv("WORLD_SIZE", "2")
  monkeypatch.setenv("LOCAL_RANK", "1")
  monkeypatch.setenv("RANK", "1")
  monkeypatch.setattr(runner_module, "is_initialized", lambda: False)
  monkeypatch.setattr(runner_module, "init_process_group", init_process_group)
  monkeypatch.setattr(torch.cuda, "set_device", set_device)

  runner._configure_multi_gpu()

  assert runner.is_distributed is True
  assert runner.gpu_world_size == 2
  assert runner.gpu_local_rank == 1
  assert runner.gpu_global_rank == 1
  assert runner.cfg["multi_gpu"] == {
    "global_rank": 1,
    "local_rank": 1,
    "world_size": 2,
  }
  init_process_group.assert_called_once_with(backend="nccl", rank=1, world_size=2)
  set_device.assert_called_once_with(1)


def test_distillation_averages_gradients_across_workers(monkeypatch) -> None:
  runner = _bare_runner()
  runner.is_distributed = True
  runner.gpu_world_size = 2
  runner.model = torch.nn.Linear(3, 2)
  original_grads = []
  for parameter in runner.model.parameters():
    parameter.grad = torch.full_like(parameter, 2.0)
    original_grads.append(parameter.grad.clone())

  all_reduce = Mock(side_effect=lambda tensor, op: tensor.mul_(2.0))
  monkeypatch.setattr(runner_module, "all_reduce", all_reduce)

  runner._reduce_gradients()

  all_reduce.assert_called_once()
  for parameter, expected_grad in zip(
    runner.model.parameters(), original_grads, strict=True
  ):
    torch.testing.assert_close(parameter.grad, expected_grad)


@pytest.mark.parametrize(
  "env_cfg_builder",
  (
    unitree_g1_flat_distillation_env_cfg,
    unitree_g1_table_tennis_distillation_env_cfg,
  ),
)
def test_distillation_wrist_impulse_is_training_only(env_cfg_builder) -> None:
  train_cfg = env_cfg_builder(play=False)
  play_cfg = env_cfg_builder(play=True)

  assert "encoder_bias" in train_cfg.events
  assert "wrist_encoder_bias" not in train_cfg.events
  assert "wrist_encoder_bias" not in play_cfg.events
  assert "right_wrist_force_impulse" in train_cfg.events
  assert "right_wrist_force_impulse" not in play_cfg.events

  impulse_cfg = train_cfg.events["right_wrist_force_impulse"]
  assert impulse_cfg.mode == "step"
  assert impulse_cfg.func is apply_body_impulse
  assert impulse_cfg.params["asset_cfg"].name == "robot"
  assert impulse_cfg.params["asset_cfg"].body_names == ("right_wrist_yaw_link",)
  assert impulse_cfg.params["force_range"] == (-5.0, 5.0)
  assert impulse_cfg.params["torque_range"] == (0.0, 0.0)
  assert impulse_cfg.params["duration_s"] == (0.05, 0.12)
  assert impulse_cfg.params["cooldown_s"] == (0.5, 1.5)
  assert impulse_cfg.params["body_point_offset"] == (
    -0.00138455,
    -0.02790999,
    0.25233888,
  )
