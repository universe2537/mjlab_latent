"""PACE-style PPO runner with an auxiliary ball predictor."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import check_nan

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.pingpong.mdp.pace import (
  get_pingpong_pace_prediction_state,
  pace_ball_position_table,
  update_pingpong_pace_prediction,
)


class _PaceBallPredictor(torch.nn.Module):
  def __init__(
    self,
    input_dim: int,
    hidden_sizes: tuple[int, int] = (64, 64),
    output_dim: int = 3,
  ) -> None:
    super().__init__()
    h0, h1 = hidden_sizes
    self.net = torch.nn.Sequential(
      torch.nn.Linear(input_dim, h0),
      torch.nn.ReLU(),
      torch.nn.Linear(h0, h1),
      torch.nn.ReLU(),
      torch.nn.Linear(h1, output_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class PingpongPaceOnPolicyRunner(MjlabOnPolicyRunner):
  """PPO runner that trains and applies the PACE learned ball predictor."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    pred_cfg = dict(train_cfg.pop("predictor", {}))
    super().__init__(env, train_cfg, log_dir, device)

    self.pred_history_len = int(pred_cfg.get("history_len", 5))
    self.pred_traj_maxlen = int(pred_cfg.get("traj_max_len", 128))
    hidden_sizes = pred_cfg.get("hidden_sizes", (64, 64))
    if len(hidden_sizes) != 2:
      raise ValueError("PACE predictor hidden_sizes must contain exactly two values.")
    self.pred_hidden = (int(hidden_sizes[0]), int(hidden_sizes[1]))
    self.pred_lr = float(pred_cfg.get("lr", 5.0e-4))
    self.pred_epochs = int(pred_cfg.get("epochs_per_update", 1))
    self.pred_batch_size = int(pred_cfg.get("batch_size", 1024))
    self.pred_train_until_iters = int(pred_cfg.get("train_until_iters", 20))

    self._pred_input_dim = 3 * self.pred_history_len
    self._predictor = _PaceBallPredictor(
      self._pred_input_dim,
      self.pred_hidden,
      output_dim=3,
    ).to(self.device)
    self._pred_optim = torch.optim.Adam(self._predictor.parameters(), lr=self.pred_lr)
    self._pred_trained = False
    self._last_pred_loss: float | None = None

    self._traj_buf_cpu = torch.zeros(
      self.pred_traj_maxlen,
      self.env.num_envs,
      3,
      dtype=torch.float32,
      device="cpu",
    )
    self._gt_buf_cpu = torch.zeros_like(self._traj_buf_cpu)
    self._traj_write_idx = 0
    self._traj_len = 0

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    """Run PPO while updating the PACE predictor between rollout steps."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    self._maybe_predict_and_update_env()
    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for rollout_step in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          self._record_ball_positions()
          self._maybe_predict_and_update_env()
          obs = self.env.get_observations()
          if self.cfg.get("check_for_nan", True):
            self._check_nan_with_diagnostics(obs, rewards, dones, it, rollout_step)
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = (
            self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
          )
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      pred_loss = None
      if it < self.pred_train_until_iters:
        pred_loss = self._train_predictor_offline()
      if pred_loss is not None:
        self._last_pred_loss = float(pred_loss)

      loss_dict = self.alg.update()
      if pred_loss is not None:
        loss_dict = dict(loss_dict)
        loss_dict["predictor_mse"] = float(pred_loss)

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it
      rnd_weight = None
      if self.cfg["algorithm"]["rnd_cfg"]:
        rnd_weight = getattr(self.alg.rnd, "weight", None)
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=rnd_weight,
      )

      if (
        self.logger.writer is not None
        and self.logger.log_dir is not None
        and it % self.cfg["save_interval"] == 0
      ):
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None and self.logger.log_dir is not None:
      self.save(
        os.path.join(
          self.logger.log_dir,
          f"model_{self.current_learning_iteration}.pt",
        )
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    """Save PPO state plus auxiliary predictor weights."""
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    saved_dict["pred_state_dict"] = self._predictor.state_dict()
    saved_dict["pred_optimizer_state_dict"] = self._pred_optim.state_dict()
    saved_dict["pred_cfg"] = {
      "history_len": self.pred_history_len,
      "traj_max_len": self.pred_traj_maxlen,
      "hidden_sizes": list(self.pred_hidden),
      "lr": self.pred_lr,
      "epochs_per_update": self.pred_epochs,
      "batch_size": self.pred_batch_size,
      "train_until_iters": self.pred_train_until_iters,
    }
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

  def _check_nan_with_diagnostics(
    self,
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    iteration: int,
    rollout_step: int,
  ) -> None:
    """Write term-level diagnostics before preserving the original NaN error."""
    if not self._contains_nan(obs, rewards, dones):
      check_nan(obs, rewards, dones)
      return

    path = self._write_nan_diagnostics(obs, rewards, dones, iteration, rollout_step)
    if path is not None:
      print(f"[PACE NaN diagnostic] wrote {path}", flush=True)
    check_nan(obs, rewards, dones)

  def _contains_nan(
    self,
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
  ) -> bool:
    for _, tensor in obs.items():
      if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any().item():
        return True
    return bool(torch.isnan(rewards).any().item() or torch.isnan(dones).any().item())

  def _write_nan_diagnostics(
    self,
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    iteration: int,
    rollout_step: int,
  ) -> Path | None:
    log_dir = getattr(self.logger, "log_dir", None)
    if log_dir is None:
      return None

    bad_env_ids = self._first_bad_env_ids(obs, rewards, dones)
    payload = {
      "error": (
        "ValueError: The observation group 'actor' returned by the environment "
        "contains NaN values. This usually indicates a bug in the environment's "
        "step() or reset() function."
      ),
      "iteration": int(iteration),
      "rollout_step": int(rollout_step),
      "current_learning_iteration": int(self.current_learning_iteration),
      "num_envs": int(self.env.num_envs),
      "bad_env_ids": bad_env_ids,
      "groups": self._observation_group_summaries(obs),
      "terms": self._observation_term_summaries(obs),
      "rewards": self._tensor_summary(rewards),
      "dones": self._tensor_summary(dones.float()),
      "env": self._env_state_snapshot(bad_env_ids),
      "predictor": self._predictor_snapshot(bad_env_ids),
    }

    dump_dir = Path(str(log_dir)) / "diagnostics"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / (
      f"pace_nan_iter{int(iteration):06d}_step{int(rollout_step):03d}.json"
    )
    with dump_path.open("w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
    return dump_path

  def _first_bad_env_ids(
    self,
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    max_envs: int = 8,
  ) -> list[int]:
    masks: list[torch.Tensor] = []
    for _, tensor in obs.items():
      if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
        masks.append(self._row_nonfinite_mask(tensor))
    masks.append(self._row_nonfinite_mask(rewards))
    masks.append(self._row_nonfinite_mask(dones.float()))
    if not masks:
      return []
    merged = masks[0].clone()
    for mask in masks[1:]:
      merged |= mask.to(device=merged.device)
    ids = torch.nonzero(merged, as_tuple=False).flatten()[:max_envs]
    return [int(i) for i in ids.detach().cpu().tolist()]

  def _observation_group_summaries(self, obs: Any) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, tensor in obs.items():
      if isinstance(tensor, torch.Tensor):
        groups[str(name)] = self._tensor_summary(tensor)
      else:
        groups[str(name)] = {"type": type(tensor).__name__}
    return groups

  def _observation_term_summaries(self, obs: Any) -> dict[str, Any]:
    env = self.env.unwrapped
    manager = getattr(env, "observation_manager", None)
    if manager is None:
      return {}

    term_names = getattr(manager, "active_terms", {})
    term_dims = getattr(manager, "group_obs_term_dim", {})
    group_concatenate = getattr(manager, "group_obs_concatenate", {})
    summaries: dict[str, Any] = {}

    for group_name, tensor in obs.items():
      if not isinstance(tensor, torch.Tensor):
        continue
      group = str(group_name)
      if not bool(group_concatenate.get(group, False)):
        summaries[group] = {"error": "group is not concatenated"}
        continue

      idx = 0
      group_terms: dict[str, Any] = {}
      names = term_names.get(group, [])
      dims = term_dims.get(group, [])
      for term_name, shape in zip(names, dims, strict=False):
        length = int(math.prod(shape))
        term_tensor = tensor[:, idx : idx + length]
        group_terms[str(term_name)] = self._tensor_summary(term_tensor)
        idx += length
      summaries[group] = group_terms
    return summaries

  def _env_state_snapshot(self, env_ids: list[int]) -> dict[str, Any]:
    env = self.env.unwrapped
    snapshot: dict[str, Any] = {
      "common_step_counter": int(getattr(env, "common_step_counter", -1)),
    }
    episode_length = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length, torch.Tensor):
      snapshot["episode_length_buf"] = self._select_rows(episode_length, env_ids)
    action = getattr(getattr(env, "action_manager", None), "action", None)
    if isinstance(action, torch.Tensor):
      snapshot["last_action"] = self._select_rows(action, env_ids, max_cols=32)

    scene = getattr(env, "scene", None)
    for entity_name in ("robot", "ball"):
      try:
        entity = scene[entity_name] if scene is not None else None
      except KeyError:
        continue
      if entity is None:
        continue
      data = getattr(entity, "data", None)
      if data is None:
        continue
      entity_snapshot: dict[str, Any] = {}
      for attr in (
        "root_link_pos_w",
        "root_link_quat_w",
        "root_link_lin_vel_w",
        "root_link_ang_vel_w",
      ):
        value = getattr(data, attr, None)
        if isinstance(value, torch.Tensor):
          entity_snapshot[attr] = self._select_rows(value, env_ids)
      for attr in ("joint_pos", "joint_vel"):
        value = getattr(data, attr, None)
        if isinstance(value, torch.Tensor):
          entity_snapshot[attr] = self._row_stats(value, env_ids)
      snapshot[entity_name] = entity_snapshot
    return snapshot

  def _predictor_snapshot(self, env_ids: list[int]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
      "trained": bool(self._pred_trained),
      "last_loss": self._last_pred_loss,
      "traj_len": int(self._traj_len),
      "traj_write_idx": int(self._traj_write_idx),
    }
    try:
      state = get_pingpong_pace_prediction_state(self.env.unwrapped)
    except RuntimeError:
      return snapshot
    for name in (
      "ball_future_pose",
      "ball_prediction",
      "robot_future_delta",
      "robot_future_vel",
      "valid_mask",
    ):
      value = getattr(state, name, None)
      if isinstance(value, torch.Tensor):
        snapshot[name] = self._select_rows(value, env_ids)
        snapshot[f"{name}_summary"] = self._tensor_summary(value)
    return snapshot

  def _tensor_summary(
    self,
    tensor: torch.Tensor,
    max_bad_envs: int = 8,
    max_bad_cols: int = 16,
  ) -> dict[str, Any]:
    data = tensor.detach()
    nan_mask = torch.isnan(data)
    inf_mask = torch.isinf(data)
    nonfinite = ~torch.isfinite(data)
    summary: dict[str, Any] = {
      "shape": list(data.shape),
      "nan_count": int(nan_mask.sum().item()),
      "inf_count": int(inf_mask.sum().item()),
      "nonfinite_count": int(nonfinite.sum().item()),
    }
    finite = data[torch.isfinite(data)]
    if finite.numel() > 0:
      summary["finite_min"] = float(finite.min().item())
      summary["finite_max"] = float(finite.max().item())
      summary["finite_mean"] = float(finite.float().mean().item())
    if data.numel() == 0:
      return summary

    bad_rows = self._row_nonfinite_mask(data)
    bad_ids = torch.nonzero(bad_rows, as_tuple=False).flatten()[:max_bad_envs]
    summary["bad_env_ids"] = [int(i) for i in bad_ids.detach().cpu().tolist()]
    bad_locations: dict[str, Any] = {}
    flattened = data.reshape(data.shape[0], -1) if data.ndim > 0 else data.reshape(1, -1)
    for env_id in summary["bad_env_ids"]:
      row = flattened[env_id]
      bad_cols = torch.nonzero(~torch.isfinite(row), as_tuple=False).flatten()
      cols = bad_cols[:max_bad_cols].detach().cpu().tolist()
      bad_locations[str(env_id)] = {
        "cols": [int(i) for i in cols],
        "values": [self._json_number(row[int(i)]) for i in cols],
      }
    if bad_locations:
      summary["bad_locations"] = bad_locations
    return summary

  def _row_nonfinite_mask(self, tensor: torch.Tensor) -> torch.Tensor:
    data = tensor.detach()
    if data.ndim == 0:
      return (~torch.isfinite(data)).reshape(1)
    return (~torch.isfinite(data)).reshape(data.shape[0], -1).any(dim=1)

  def _select_rows(
    self,
    tensor: torch.Tensor,
    env_ids: list[int],
    max_cols: int = 16,
  ) -> dict[str, Any]:
    if not env_ids:
      return {"summary": self._tensor_summary(tensor)}
    selected = tensor.detach()[env_ids]
    rows = selected.reshape(selected.shape[0], -1)
    values: dict[str, Any] = {}
    for row_idx, env_id in enumerate(env_ids):
      row = rows[row_idx, :max_cols]
      values[str(env_id)] = [self._json_number(v) for v in row]
    return {"shape": list(tensor.shape), "values": values}

  def _row_stats(self, tensor: torch.Tensor, env_ids: list[int]) -> dict[str, Any]:
    if not env_ids:
      return {"summary": self._tensor_summary(tensor)}
    selected = tensor.detach()[env_ids].reshape(len(env_ids), -1)
    stats: dict[str, Any] = {"shape": list(tensor.shape), "rows": {}}
    for row_idx, env_id in enumerate(env_ids):
      row = selected[row_idx]
      finite = row[torch.isfinite(row)]
      row_stats: dict[str, Any] = {
        "nan_count": int(torch.isnan(row).sum().item()),
        "inf_count": int(torch.isinf(row).sum().item()),
      }
      if finite.numel() > 0:
        row_stats["finite_min"] = float(finite.min().item())
        row_stats["finite_max"] = float(finite.max().item())
        row_stats["finite_mean"] = float(finite.float().mean().item())
      stats["rows"][str(env_id)] = row_stats
    return stats

  def _json_number(self, value: torch.Tensor) -> float | str:
    scalar = float(value.detach().cpu().item())
    if math.isnan(scalar):
      return "NaN"
    if math.isinf(scalar):
      return "Infinity" if scalar > 0 else "-Infinity"
    return scalar

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
    )
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    if "pred_state_dict" not in loaded_dict:
      self._pred_trained = False
      return infos
    try:
      self._predictor.load_state_dict(loaded_dict["pred_state_dict"])
      if load_cfg is None or bool(load_cfg.get("optimizer", True)):
        if "pred_optimizer_state_dict" in loaded_dict:
          self._pred_optim.load_state_dict(loaded_dict["pred_optimizer_state_dict"])
      self._pred_trained = True
    except RuntimeError:
      self._pred_trained = False
    return infos

  def get_inference_policy(self, device: str | None = None):
    base_policy = super().get_inference_policy(device=device)
    policy_device = torch.device(device) if device is not None else self.device
    runner = self

    class _PaceInferencePolicy:
      def __call__(self, obs):
        del obs
        with torch.no_grad():
          runner._record_ball_positions()
          runner._maybe_predict_and_update_env()
          fresh_obs = runner.env.get_observations().to(policy_device)
          return base_policy(fresh_obs)

    return _PaceInferencePolicy()

  def _record_ball_positions(self) -> None:
    env = self.env.unwrapped
    with torch.no_grad():
      ball_pos = pace_ball_position_table(env).detach().to("cpu")
      state = get_pingpong_pace_prediction_state(env)
      state.update()
      future_pose = state.ball_future_pose.detach().to("cpu")
      self._traj_buf_cpu[self._traj_write_idx].copy_(ball_pos)
      self._gt_buf_cpu[self._traj_write_idx].copy_(future_pose)
      self._traj_write_idx = (self._traj_write_idx + 1) % self.pred_traj_maxlen
      self._traj_len = min(self._traj_len + 1, self.pred_traj_maxlen)

  def _maybe_predict_and_update_env(self) -> None:
    if not self._pred_trained or self._traj_len < self.pred_history_len:
      return
    h = self.pred_history_len
    idxs = (torch.arange(-h, 0) + self._traj_write_idx) % self.pred_traj_maxlen
    hist = self._traj_buf_cpu[idxs]
    x = hist.permute(1, 0, 2).reshape(self.env.num_envs, -1).to(self.device)
    with torch.no_grad():
      preds = self._predictor(x)
    update_pingpong_pace_prediction(self.env.unwrapped, preds)

  def _train_predictor_offline(self) -> float | None:
    h = self.pred_history_len
    length = self._traj_len
    if length < h + 1:
      return None

    idxs = (torch.arange(-length, 0) + self._traj_write_idx) % self.pred_traj_maxlen
    seq = self._traj_buf_cpu[idxs]
    gt_seq = self._gt_buf_cpu[idxs]
    x_parts = []
    y_parts = []
    for t in range(h, length):
      hist = seq[t - h : t]
      x_parts.append(hist.permute(1, 0, 2).reshape(self.env.num_envs, -1))
      y_parts.append(gt_seq[t - 1])
    if not x_parts:
      return None

    x = torch.cat(x_parts, dim=0).to(self.device)
    y = torch.cat(y_parts, dim=0).to(self.device)
    finite = torch.isfinite(x).all(dim=1) & torch.isfinite(y).all(dim=1)
    if not finite.any():
      return None
    x = x[finite]
    y = y[finite]
    total_loss = 0.0
    n_batches = 0
    batch_size = max(1, self.pred_batch_size)
    self._predictor.train()
    for _ in range(max(1, self.pred_epochs)):
      perm = torch.randperm(x.shape[0], device=self.device)
      for start in range(0, x.shape[0], batch_size):
        ids = perm[start : start + batch_size]
        pred = self._predictor(x[ids])
        loss = torch.nn.functional.mse_loss(pred, y[ids])
        self._pred_optim.zero_grad()
        loss.backward()
        self._pred_optim.step()
        total_loss += float(loss.detach().item())
        n_batches += 1
    self._predictor.eval()
    self._pred_trained = True
    return total_loss / max(1, n_batches)
