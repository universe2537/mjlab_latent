"""Debug visualization helpers for table-tennis play."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.pingpong.mdp.pace import get_pingpong_pace_prediction_state
from mjlab.tasks.pingpong.mdp.state import get_pingpong_rally_state
from mjlab.tasks.pingpong.pace_geometry import G1_PACE_GEOMETRY
from mjlab.tasks.pingpong.scene import BALL_CENTER_TABLE_Z, NET_TOP_Z, NET_X
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_BALL_CFG = SceneEntityCfg("ball")
_ROBOT_CFG = SceneEntityCfg("robot")
_PADDLE_CFG = SceneEntityCfg("robot", site_names=("pingpong_paddle_center",))
_PADDLE_GEOM_CFG = SceneEntityCfg("robot", geom_names=("pingpong_paddle_collision",))
_PADDLE_SENSOR_NAME = "paddle_ball_contact"
_NET_SENSOR_NAME = "pingpong_ball_net_contact"
_BODY_BALL_SENSOR_NAME = "robot_ball_contact"
_INSTALLED_ATTR = "_pingpong_debug_overlay_installed"

_ORANGE = (1.0, 0.55, 0.0, 0.95)
_YELLOW = (1.0, 0.95, 0.0, 0.95)
_GREEN = (0.0, 0.95, 0.2, 0.95)
_CYAN = (0.0, 0.85, 1.0, 0.95)
_BLUE = (0.05, 0.3, 1.0, 0.9)
_MAGENTA = (1.0, 0.0, 0.95, 0.9)
_RED = (1.0, 0.08, 0.02, 0.9)
_WHITE = (1.0, 1.0, 1.0, 0.65)


@dataclass
class PingpongDebugOverlay:
  """Draw ball, paddle, and PACE target diagnostics into a DebugVisualizer."""

  env: ManagerBasedRlEnv
  include_pace: bool = True
  trajectory_steps: int = 18
  trajectory_dt: float = 0.035

  def __post_init__(self) -> None:
    self._paddle_cfg = SceneEntityCfg(
      "robot", site_names=("pingpong_paddle_center",)
    )
    self._paddle_geom_cfg = SceneEntityCfg(
      "robot", geom_names=("pingpong_paddle_collision",)
    )
    self._paddle_cfg.resolve(self.env.scene)
    self._paddle_geom_cfg.resolve(self.env.scene)

  def draw(self, visualizer: DebugVisualizer) -> None:
    env_indices = list(visualizer.get_env_indices(self.env.num_envs))
    if not env_indices:
      return

    try:
      ball = self.env.scene[_BALL_CFG.name]
      robot = self.env.scene[_ROBOT_CFG.name]
    except (KeyError, AttributeError):
      return

    origins = self.env.scene.env_origins
    ball_pos_w = ball.data.root_link_pos_w
    ball_pos_table = ball_pos_w - origins
    ball_vel = ball.data.root_link_lin_vel_w
    root_pos_w = robot.data.root_link_pos_w
    root_quat = robot.data.root_link_quat_w
    paddle_center = (
      robot.data.site_pos_w[:, self._paddle_cfg.site_ids].squeeze(1)
    )
    paddle_normal = self._paddle_normal(robot)
    desired_paddle = self._desired_forehand_paddle(root_pos_w, root_quat)

    rally = self._rally_state()
    rally.update()
    pace_state = None
    if self.include_pace:
      try:
        pace_state = get_pingpong_pace_prediction_state(
          self.env,
          paddle_sensor_name=_PADDLE_SENSOR_NAME,
          net_sensor_name=_NET_SENSOR_NAME,
          body_ball_sensor_name=_BODY_BALL_SENSOR_NAME,
          ball_cfg=_BALL_CFG,
          paddle_cfg=self._paddle_cfg,
          paddle_geom_cfg=self._paddle_geom_cfg,
          robot_cfg=_ROBOT_CFG,
        )
        pace_state.update()
      except Exception:
        pace_state = None

    for env_idx in env_indices:
      idx = int(env_idx)
      origin = origins[idx]
      b = ball_pos_w[idx]
      p = paddle_center[idx]
      n = paddle_normal[idx]
      desired = desired_paddle[idx]

      visualizer.add_sphere(self._np(b), 0.035, _ORANGE, label="ball")
      visualizer.add_arrow(
        self._np(b),
        self._np(b + ball_vel[idx] * 0.12),
        _ORANGE,
        width=0.012,
        label="ball_velocity",
      )
      self._draw_ball_trajectory(visualizer, b, ball_vel[idx])

      visualizer.add_sphere(self._np(p), 0.045, _GREEN, label="paddle_center")
      visualizer.add_arrow(
        self._np(p),
        self._np(p + n * 0.28),
        _GREEN,
        width=0.014,
        label="paddle_normal",
      )

      visualizer.add_sphere(
        self._np(desired), 0.045, _CYAN, label="forehand_paddle_target"
      )
      visualizer.add_cylinder(
        self._np(p),
        self._np(desired),
        0.006,
        _CYAN,
        label="paddle_to_forehand_target",
      )

      if bool(rally.impact_window_active[idx].item()):
        impact_ball_pos_w = rally.impact_ball_pos[idx] + origin
        visualizer.add_sphere(
          self._np(impact_ball_pos_w),
          0.045,
          _RED,
          label="impact_window_ball",
        )
        visualizer.add_arrow(
          self._np(impact_ball_pos_w),
          self._np(
            impact_ball_pos_w + rally.impact_desired_outgoing_dir[idx] * 0.35
          ),
          _RED,
          width=0.012,
          label="desired_outgoing_dir",
        )

      if pace_state is not None:
        future = pace_state.ball_future_pose[idx] + origin
        target_base = pace_state.robot_future_pos[idx] + origin
        predicted = pace_state.actor_prediction[idx]
        future_valid = bool(pace_state.ball_future_valid[idx].item())
        reward_active = bool(pace_state.reward_active[idx].item())
        future_color = _YELLOW if future_valid else _WHITE
        target_color = _MAGENTA if reward_active else _WHITE
        visualizer.add_sphere(
          self._np(future), 0.05, future_color, label="pace_future_ball_pose"
        )
        visualizer.add_cylinder(
          self._np(p),
          self._np(future),
          0.005,
          future_color,
          label="paddle_to_future_ball_pose",
        )
        visualizer.add_sphere(
          self._np(target_base),
          0.055,
          target_color,
          label="pace_target_base",
        )
        visualizer.add_cylinder(
          self._np(future),
          self._np(target_base),
          0.006,
          target_color,
          label="future_ball_to_target_base",
        )
        if torch.linalg.vector_norm(predicted).item() > 1.0e-6:
          visualizer.add_sphere(
            self._np(predicted + origin),
            0.045,
            _BLUE,
            label="learned_ball_prediction",
          )

        if bool(pace_state.predict_landing_valid[idx].item()):
          landing = torch.stack(
            (
              pace_state.predict_landing_xy[idx, 0],
              pace_state.predict_landing_xy[idx, 1],
              torch.as_tensor(
                BALL_CENTER_TABLE_Z, device=self.env.device, dtype=ball_pos_w.dtype
              ),
            )
          ) + origin
          visualizer.add_sphere(
            self._np(landing), 0.045, _BLUE, label="predicted_landing"
          )
        if bool(pace_state.predict_net_valid[idx].item()):
          net_point = torch.stack(
            (
              torch.as_tensor(NET_X, device=self.env.device, dtype=ball_pos_w.dtype),
              ball_pos_table[idx, 1],
              pace_state.predict_net_height[idx],
            )
          ) + origin
          visualizer.add_sphere(
            self._np(net_point), 0.04, _BLUE, label="predicted_net_height"
          )
          net_top = torch.stack(
            (
              torch.as_tensor(NET_X, device=self.env.device, dtype=ball_pos_w.dtype),
              ball_pos_table[idx, 1],
              torch.as_tensor(
                NET_TOP_Z, device=self.env.device, dtype=ball_pos_w.dtype
              ),
            )
          ) + origin
          visualizer.add_cylinder(
            self._np(net_top),
            self._np(net_point),
            0.006,
            _BLUE,
            label="net_clearance_segment",
          )

  def _paddle_normal(self, robot: Any) -> torch.Tensor:
    geom_quat = robot.data.geom_quat_w[:, self._paddle_geom_cfg.geom_ids]
    quat = geom_quat[:, 0]
    local_z = torch.zeros_like(quat[:, :3])
    local_z[:, 2] = 1.0
    normal = quat_apply(quat, local_z)
    return torch.nn.functional.normalize(normal, dim=-1, eps=1.0e-6)

  def _desired_forehand_paddle(
    self,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
  ) -> torch.Tensor:
    offset = torch.tensor(
      G1_PACE_GEOMETRY.forehand_paddle_offset,
      device=root_pos.device,
      dtype=root_pos.dtype,
    ).expand_as(root_pos)
    return root_pos + quat_apply(root_quat, offset)

  def _draw_ball_trajectory(
    self,
    visualizer: DebugVisualizer,
    ball_pos: torch.Tensor,
    ball_vel: torch.Tensor,
  ) -> None:
    gravity = torch.tensor((0.0, 0.0, -9.81), device=ball_pos.device, dtype=ball_pos.dtype)
    prev = ball_pos
    for i in range(1, self.trajectory_steps + 1):
      t = self.trajectory_dt * i
      pos = ball_pos + ball_vel * t + 0.5 * gravity * (t * t)
      if not torch.isfinite(pos).all():
        break
      color = _ORANGE if pos[2].item() >= BALL_CENTER_TABLE_Z else _WHITE
      visualizer.add_cylinder(
        self._np(prev),
        self._np(pos),
        0.004,
        color,
        label="ball_trajectory",
      )
      prev = pos

  def _rally_state(self):
    return get_pingpong_rally_state(
      self.env,
      paddle_sensor_name=_PADDLE_SENSOR_NAME,
      net_sensor_name=_NET_SENSOR_NAME,
      body_ball_sensor_name=_BODY_BALL_SENSOR_NAME,
      ball_cfg=_BALL_CFG,
      paddle_cfg=self._paddle_cfg,
      paddle_geom_cfg=self._paddle_geom_cfg,
      robot_cfg=_ROBOT_CFG,
    )

  @staticmethod
  def _np(value: torch.Tensor):
    return value.detach().cpu().numpy()


def install_pingpong_debug_overlay(
  env: ManagerBasedRlEnv,
  *,
  include_pace: bool = True,
) -> None:
  """Install Pingpong debug overlay drawing into ``env.update_visualizers``."""
  if getattr(env, _INSTALLED_ATTR, False):
    return

  overlay = PingpongDebugOverlay(env, include_pace=include_pace)
  original_update: Callable[[DebugVisualizer], None] = env.update_visualizers

  def _update_visualizers(visualizer: DebugVisualizer) -> None:
    original_update(visualizer)
    overlay.draw(visualizer)

  env.update_visualizers = _update_visualizers  # type: ignore[method-assign]
  env._pingpong_debug_overlay = overlay  # type: ignore[attr-defined]
  setattr(env, _INSTALLED_ATTR, True)


__all__ = [
  "PingpongDebugOverlay",
  "install_pingpong_debug_overlay",
]
