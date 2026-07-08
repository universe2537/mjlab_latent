"""Calibrate G1 Pingpong PACE geometry with FK metrics and pose renders."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import mediapy as media
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1_w_pingpong_paddle import (
  get_g1_w_pingpong_paddle_spec,
)
from mjlab.asset_zoo.robots.unitree_g1_w_racket.g1_constants import (
  HOME_KEYFRAME,
  KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg
from mjlab.tasks.pingpong.pace_geometry import G1_PACE_GEOMETRY
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

_BODY_NAMES = (
  "pelvis",
  "torso_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)
_SITE_NAMES = ("pingpong_paddle_center",)
_VIEW_SPECS = {
  "front": (180.0, -12.0, 2.3),
  "side": (90.0, -12.0, 2.3),
  "top": (180.0, -88.0, 2.8),
  "iso": (135.0, -22.0, 2.5),
  "hand_closeup": (125.0, -18.0, 0.75),
}
_PLACEHOLDER_COLORS = {
  "front": (55, 80, 125),
  "side": (70, 105, 80),
  "top": (110, 90, 70),
  "iso": (95, 75, 110),
  "hand_closeup": (120, 70, 75),
}
_TARGET_REACH_COLOR = (0.95, 0.05, 1.0, 1.0)
_TARGET_REACH_LINE_COLOR = (0.85, 0.0, 1.0, 0.75)
_REACH_ERROR_LINE_COLOR = (0.2, 0.6, 1.0, 0.85)
_STRIKE_DIRECTION_COLOR = (1.0, 0.55, 0.05, 1.0)
_STRIKE_DIRECTION_LENGTH = 0.34


@dataclass(frozen=True)
class PoseSpec:
  name: str
  base: str
  description: str
  joint_overrides: dict[str, float]


def _default_pose_specs() -> list[PoseSpec]:
  return [
    PoseSpec("zero_pose", "zero", "Raw XML qpos0 pose.", {}),
    PoseSpec("home", "home", "G1 HOME_KEYFRAME posture.", {}),
    PoseSpec("knees_bent", "knees_bent", "Current G1 pingpong reset posture.", {}),
    PoseSpec(
      "forehand_open_light",
      "knees_bent",
      "Knees-bent stance with a lightly opened right forehand reach.",
      {
        "right_shoulder_pitch_joint": 0.12,
        "right_shoulder_roll_joint": -0.30,
        "right_shoulder_yaw_joint": 0.10,
        "right_elbow_joint": 0.78,
        "right_wrist_pitch_joint": 0.04,
        "right_wrist_yaw_joint": -0.03,
      },
    ),
    PoseSpec(
      "forehand_open_medium",
      "knees_bent",
      "Knees-bent stance with a more open right forehand reach.",
      {
        "right_shoulder_pitch_joint": 0.04,
        "right_shoulder_roll_joint": -0.42,
        "right_shoulder_yaw_joint": 0.18,
        "right_elbow_joint": 0.92,
        "right_wrist_pitch_joint": 0.08,
        "right_wrist_yaw_joint": -0.06,
      },
    ),
  ]


def _load_custom_pose_specs(path: Path | None) -> list[PoseSpec]:
  if path is None:
    return []
  with path.open("r", encoding="utf-8") as f:
    data = json.load(f)
  raw_poses = data["poses"] if isinstance(data, dict) and "poses" in data else data
  poses = []
  for raw in raw_poses:
    poses.append(
      PoseSpec(
        name=str(raw["name"]),
        base=str(raw.get("base", "knees_bent")),
        description=str(raw.get("description", "Custom calibration pose.")),
        joint_overrides={
          str(name): float(value)
          for name, value in dict(raw.get("joint_overrides", {})).items()
        },
      )
    )
  return poses


def _joint_names(model: mujoco.MjModel) -> list[str]:
  names = []
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    names.append("" if name is None else name)
  return names


def _object_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
  obj_id = mujoco.mj_name2id(model, obj_type, name)
  if obj_id < 0:
    raise ValueError(f"Could not find {obj_type.name} named {name!r}.")
  return int(obj_id)


def _set_root_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  pos: Iterable[float],
  quat: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
  for joint_id in range(model.njnt):
    if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
      continue
    qadr = int(model.jnt_qposadr[joint_id])
    data.qpos[qadr : qadr + 3] = np.asarray(tuple(pos), dtype=np.float64)
    data.qpos[qadr + 3 : qadr + 7] = np.asarray(tuple(quat), dtype=np.float64)
    return


def _set_joint_value(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float):
  joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
  if int(model.jnt_type[joint_id]) not in (
    int(mujoco.mjtJoint.mjJNT_HINGE),
    int(mujoco.mjtJoint.mjJNT_SLIDE),
  ):
    return
  qadr = int(model.jnt_qposadr[joint_id])
  limited = bool(model.jnt_limited[joint_id])
  if limited:
    lo, hi = model.jnt_range[joint_id]
    value = float(np.clip(value, lo, hi))
  data.qpos[qadr] = value


def _apply_keyframe(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  init_state: EntityCfg.InitialStateCfg,
) -> None:
  _set_root_pose(model, data, init_state.pos, init_state.rot)
  names = _joint_names(model)
  for pattern, value in (init_state.joint_pos or {}).items():
    regex = re.compile(pattern)
    for name in names:
      if name and regex.fullmatch(name):
        _set_joint_value(model, data, name, float(value))


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseSpec) -> None:
  data.qpos[:] = model.qpos0
  data.qvel[:] = 0.0
  if pose.base == "home":
    _apply_keyframe(model, data, HOME_KEYFRAME)
  elif pose.base == "knees_bent":
    _apply_keyframe(model, data, KNEES_BENT_KEYFRAME)
  elif pose.base != "zero":
    raise ValueError(f"Unsupported pose base {pose.base!r}.")
  for name, value in pose.joint_overrides.items():
    _set_joint_value(model, data, name, value)
  mujoco.mj_forward(model, data)


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
  body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  return np.array(data.xpos[body_id], dtype=np.float64)


def _body_mat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
  body_id = _object_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
  return np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
  site_id = _object_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
  return np.array(data.site_xpos[site_id], dtype=np.float64)


def _geom_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
  geom_id = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
  return np.array(data.geom_xpos[geom_id], dtype=np.float64)


def _as_list(values: np.ndarray) -> list[float]:
  return [float(v) for v in values.reshape(-1)]


def _foot_center(
  model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> np.ndarray:
  points = [
    _geom_pos(model, data, f"{side}_foot{i}_collision") for i in range(1, 8)
  ]
  return np.mean(np.stack(points, axis=0), axis=0)


def _desired_reach_world(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  reach_offset_pelvis: tuple[float, float, float],
) -> np.ndarray:
  pelvis = _body_pos(model, data, "pelvis")
  pelvis_mat = _body_mat(model, data, "pelvis")
  offset = np.asarray(reach_offset_pelvis, dtype=np.float64)
  return pelvis + pelvis_mat @ offset


def _strike_direction_world(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  strike_direction_pelvis: tuple[float, float, float],
) -> np.ndarray:
  pelvis_mat = _body_mat(model, data, "pelvis")
  direction = pelvis_mat @ np.asarray(strike_direction_pelvis, dtype=np.float64)
  norm = np.linalg.norm(direction)
  if norm < 1.0e-9:
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)
  return direction / norm


def _pose_metrics(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  pose: PoseSpec,
  *,
  reach_offset_pelvis: tuple[float, float, float],
  strike_direction_pelvis: tuple[float, float, float],
) -> dict[str, Any]:
  body_positions = {name: _as_list(_body_pos(model, data, name)) for name in _BODY_NAMES}
  site_positions = {name: _as_list(_site_pos(model, data, name)) for name in _SITE_NAMES}
  foot_geom_positions = {
    name: _as_list(_geom_pos(model, data, name))
    for name in G1_PACE_GEOMETRY.foot_geom_names
  }

  pelvis = _body_pos(model, data, "pelvis")
  pelvis_mat = _body_mat(model, data, "pelvis")
  torso = _body_pos(model, data, "torso_link")
  paddle = _site_pos(model, data, "pingpong_paddle_center")
  desired_reach = _desired_reach_world(model, data, reach_offset_pelvis)
  strike_direction = _strike_direction_world(model, data, strike_direction_pelvis)
  rel_world = paddle - pelvis
  rel_pelvis = pelvis_mat.T @ rel_world
  reach_error_world = paddle - desired_reach
  reach_error_pelvis = rel_pelvis - np.asarray(reach_offset_pelvis, dtype=np.float64)

  shoulder = _body_pos(model, data, "right_shoulder_pitch_link")
  elbow = _body_pos(model, data, "right_elbow_link")
  wrist = _body_pos(model, data, "right_wrist_yaw_link")
  upper = float(np.linalg.norm(elbow - shoulder))
  lower = float(np.linalg.norm(wrist - elbow))
  span = float(np.linalg.norm(wrist - shoulder))
  ratio = span / max(upper + lower, 1.0e-9)

  left_foot = _foot_center(model, data, "left")
  right_foot = _foot_center(model, data, "right")
  foot_delta = left_foot - right_foot

  return {
    "name": pose.name,
    "base": pose.base,
    "description": pose.description,
    "joint_overrides": pose.joint_overrides,
    "root_height": float(pelvis[2]),
    "torso_height": float(torso[2]),
    "paddle_offset_world": _as_list(rel_world),
    "paddle_offset_pelvis": _as_list(rel_pelvis),
    "target_base_offset_xy": _as_list(-rel_pelvis[:2]),
    "desired_reach_offset_pelvis": _as_list(
      np.asarray(reach_offset_pelvis, dtype=np.float64)
    ),
    "desired_reach_position_world": _as_list(desired_reach),
    "desired_reach_error_world": _as_list(reach_error_world),
    "desired_reach_error_pelvis": _as_list(reach_error_pelvis),
    "desired_reach_error_norm": float(np.linalg.norm(reach_error_pelvis)),
    "desired_strike_direction_pelvis": _as_list(
      np.asarray(strike_direction_pelvis, dtype=np.float64)
    ),
    "desired_strike_direction_world": _as_list(strike_direction),
    "elbow_extension": {
      "upper": upper,
      "lower": lower,
      "span": span,
      "ratio": ratio,
    },
    "foot_center_left": _as_list(left_foot),
    "foot_center_right": _as_list(right_foot),
    "foot_spacing_xy": float(np.linalg.norm(foot_delta[:2])),
    "foot_spacing_xyz": float(np.linalg.norm(foot_delta)),
    "body_positions": body_positions,
    "site_positions": site_positions,
    "foot_geom_positions": foot_geom_positions,
  }


def _make_camera(
  model: mujoco.MjModel,
  lookat: np.ndarray,
  azimuth: float,
  elevation: float,
  distance: float,
) -> mujoco.MjvCamera:
  camera = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, camera)
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
  camera.lookat[:] = lookat
  camera.distance = distance
  camera.azimuth = azimuth
  camera.elevation = elevation
  return camera


def _add_pose_overlay(
  visualizer: MujocoNativeDebugVisualizer,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  reach_offset_pelvis: tuple[float, float, float],
  strike_direction_pelvis: tuple[float, float, float],
) -> None:
  points = {
    "pelvis": (_body_pos(model, data, "pelvis"), (0.1, 0.35, 1.0, 1.0), 0.025),
    "torso": (_body_pos(model, data, "torso_link"), (0.1, 0.8, 1.0, 1.0), 0.02),
    "shoulder": (
      _body_pos(model, data, "right_shoulder_pitch_link"),
      (1.0, 0.9, 0.1, 1.0),
      0.018,
    ),
    "elbow": (_body_pos(model, data, "right_elbow_link"), (1.0, 0.45, 0.1, 1.0), 0.018),
    "wrist": (
      _body_pos(model, data, "right_wrist_yaw_link"),
      (1.0, 0.15, 0.1, 1.0),
      0.018,
    ),
    "paddle": (
      _site_pos(model, data, "pingpong_paddle_center"),
      (0.0, 1.0, 0.2, 1.0),
      0.022,
    ),
  }
  for _, (center, color, radius) in points.items():
    visualizer.add_sphere(center, radius=radius, color=color)

  shoulder = points["shoulder"][0]
  elbow = points["elbow"][0]
  wrist = points["wrist"][0]
  paddle = points["paddle"][0]
  visualizer.add_cylinder(shoulder, elbow, radius=0.008, color=(1.0, 0.8, 0.0, 0.9))
  visualizer.add_cylinder(elbow, wrist, radius=0.008, color=(1.0, 0.45, 0.0, 0.9))
  visualizer.add_cylinder(wrist, paddle, radius=0.006, color=(0.0, 1.0, 0.3, 0.9))

  pelvis = points["pelvis"][0]
  pelvis_mat = _body_mat(model, data, "pelvis")
  desired_reach = _desired_reach_world(model, data, reach_offset_pelvis)
  strike_direction = _strike_direction_world(model, data, strike_direction_pelvis)
  visualizer.add_sphere(
    desired_reach,
    radius=0.028,
    color=_TARGET_REACH_COLOR,
  )
  visualizer.add_cylinder(
    pelvis,
    desired_reach,
    radius=0.005,
    color=_TARGET_REACH_LINE_COLOR,
  )
  visualizer.add_cylinder(
    paddle,
    desired_reach,
    radius=0.004,
    color=_REACH_ERROR_LINE_COLOR,
  )
  visualizer.add_arrow(
    start=desired_reach,
    end=desired_reach + strike_direction * _STRIKE_DIRECTION_LENGTH,
    color=_STRIKE_DIRECTION_COLOR,
    width=0.018,
  )
  visualizer.add_frame(pelvis, pelvis_mat, scale=0.22, axis_radius=0.008)


def _write_placeholder(path: Path, view_name: str, width: int, height: int) -> None:
  rgb = np.asarray(_PLACEHOLDER_COLORS[view_name], dtype=np.uint8)
  image = np.zeros((height, width, 3), dtype=np.uint8)
  image[:] = rgb
  media.write_image(path, image)


def _render_pose_images(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  pose_dir: Path,
  *,
  reach_offset_pelvis: tuple[float, float, float],
  strike_direction_pelvis: tuple[float, float, float],
  width: int,
  height: int,
  render: bool,
) -> dict[str, str]:
  image_paths: dict[str, str] = {}
  pose_dir.mkdir(parents=True, exist_ok=True)
  if not render:
    for view_name in _VIEW_SPECS:
      path = pose_dir / f"{view_name}.png"
      overlay_path = pose_dir / f"{view_name}_overlay.png"
      _write_placeholder(path, view_name, width, height)
      _write_placeholder(overlay_path, view_name, width, height)
      image_paths[view_name] = str(path)
      image_paths[f"{view_name}_overlay"] = str(overlay_path)
    return image_paths

  opt = mujoco.MjvOption()
  opt.geomgroup[:] = 1
  model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
  model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
  renderer = mujoco.Renderer(model, height=height, width=width)
  try:
    pelvis = _body_pos(model, data, "pelvis")
    paddle = _site_pos(model, data, "pingpong_paddle_center")
    for view_name, (azimuth, elevation, distance) in _VIEW_SPECS.items():
      lookat = pelvis.copy()
      lookat[2] += 0.10
      view_distance = distance
      if view_name == "hand_closeup":
        lookat = 0.65 * paddle + 0.35 * _body_pos(model, data, "right_elbow_link")
      camera = _make_camera(model, lookat, azimuth, elevation, view_distance)

      renderer.update_scene(data, camera=camera, scene_option=opt)
      image = renderer.render()
      path = pose_dir / f"{view_name}.png"
      media.write_image(path, image)
      image_paths[view_name] = str(path)

      renderer.update_scene(data, camera=camera, scene_option=opt)
      visualizer = MujocoNativeDebugVisualizer(renderer.scene, model, env_idx=0)
      _add_pose_overlay(
        visualizer,
        model,
        data,
        reach_offset_pelvis,
        strike_direction_pelvis,
      )
      overlay_image = renderer.render()
      overlay_path = pose_dir / f"{view_name}_overlay.png"
      media.write_image(overlay_path, overlay_image)
      image_paths[f"{view_name}_overlay"] = str(overlay_path)
  finally:
    renderer.close()
  return image_paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  fields = (
    "name",
    "base",
    "root_height",
    "torso_height",
    "paddle_offset_pelvis_x",
    "paddle_offset_pelvis_y",
    "paddle_offset_pelvis_z",
    "target_base_offset_x",
    "target_base_offset_y",
    "desired_reach_x",
    "desired_reach_y",
    "desired_reach_z",
    "desired_reach_error_norm",
    "desired_strike_dir_x",
    "desired_strike_dir_y",
    "desired_strike_dir_z",
    "elbow_extension_ratio",
    "foot_spacing_xy",
  )
  with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      paddle_offset = row["paddle_offset_pelvis"]
      target_offset = row["target_base_offset_xy"]
      desired_reach = row["desired_reach_position_world"]
      strike_direction = row["desired_strike_direction_world"]
      writer.writerow(
        {
          "name": row["name"],
          "base": row["base"],
          "root_height": row["root_height"],
          "torso_height": row["torso_height"],
          "paddle_offset_pelvis_x": paddle_offset[0],
          "paddle_offset_pelvis_y": paddle_offset[1],
          "paddle_offset_pelvis_z": paddle_offset[2],
          "target_base_offset_x": target_offset[0],
          "target_base_offset_y": target_offset[1],
          "desired_reach_x": desired_reach[0],
          "desired_reach_y": desired_reach[1],
          "desired_reach_z": desired_reach[2],
          "desired_reach_error_norm": row["desired_reach_error_norm"],
          "desired_strike_dir_x": strike_direction[0],
          "desired_strike_dir_y": strike_direction[1],
          "desired_strike_dir_z": strike_direction[2],
          "elbow_extension_ratio": row["elbow_extension"]["ratio"],
          "foot_spacing_xy": row["foot_spacing_xy"],
        }
      )


def _write_readme(
  path: Path,
  rows: list[dict[str, Any]],
  *,
  render: bool,
  reach_offset_pelvis: tuple[float, float, float],
  strike_direction_pelvis: tuple[float, float, float],
) -> None:
  lines = [
    "# G1 PACE Pose Calibration",
    "",
    "This folder was generated by `contact_test/run_g1_pace_pose_calibration.py`.",
    "",
    f"- Rendered images: `{render}`",
    f"- Default PACE target base offset: `{G1_PACE_GEOMETRY.target_base_offset_xy}`",
    f"- Default PACE root height: `{G1_PACE_GEOMETRY.target_root_height}`",
    f"- Default PACE paddle offset: `{G1_PACE_GEOMETRY.forehand_paddle_offset}`",
    f"- Default PACE strike direction in pelvis frame: `{strike_direction_pelvis}`",
    f"- Default PACE strike upward angle: `{G1_PACE_GEOMETRY.strike_upward_angle}` rad",
    f"- Visualized desired reach offset in pelvis frame: `{reach_offset_pelvis}`",
    "- Overlay legend: green sphere = current paddle center; purple sphere =",
    "  desired reach center; purple line = pelvis-to-target reach vector; blue",
    "  line = current paddle-to-target error; orange arrow = desired strike",
    "  direction.",
    "",
    "Pose base names map to the current G1 asset setup: `home` uses",
    "`HOME_KEYFRAME`, and `knees_bent` uses `KNEES_BENT_KEYFRAME`.",
    "",
    "## Pose Summary",
    "",
    "| pose | base | root z | torso z | paddle offset pelvis | reach error | elbow ratio |",
    "| --- | --- | ---: | ---: | --- | ---: | ---: |",
  ]
  for row in rows:
    offset = tuple(round(float(v), 4) for v in row["paddle_offset_pelvis"])
    lines.append(
      (
        "| {name} | {base} | {root:.4f} | {torso:.4f} | {offset} | "
        "{error:.4f} | {ratio:.4f} |"
      ).format(
        name=row["name"],
        base=row["base"],
        root=row["root_height"],
        torso=row["torso_height"],
        offset=offset,
        error=row["desired_reach_error_norm"],
        ratio=row["elbow_extension"]["ratio"],
      )
    )
  lines.extend(
    [
      "",
      "Use `calibration.json` as the source of truth when selecting a new G1",
      "PACE geometry target. The script does not mutate training configs.",
    ]
  )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(
  *,
  output_dir: Path,
  pose_set: str = "default",
  custom_pose_file: Path | None = None,
  reach_offset_pelvis: tuple[float, float, float] = (
    G1_PACE_GEOMETRY.forehand_paddle_offset
  ),
  strike_direction_pelvis: tuple[float, float, float] = (
    G1_PACE_GEOMETRY.strike_direction_pelvis
  ),
  width: int = 1280,
  height: int = 720,
  render: bool = True,
) -> Path:
  if pose_set != "default":
    raise ValueError("Only --pose-set default is currently supported.")
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_dir = output_dir / f"{timestamp}_g1_pace_pose_calibration"
  run_dir.mkdir(parents=True, exist_ok=False)

  spec = get_g1_w_pingpong_paddle_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  poses = _default_pose_specs() + _load_custom_pose_specs(custom_pose_file)

  rows = []
  for pose in poses:
    _apply_pose(model, data, pose)
    metrics = _pose_metrics(
      model,
      data,
      pose,
      reach_offset_pelvis=reach_offset_pelvis,
      strike_direction_pelvis=strike_direction_pelvis,
    )
    image_paths = _render_pose_images(
      model,
      data,
      run_dir / pose.name,
      reach_offset_pelvis=reach_offset_pelvis,
      strike_direction_pelvis=strike_direction_pelvis,
      width=width,
      height=height,
      render=render,
    )
    metrics["images"] = image_paths
    rows.append(metrics)

  payload = {
    "schema_version": 1,
    "generated_at": timestamp,
    "pose_set": pose_set,
    "rendered": render,
    "image_size": {"width": width, "height": height},
    "visualized_reach_offset_pelvis": reach_offset_pelvis,
    "visualized_strike_direction_pelvis": strike_direction_pelvis,
    "pace_geometry_default": {
      "target_base_offset_xy": G1_PACE_GEOMETRY.target_base_offset_xy,
      "target_root_height": G1_PACE_GEOMETRY.target_root_height,
      "forehand_paddle_offset": G1_PACE_GEOMETRY.forehand_paddle_offset,
      "forehand_paddle_offset_std": G1_PACE_GEOMETRY.forehand_paddle_offset_std,
      "forehand_elbow_target_ratio": G1_PACE_GEOMETRY.forehand_elbow_target_ratio,
      "strike_direction_pelvis": G1_PACE_GEOMETRY.strike_direction_pelvis,
      "strike_upward_angle": G1_PACE_GEOMETRY.strike_upward_angle,
      "bad_orientation_limit": G1_PACE_GEOMETRY.bad_orientation_limit,
      "root_height_minimum": G1_PACE_GEOMETRY.root_height_minimum,
    },
    "poses": rows,
  }
  (run_dir / "calibration.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
  )
  _write_csv(run_dir / "calibration.csv", rows)
  _write_readme(
    run_dir / "README.md",
    rows,
    render=render,
    reach_offset_pelvis=reach_offset_pelvis,
    strike_direction_pelvis=strike_direction_pelvis,
  )
  return run_dir


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("--output-dir", type=Path, default=Path("contact_test/results"))
  parser.add_argument("--pose-set", default="default")
  parser.add_argument("--custom-pose-file", type=Path, default=None)
  parser.add_argument(
    "--reach-offset-pelvis",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=G1_PACE_GEOMETRY.forehand_paddle_offset,
    help="Desired paddle-center reach offset in the pelvis frame.",
  )
  parser.add_argument(
    "--strike-direction-pelvis",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=G1_PACE_GEOMETRY.strike_direction_pelvis,
    help="Desired strike direction in the pelvis frame.",
  )
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument(
    "--skip-render",
    action="store_true",
    help="Write placeholder PNGs and metrics only; useful for CPU unit tests.",
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  run_dir = run_calibration(
    output_dir=args.output_dir,
    pose_set=args.pose_set,
    custom_pose_file=args.custom_pose_file,
    reach_offset_pelvis=tuple(args.reach_offset_pelvis),
    strike_direction_pelvis=tuple(args.strike_direction_pelvis),
    width=args.width,
    height=args.height,
    render=not args.skip_render,
  )
  print(run_dir)


if __name__ == "__main__":
  main()
