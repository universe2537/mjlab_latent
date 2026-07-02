"""Convert table-tennis pkl motions into tracking motion NPZ files."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from mjlab.scene import Scene
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_table_tennis_tracking_env_cfg,
)

TABLE_TENNIS_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class TablePklMotion:
  root_pos: np.ndarray
  root_rot_xyzw: np.ndarray
  dof_pos: np.ndarray
  fps: float


@dataclass(frozen=True)
class ConversionConfig:
  input_dir: Path = Path("table_data")
  output_dir: Path = Path("artifacts/table_tennis")
  output_fps: float = 50.0
  device: str = "cpu"
  overwrite: bool = False


def motion_name_from_path(path: Path) -> str:
  """Return the artifact motion name for a table-data pkl path."""
  name = path.name
  suffix = ".bvh_wxy.pkl"
  if name.endswith(suffix):
    return name[: -len(suffix)]
  return path.stem


def _as_float_array(data: Any, key: str, ndim: int, last_dim: int) -> np.ndarray:
  arr = np.asarray(data[key], dtype=np.float64)
  if arr.ndim != ndim or arr.shape[-1] != last_dim:
    raise ValueError(
      f"Expected {key!r} shape (*, {last_dim}) with ndim={ndim}, got {arr.shape}."
    )
  return arr


def load_table_pkl(path: Path) -> TablePklMotion:
  """Load and validate a table-tennis pkl motion file."""
  with path.open("rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Expected {path} to contain a dict, got {type(data)!r}.")

  required = ("root_pos", "root_rot", "dof_pos", "local_body_pos", "fps")
  missing = [key for key in required if key not in data]
  if missing:
    raise ValueError(f"{path} is missing required pkl keys: {missing}.")

  root_pos = _as_float_array(data, "root_pos", ndim=2, last_dim=3)
  root_rot = _as_float_array(data, "root_rot", ndim=2, last_dim=4)
  dof_pos = _as_float_array(data, "dof_pos", ndim=2, last_dim=29)
  if root_pos.shape[0] != root_rot.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
    raise ValueError(
      f"{path} has inconsistent frame counts: "
      f"root_pos={root_pos.shape}, root_rot={root_rot.shape}, dof_pos={dof_pos.shape}."
    )
  if root_pos.shape[0] < 2:
    raise ValueError(f"{path} must contain at least two frames.")

  fps = float(data["fps"])
  if not np.isfinite(fps) or fps <= 0:
    raise ValueError(f"{path} has invalid fps={data['fps']!r}.")

  quat_norm = np.linalg.norm(root_rot, axis=1)
  if np.any(quat_norm < 1.0e-8):
    raise ValueError(f"{path} contains a near-zero root quaternion.")
  root_rot = root_rot / quat_norm[:, None]
  return TablePklMotion(
    root_pos=root_pos,
    root_rot_xyzw=root_rot,
    dof_pos=dof_pos,
    fps=fps,
  )


def _resample_linear(
  values: np.ndarray, input_times: np.ndarray, output_times: np.ndarray
):
  out = np.empty((output_times.shape[0], values.shape[1]), dtype=np.float64)
  for dim in range(values.shape[1]):
    out[:, dim] = np.interp(output_times, input_times, values[:, dim])
  return out


def _resample_motion(motion: TablePklMotion, output_fps: float) -> TablePklMotion:
  if output_fps <= 0:
    raise ValueError(f"output_fps must be positive, got {output_fps}.")
  duration = (motion.root_pos.shape[0] - 1) / motion.fps
  input_times = np.arange(motion.root_pos.shape[0], dtype=np.float64) / motion.fps
  output_times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float64)
  if output_times.shape[0] == 0:
    output_times = np.array([0.0], dtype=np.float64)

  root_pos = _resample_linear(motion.root_pos, input_times, output_times)
  dof_pos = _resample_linear(motion.dof_pos, input_times, output_times)
  slerp = Slerp(input_times, Rotation.from_quat(motion.root_rot_xyzw))
  root_rot_xyzw = slerp(output_times).as_quat()
  return TablePklMotion(
    root_pos=root_pos,
    root_rot_xyzw=root_rot_xyzw,
    dof_pos=dof_pos,
    fps=output_fps,
  )


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
  return quat[:, [3, 0, 1, 2]]


def _quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
  w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
  return np.stack(
    (
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ),
    axis=-1,
  )


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
  out = quat.copy()
  out[..., 1:] *= -1.0
  return out


def _axis_angle_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
  quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
  quat = np.where(quat[..., :1] < 0.0, -quat, quat)
  vec = quat[..., 1:]
  vec_norm = np.linalg.norm(vec, axis=-1)
  angle = 2.0 * np.arctan2(vec_norm, np.clip(quat[..., 0], -1.0, 1.0))
  axis = np.zeros_like(vec)
  valid = vec_norm > 1.0e-8
  axis[valid] = vec[valid] / vec_norm[valid, None]
  return axis * angle[..., None]


def _quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
  return rot.inv().apply(vec)


def _angular_velocity_from_quat_wxyz(quat: np.ndarray, dt: float) -> np.ndarray:
  if quat.shape[0] == 1:
    return np.zeros((1, 3), dtype=np.float64)
  if quat.shape[0] == 2:
    rel = _quat_mul_wxyz(quat[1:2], _quat_conj_wxyz(quat[0:1]))
    omega = _axis_angle_from_quat_wxyz(rel) / dt
    return np.repeat(omega, 2, axis=0)

  rel = _quat_mul_wxyz(quat[2:], _quat_conj_wxyz(quat[:-2]))
  omega = _axis_angle_from_quat_wxyz(rel) / (2.0 * dt)
  return np.concatenate((omega[:1], omega, omega[-1:]), axis=0)


def _gradient(values: np.ndarray, dt: float) -> np.ndarray:
  if values.shape[0] == 1:
    return np.zeros_like(values)
  return np.gradient(values, dt, axis=0, edge_order=1)


def _require_mujoco_id(
  model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str
) -> int:
  obj_id = mujoco.mj_name2id(model, obj_type, name)
  if obj_id < 0:
    raise ValueError(f"MuJoCo model is missing {obj_type.name} named {name!r}.")
  return int(obj_id)


def _compute_body_link_velocities(
  data: mujoco.MjData,
  body_ids: np.ndarray,
  root_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
  pos = data.xpos[body_ids]
  cvel = data.cvel[body_ids]
  lin_vel_c = cvel[:, 3:6]
  ang_vel_w = cvel[:, 0:3]
  offset = data.subtree_com[root_body_id][None, :] - pos
  lin_vel_w = lin_vel_c - np.cross(ang_vel_w, offset)
  return lin_vel_w.copy(), ang_vel_w.copy()


class TablePklConverter:
  """Stateful converter that reuses one MuJoCo model across input files."""

  def __init__(self, output_fps: float, device: str = "cuda:0") -> None:
    self.output_fps = float(output_fps)
    if device != "cpu":
      print(
        f"[INFO]: table pkl conversion uses direct MuJoCo on CPU; ignoring {device!r}."
      )

    scene_cfg = unitree_g1_table_tennis_tracking_env_cfg().scene
    self.model = Scene(scene_cfg, device="cpu").compile()
    self.model.opt.timestep = 1.0 / self.output_fps
    self.data = mujoco.MjData(self.model)

    root_joint_id = _require_mujoco_id(
      self.model,
      mujoco.mjtObj.mjOBJ_JOINT,
      "robot/floating_base_joint",
    )
    self.root_qpos_adr = int(self.model.jnt_qposadr[root_joint_id])
    self.root_dof_adr = int(self.model.jnt_dofadr[root_joint_id])

    joint_ids = np.array(
      [
        _require_mujoco_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
        for name in TABLE_TENNIS_JOINT_NAMES
      ],
      dtype=np.int32,
    )
    self.joint_qpos_adrs = self.model.jnt_qposadr[joint_ids].astype(np.int64)
    self.joint_dof_adrs = self.model.jnt_dofadr[joint_ids].astype(np.int64)

    self.body_ids = np.array(
      [
        body_id
        for body_id in range(self.model.nbody)
        if (
          mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        ).startswith("robot/")
      ],
      dtype=np.int32,
    )
    if self.body_ids.size == 0:
      raise ValueError("MuJoCo model does not contain any robot bodies.")
    self.root_body_id = int(self.body_ids[0])

  def convert_file(self, input_file: Path, output_file: Path) -> None:
    motion = _resample_motion(load_table_pkl(input_file), self.output_fps)
    root_quat_wxyz = _xyzw_to_wxyz(motion.root_rot_xyzw)
    dt = 1.0 / self.output_fps

    root_lin_vel = _gradient(motion.root_pos, dt)
    root_ang_vel_w = _angular_velocity_from_quat_wxyz(root_quat_wxyz, dt)
    root_ang_vel_b = _quat_apply_inverse_wxyz(root_quat_wxyz, root_ang_vel_w)
    dof_vel = _gradient(motion.dof_pos, dt)

    log: dict[str, Any] = {
      "fps": [self.output_fps],
      "joint_pos": [],
      "joint_vel": [],
      "body_pos_w": [],
      "body_quat_w": [],
      "body_lin_vel_w": [],
      "body_ang_vel_w": [],
    }
    for frame_id in tqdm(
      range(motion.root_pos.shape[0]),
      desc=f"Converting {input_file.name}",
      unit="frame",
      leave=False,
    ):
      self.data.qpos[:] = self.model.qpos0
      self.data.qvel[:] = 0.0
      self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = motion.root_pos[
        frame_id
      ]
      self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = root_quat_wxyz[
        frame_id
      ]
      self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3] = root_lin_vel[frame_id]
      self.data.qvel[self.root_dof_adr + 3 : self.root_dof_adr + 6] = root_ang_vel_b[
        frame_id
      ]
      self.data.qpos[self.joint_qpos_adrs] = motion.dof_pos[frame_id]
      self.data.qvel[self.joint_dof_adrs] = dof_vel[frame_id]

      mujoco.mj_forward(self.model, self.data)
      body_lin_vel_w, body_ang_vel_w = _compute_body_link_velocities(
        self.data,
        self.body_ids,
        self.root_body_id,
      )

      log["joint_pos"].append(self.data.qpos[self.joint_qpos_adrs].copy())
      log["joint_vel"].append(self.data.qvel[self.joint_dof_adrs].copy())
      log["body_pos_w"].append(self.data.xpos[self.body_ids].copy())
      log["body_quat_w"].append(self.data.xquat[self.body_ids].copy())
      log["body_lin_vel_w"].append(body_lin_vel_w)
      log["body_ang_vel_w"].append(body_ang_vel_w)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    for key in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      log[key] = np.stack(log[key], axis=0)
    np.savez(output_file, **log)


def convert_directory(cfg: ConversionConfig) -> list[Path]:
  input_files = sorted(cfg.input_dir.glob("*.pkl"))
  if not input_files:
    raise FileNotFoundError(f"No .pkl files found under {cfg.input_dir}.")

  converter = TablePklConverter(output_fps=cfg.output_fps, device=cfg.device)
  output_files = []
  for input_file in input_files:
    output_file = cfg.output_dir / motion_name_from_path(input_file) / "motion.npz"
    if output_file.exists() and not cfg.overwrite:
      print(f"[SKIP] {output_file} already exists. Use --overwrite to replace it.")
      output_files.append(output_file)
      continue
    print(f"[CONVERT] {input_file} -> {output_file}")
    converter.convert_file(input_file, output_file)
    output_files.append(output_file)
  print(f"Done. Converted or found: {len(output_files)} motion file(s).")
  return output_files


def main() -> None:
  cfg = tyro.cli(ConversionConfig)
  convert_directory(cfg)


if __name__ == "__main__":
  main()
