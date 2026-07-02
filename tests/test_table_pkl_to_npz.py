import pickle
from pathlib import Path

import numpy as np

from mjlab.scripts.table_pkl_to_npz import (
  TablePklConverter,
  load_table_pkl,
  motion_name_from_path,
)


def _write_tiny_table_pkl(path: Path, *, dof_dim: int = 29) -> None:
  frames = 5
  root_pos = np.zeros((frames, 3), dtype=np.float64)
  root_pos[:, 2] = 0.76
  root_pos[:, 0] = np.linspace(0.0, 0.04, frames)
  root_rot_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (frames, 1))
  dof_pos = np.zeros((frames, dof_dim), dtype=np.float64)
  if dof_dim == 29:
    dof_pos[:, 0] = np.linspace(0.0, 0.01, frames)
  data = {
    "root_pos": root_pos,
    "root_rot": root_rot_xyzw,
    "dof_pos": dof_pos,
    "local_body_pos": np.zeros((frames, 38, 3), dtype=np.float32),
    "fps": 4,
    "link_body_list": ["pelvis"],
  }
  with path.open("wb") as f:
    pickle.dump(data, f)


def test_motion_name_strips_bvh_wxy_suffix() -> None:
  assert (
    motion_name_from_path(Path("table_data/zhengshou_a_001.bvh_wxy.pkl"))
    == "zhengshou_a_001"
  )


def test_table_pkl_loader_validates_joint_dim(tmp_path: Path) -> None:
  pkl_path = tmp_path / "bad.bvh_wxy.pkl"
  _write_tiny_table_pkl(pkl_path, dof_dim=28)

  try:
    load_table_pkl(pkl_path)
  except ValueError as exc:
    assert "dof_pos" in str(exc)
  else:
    raise AssertionError("Expected loader to reject invalid dof_pos width.")


def test_table_pkl_converter_writes_tracking_npz(tmp_path: Path) -> None:
  pkl_path = tmp_path / "tiny.bvh_wxy.pkl"
  output_path = tmp_path / "tiny" / "motion.npz"
  _write_tiny_table_pkl(pkl_path)

  converter = TablePklConverter(output_fps=2.0, device="cpu")
  converter.convert_file(pkl_path, output_path)

  data = np.load(output_path)
  assert set(data.files) == {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  }
  assert data["fps"][0] == 2.0
  assert data["joint_pos"].shape == (2, 29)
  assert data["joint_vel"].shape == (2, 29)
  assert data["body_pos_w"].shape[0] == 2
  np.testing.assert_allclose(data["body_quat_w"][0, 0], [1.0, 0.0, 0.0, 0.0])
  for key in data.files:
    assert np.isfinite(data[key]).all()
