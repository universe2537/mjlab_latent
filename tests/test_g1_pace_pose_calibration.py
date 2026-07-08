import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
  Path(__file__).resolve().parents[1]
  / "contact_test"
  / "run_g1_pace_pose_calibration.py"
)
_SPEC = importlib.util.spec_from_file_location("g1_pace_pose_calibration", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_CALIBRATION_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CALIBRATION_MODULE
_SPEC.loader.exec_module(_CALIBRATION_MODULE)
run_calibration = _CALIBRATION_MODULE.run_calibration


def _assert_finite_numbers(value: Any) -> None:
  if isinstance(value, bool) or value is None:
    return
  if isinstance(value, int | float):
    assert math.isfinite(float(value))
    return
  if isinstance(value, dict):
    for item in value.values():
      _assert_finite_numbers(item)
    return
  if isinstance(value, list | tuple):
    for item in value:
      _assert_finite_numbers(item)


def test_g1_pace_pose_calibration_writes_metrics_and_images(
  tmp_path: Path,
) -> None:
  run_dir = run_calibration(
    output_dir=tmp_path,
    width=64,
    height=48,
    render=False,
  )

  json_path = run_dir / "calibration.json"
  csv_path = run_dir / "calibration.csv"
  readme_path = run_dir / "README.md"
  assert json_path.exists()
  assert csv_path.exists()
  assert readme_path.exists()

  payload = json.loads(json_path.read_text(encoding="utf-8"))
  assert payload["schema_version"] == 1
  assert payload["pose_set"] == "default"
  assert payload["rendered"] is False
  assert payload["image_size"] == {"width": 64, "height": 48}
  assert payload["pace_geometry_default"]["target_base_offset_xy"] == pytest.approx(
    [-0.3112, 0.4510]
  )
  assert payload["visualized_reach_offset_pelvis"] == pytest.approx(
    [0.3112, -0.4510, 0.0290]
  )
  assert payload["visualized_strike_direction_pelvis"] == pytest.approx(
    [math.cos(math.radians(15.0)), 0.0, math.sin(math.radians(15.0))]
  )

  poses = {pose["name"]: pose for pose in payload["poses"]}
  assert {
    "zero_pose",
    "home",
    "knees_bent",
    "forehand_open_light",
    "forehand_open_medium",
  } <= set(poses)

  for pose in poses.values():
    _assert_finite_numbers(pose)
    assert len(pose["paddle_offset_pelvis"]) == 3
    assert len(pose["target_base_offset_xy"]) == 2
    assert len(pose["desired_reach_offset_pelvis"]) == 3
    assert len(pose["desired_reach_position_world"]) == 3
    assert len(pose["desired_reach_error_world"]) == 3
    assert len(pose["desired_reach_error_pelvis"]) == 3
    assert len(pose["desired_strike_direction_pelvis"]) == 3
    assert len(pose["desired_strike_direction_world"]) == 3
    assert pose["desired_reach_error_norm"] >= 0.0
    assert pose["root_height"] > 0.0
    assert pose["torso_height"] > pose["root_height"]
    assert pose["foot_spacing_xy"] > 0.0
    assert set(pose["foot_geom_positions"]) == {
      f"{side}_foot{i}_collision"
      for side in ("left", "right")
      for i in range(1, 8)
    }
    for view in ("front", "side", "top", "iso", "hand_closeup"):
      assert Path(pose["images"][view]).exists()
      assert Path(pose["images"][f"{view}_overlay"]).exists()

  assert poses["zero_pose"]["paddle_offset_pelvis"] != pytest.approx(
    poses["knees_bent"]["paddle_offset_pelvis"]
  )
  readme = readme_path.read_text(encoding="utf-8")
  assert "KNEES_BENT" in readme
  assert "purple sphere" in readme
  assert "desired reach" in readme
  assert "orange arrow" in readme
