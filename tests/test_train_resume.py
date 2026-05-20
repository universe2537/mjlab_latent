from pathlib import Path

import pytest

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.scripts import train
from mjlab.scripts.train import TrainConfig
from mjlab.terrains import TerrainEntityCfg


def _train_config(agent: RslRlOnPolicyRunnerCfg) -> TrainConfig:
  return TrainConfig(
    env=ManagerBasedRlEnvCfg(
      decimation=5,
      scene=SceneCfg(terrain=TerrainEntityCfg()),
    ),
    agent=agent,
  )


def test_resolve_resume_path_uses_explicit_checkpoint_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  checkpoint = tmp_path / "model_29999.pt"
  checkpoint.write_bytes(b"checkpoint")
  agent = RslRlOnPolicyRunnerCfg(
    resume=True,
    load_run="must_not_be_used",
    load_checkpoint_file=str(checkpoint),
  )

  def fail_get_checkpoint_path(*args, **kwargs):
    raise AssertionError("load_run/load_checkpoint fallback should not be used")

  monkeypatch.setattr(train, "get_checkpoint_path", fail_get_checkpoint_path)

  cfg = _train_config(agent)
  assert train._resolve_resume_path(cfg, tmp_path) == checkpoint


def test_resolve_resume_path_returns_none_when_resume_is_disabled(
  tmp_path: Path,
) -> None:
  checkpoint = tmp_path / "model_29999.pt"
  agent = RslRlOnPolicyRunnerCfg(
    resume=False,
    load_checkpoint_file=str(checkpoint),
  )

  cfg = _train_config(agent)
  assert train._resolve_resume_path(cfg, tmp_path) is None
