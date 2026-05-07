"""ONNX export helpers for the latent student policy.

Deployment only sees the prior ``P(z|s)``, so the exported graph slices the
state out of the full actor observation, samples (deterministically) from
the prior, and decodes the action. No target / encoder is needed at runtime.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from torch import nn

from mjlab.tasks.distillation.rl.models import LatentStudentModel


class _OnnxStudentModel(nn.Module):
  """Inference-only wrapper consumed by ``torch.onnx.export``."""

  def __init__(self, model: LatentStudentModel, state_indices: torch.Tensor) -> None:
    super().__init__()
    self.model = model
    self.register_buffer("state_indices", state_indices.cpu())

  def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
    state = actor_obs[:, self.state_indices]  # type: ignore[index]
    prior = self.model.prior_distribution(state)
    return self.model.decode(state, prior.mean)


def export_student_to_onnx(
  *,
  model: LatentStudentModel,
  state_indices: torch.Tensor,
  obs_dim: int,
  path: str,
  filename: str = "policy.onnx",
  verbose: bool = False,
) -> Path:
  """Serialise the prior-only student to ONNX. Returns the output file path."""
  os.makedirs(path, exist_ok=True)
  # deepcopy avoids moving the live training model to CPU.
  onnx_model = _OnnxStudentModel(copy.deepcopy(model), state_indices)
  onnx_model.to("cpu")
  onnx_model.eval()
  dummy_obs = torch.zeros(1, obs_dim)
  out_path = Path(path) / filename
  torch.onnx.export(
    onnx_model,
    (dummy_obs,),
    str(out_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=["actor_obs"],
    output_names=["actions"],
    dynamic_axes={},
    dynamo=False,
  )
  return out_path
