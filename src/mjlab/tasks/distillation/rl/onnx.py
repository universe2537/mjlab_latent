"""latent student 的 ONNX 导出辅助函数。

Deployment only sees the prior ``P(z|s)``, so the exported graph slices the
state out of the full actor observation, samples (deterministically) from
the prior, and decodes the action. 运行时不需要 target，也不需要 posterior。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
from torch import nn

from mjlab.tasks.distillation.rl.models import LatentStudentModel


class _OnnxStudentModel(nn.Module):
  """给 ``torch.onnx.export`` 使用的推理包装器。

  这个包装器的职责是把完整 actor observation 裁成 ``state``，
  然后走 prior-only 路径得到部署时动作。
  """

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
  """导出 prior-only student 为 ONNX。

  参数:
    model: 要导出的 student 模型。
    state_indices: 从 actor_obs 中取出 ``state`` 的索引。
    obs_dim: 完整 actor observation 维度，用于构造 dummy input。
    path: 导出目录。
    filename: 导出文件名。
    verbose: 是否打印导出过程中的详细信息。

  返回:
    导出的 ONNX 文件路径。
  """
  os.makedirs(path, exist_ok=True)
  # deepcopy 避免把训练中的 live model 直接挪到 CPU，影响后续训练状态。
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
