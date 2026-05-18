"""PHC/PULSE 风格的 latent VAE 蒸馏模型。

Mirrors the architecture used by PULSE / PHC's ``amp_network_z_builder.py``:
https://github.com/ZhengyiLuo/PULSE/blob/master/phc/learning/amp_network_z_builder.py

Realises the conditional variational bottleneck described in LATENT §3.2.2:

  * Posterior  ``E(z | s, s̃) = N(mu_e(s, s̃), sigma_e(s, s̃))``
  * Prior      ``P(z | s)     = N(mu_p(s),    sigma_p(s))``    (state-conditioned, learnable)
  * Decoder    ``D(a | s, z)``                                  (or ``D(a | z)`` when ``z_all=True``)

Training loss (eq. 1-3 of LATENT):

  ``L = lambda_action * MSE(D(s, z_q), a_teacher) + lambda_kl * KL(E || P)``

该模块刻意保持自包含：

1. 不依赖 runner 训练循环。
2. 不依赖任务配置系统。
3. 可以单独做单元测试或 smoke test。

因此这里的职责仅限于“定义分布和网络”，不处理数据采样、teacher 加载、
日志记录或 checkpoint 逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

#
# Helpers
#


def _activation_cls(name: str) -> type[nn.Module]:
  """按字符串名称解析激活函数。

  参数:
    name: 配置里写入的激活函数名称，例如 ``elu`` / ``relu``。

  返回:
    对应的 ``torch.nn`` 模块类型，而不是实例。
  """
  match name.lower():
    case "elu":
      return nn.ELU
    case "relu":
      return nn.ReLU
    case "gelu":
      return nn.GELU
    case "tanh":
      return nn.Tanh
    case _:
      raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(
  input_dim: int,
  hidden_dims: tuple[int, ...],
  activation: str,
  *,
  output_dim: int | None = None,
  output_activation: bool = False,
) -> nn.Sequential:
  """构造一个多层感知机。

  When ``output_dim`` is ``None`` the trunk ends in an activated hidden layer
  (PHC's ``z_mlp`` style); otherwise a final ``Linear(hidden_dims[-1], output_dim)``
  is appended without activation by default.

  参数:
    input_dim: 输入向量维度。
    hidden_dims: 每层隐层宽度。
    activation: 激活函数名称。
    output_dim: 若不为 ``None``，则在最后追加一个输出线性层。
    output_activation: 是否在输出线性层后继续接激活函数。

  返回:
    按当前配置拼好的 ``nn.Sequential``。
  """
  layers: list[nn.Module] = []
  act_cls = _activation_cls(activation)
  last_dim = input_dim
  for hidden_dim in hidden_dims:
    layers.append(nn.Linear(last_dim, hidden_dim))
    layers.append(act_cls())
    last_dim = hidden_dim
  if output_dim is not None:
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation:
      layers.append(act_cls())
  return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Diagonal Gaussian
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagGaussian:
  """轻量的对角高斯分布封装，参数化方式为 ``(mean, log_std)``。

  ``log_std`` is preferred over ``log_var`` for numerical symmetry with the
  KL formula and rsl_rl's distribution head; both representations are
  related by ``log_var = 2 * log_std``.

  这里不直接继承 ``torch.distributions.Normal``，是为了让接口更轻、
  更直观，并避免在训练/导出时引入多余依赖。
  """

  mean: torch.Tensor
  log_std: torch.Tensor

  @property
  def std(self) -> torch.Tensor:
    return torch.exp(self.log_std)

  @property
  def log_var(self) -> torch.Tensor:
    return 2.0 * self.log_std

  def rsample(self) -> torch.Tensor:
    """Reparameterised sample: ``mu + sigma * eps`` with grads through both."""
    return self.mean + self.std * torch.randn_like(self.mean)

  def sample(self) -> torch.Tensor:
    """Detached sample for inference / rollout."""
    with torch.no_grad():
      return self.mean + self.std * torch.randn_like(self.mean)


def diagonal_gaussian_kl(q: DiagGaussian, p: DiagGaussian) -> torch.Tensor:
  """计算逐样本的 ``KL(q || p)``。

  参数:
    q: 通常表示 posterior ``q(z|s,s_tilde)``。
    p: 通常表示 prior ``p(z|s)``。

  返回:
    shape 为 ``[batch]`` 的张量；latent 维度内部已经求和。
  """
  q_var = torch.exp(2.0 * q.log_std)
  p_var = torch.exp(2.0 * p.log_std)
  kl = (
    p.log_std - q.log_std + (q_var + (q.mean - p.mean).square()) / (2.0 * p_var) - 0.5
  )
  return kl.sum(dim=-1)


# ---------------------------------------------------------------------------
# Gaussian heads (PHC style: trunk + separate mu / log_std linear heads)
# ---------------------------------------------------------------------------


class _GaussianHead(nn.Module):
  """PHC 风格的高斯分布头。

  Architecture mirrors ``amp_network_z_buprior_hidden_dimsilder._build_z_mlp`` for the posterior
  branch (and the prior branch when ``feature_multiplier == 1``):

    ``trunk: [Linear, Act] * n``
    optional ``feature_proj: Linear(hidden, latent_dim * feature_multiplier) + Act``
    ``mu_head:      Linear(feature, latent_dim)``
    ``log_std_head: Linear(feature, latent_dim)``  (clamped to [min, max])

  PHC sets ``feature_multiplier = 5`` for the posterior so the bottleneck
  (``z_mu`` / ``z_logvar``) is fed by an over-parameterised feature vector
  of size ``embedding_size * 5``. The prior uses ``feature_multiplier = 1``,
  i.e. mu / log_std heads sit directly on the trunk's last hidden layer.

  直观理解：

  1. ``trunk`` 负责提取语义特征。
  2. ``feature_proj`` 决定在输出分布头之前是否额外扩展特征宽度。
  3. ``mu_head`` / ``log_std_head`` 最终给出高斯分布参数。
  """

  def __init__(
    self,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    *,
    min_log_std: float,
    max_log_std: float,
    feature_multiplier: int = 1,
  ) -> None:
    super().__init__()
    if feature_multiplier < 1:
      raise ValueError("feature_multiplier must be >= 1")
    self.trunk = _build_mlp(input_dim, hidden_dims, activation)
    trunk_out = hidden_dims[-1]
    if feature_multiplier > 1:
      feature_dim = latent_dim * feature_multiplier
      self.feature_proj: nn.Module = nn.Sequential(
        nn.Linear(trunk_out, feature_dim),
        _activation_cls(activation)(),
      )
    else:
      feature_dim = trunk_out
      self.feature_proj = nn.Identity()
    self.mu_head = nn.Linear(feature_dim, latent_dim)
    self.log_std_head = nn.Linear(feature_dim, latent_dim)
    self.min_log_std = float(min_log_std)
    self.max_log_std = float(max_log_std)

  def forward(self, x: torch.Tensor) -> DiagGaussian:
    h = self.feature_proj(self.trunk(x))
    mean = self.mu_head(h)
    log_std = torch.clamp(self.log_std_head(h), self.min_log_std, self.max_log_std)
    return DiagGaussian(mean=mean, log_std=log_std)


# ---------------------------------------------------------------------------
# LATENT student VAE (PHC / PULSE topology)
# ---------------------------------------------------------------------------


class LatentStudentModel(nn.Module):
  """LATENT 在线蒸馏使用的 conditional VAE student。

  Faithfully follows PULSE's encoder / state-conditioned learnable prior /
  decoder topology, with an optional ``z_all`` switch matching PHC's
  ``z_all`` flag (decoder consumes only ``z`` when true, else ``[s, z]``).

  The module exposes three orthogonal entry points so callers can compose
  whichever sampling strategy they need without touching internals:

    * :meth:`forward_train` -- training step (posterior reparam + decode).
    * :meth:`act`           -- inference rollout (prior or posterior).
    * :meth:`prior_distribution` / :meth:`posterior_distribution` /
      :meth:`decode` -- low-level building blocks for custom pipelines
      (e.g. ONNX export wrappers).

  参数说明:
    state_dim: ``state`` 向量维度，供 prior 和 decoder 使用。
    target_dim: ``target`` 向量维度，仅训练时 posterior 使用。
    action_dim: 动作维度，通常等于环境 action space 维度。
    latent_dim: latent 变量 ``z`` 的维度。
    encoder_hidden_dims: posterior 编码器隐层宽度。
    prior_hidden_dims: prior 网络隐层宽度。
    decoder_hidden_dims: 动作解码器隐层宽度。
    activation: 网络激活函数。
    min_log_std: 对高斯 ``log_std`` 的下界裁剪。
    max_log_std: 对高斯 ``log_std`` 的上界裁剪。
    posterior_feature_multiplier: posterior 输出头前的宽度放大倍数。
    prior_feature_multiplier: prior 输出头前的宽度放大倍数。
    z_all: 若为 True，decoder 只接收 ``z``；否则接收 ``[state, z]``。
  """

  def __init__(
    self,
    *,
    state_dim: int,
    target_dim: int,
    action_dim: int,
    latent_dim: int,
    encoder_hidden_dims: tuple[int, ...],
    prior_hidden_dims: tuple[int, ...],
    decoder_hidden_dims: tuple[int, ...],
    activation: str = "elu",
    min_log_std: float = -5.0,
    max_log_std: float = 2.0,
    posterior_feature_multiplier: int = 5,
    prior_feature_multiplier: int = 1,
    z_all: bool = False,
  ) -> None:
    super().__init__()
    self.state_dim = int(state_dim)
    self.target_dim = int(target_dim)
    self.action_dim = int(action_dim)
    self.latent_dim = int(latent_dim)
    self.z_all = bool(z_all)

    # Posterior  E(z | s, s̃)  -- PHC's z_mlp + z_mu + z_logvar
    self.posterior = _GaussianHead(
      input_dim=self.state_dim + self.target_dim,
      latent_dim=self.latent_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
      min_log_std=min_log_std,
      max_log_std=max_log_std,
      feature_multiplier=posterior_feature_multiplier,
    )

    # Prior  P(z | s)  -- PHC's z_prior + z_prior_mu + z_prior_logvar
    self.prior = _GaussianHead(
      input_dim=self.state_dim,
      latent_dim=self.latent_dim,
      hidden_dims=prior_hidden_dims,
      activation=activation,
      min_log_std=min_log_std,
      max_log_std=max_log_std,
      feature_multiplier=prior_feature_multiplier,
    )

    # Decoder  D(a | s, z)  (or D(a | z) when z_all=True)
    decoder_input_dim = (
      self.latent_dim if self.z_all else self.state_dim + self.latent_dim
    )
    self.decoder = _build_mlp(
      decoder_input_dim,
      decoder_hidden_dims,
      activation,
      output_dim=self.action_dim,
    )

  # -- distribution heads ----------------------------------------------------

  def posterior_distribution(
    self, state: torch.Tensor, target: torch.Tensor
  ) -> DiagGaussian:
    """根据 ``(state, target)`` 计算 posterior 分布。"""
    return self.posterior(torch.cat([state, target], dim=-1))

  def prior_distribution(self, state: torch.Tensor) -> DiagGaussian:
    """仅根据 ``state`` 计算可部署的先验分布。"""
    return self.prior(state)

  # -- decoder ---------------------------------------------------------------

  def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """将 latent 解码为动作。

    当 ``z_all=False`` 时，动作由 ``[state, z]`` 决定；
    当 ``z_all=True`` 时，动作只由 ``z`` 决定。
    """
    if self.z_all:
      return self.decoder(latent)
    return self.decoder(torch.cat([state, latent], dim=-1))

  @staticmethod
  def reparameterize(dist: DiagGaussian) -> torch.Tensor:
    """Explicit reparameterisation trick (mirrors PHC's ``reparameterize``)."""
    return dist.rsample()

  # -- training & inference --------------------------------------------------

  def forward_train(
    self, state: torch.Tensor, target: torch.Tensor
  ) -> tuple[torch.Tensor, DiagGaussian, DiagGaussian]:
    """Training step: encode, reparameterise, decode.

    返回值:
      reconstructed_action, posterior, prior

    runner 会基于这三个量自行计算：

    1. 动作蒸馏损失 ``MSE(action, teacher_action)``
    2. 变分正则 ``KL(posterior || prior)``
    """
    posterior = self.posterior_distribution(state, target)
    prior = self.prior_distribution(state)
    latent = self.reparameterize(posterior)
    action = self.decode(state, latent)
    return action, posterior, prior

  @torch.no_grad()
  def act(
    self,
    state: torch.Tensor,
    target: torch.Tensor | None = None,
    *,
    deterministic: bool = True,
    source: str = "prior",
  ) -> torch.Tensor:
    """推理时生成动作。

    ``source='prior'``    -- deployment-time path (no target available).
    ``source='posterior'`` -- evaluation when a target is still present
    (mirrors PHC's ``flags.test`` branch which uses ``vae_mu``).

    参数:
      state: 当前状态输入。
      target: 若使用 posterior 推理，则需要提供对应目标输入。
      deterministic: True 时使用分布均值；False 时执行随机采样。
      source: ``prior`` 或 ``posterior``，决定采用哪条分布支路。
    """
    if source == "prior":
      dist = self.prior_distribution(state)
    elif source == "posterior":
      if target is None:
        raise ValueError("source='posterior' requires a target tensor")
      dist = self.posterior_distribution(state, target)
    else:
      raise ValueError(f"Unknown sampling source: {source!r}")
    latent = dist.mean if deterministic else dist.sample()
    return self.decode(state, latent)
