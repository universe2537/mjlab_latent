"""PHC/PULSE-style latent VAE for online distillation.

Mirrors the architecture used by PULSE / PHC's ``amp_network_z_builder.py``:
https://github.com/ZhengyiLuo/PULSE/blob/master/phc/learning/amp_network_z_builder.py

Realises the conditional variational bottleneck described in LATENT §3.2.2:

  * Posterior  ``E(z | s, s̃) = N(mu_e(s, s̃), sigma_e(s, s̃))``
  * Prior      ``P(z | s)     = N(mu_p(s),    sigma_p(s))``    (state-conditioned, learnable)
  * Decoder    ``D(a | s, z)``                                  (or ``D(a | z)`` when ``z_all=True``)

Training loss (eq. 1-3 of LATENT):

  ``L = lambda_action * MSE(D(s, z_q), a_teacher) + lambda_kl * KL(E || P)``

The module is intentionally self-contained -- it has no dependency on the
runner / config layer and can be unit-tested or reused by downstream policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _activation_cls(name: str) -> type[nn.Module]:
  """Return an activation class by lowercase name."""
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
  """Build an MLP ``[Linear, Act] * n`` with optional final linear projection.

  When ``output_dim`` is ``None`` the trunk ends in an activated hidden layer
  (PHC's ``z_mlp`` style); otherwise a final ``Linear(hidden_dims[-1], output_dim)``
  is appended without activation by default.
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
  """Lightweight diagonal-Gaussian distribution parameterised by log-std.

  ``log_std`` is preferred over ``log_var`` for numerical symmetry with the
  KL formula and rsl_rl's distribution head; both representations are
  related by ``log_var = 2 * log_std``.
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
  """Per-sample ``KL(q || p)`` for diagonal Gaussians, summed over the latent dim."""
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
  """PHC-style Gaussian head.

  Architecture mirrors ``amp_network_z_builder._build_z_mlp`` for the posterior
  branch (and the prior branch when ``feature_multiplier == 1``):

    ``trunk: [Linear, Act] * n``
    optional ``feature_proj: Linear(hidden, latent_dim * feature_multiplier) + Act``
    ``mu_head:      Linear(feature, latent_dim)``
    ``log_std_head: Linear(feature, latent_dim)``  (clamped to [min, max])

  PHC sets ``feature_multiplier = 5`` for the posterior so the bottleneck
  (``z_mu`` / ``z_logvar``) is fed by an over-parameterised feature vector
  of size ``embedding_size * 5``. The prior uses ``feature_multiplier = 1``,
  i.e. mu / log_std heads sit directly on the trunk's last hidden layer.
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
  """Conditional VAE student used by LATENT online distillation.

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
    return self.posterior(torch.cat([state, target], dim=-1))

  def prior_distribution(self, state: torch.Tensor) -> DiagGaussian:
    return self.prior(state)

  # -- decoder ---------------------------------------------------------------

  def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
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

    Returns the reconstructed action together with both distributions so the
    runner can compute ``L_action`` and ``KL(posterior || prior)`` itself.
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
    """Inference rollout.

    ``source='prior'``    -- deployment-time path (no target available).
    ``source='posterior'`` -- evaluation when a target is still present
    (mirrors PHC's ``flags.test`` branch which uses ``vae_mu``).
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
