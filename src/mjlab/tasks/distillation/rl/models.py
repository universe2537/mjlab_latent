"""Latent action models for online distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _activation(name: str) -> type[nn.Module]:
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


def _mlp(
  input_dim: int,
  output_dim: int,
  hidden_dims: tuple[int, ...],
  activation: str,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  act_cls = _activation(activation)
  last_dim = input_dim
  for hidden_dim in hidden_dims:
    layers.append(nn.Linear(last_dim, hidden_dim))
    layers.append(act_cls())
    last_dim = hidden_dim
  layers.append(nn.Linear(last_dim, output_dim))
  return nn.Sequential(*layers)


@dataclass(frozen=True)
class DiagGaussian:
  mean: torch.Tensor
  log_std: torch.Tensor

  @property
  def std(self) -> torch.Tensor:
    return torch.exp(self.log_std)

  def sample(self) -> torch.Tensor:
    return self.mean + self.std * torch.randn_like(self.mean)


class GaussianHead(nn.Module):
  def __init__(
    self,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    activation: str,
    min_log_std: float,
    max_log_std: float,
  ) -> None:
    super().__init__()
    self.net = _mlp(input_dim, 2 * latent_dim, hidden_dims, activation)
    self.min_log_std = min_log_std
    self.max_log_std = max_log_std

  def forward(self, x: torch.Tensor) -> DiagGaussian:
    mean, log_std = torch.chunk(self.net(x), 2, dim=-1)
    log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
    return DiagGaussian(mean=mean, log_std=log_std)


class LatentStudentModel(nn.Module):
  """Conditional VAE-style student used by LATENT online distillation."""

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
  ) -> None:
    super().__init__()
    self.state_dim = state_dim
    self.target_dim = target_dim
    self.action_dim = action_dim
    self.latent_dim = latent_dim
    self.encoder = GaussianHead(
      state_dim + target_dim,
      latent_dim,
      encoder_hidden_dims,
      activation,
      min_log_std,
      max_log_std,
    )
    self.prior = GaussianHead(
      state_dim,
      latent_dim,
      prior_hidden_dims,
      activation,
      min_log_std,
      max_log_std,
    )
    self.decoder = _mlp(
      state_dim + latent_dim,
      action_dim,
      decoder_hidden_dims,
      activation,
    )

  def posterior(self, state: torch.Tensor, target: torch.Tensor) -> DiagGaussian:
    return self.encoder(torch.cat([state, target], dim=-1))

  def prior_distribution(self, state: torch.Tensor) -> DiagGaussian:
    return self.prior(state)

  def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat([state, latent], dim=-1))

  def act(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    prior = self.prior_distribution(state)
    latent = prior.mean if deterministic else prior.sample()
    return self.decode(state, latent)

  def forward_train(
    self, state: torch.Tensor, target: torch.Tensor
  ) -> tuple[torch.Tensor, DiagGaussian, DiagGaussian]:
    posterior = self.posterior(state, target)
    prior = self.prior_distribution(state)
    action = self.decode(state, posterior.sample())
    return action, posterior, prior


def diagonal_gaussian_kl(q: DiagGaussian, p: DiagGaussian) -> torch.Tensor:
  """Return per-sample KL(q || p) for diagonal Gaussian distributions."""
  q_var = torch.exp(2.0 * q.log_std)
  p_var = torch.exp(2.0 * p.log_std)
  kl = (
    p.log_std - q.log_std + (q_var + (q.mean - p.mean).square()) / (2.0 * p_var) - 0.5
  )
  return kl.sum(dim=-1)
