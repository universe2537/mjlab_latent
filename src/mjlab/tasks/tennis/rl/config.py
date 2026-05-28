"""Runner config for tennis latent-control tasks."""

from dataclasses import dataclass

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class TennisLatentOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  """PPO runner config with frozen-decoder validation."""

  require_decoder_checkpoint: bool = True
  """Raise during runner construction when no decoder checkpoint is configured."""

  reset_resume_progress: bool = False
  """Load policy weights while resetting iteration and environment progress."""
