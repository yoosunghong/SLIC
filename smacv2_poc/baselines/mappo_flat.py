"""
Condition 6 — MAPPO Vanilla with flat raw SMACv2 observation.

Architecture: 3-layer MLP over the raw SMACv2 obs vector (no entity-centric
reformatting, no role labels, no attention).  This establishes the benefit of
the entity-centric observation format independent of the attention architecture.

Interface mirrors BasePolicy: forward(obs) → (logits, None)
"""
import os
import sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE = os.path.dirname(_ROOT)
if _WORKSPACE not in sys.path:
    sys.path.insert(0, _WORKSPACE)

from validation_poc.models.base_policy import BasePolicy


class FlatObsMLPModel(BasePolicy):
    """
    3-layer MLP baseline over raw SMACv2 flat observation.

    Parameters
    ----------
    obs_dim : int    Raw SMACv2 obs dimension (auto-detected at runtime).
    action_dim : int SMACv2 discrete action count (default 6).
    hidden_dim : int Hidden layer width (default 256).
    """
    def __init__(self, obs_dim: int, action_dim: int = 6, hidden_dim: int = 256,
                 embed_dim: int = 64, num_heads: int = 2):
        # Pass obs_dim/action_dim to BasePolicy; embed_dim/num_heads are unused but
        # accepted to match the build_policies(**kwargs) call signature.
        super().__init__(obs_dim, action_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits = self.net(obs)
        return logits, None


def build_flat_policies(n_agents: int, obs_dim: int, action_dim: int = 6,
                        hidden_dim: int = 256) -> list:
    """Create independent FlatObsMLP policies, one per agent."""
    return [FlatObsMLPModel(obs_dim=obs_dim, action_dim=action_dim,
                            hidden_dim=hidden_dim) for _ in range(n_agents)]
