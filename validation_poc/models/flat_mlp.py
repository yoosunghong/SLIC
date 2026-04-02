"""Condition 5 — Flat MLP baseline (no role labels, no attention)."""
import torch
import torch.nn as nn
from .base_policy import BasePolicy
from .no_role_label import zero_role_labels


class FlatMLPModel(BasePolicy):
    def __init__(self, obs_dim: int = 65, action_dim: int = 6, **kwargs):
        super().__init__(obs_dim, action_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = zero_role_labels(obs)
        logits = self.net(obs)
        return logits, None
