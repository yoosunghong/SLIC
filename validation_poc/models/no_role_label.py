"""Condition 4 — No Role Label + Self-Attn + Cross-Attn (shared policy).
Role onehot dims are zeroed in self_tok[5:8] and in each ally_tok slot[4:7].
"""
import torch
from .base_policy import BasePolicy, _safe_mask
from .full_model import FullModel, parse_obs, N_ALLY_SLOTS

# Role onehot positions in self_tok
SELF_ROLE_DIMS = slice(5, 8)
# Within each 8-dim ally token, role is at dims 4:7
ALLY_ROLE_START = 4
ALLY_ROLE_END = 7


def zero_role_labels(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    # self_tok role dims: global indices 5,6,7
    obs[:, 5:8] = 0.0
    # ally_tok: 2 slots starting at index 8, each 8 dims; role at slot_offset+4:slot_offset+7
    for slot in range(N_ALLY_SLOTS):
        base = 8 + slot * 8
        obs[:, base + ALLY_ROLE_START:base + ALLY_ROLE_END] = 0.0
    return obs


class NoRoleLabelModel(BasePolicy):
    """Shared policy — same network used for all three agent roles."""

    def __init__(self, obs_dim: int = 65, action_dim: int = 6, embed_dim: int = 64, num_heads: int = 2):
        super().__init__(obs_dim, action_dim)
        self._inner = FullModel(obs_dim, action_dim, embed_dim, num_heads)

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = zero_role_labels(obs)
        return self._inner(obs, return_attn=return_attn)
