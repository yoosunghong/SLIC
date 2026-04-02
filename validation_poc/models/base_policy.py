import torch
import torch.nn as nn


def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
    """If all slots are masked, force-unmask slot 0 to prevent NaN in attention."""
    mask = mask.clone()
    all_masked = mask.all(dim=-1, keepdim=True)  # (B, 1)
    mask[:, 0] = mask[:, 0] & ~all_masked.squeeze(-1)
    return mask


class BasePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        raise NotImplementedError

    def act(self, obs: torch.Tensor):
        """Returns action logits (no sampling)."""
        logits, _ = self.forward(obs, return_attn=False)
        return logits


class CentralizedCritic(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int):
        super().__init__()
        in_dim = n_agents * obs_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs)
