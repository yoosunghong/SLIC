from .base_policy import BasePolicy, CentralizedCritic
from .full_model import FullModel
from .no_self_attn import NoSelfAttnModel
from .role_mlp import RoleMLPModel
from .no_role_label import NoRoleLabelModel
from .flat_mlp import FlatMLPModel


def build_policies(condition: int, obs_dim: int = 65, action_dim: int = 6,
                   embed_dim: int = 64, num_heads: int = 2):
    kwargs = dict(obs_dim=obs_dim, action_dim=action_dim, embed_dim=embed_dim, num_heads=num_heads)
    if condition == 1:
        return [FullModel(**kwargs) for _ in range(3)]
    elif condition == 2:
        return [NoSelfAttnModel(**kwargs) for _ in range(3)]
    elif condition == 3:
        return [RoleMLPModel(**kwargs) for _ in range(3)]
    elif condition == 4:
        shared = NoRoleLabelModel(**kwargs)
        return [shared, shared, shared]
    elif condition == 5:
        return [FlatMLPModel(**kwargs) for _ in range(3)]
    else:
        raise ValueError(f"Unknown condition: {condition}")
