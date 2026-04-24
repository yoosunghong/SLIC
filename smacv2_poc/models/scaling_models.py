"""
Scaling-study model variants.

Padded-observation models for variable team size.
Obs layout matches smac_wrapper_scaling.OBS_DIM = 182.

  self_tok   [0:8]
  ally_tok   [8:72]      8 slots × 8
  enemy_tok  [72:144]    9 slots × 8
  zone_tok   [144:162]   3 slots × 6
  ally_mask  [162:170]   8
  enemy_mask [170:179]   9
  zone_mask  [179:182]   3

One policy per role (3 networks total) is shared across all agents of that role
at build time. This matches the existing smacv2_poc 3v3 setup (independent
policies, one per role) while scaling to teams of 6 or 9.
"""
import torch
import torch.nn as nn

# Match base_policy without importing (keeps this file self-contained under
# smacv2_poc without touching validation_poc).
from validation_poc.models.base_policy import BasePolicy, _safe_mask


# Slot counts (padded ceiling)
N_ALLY_SLOTS  = 8
N_ENEMY_SLOTS = 9
N_ZONE_SLOTS  = 3

SELF_DIM, ALLY_DIM, ENEMY_DIM, ZONE_DIM = 8, 8, 8, 6

_SELF_END   = SELF_DIM
_ALLY_END   = _SELF_END  + N_ALLY_SLOTS  * ALLY_DIM
_ENEMY_END  = _ALLY_END  + N_ENEMY_SLOTS * ENEMY_DIM
_ZONE_END   = _ENEMY_END + N_ZONE_SLOTS  * ZONE_DIM
_AMASK_END  = _ZONE_END  + N_ALLY_SLOTS
_EMASK_END  = _AMASK_END + N_ENEMY_SLOTS
OBS_DIM     = _EMASK_END + N_ZONE_SLOTS     # 182


def parse_obs_scaling(obs: torch.Tensor):
    B = obs.shape[0]
    self_tok  = obs[:, :_SELF_END]
    ally_tok  = obs[:, _SELF_END:_ALLY_END ].view(B, N_ALLY_SLOTS,  ALLY_DIM)
    enemy_tok = obs[:, _ALLY_END:_ENEMY_END].view(B, N_ENEMY_SLOTS, ENEMY_DIM)
    zone_tok  = obs[:, _ENEMY_END:_ZONE_END].view(B, N_ZONE_SLOTS,  ZONE_DIM)
    ally_mask  = obs[:, _ZONE_END:_AMASK_END].bool()
    enemy_mask = obs[:, _AMASK_END:_EMASK_END].bool()
    zone_mask  = obs[:, _EMASK_END:OBS_DIM].bool()
    return self_tok, ally_tok, enemy_tok, zone_tok, ally_mask, enemy_mask, zone_mask


class ScalingFullModel(BasePolicy):
    """Condition 1: Intra-Set Self-Attn + Cross-Attn, padded to 182-dim obs."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = 6,
                 embed_dim: int = 64, num_heads: int = 2):
        super().__init__(obs_dim, action_dim)
        assert obs_dim == OBS_DIM, f"ScalingFullModel expects obs_dim={OBS_DIM}"
        self.embed_dim = embed_dim

        self.self_enc  = nn.Linear(SELF_DIM,  embed_dim)
        self.ally_enc  = nn.Linear(ALLY_DIM,  embed_dim)
        self.enemy_enc = nn.Linear(ENEMY_DIM, embed_dim)
        self.zone_enc  = nn.Linear(ZONE_DIM,  embed_dim)

        self.ally_self_attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enemy_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.zone_self_attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ally_ln, self.enemy_ln, self.zone_ln = (
            nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        )

        self.ally_cross_attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enemy_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.zone_cross_attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.fc1 = nn.Linear(embed_dim * 4, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self_tok, ally_tok, enemy_tok, zone_tok, ally_m, enemy_m, zone_m = parse_obs_scaling(obs)
        ally_m  = _safe_mask(ally_m)
        enemy_m = _safe_mask(enemy_m)
        zone_m  = _safe_mask(zone_m)

        self_emb  = self.relu(self.self_enc(self_tok))
        ally_emb  = self.relu(self.ally_enc(ally_tok))
        enemy_emb = self.relu(self.enemy_enc(enemy_tok))
        zone_emb  = self.relu(self.zone_enc(zone_tok))

        ally_sa, ally_sa_w = self.ally_self_attn(
            ally_emb, ally_emb, ally_emb,
            key_padding_mask=ally_m, need_weights=True, average_attn_weights=True
        )
        ally_emb = self.ally_ln(ally_emb + ally_sa)

        enemy_sa, enemy_sa_w = self.enemy_self_attn(
            enemy_emb, enemy_emb, enemy_emb,
            key_padding_mask=enemy_m, need_weights=True, average_attn_weights=True
        )
        enemy_emb = self.enemy_ln(enemy_emb + enemy_sa)

        zone_sa, zone_sa_w = self.zone_self_attn(
            zone_emb, zone_emb, zone_emb,
            key_padding_mask=zone_m, need_weights=True, average_attn_weights=True
        )
        zone_emb = self.zone_ln(zone_emb + zone_sa)

        q = self_emb.unsqueeze(1)
        ally_ctx,  ally_ca_w  = self.ally_cross_attn (q, ally_emb,  ally_emb,  key_padding_mask=ally_m,  need_weights=True, average_attn_weights=True)
        enemy_ctx, enemy_ca_w = self.enemy_cross_attn(q, enemy_emb, enemy_emb, key_padding_mask=enemy_m, need_weights=True, average_attn_weights=True)
        zone_ctx,  zone_ca_w  = self.zone_cross_attn (q, zone_emb,  zone_emb,  key_padding_mask=zone_m,  need_weights=True, average_attn_weights=True)

        ally_ctx, enemy_ctx, zone_ctx = ally_ctx.squeeze(1), enemy_ctx.squeeze(1), zone_ctx.squeeze(1)

        x = torch.cat([self_emb, ally_ctx, enemy_ctx, zone_ctx], dim=-1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        aw = None
        if return_attn:
            aw = {
                'ally_self_attn':  ally_sa_w.detach(),
                'enemy_self_attn': enemy_sa_w.detach(),
                'zone_self_attn':  zone_sa_w.detach(),
                'ally_cross_attn':  ally_ca_w.detach(),
                'enemy_cross_attn': enemy_ca_w.detach(),
                'zone_cross_attn':  zone_ca_w.detach(),
            }
        return logits, aw


class ScalingRoleMLPModel(BasePolicy):
    """Condition 3: Pure MLP with role label present in flat obs."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = 6, **kwargs):
        super().__init__(obs_dim, action_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),     nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs), None


def build_scaling_policies(condition: int, agent_roles: list,
                           action_dim: int = 6, embed_dim: int = 64, num_heads: int = 2):
    """
    Build one policy per role (3 networks), then map each agent index to its
    role's policy. Agents sharing a role share weights — matches the
    independent-policy-per-role setup.

    Returns: list of length len(agent_roles), with shared references.
    """
    kwargs = dict(obs_dim=OBS_DIM, action_dim=action_dim,
                  embed_dim=embed_dim, num_heads=num_heads)
    if condition == 1:
        role_policies = [ScalingFullModel(**kwargs) for _ in range(3)]
    elif condition == 3:
        role_policies = [ScalingRoleMLPModel(obs_dim=OBS_DIM, action_dim=action_dim) for _ in range(3)]
    else:
        raise ValueError(f"Scaling study only supports conditions 1 and 3, got {condition}")
    return [role_policies[r] for r in agent_roles]
