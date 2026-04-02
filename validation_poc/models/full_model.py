"""Condition 1 — Full model: Intra-Set Self-Attention + Cross-Attention."""
import torch
import torch.nn as nn
from .base_policy import BasePolicy, _safe_mask

# Observation layout constants
SELF_TOK_SLICE = slice(0, 8)
ALLY_TOK_SLICE = slice(8, 24)
ENEMY_TOK_SLICE = slice(24, 40)
ZONE_TOK_SLICE = slice(40, 58)
ALLY_MASK_SLICE = slice(58, 60)
ENEMY_MASK_SLICE = slice(60, 62)
ZONE_MASK_SLICE = slice(62, 65)

N_ALLY_SLOTS = 2
N_ENEMY_SLOTS = 2
N_ZONE_SLOTS = 3
SELF_DIM = 8
ALLY_DIM = 8
ENEMY_DIM = 8
ZONE_DIM = 6


def parse_obs(obs: torch.Tensor):
    """Parse flat obs into entity tensors."""
    B = obs.shape[0]
    self_tok = obs[:, SELF_TOK_SLICE]                              # (B, 8)
    ally_tok = obs[:, ALLY_TOK_SLICE].view(B, N_ALLY_SLOTS, ALLY_DIM)    # (B, 2, 8)
    enemy_tok = obs[:, ENEMY_TOK_SLICE].view(B, N_ENEMY_SLOTS, ENEMY_DIM)  # (B, 2, 8)
    zone_tok = obs[:, ZONE_TOK_SLICE].view(B, N_ZONE_SLOTS, ZONE_DIM)     # (B, 3, 6)
    ally_mask = obs[:, ALLY_MASK_SLICE].bool()                    # (B, 2)
    enemy_mask = obs[:, ENEMY_MASK_SLICE].bool()                  # (B, 2)
    zone_mask = obs[:, ZONE_MASK_SLICE].bool()                    # (B, 3)
    return self_tok, ally_tok, enemy_tok, zone_tok, ally_mask, enemy_mask, zone_mask


class FullModel(BasePolicy):
    def __init__(self, obs_dim: int = 65, action_dim: int = 6, embed_dim: int = 64, num_heads: int = 2):
        super().__init__(obs_dim, action_dim)
        self.embed_dim = embed_dim

        # Encoders
        self.self_enc = nn.Linear(SELF_DIM, embed_dim)
        self.ally_enc = nn.Linear(ALLY_DIM, embed_dim)
        self.enemy_enc = nn.Linear(ENEMY_DIM, embed_dim)
        self.zone_enc = nn.Linear(ZONE_DIM, embed_dim)

        # Intra-set self-attention
        self.ally_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enemy_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.zone_self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ally_ln = nn.LayerNorm(embed_dim)
        self.enemy_ln = nn.LayerNorm(embed_dim)
        self.zone_ln = nn.LayerNorm(embed_dim)

        # Cross-attention (self as query)
        self.ally_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enemy_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.zone_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Output MLP
        self.fc1 = nn.Linear(embed_dim * 4, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]

        self_tok, ally_tok, enemy_tok, zone_tok, ally_mask, enemy_mask, zone_mask = parse_obs(obs)

        ally_mask = _safe_mask(ally_mask)
        enemy_mask = _safe_mask(enemy_mask)
        zone_mask = _safe_mask(zone_mask)

        # Encode
        self_emb = self.relu(self.self_enc(self_tok))          # (B, 64)
        ally_emb = self.relu(self.ally_enc(ally_tok))          # (B, 2, 64)
        enemy_emb = self.relu(self.enemy_enc(enemy_tok))       # (B, 2, 64)
        zone_emb = self.relu(self.zone_enc(zone_tok))          # (B, 3, 64)

        # Intra-set self-attention
        ally_sa, ally_sa_w = self.ally_self_attn(
            ally_emb, ally_emb, ally_emb,
            key_padding_mask=ally_mask, need_weights=True, average_attn_weights=True
        )
        ally_emb = self.ally_ln(ally_emb + ally_sa)

        enemy_sa, enemy_sa_w = self.enemy_self_attn(
            enemy_emb, enemy_emb, enemy_emb,
            key_padding_mask=enemy_mask, need_weights=True, average_attn_weights=True
        )
        enemy_emb = self.enemy_ln(enemy_emb + enemy_sa)

        zone_sa, zone_sa_w = self.zone_self_attn(
            zone_emb, zone_emb, zone_emb,
            key_padding_mask=zone_mask, need_weights=True, average_attn_weights=True
        )
        zone_emb = self.zone_ln(zone_emb + zone_sa)

        # Cross-attention
        q = self_emb.unsqueeze(1)  # (B, 1, 64)

        ally_ctx, ally_ca_w = self.ally_cross_attn(
            q, ally_emb, ally_emb,
            key_padding_mask=ally_mask, need_weights=True, average_attn_weights=True
        )
        enemy_ctx, enemy_ca_w = self.enemy_cross_attn(
            q, enemy_emb, enemy_emb,
            key_padding_mask=enemy_mask, need_weights=True, average_attn_weights=True
        )
        zone_ctx, zone_ca_w = self.zone_cross_attn(
            q, zone_emb, zone_emb,
            key_padding_mask=zone_mask, need_weights=True, average_attn_weights=True
        )

        ally_ctx = ally_ctx.squeeze(1)   # (B, 64)
        enemy_ctx = enemy_ctx.squeeze(1)
        zone_ctx = zone_ctx.squeeze(1)

        # MLP output
        x = torch.cat([self_emb, ally_ctx, enemy_ctx, zone_ctx], dim=-1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        attn_weights = None
        if return_attn:
            attn_weights = {
                'ally_self_attn': ally_sa_w.detach(),    # (B, 2, 2)
                'enemy_self_attn': enemy_sa_w.detach(),  # (B, 2, 2)
                'zone_self_attn': zone_sa_w.detach(),    # (B, 3, 3)
                'ally_cross_attn': ally_ca_w.detach(),   # (B, 1, 2)
                'enemy_cross_attn': enemy_ca_w.detach(), # (B, 1, 2)
                'zone_cross_attn': zone_ca_w.detach(),   # (B, 1, 3)
            }

        return logits, attn_weights
