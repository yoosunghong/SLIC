"""Condition 2 — No Self-Attn: MLP encoder + Cross-Attention (no intra-set self-attention)."""
import torch
import torch.nn as nn
from .base_policy import BasePolicy, _safe_mask
from .full_model import parse_obs, SELF_DIM, ALLY_DIM, ENEMY_DIM, ZONE_DIM


class NoSelfAttnModel(BasePolicy):
    def __init__(self, obs_dim: int = 65, action_dim: int = 6, embed_dim: int = 64, num_heads: int = 2):
        super().__init__(obs_dim, action_dim)

        self.self_enc = nn.Linear(SELF_DIM, embed_dim)
        self.ally_enc = nn.Linear(ALLY_DIM, embed_dim)
        self.enemy_enc = nn.Linear(ENEMY_DIM, embed_dim)
        self.zone_enc = nn.Linear(ZONE_DIM, embed_dim)

        self.ally_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.enemy_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.zone_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.fc1 = nn.Linear(embed_dim * 4, 128)
        self.fc2 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor, return_attn: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        self_tok, ally_tok, enemy_tok, zone_tok, ally_mask, enemy_mask, zone_mask = parse_obs(obs)
        ally_mask = _safe_mask(ally_mask)
        enemy_mask = _safe_mask(enemy_mask)
        zone_mask = _safe_mask(zone_mask)

        self_emb = self.relu(self.self_enc(self_tok))
        ally_emb = self.relu(self.ally_enc(ally_tok))
        enemy_emb = self.relu(self.enemy_enc(enemy_tok))
        zone_emb = self.relu(self.zone_enc(zone_tok))

        q = self_emb.unsqueeze(1)
        ally_ctx, ally_ca_w = self.ally_cross_attn(q, ally_emb, ally_emb, key_padding_mask=ally_mask, need_weights=True, average_attn_weights=True)
        enemy_ctx, enemy_ca_w = self.enemy_cross_attn(q, enemy_emb, enemy_emb, key_padding_mask=enemy_mask, need_weights=True, average_attn_weights=True)
        zone_ctx, zone_ca_w = self.zone_cross_attn(q, zone_emb, zone_emb, key_padding_mask=zone_mask, need_weights=True, average_attn_weights=True)

        ally_ctx = ally_ctx.squeeze(1)
        enemy_ctx = enemy_ctx.squeeze(1)
        zone_ctx = zone_ctx.squeeze(1)

        x = torch.cat([self_emb, ally_ctx, enemy_ctx, zone_ctx], dim=-1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        attn_weights = None
        if return_attn:
            attn_weights = {
                'ally_cross_attn': ally_ca_w.detach(),
                'enemy_cross_attn': enemy_ca_w.detach(),
                'zone_cross_attn': zone_ca_w.detach(),
            }

        return logits, attn_weights
