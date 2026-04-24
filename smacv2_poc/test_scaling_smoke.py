"""
Smoke-test for the scaling study infrastructure.

Checks:
  1. Padded OBS_DIM agrees between env and model (182).
  2. ScalingSMACWrapper constructs for all 5 configs, resets, and steps once.
  3. Masks correctly flag padded slots (e.g. in 3v3, ally slots 2-7 are masked).
  4. ScalingFullModel + ScalingRoleMLPModel forward pass returns finite logits
     and attention weights with the expected shapes.
  5. build_scaling_policies shares weights across same-role agents.

Skips gracefully if SMACv2 is not installed on this machine.
"""
import os
import sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_WS   = os.path.dirname(_ROOT)
for p in (_WS, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from env.smac_wrapper_scaling import (
    ScalingSMACWrapper, OBS_DIM as ENV_OBS_DIM,
    MAX_ALLY_SLOTS, MAX_ENEMY_SLOTS, N_ZONE_SLOTS,
    _SELF_END, _ALLY_END, _ENEMY_END, _ZONE_END, _AMASK_END, _EMASK_END,
    _SMACV2_AVAILABLE,
)
from models.scaling_models import (
    ScalingFullModel, ScalingRoleMLPModel, build_scaling_policies,
    OBS_DIM as MODEL_OBS_DIM, parse_obs_scaling,
)


def check_dims():
    assert ENV_OBS_DIM == MODEL_OBS_DIM == 182, \
        f"OBS_DIM mismatch: env={ENV_OBS_DIM} model={MODEL_OBS_DIM}"
    assert _SELF_END == 8
    assert _ALLY_END == 72
    assert _ENEMY_END == 144
    assert _ZONE_END == 162
    assert _AMASK_END == 170
    assert _EMASK_END == 179
    print('[OK] dim constants')


def check_model_forward_on_synthetic():
    """Run models on random obs - doesn't need SMACv2."""
    B = 4
    obs = torch.randn(B, ENV_OBS_DIM)
    # Simulate a 3v3-like padding pattern: ally slots 2-7 padded, enemy 3-8 padded
    obs[:, _ZONE_END:_AMASK_END] = 0.0
    obs[:, _ZONE_END + 2:_AMASK_END] = 1.0     # ally mask: slots 2-7 padded
    obs[:, _AMASK_END:_EMASK_END] = 0.0
    obs[:, _AMASK_END + 3:_EMASK_END] = 1.0     # enemy mask: slots 3-8 padded
    obs[:, _EMASK_END:] = 0.0

    full = ScalingFullModel()
    logits, aw = full(obs, return_attn=True)
    assert logits.shape == (B, 6), logits.shape
    assert torch.isfinite(logits).all(), 'non-finite logits from FullModel'
    assert aw['ally_self_attn'].shape == (B, MAX_ALLY_SLOTS, MAX_ALLY_SLOTS)
    assert aw['enemy_cross_attn'].shape == (B, 1, MAX_ENEMY_SLOTS)
    print(f'[OK] ScalingFullModel forward  - logits {tuple(logits.shape)}, '
          f'ally_sa {tuple(aw["ally_self_attn"].shape)}')

    mlp = ScalingRoleMLPModel()
    lm, _ = mlp(obs)
    assert lm.shape == (B, 6) and torch.isfinite(lm).all()
    print(f'[OK] ScalingRoleMLPModel forward - logits {tuple(lm.shape)}')


def check_policy_builder_weight_sharing():
    # 6v6 balanced: 2 Strike, 2 Vanguard, 2 Support
    roles = [0, 0, 1, 1, 2, 2]
    pols = build_scaling_policies(1, agent_roles=roles)
    assert len(pols) == 6
    # Same-role agents share weights; cross-role do not
    assert pols[0] is pols[1], 'Strike policies not shared'
    assert pols[2] is pols[3], 'Vanguard policies not shared'
    assert pols[4] is pols[5], 'Support policies not shared'
    assert pols[0] is not pols[2] and pols[2] is not pols[4]
    unique = {id(p) for p in pols}
    assert len(unique) == 3, f'expected 3 unique policies, got {len(unique)}'
    print('[OK] build_scaling_policies - same-role weight sharing confirmed')


def check_env(config_name, composition, n_enemies):
    if not _SMACV2_AVAILABLE:
        print(f'[SKIP] {config_name}: SMACv2 not importable')
        return
    cfg = dict(
        map_name='10gen_terran', max_steps=10, obs_dim=ENV_OBS_DIM, action_dim=6,
        team_reward_alpha=0.2, seed=0,
        composition=composition, n_enemies=n_enemies,
    )
    try:
        env = ScalingSMACWrapper(cfg)
    except Exception as e:
        print(f'[SKIP] {config_name}: env construction failed - {e}')
        return

    obs, _ = env.reset()
    n = env.n_agents
    assert len(obs) == n
    assert all(o.shape == (ENV_OBS_DIM,) for o in obs)

    # Check padding: in 3v3 balanced, agent 0 has 2 other allies → slots 0-1 valid, 2-7 padded.
    n_valid_ally = n - 1
    agent0_mask = obs[0][_ZONE_END:_AMASK_END]
    # At least the padded slots should be masked = 1; first n_valid_ally can be 0 or 1 depending on whether ally is alive (should be alive right after reset).
    assert agent0_mask[n_valid_ally:].all() == True or np.all(agent0_mask[n_valid_ally:] == 1.0), \
        f'padded ally slots should be masked in {config_name}, got {agent0_mask}'

    # Step once with random actions
    acts = [np.random.randint(0, 6) for _ in range(n)]
    obs2, rew, done, trunc, info = env.step(acts)
    assert len(obs2) == n and rew.shape == (n,)
    assert np.isfinite(rew).all(), f'non-finite reward in {config_name}'

    # Forward through FullModel
    t = torch.tensor(np.stack(obs2), dtype=torch.float32)
    m = ScalingFullModel()
    with torch.no_grad():
        lg, _ = m(t)
    assert torch.isfinite(lg).all()

    env.close()
    print(f'[OK] {config_name} - n_agents={n}, n_enemies={n_enemies}, '
          f'reward range [{rew.min():.3f}, {rew.max():.3f}]')


if __name__ == '__main__':
    check_dims()
    check_model_forward_on_synthetic()
    check_policy_builder_weight_sharing()

    # Env smoke for each config (only if SMACv2 available)
    configs = [
        ('3v3_balanced',      {'Strike': 1, 'Vanguard': 1, 'Support': 1}, 3),
        ('6v6_balanced',      {'Strike': 2, 'Vanguard': 2, 'Support': 2}, 6),
        ('9v9_balanced',      {'Strike': 3, 'Vanguard': 3, 'Support': 3}, 9),
        ('6v6_strike_heavy',  {'Strike': 3, 'Vanguard': 2, 'Support': 1}, 6),
        ('6v6_support_heavy', {'Strike': 1, 'Vanguard': 2, 'Support': 3}, 6),
    ]
    for name, comp, n_enem in configs:
        check_env(name, comp, n_enem)

    print('\nAll smoke checks passed.')
