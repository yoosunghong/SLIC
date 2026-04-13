"""
Track 2 analysis for SMACv2 checkpoints.

Usage:
  python analyze_attention.py \
    --cond1_ckpt results/checkpoints/cond1_seed0_step2000000.pt \
    --cond2_ckpt results/checkpoints/cond2_seed0_step2000000.pt \
    --cond3_ckpt results/checkpoints/cond3_seed0_step2000000.pt

Identical logic to validation_poc/analysis/analyze_attention.py except:
  - Uses smacv2_poc's scenario_builder (same interface, SMACv2 coordinates)
  - Reports SMACv2-specific role names (Stalker/Zealot/Colossus)
"""
import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE = os.path.dirname(_ROOT)
for p in [_WORKSPACE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from smacv2_poc.models import build_policies, CentralizedCritic
from smacv2_poc.analysis.scenario_builder import build_scenario_obs, ROLES, ALLY_SLOT_ROLES
from smacv2_poc.analysis.visualize import (
    plot_self_attn_heatmap,
    plot_role_routing_matrix,
    plot_delta_max_comparison,
    plot_gradient_attribution,
    plot_win_rate_curves,
)

ROLE_DISPLAY = {
    'strike':   'Strike (Stalker)',
    'vanguard': 'Vanguard (Zealot)',
    'support':  'Support (Colossus)',
}


def load_condition(condition: int, ckpt_path: str, device: torch.device,
                   obs_dim: int = 65, action_dim: int = 6):
    policies = build_policies(condition, obs_dim=obs_dim, action_dim=action_dim)
    unique = list({id(p): p for p in policies}.values())
    for p in unique:
        p.to(device)
        p.eval()
    critic = CentralizedCritic(n_agents=1, obs_dim=obs_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    for p, sd in zip(unique, ckpt['policies']):
        p.load_state_dict(sd)
    critic.load_state_dict(ckpt['critic'])
    return policies


def extract_attn_weights(policy, obs_np: np.ndarray, device: torch.device):
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, attn = policy(obs, return_attn=True)
    return attn


def compute_role_routing_matrix(policies, device: torch.device) -> np.ndarray:
    """3×3 cross-attention role-routing matrix, averaged over both scenarios."""
    matrix = np.zeros((3, 3), dtype=np.float32)
    counts = np.zeros((3, 3), dtype=np.float32)

    for scenario in ['clustered', 'spread']:
        scenario_obs = build_scenario_obs(scenario)
        for i, role in enumerate(ROLES):
            attn = extract_attn_weights(policies[i], scenario_obs[role], device)
            if attn is None or 'ally_cross_attn' not in attn:
                continue
            ca = attn['ally_cross_attn'].squeeze().cpu().numpy()
            if ca.ndim == 0:
                ca = np.array([ca.item()])
            for slot_idx, ally_role_idx in enumerate(ALLY_SLOT_ROLES[i]):
                if slot_idx < len(ca):
                    matrix[i, ally_role_idx] += ca[slot_idx]
                    counts[i, ally_role_idx] += 1

    counts = np.where(counts == 0, 1, counts)
    return matrix / counts


def compute_delta_max(policies, device: torch.device) -> dict:
    """Δmax per role from ally self-attention weights."""
    delta_max = {}
    clustered_obs = build_scenario_obs('clustered')
    spread_obs = build_scenario_obs('spread')

    for i, role in enumerate(ROLES):
        attn_c = extract_attn_weights(policies[i], clustered_obs[role], device)
        attn_s = extract_attn_weights(policies[i], spread_obs[role], device)
        if attn_c is None or 'ally_self_attn' not in attn_c:
            delta_max[role] = 0.0
            continue
        wc = attn_c['ally_self_attn'].squeeze().cpu().numpy()
        ws = attn_s['ally_self_attn'].squeeze().cpu().numpy()
        delta_max[role] = float(np.max(np.abs(wc - ws)))

    return delta_max


def compute_mlp_gradient_attribution(policy, obs_np: np.ndarray, device: torch.device):
    """Spatial vs. role-dim gradient magnitudes for Condition 3 MLP."""
    obs_leaf = torch.tensor(obs_np, dtype=torch.float32, device=device, requires_grad=True)
    logits, _ = policy(obs_leaf.unsqueeze(0))
    logits.max(dim=-1).values.sum().backward()
    grad = obs_leaf.grad.abs().cpu().numpy()

    # Role dims (unit_type one-hots):
    #   self_tok:       [4:7]
    #   ally slot 0:    [12:15]
    #   ally slot 1:    [20:23]
    role_dims = list(range(4, 7)) + list(range(12, 15)) + list(range(20, 23))
    spatial_dims = [d for d in range(len(grad)) if d not in role_dims]
    return float(np.mean(grad[spatial_dims])), float(np.mean(grad[role_dims]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond1_ckpt', type=str, default=None)
    parser.add_argument('--cond2_ckpt', type=str, default=None)
    parser.add_argument('--cond3_ckpt', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--obs_dim', type=int, default=65)
    parser.add_argument('--action_dim', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plots_dir = os.path.join(_ROOT, args.results_dir, 'plots')
    results_dir = os.path.join(_ROOT, args.results_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # Win rate curves from CSV logs
    plot_win_rate_curves(results_dir, plots_dir)

    checks = {}

    # ---- Condition 1 (Full) ----
    delta_max1 = {r: 0.0 for r in ROLES}
    routing1 = None

    if args.cond1_ckpt and os.path.exists(args.cond1_ckpt):
        print('\n--- Condition 1 (Full — Self-Attn + Cross-Attn) ---')
        policies1 = load_condition(1, args.cond1_ckpt, device, args.obs_dim, args.action_dim)

        clustered_obs = build_scenario_obs('clustered')
        spread_obs = build_scenario_obs('spread')

        for i, role in enumerate(ROLES):
            attn_c = extract_attn_weights(policies1[i], clustered_obs[role], device)
            attn_s = extract_attn_weights(policies1[i], spread_obs[role], device)
            if attn_c and 'ally_self_attn' in attn_c:
                wc = attn_c['ally_self_attn'].squeeze().cpu().numpy()
                ws = attn_s['ally_self_attn'].squeeze().cpu().numpy()
                plot_self_attn_heatmap(wc, ws, ROLE_DISPLAY[role], plots_dir)

        delta_max1 = compute_delta_max(policies1, device)
        print(f'  Δmax per role (Cond1): {delta_max1}')

        routing1 = compute_role_routing_matrix(policies1, device)
        print(f'  Role routing matrix (Cond1):\n{routing1}')
        plot_role_routing_matrix(routing1, 'Cond1_Full_SMACv2', plots_dir)

        checks['vanguard_highest_delta'] = (
            delta_max1.get('vanguard', 0) >= max(delta_max1.values())
        )
        checks['support_lowest_delta'] = (
            delta_max1.get('support', 0) <= min(delta_max1.values())
        )
        # Strike→Support (row 0, col 2) > Strike→Vanguard (row 0, col 1)
        checks['strike_support_routing'] = routing1[0, 2] > routing1[0, 1]
    else:
        print('Condition 1 checkpoint not provided — skipping.')

    # ---- Condition 2 (No Self-Attn) ----
    delta_max2 = {r: 0.0 for r in ROLES}
    if args.cond2_ckpt and os.path.exists(args.cond2_ckpt):
        print('\n--- Condition 2 (No Self-Attn) ---')
        policies2 = load_condition(2, args.cond2_ckpt, device, args.obs_dim, args.action_dim)
        delta_max2 = compute_delta_max(policies2, device)
        print(f'  Δmax per role (Cond2): {delta_max2}')

    plot_delta_max_comparison(delta_max1, delta_max2, plots_dir)

    # ---- Condition 3 (Role+MLP) ----
    if args.cond3_ckpt and os.path.exists(args.cond3_ckpt):
        print('\n--- Condition 3 (Role+MLP) ---')
        policies3 = load_condition(3, args.cond3_ckpt, device, args.obs_dim, args.action_dim)

        routing3 = np.zeros((3, 3), dtype=np.float32)
        clustered_obs = build_scenario_obs('clustered')

        for i, role in enumerate(ROLES):
            sp_grad, role_grad = compute_mlp_gradient_attribution(
                policies3[i], clustered_obs[role], device
            )
            print(f'  {ROLE_DISPLAY[role]}: spatial_grad={sp_grad:.4f}, role_grad={role_grad:.4f}')
            plot_gradient_attribution(ROLE_DISPLAY[role], sp_grad, role_grad, plots_dir)

            # Gradient-based pseudo routing matrix
            obs_leaf = torch.tensor(
                clustered_obs[role], dtype=torch.float32, device=device, requires_grad=True
            )
            logits, _ = policies3[i](obs_leaf.unsqueeze(0))
            logits.max(dim=-1).values.sum().backward()
            grad = obs_leaf.grad.abs().cpu().numpy()
            for slot_idx, ally_role_idx in enumerate(ALLY_SLOT_ROLES[i]):
                # ally role dims: base = 8 + slot*8 + 4, length 3
                role_start = 8 + slot_idx * 8 + 4
                routing3[i, ally_role_idx] = float(np.mean(grad[role_start:role_start + 3]))

        print(f'  Gradient routing matrix (Cond3):\n{routing3}')
        plot_role_routing_matrix(routing3, 'Cond3_MLP_Gradient_SMACv2', plots_dir)

        if routing1 is not None:
            r1_std = np.std(routing1)
            r3_std = np.std(routing3)
            checks['mlp_routing_weaker'] = r3_std < r1_std
            print(f'  Routing structure: Cond1 std={r1_std:.4f}, Cond3 std={r3_std:.4f}')
    else:
        print('Condition 3 checkpoint not provided — skipping.')

    # ---- Claim Confirmation Checklist ----
    print('\n' + '=' * 55)
    print('CLAIM CONFIRMATION CHECKLIST  (SMACv2)')
    print('=' * 55)

    def report(key, label):
        result = checks.get(key)
        status = 'SKIP' if result is None else ('PASS' if result else 'FAIL')
        print(f'[{status}] {label}')

    report('vanguard_highest_delta',
           'Zealot (Vanguard) Δmax is highest among all roles')
    report('support_lowest_delta',
           'Colossus (Support) Δmax is lowest among all roles')
    report('strike_support_routing',
           'Stalker→Colossus routing weight > Stalker→Zealot routing weight')
    report('mlp_routing_weaker',
           'Role routing is weaker/unstructured in Condition 3 (MLP)')
    print('[SKIP] No_RoleLabel win rate < Full win rate by >5pp  '
          '(requires training logs)')
    print('=' * 55)


if __name__ == '__main__':
    main()
