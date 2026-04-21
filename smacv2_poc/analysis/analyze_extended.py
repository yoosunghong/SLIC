"""
Extended Track 2 analysis:
  Task A — Cond4 (No-RoleLabel + Attn) and Cond5 (Flat-MLP) gradient attribution
  Task B — Multi-seed Cond1 routing matrix (seed0, seed1, seed2), mean ± std

Usage:
  python -m smacv2_poc.analysis.analyze_extended \
    --cond1_seed0 smacv2_poc/results/checkpoints/cond1_seed0_step2000384.pt \
    --cond1_seed1 smacv2_poc/results/checkpoints/cond1_seed1_step1801728.pt \
    --cond1_seed2 smacv2_poc/results/checkpoints/cond1_seed2_step2000384.pt \
    --cond4_ckpt  smacv2_poc/results/checkpoints/cond4_seed1_step2000384.pt \
    --cond5_ckpt  smacv2_poc/results/checkpoints/cond5_seed1_step2000384.pt \
    --results_dir smacv2_poc/results
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from validation_poc.analysis.visualize import CB_COLORS, _apply_style, _savefig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE = os.path.dirname(_ROOT)
for p in [_WORKSPACE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from smacv2_poc.models import build_policies, CentralizedCritic
from smacv2_poc.analysis.scenario_builder import build_scenario_obs, ROLES, ALLY_SLOT_ROLES
from smacv2_poc.analysis.analyze_attention import (
    load_condition,
    compute_role_routing_matrix,
    compute_mlp_gradient_attribution,
)

ROLE_DISPLAY = {
    'strike':   'Strike (Marine)',
    'vanguard': 'Vanguard (Marauder)',
    'support':  'Support (Medivac)',
}


def compute_gradient_attribution_cond(policies, device, label='Cond'):
    """Run spatial vs. role-dim gradient attribution for any condition."""
    results = {}
    clustered_obs = build_scenario_obs('clustered')
    for i, role in enumerate(ROLES):
        sp_grad, role_grad = compute_mlp_gradient_attribution(
            policies[i], clustered_obs[role], device
        )
        results[role] = {'spatial': sp_grad, 'role': role_grad,
                         'ratio': sp_grad / max(role_grad, 1e-8)}
        print(f'  {label} {ROLE_DISPLAY[role]}: '
              f'spatial={sp_grad:.4f}, role={role_grad:.4f}, '
              f'ratio={sp_grad / max(role_grad, 1e-8):.2f}x')
    return results


def plot_attribution_comparison(grad_cond3, grad_cond4, grad_cond5, plots_dir):
    """Bar chart comparing role-dim gradient across Cond3/4/5 per role."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    width = 0.25
    roles_display = [ROLE_DISPLAY[r] for r in ROLES]

    c3 = [grad_cond3[r]['role'] for r in ROLES]
    c4 = [grad_cond4[r]['role'] for r in ROLES]
    c5 = [grad_cond5[r]['role'] for r in ROLES]

    ax.bar(x - width, c3, width, label='Cond3 (Role+MLP)',
           color=CB_COLORS[0], hatch='//', edgecolor='white')
    ax.bar(x,         c4, width, label='Cond4 (No-Label+Attn)',
           color=CB_COLORS[1], hatch='xx', edgecolor='white')
    ax.bar(x + width, c5, width, label='Cond5 (Flat-MLP)',
           color=CB_COLORS[2], hatch='..', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(roles_display)
    ax.set_ylabel(r'Mean $|\partial \mathrm{logit} / \partial \mathrm{role\_dim}|$')
    ax.set_title('Role-Dim Gradient Magnitude: Cond3 vs Cond4 vs Cond5')
    ax.legend()
    plt.tight_layout()
    stem = os.path.join(plots_dir, 'role_grad_attribution_comparison')
    saved = _savefig(fig, stem)
    plt.close()
    print(f'Saved: {saved}  (+.png)')


def plot_multiseed_routing(mean_matrix, std_matrix, plots_dir):
    """Heatmap of 3×3 routing matrix with ± std annotations."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    role_labels = ['Strike\n(Marine)', 'Vanguard\n(Marauder)', 'Support\n(Medivac)']

    mask = np.eye(3, dtype=bool)
    annot = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            if i == j:
                annot[i, j] = '—'
            else:
                annot[i, j] = f'{mean_matrix[i,j]:.3f}\n\u00b1{std_matrix[i,j]:.3f}'

    display = np.where(mask, np.nan, mean_matrix)
    sns.heatmap(display, annot=annot, fmt='', cmap='Blues',
                xticklabels=role_labels, yticklabels=role_labels,
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                mask=mask)
    ax.set_xlabel('Attended Ally Role')
    ax.set_ylabel('Agent Role')
    ax.set_title('Multi-Seed Routing Matrix (Cond1, 5 seeds)')
    plt.tight_layout()
    stem = os.path.join(plots_dir, 'role_routing_matrix_cond1_multiseed')
    saved = _savefig(fig, stem)
    plt.close()
    print(f'Saved: {saved}  (+.png)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cond1_seed0', type=str, default=None)
    parser.add_argument('--cond1_seed1', type=str, default=None)
    parser.add_argument('--cond1_seed2', type=str, default=None)
    parser.add_argument('--cond4_ckpt',  type=str, default=None)
    parser.add_argument('--cond5_ckpt',  type=str, default=None)
    parser.add_argument('--cond3_ckpt',  type=str, default=None,
                        help='Optional: Cond3 for comparison baseline in attribution chart')
    parser.add_argument('--results_dir', type=str, default='smacv2_poc/results')
    parser.add_argument('--obs_dim',    type=int, default=65)
    parser.add_argument('--action_dim', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plots_dir = os.path.join(_ROOT, args.results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ---- Task A: Cond4 gradient attribution ----
    grad_cond4 = None
    if args.cond4_ckpt and os.path.exists(args.cond4_ckpt):
        print('\n--- Condition 4 (No-RoleLabel + Attn) ---')
        policies4 = load_condition(4, args.cond4_ckpt, device, args.obs_dim, args.action_dim)
        grad_cond4 = compute_gradient_attribution_cond(policies4, device, label='Cond4')
        print('  Note: role-dim gradient expected ~0 (labels zeroed in obs)')
    else:
        print('Cond4 checkpoint not provided — skipping.')

    # ---- Task A: Cond5 gradient attribution ----
    grad_cond5 = None
    if args.cond5_ckpt and os.path.exists(args.cond5_ckpt):
        print('\n--- Condition 5 (Flat-MLP) ---')
        policies5 = load_condition(5, args.cond5_ckpt, device, args.obs_dim, args.action_dim)
        grad_cond5 = compute_gradient_attribution_cond(policies5, device, label='Cond5')
        print('  Note: role-dim gradient expected ~0 (labels zeroed, no attention)')
    else:
        print('Cond5 checkpoint not provided — skipping.')

    # Optional Cond3 for comparison chart
    grad_cond3 = None
    if args.cond3_ckpt and os.path.exists(args.cond3_ckpt):
        print('\n--- Condition 3 (Role+MLP, for comparison) ---')
        policies3 = load_condition(3, args.cond3_ckpt, device, args.obs_dim, args.action_dim)
        grad_cond3 = compute_gradient_attribution_cond(policies3, device, label='Cond3')

    if grad_cond3 and grad_cond4 and grad_cond5:
        plot_attribution_comparison(grad_cond3, grad_cond4, grad_cond5, plots_dir)

    # ---- Task B: Multi-seed Cond1 routing matrix ----
    seed_ckpts = {
        'seed0': args.cond1_seed0,
        'seed1': args.cond1_seed1,
        'seed2': args.cond1_seed2,
    }
    matrices = []
    print('\n--- Task B: Multi-seed Cond1 Routing Matrix ---')
    for seed_name, ckpt in seed_ckpts.items():
        if ckpt and os.path.exists(ckpt):
            print(f'  Loading {seed_name}: {os.path.basename(ckpt)}')
            policies = load_condition(1, ckpt, device, args.obs_dim, args.action_dim)
            m = compute_role_routing_matrix(policies, device)
            matrices.append(m)
            print(f'  {seed_name} routing matrix:\n{m}')
        else:
            print(f'  {seed_name} checkpoint not provided — skipping.')

    if len(matrices) >= 2:
        mean_m = np.mean(matrices, axis=0)
        std_m  = np.std(matrices, axis=0)
        print(f'\n  Multi-seed mean routing matrix ({len(matrices)} seeds):\n{mean_m}')
        print(f'  Multi-seed std:\n{std_m}')
        plot_multiseed_routing(mean_m, std_m, plots_dir)

        # Summary table
        role_labels = ['Strike', 'Vanguard', 'Support']
        print('\n  Per-cell summary (mean ± std):')
        print(f'  {"":10s}  {"Strike":>14s}  {"Vanguard":>14s}  {"Support":>14s}')
        for i, row_role in enumerate(role_labels):
            cells = []
            for j in range(3):
                if i == j:
                    cells.append(f'{"--":>14s}')
                else:
                    cells.append(f'{mean_m[i,j]:.3f}±{std_m[i,j]:.3f}'.rjust(14))
            print(f'  {row_role:10s}  {"  ".join(cells)}')
    elif len(matrices) == 1:
        print('  Only 1 seed loaded — no multi-seed average computed.')
    else:
        print('  No Cond1 seeds loaded.')


if __name__ == '__main__':
    main()
