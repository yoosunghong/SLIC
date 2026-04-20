"""
Regenerate all Track 1 & 2 plots with updated styling into results/plots/new/.

Usage (from repo root):
  python smacv2_poc/regen_styled_plots.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.dirname(_ROOT)
for p in [_WORKSPACE, _ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from smacv2_poc.models import build_policies, CentralizedCritic
from smacv2_poc.analysis.scenario_builder import build_scenario_obs, ROLES, ALLY_SLOT_ROLES
from smacv2_poc.analysis.analyze_attention import (
    load_condition,
    extract_attn_weights,
    compute_role_routing_matrix,
    compute_delta_max,
    compute_mlp_gradient_attribution,
    check_slot_ordering_bias,
)
from smacv2_poc.analysis.analyze_extended import (
    compute_gradient_attribution_cond,
    plot_attribution_comparison,
    plot_multiseed_routing,
)
from validation_poc.analysis.visualize import (
    plot_self_attn_heatmap,
    plot_role_routing_matrix,
    plot_delta_max_comparison,
    plot_gradient_attribution,
    plot_win_rate_curves,
)

RESULTS_DIR = os.path.join(_ROOT, 'results')
PLOTS_NEW   = os.path.join(RESULTS_DIR, 'plots', 'new')
CKPT_DIR    = os.path.join(RESULTS_DIR, 'checkpoints')

ROLE_DISPLAY = {
    'strike':   'Strike (Marine)',
    'vanguard': 'Vanguard (Marauder)',
    'support':  'Support (Medivac)',
}


def ckpt(cond, seed, step='2000384'):
    path = os.path.join(CKPT_DIR, f'cond{cond}_seed{seed}_step{step}.pt')
    if not os.path.exists(path):
        # fallback to any step for this cond/seed
        import glob
        candidates = sorted(glob.glob(
            os.path.join(CKPT_DIR, f'cond{cond}_seed{seed}_step*.pt')))
        return candidates[-1] if candidates else None
    return path


def main():
    os.makedirs(PLOTS_NEW, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Output dir : {PLOTS_NEW}')
    print(f'Device     : {device}')

    # ── Track 1: Win Rate Curves ──────────────────────────────────────────
    print('\n[Track 1] Win rate curves')
    plot_win_rate_curves(RESULTS_DIR, PLOTS_NEW)

    # ── Track 2 — Condition 1 ─────────────────────────────────────────────
    ckpt1 = ckpt(1, 0)
    delta_max1 = {r: 0.0 for r in ROLES}
    routing1   = None

    if ckpt1:
        print(f'\n[Cond1] Loading {os.path.basename(ckpt1)}')
        policies1 = load_condition(1, ckpt1, device)

        clustered_obs = build_scenario_obs('clustered')
        spread_obs    = build_scenario_obs('spread')

        for i, role in enumerate(ROLES):
            attn_c = extract_attn_weights(policies1[i], clustered_obs[role], device)
            attn_s = extract_attn_weights(policies1[i], spread_obs[role],    device)
            if attn_c and 'ally_self_attn' in attn_c:
                wc = attn_c['ally_self_attn'].squeeze().cpu().numpy()
                ws = attn_s['ally_self_attn'].squeeze().cpu().numpy()
                plot_self_attn_heatmap(wc, ws, ROLE_DISPLAY[role], PLOTS_NEW)

        delta_max1 = compute_delta_max(policies1, device)
        print(f'  Δmax (Cond1): {delta_max1}')

        bias = check_slot_ordering_bias(policies1, device)
        routing1 = bias['default_matrix']
        print(f'  Routing matrix (Cond1):\n{routing1}')
        plot_role_routing_matrix(routing1, 'Cond1_Full_SMACv2', PLOTS_NEW)

    # ── Track 2 — Condition 2 ─────────────────────────────────────────────
    ckpt2 = ckpt(2, 0)
    delta_max2 = {r: 0.0 for r in ROLES}

    if ckpt2:
        print(f'\n[Cond2] Loading {os.path.basename(ckpt2)}')
        policies2  = load_condition(2, ckpt2, device)
        delta_max2 = compute_delta_max(policies2, device)
        print(f'  Δmax (Cond2): {delta_max2}')

    plot_delta_max_comparison(delta_max1, delta_max2, PLOTS_NEW)

    # ── Track 2 — Condition 3 ─────────────────────────────────────────────
    ckpt3    = ckpt(3, 0)
    routing3 = None
    grad_cond3 = None

    if ckpt3:
        print(f'\n[Cond3] Loading {os.path.basename(ckpt3)}')
        policies3 = load_condition(3, ckpt3, device)

        routing3 = np.zeros((3, 3), dtype=np.float32)
        clustered_obs = build_scenario_obs('clustered')
        grad_cond3 = {}

        for i, role in enumerate(ROLES):
            sp_grad, role_grad = compute_mlp_gradient_attribution(
                policies3[i], clustered_obs[role], device)
            print(f'  {ROLE_DISPLAY[role]}: spatial={sp_grad:.4f}, role={role_grad:.4f}')
            plot_gradient_attribution(ROLE_DISPLAY[role], sp_grad, role_grad, PLOTS_NEW)
            grad_cond3[role] = {'spatial': sp_grad, 'role': role_grad,
                                'ratio': sp_grad / max(role_grad, 1e-8)}

            obs_leaf = torch.tensor(
                clustered_obs[role], dtype=torch.float32, device=device, requires_grad=True)
            logits, _ = policies3[i](obs_leaf.unsqueeze(0))
            logits.max(dim=-1).values.sum().backward()
            grad = obs_leaf.grad.abs().cpu().numpy()
            for slot_idx, ally_role_idx in enumerate(ALLY_SLOT_ROLES[i]):
                role_start = 8 + slot_idx * 8 + 4
                routing3[i, ally_role_idx] = float(np.mean(grad[role_start:role_start + 3]))

        print(f'  Routing matrix (Cond3 gradient):\n{routing3}')
        plot_role_routing_matrix(routing3, 'Cond3_MLP_Gradient_SMACv2', PLOTS_NEW)

    # ── Extended: Multi-seed Cond1 routing matrix ─────────────────────────
    print('\n[Extended] Multi-seed Cond1 routing matrix')
    matrices = []
    for seed in range(3):
        c = ckpt(1, seed)
        if c:
            print(f'  seed{seed}: {os.path.basename(c)}')
            p = load_condition(1, c, device)
            matrices.append(compute_role_routing_matrix(p, device))

    if len(matrices) >= 2:
        mean_m = np.mean(matrices, axis=0)
        std_m  = np.std(matrices,  axis=0)
        plot_multiseed_routing(mean_m, std_m, PLOTS_NEW)

    # ── Extended: Cond4 / Cond5 gradient attribution comparison ──────────
    ckpt4 = ckpt(4, 1)
    ckpt5 = ckpt(5, 1)
    grad_cond4 = grad_cond5 = None

    if ckpt4:
        print(f'\n[Cond4] Loading {os.path.basename(ckpt4)}')
        policies4 = load_condition(4, ckpt4, device)
        grad_cond4 = compute_gradient_attribution_cond(policies4, device, label='Cond4')

    if ckpt5:
        print(f'\n[Cond5] Loading {os.path.basename(ckpt5)}')
        policies5 = load_condition(5, ckpt5, device)
        grad_cond5 = compute_gradient_attribution_cond(policies5, device, label='Cond5')

    if grad_cond3 and grad_cond4 and grad_cond5:
        plot_attribution_comparison(grad_cond3, grad_cond4, grad_cond5, PLOTS_NEW)

    print(f'\nDone. All styled plots saved to: {PLOTS_NEW}')


if __name__ == '__main__':
    main()
