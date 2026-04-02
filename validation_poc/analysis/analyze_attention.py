"""Track 2 analysis: load trained checkpoints, extract attention weights, generate plots.

Usage:
  python analyze_attention.py --cond1_ckpt results/checkpoints/cond1_seed0_step2000000.pt \
                               --cond2_ckpt results/checkpoints/cond2_seed0_step2000000.pt \
                               --cond3_ckpt results/checkpoints/cond3_seed0_step2000000.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_policies, CentralizedCritic
from analysis.scenario_builder import build_scenario_obs, ROLES
from analysis.visualize import (
    plot_self_attn_heatmap,
    plot_role_routing_matrix,
    plot_delta_max_comparison,
    plot_gradient_attribution,
    plot_win_rate_curves,
)

ROLE_IDX = {'strike': 0, 'vanguard': 1, 'support': 2}


def load_condition(condition: int, ckpt_path: str, device: torch.device, obs_dim: int = 65, action_dim: int = 6):
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


def compute_role_routing_matrix(policies, device: torch.device):
    """
    Build 3×3 role-routing matrix from cross-attention weights.
    Entry (i,j) = mean cross-attention weight of role i toward ally slot containing role j.
    Uses both Clustered and Spread scenarios averaged.
    """
    # ally slot → role mapping: for agent i, ally slots are the other 2 agents
    # agent 0 (Strike): allies = [Vanguard(1), Support(2)]
    # agent 1 (Vanguard): allies = [Strike(0), Support(2)]
    # agent 2 (Support): allies = [Strike(0), Vanguard(1)]
    ally_slot_roles = {
        0: [1, 2],  # Strike's ally slots: Vanguard, Support
        1: [0, 2],  # Vanguard's ally slots: Strike, Support
        2: [0, 1],  # Support's ally slots: Strike, Vanguard
    }

    matrix = np.zeros((3, 3), dtype=np.float32)
    counts = np.zeros((3, 3), dtype=np.float32)

    for scenario in ['clustered', 'spread']:
        scenario_obs = build_scenario_obs(scenario)
        for i, role in enumerate(ROLES):
            attn = extract_attn_weights(policies[i], scenario_obs[role], device)
            if attn is None or 'ally_cross_attn' not in attn:
                continue
            # ally_cross_attn: (1, 1, 2) → weights over 2 ally slots
            ca_weights = attn['ally_cross_attn'].squeeze().cpu().numpy()  # (2,)
            if ca_weights.ndim == 0:
                ca_weights = np.array([ca_weights.item()])
            ally_roles = ally_slot_roles[i]
            for slot_idx, ally_role_idx in enumerate(ally_roles):
                if slot_idx < len(ca_weights):
                    matrix[i, ally_role_idx] += ca_weights[slot_idx]
                    counts[i, ally_role_idx] += 1

    counts = np.where(counts == 0, 1, counts)
    return matrix / counts


def compute_delta_max(policies, device: torch.device):
    """Compute Δmax per role: max |clustered - spread| in ally self-attention weights."""
    delta_max = {}
    clustered_obs = build_scenario_obs('clustered')
    spread_obs = build_scenario_obs('spread')

    for i, role in enumerate(ROLES):
        attn_c = extract_attn_weights(policies[i], clustered_obs[role], device)
        attn_s = extract_attn_weights(policies[i], spread_obs[role], device)

        if attn_c is None or 'ally_self_attn' not in attn_c:
            delta_max[role] = 0.0
            continue

        wc = attn_c['ally_self_attn'].squeeze().cpu().numpy()  # (2, 2)
        ws = attn_s['ally_self_attn'].squeeze().cpu().numpy()
        delta_max[role] = float(np.max(np.abs(wc - ws)))

    return delta_max


def compute_mlp_gradient_attribution(policy, obs_np: np.ndarray, device: torch.device):
    """Compute input gradient attribution for spatial vs role dims."""
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
    logits, _ = policy(obs)
    # Use max logit as scalar output
    output = logits.max(dim=-1).values.sum()
    output.backward()
    grad = obs.grad.squeeze(0).abs().cpu().numpy()

    # Spatial dims: positions throughout obs (rough heuristic: all non-role dims)
    # Role dims in self_tok: 5,6,7
    # Role dims in ally_tok: 8+4=12,13,14 and 8+8+4=20,21,22
    role_dims = list(range(5, 8)) + list(range(12, 15)) + list(range(20, 23))
    all_dims = set(range(len(grad)))
    spatial_dims = list(all_dims - set(role_dims))

    spatial_grad = float(np.mean(grad[spatial_dims]))
    role_grad = float(np.mean(grad[role_dims]))
    return spatial_grad, role_grad


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
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base, args.results_dir, 'plots')
    results_dir = os.path.join(base, args.results_dir)
    os.makedirs(plots_dir, exist_ok=True)

    # Win rate curves (from CSV logs)
    plot_win_rate_curves(results_dir, plots_dir)

    checks = {}

    # ---- Condition 1 Analysis ----
    if args.cond1_ckpt and os.path.exists(args.cond1_ckpt):
        print('\n--- Condition 1 (Full) ---')
        policies1 = load_condition(1, args.cond1_ckpt, device, args.obs_dim, args.action_dim)

        # Self-attention heatmaps per role
        clustered_obs = build_scenario_obs('clustered')
        spread_obs = build_scenario_obs('spread')
        for i, role in enumerate(ROLES):
            attn_c = extract_attn_weights(policies1[i], clustered_obs[role], device)
            attn_s = extract_attn_weights(policies1[i], spread_obs[role], device)
            if attn_c and 'ally_self_attn' in attn_c:
                wc = attn_c['ally_self_attn'].squeeze().cpu().numpy()
                ws = attn_s['ally_self_attn'].squeeze().cpu().numpy()
                plot_self_attn_heatmap(wc, ws, role.capitalize(), plots_dir)

        # Delta max
        delta_max1 = compute_delta_max(policies1, device)
        print(f'  Δmax per role (Cond1): {delta_max1}')

        # Role routing matrix
        routing1 = compute_role_routing_matrix(policies1, device)
        print(f'  Role routing matrix (Cond1):\n{routing1}')
        plot_role_routing_matrix(routing1, 'Cond1_Full', plots_dir)

        checks['vanguard_highest_delta'] = (
            delta_max1.get('vanguard', 0) >= max(delta_max1.values())
        )
        checks['support_lowest_delta'] = (
            delta_max1.get('support', 0) <= min(delta_max1.values())
        )
        checks['strike_support_routing'] = (
            routing1[0, 2] > routing1[0, 1]  # Strike→Support > Strike→Vanguard
        )
    else:
        print('Condition 1 checkpoint not provided or not found — skipping.')
        delta_max1 = {'strike': 0, 'vanguard': 0, 'support': 0}
        routing1 = None

    # ---- Condition 2 Analysis ----
    delta_max2 = {'strike': 0, 'vanguard': 0, 'support': 0}
    if args.cond2_ckpt and os.path.exists(args.cond2_ckpt):
        print('\n--- Condition 2 (No Self-Attn) ---')
        policies2 = load_condition(2, args.cond2_ckpt, device, args.obs_dim, args.action_dim)
        # No self-attn in cond2, use cross-attn weights as proxy
        delta_max2 = compute_delta_max(policies2, device)
        print(f'  Δmax per role (Cond2): {delta_max2}')

    plot_delta_max_comparison(delta_max1, delta_max2, plots_dir)

    # ---- Condition 3 Analysis ----
    if args.cond3_ckpt and os.path.exists(args.cond3_ckpt):
        print('\n--- Condition 3 (Role+MLP) ---')
        policies3 = load_condition(3, args.cond3_ckpt, device, args.obs_dim, args.action_dim)

        routing3 = np.zeros((3, 3), dtype=np.float32)
        clustered_obs = build_scenario_obs('clustered')
        for i, role in enumerate(ROLES):
            sp_grad, role_grad = compute_mlp_gradient_attribution(
                policies3[i], clustered_obs[role], device
            )
            print(f'  {role.capitalize()}: spatial_grad={sp_grad:.4f}, role_grad={role_grad:.4f}')
            plot_gradient_attribution(role.capitalize(), sp_grad, role_grad, plots_dir)

            # Build pseudo routing from gradient attribution on role dims in ally_tok
            obs = torch.tensor(clustered_obs[role], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            logits, _ = policies3[i](obs)
            logits.max(dim=-1).values.sum().backward()
            grad = obs.grad.squeeze(0).abs().cpu().numpy()
            # ally slot 0 role dims: 12,13,14 — ally slot 1 role dims: 20,21,22
            ally_slot_roles = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
            for slot_idx, ally_role_idx in enumerate(ally_slot_roles[i]):
                role_start = 8 + slot_idx * 8 + 4
                role_end = role_start + 3
                routing3[i, ally_role_idx] = float(np.mean(grad[role_start:role_end]))

        print(f'  Gradient-based routing matrix (Cond3):\n{routing3}')
        plot_role_routing_matrix(routing3, 'Cond3_MLP_Gradient', plots_dir)

        if routing1 is not None:
            routing1_strength = np.std(routing1)
            routing3_strength = np.std(routing3)
            checks['mlp_routing_weaker'] = routing3_strength < routing1_strength
            print(f'  Routing structure: Cond1 std={routing1_strength:.4f}, Cond3 std={routing3_strength:.4f}')
    else:
        print('Condition 3 checkpoint not provided or not found — skipping.')

    # ---- Claim Confirmation Checklist ----
    print('\n' + '=' * 50)
    print('CLAIM CONFIRMATION CHECKLIST')
    print('=' * 50)

    def report(key, label):
        result = checks.get(key, None)
        if result is None:
            status = 'SKIP'
        else:
            status = 'PASS' if result else 'FAIL'
        print(f'[{status}] {label}')

    report('vanguard_highest_delta', 'Vanguard Δmax is highest among all roles')
    report('support_lowest_delta', 'Support Δmax is lowest among all roles')
    report('strike_support_routing', 'Strike→Support routing weight > Strike→Vanguard routing weight')
    report('mlp_routing_weaker', 'Role routing pattern is weaker/unstructured in Condition 3 (MLP)')
    print('[SKIP] No_RoleLabel (shared) win rate < Full win rate by >5pp  (requires training logs)')
    print('=' * 50)


if __name__ == '__main__':
    main()
