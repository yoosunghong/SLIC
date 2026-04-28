"""
PLAN_V2 item A — Track 2 at scale.

Extract attention-based 3×3 role-routing matrices from scaling-study Cond1
checkpoints (3v3 / 6v6 / 9v9, multiple seeds), and compare side-by-side
with 3v3 to demonstrate routing-structure preservation under team scaling.

Output:
  results_scaling/analysis/routing_<config>_seed<N>_step<step>.npy
  results_scaling/analysis/routing_3v3_vs_9v9.png
  results_scaling/analysis/delta_max_3v3_vs_9v9.png
  results_scaling/analysis/routing_summary.csv

Usage:
  python -m smacv2_poc.analysis.analyze_attention_scaling \
      --configs 3v3_balanced 9v9_balanced \
      --seeds 0 1 2 \
      --step 2000384

Cost: GPU 0 (inference only).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent          # smacv2_poc/
_WORKSPACE = _ROOT.parent
for p in (_WORKSPACE, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from smacv2_poc.models.scaling_models import build_scaling_policies, OBS_DIM
from smacv2_poc.analysis.scenario_builder_scaling import (
    build_scenario_obs_scaling, composition_to_roles, ROLES,
)

ROLE_DISPLAY = ['Strike', 'Vanguard', 'Support']


def _load_config(config_name: str, config_file: Path) -> dict:
    with open(config_file, encoding='utf-8') as f:
        full = yaml.safe_load(f)
    return full['configs'][config_name]


def _find_checkpoint(ckpt_dir: Path, step: int = None) -> Path:
    if not ckpt_dir.exists():
        return None
    def _step_num(p):
        try:
            return int(p.stem.replace('step', ''))
        except ValueError:
            return -1
    ckpts = sorted(ckpt_dir.glob('step*.pt'), key=_step_num)
    if not ckpts:
        return None
    if step is None:
        return ckpts[-1]
    for c in ckpts:
        if f'step{step}.pt' in c.name:
            return c
    return ckpts[-1]


def load_cond1_policies(ckpt_path: Path, agent_roles: list, device: torch.device):
    policies = build_scaling_policies(condition=1, agent_roles=agent_roles)
    unique = list({id(p): p for p in policies}.values())
    for p in unique:
        p.to(device)
        p.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for p, sd in zip(unique, ckpt['policies']):
        p.load_state_dict(sd)
    return policies, unique


def compute_routing_matrix(unique_policies, scenario_obs: dict,
                           device: torch.device) -> np.ndarray:
    """
    3×3 role-routing matrix: rows = agent role, cols = attended ally role.
    Aggregates ally cross-attention weights, grouping ally slots by their role.
    """
    matrix = np.zeros((3, 3), dtype=np.float32)
    counts = np.zeros((3, 3), dtype=np.float32)

    slot_roles = scenario_obs['ally_slot_roles']
    for role_idx, role_name in enumerate(ROLES):
        policy = unique_policies[role_idx]
        obs = torch.tensor(scenario_obs[role_name], dtype=torch.float32,
                            device=device).unsqueeze(0)
        with torch.no_grad():
            _, attn = policy(obs, return_attn=True)
        if attn is None or 'ally_cross_attn' not in attn:
            continue
        ca = attn['ally_cross_attn'].squeeze().cpu().numpy()
        if ca.ndim == 0:
            ca = np.array([ca.item()])
        for slot_idx, ally_role in enumerate(slot_roles[role_name]):
            if slot_idx >= len(ca):
                break
            matrix[role_idx, ally_role] += ca[slot_idx]
            counts[role_idx, ally_role] += 1

    counts = np.where(counts == 0, 1, counts)
    return matrix / counts


def compute_delta_max(unique_policies, clustered_obs, spread_obs,
                       device: torch.device) -> dict:
    delta = {}
    for role_idx, role_name in enumerate(ROLES):
        policy = unique_policies[role_idx]
        oc = torch.tensor(clustered_obs[role_name], dtype=torch.float32,
                           device=device).unsqueeze(0)
        os_ = torch.tensor(spread_obs[role_name], dtype=torch.float32,
                            device=device).unsqueeze(0)
        with torch.no_grad():
            _, attn_c = policy(oc, return_attn=True)
            _, attn_s = policy(os_, return_attn=True)
        if attn_c is None or 'ally_self_attn' not in attn_c:
            delta[role_name] = 0.0
            continue
        wc = attn_c['ally_self_attn'].squeeze().cpu().numpy()
        ws = attn_s['ally_self_attn'].squeeze().cpu().numpy()
        delta[role_name] = float(np.max(np.abs(wc - ws)))
    return delta


def analyze_run(config_name: str, seed: int, step: int,
                config_file: Path, ckpt_root: Path,
                device: torch.device) -> dict:
    cfg = _load_config(config_name, config_file)
    composition = cfg['composition']
    n_enemies = cfg.get('n_enemies', sum(composition.values()))
    agent_roles = composition_to_roles(composition)

    ckpt_dir = ckpt_root / f'{config_name}_cond1_seed{seed}'
    ckpt_path = _find_checkpoint(ckpt_dir, step)
    if ckpt_path is None:
        print(f'  [skip] no checkpoint at {ckpt_dir}')
        return None
    print(f'  loading {ckpt_path.name}')

    policies, unique = load_cond1_policies(ckpt_path, agent_roles, device)

    clustered = build_scenario_obs_scaling('clustered', composition, n_enemies)
    spread    = build_scenario_obs_scaling('spread',    composition, n_enemies)

    # Average routing matrix over both scenarios for stability.
    rm_c = compute_routing_matrix(unique, clustered, device)
    rm_s = compute_routing_matrix(unique, spread,    device)
    routing = 0.5 * (rm_c + rm_s)

    delta = compute_delta_max(unique, clustered, spread, device)
    return {
        'config': config_name,
        'seed': seed,
        'ckpt': ckpt_path.name,
        'routing_clustered': rm_c,
        'routing_spread': rm_s,
        'routing_mean': routing,
        'delta_max': delta,
    }


def plot_comparison(results_by_config: dict, out_path: Path, title: str):
    import matplotlib.pyplot as plt
    cfgs = list(results_by_config.keys())
    n_cfg = len(cfgs)
    fig, axes = plt.subplots(1, n_cfg, figsize=(4.2 * n_cfg, 3.6),
                              constrained_layout=True)
    if n_cfg == 1:
        axes = [axes]
    vmax = max(np.max(r['routing_mean']) for r in results_by_config.values()) or 1.0
    for ax, cfg_name in zip(axes, cfgs):
        rm = results_by_config[cfg_name]['routing_mean']
        im = ax.imshow(rm, cmap='magma', vmin=0, vmax=vmax)
        ax.set_xticks(range(3)); ax.set_xticklabels(ROLE_DISPLAY, rotation=20)
        ax.set_yticks(range(3)); ax.set_yticklabels(ROLE_DISPLAY)
        ax.set_xlabel('Attended ally role')
        if ax is axes[0]:
            ax.set_ylabel('Agent role')
        ax.set_title(f'{cfg_name}\n(seed-mean, scenario-mean)')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{rm[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if rm[i, j] < vmax * 0.55 else 'black',
                         fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_bar(results_by_config: dict, out_path: Path):
    import matplotlib.pyplot as plt
    cfgs = list(results_by_config.keys())
    width = 0.25
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.0, 3.6), constrained_layout=True)
    for k, cfg_name in enumerate(cfgs):
        deltas = results_by_config[cfg_name]['delta_max_mean']
        vals = [deltas[r] for r in ROLES]
        ax.bar(x + (k - (len(cfgs) - 1) / 2.0) * width, vals,
                width=width, label=cfg_name)
    ax.set_xticks(x); ax.set_xticklabels(ROLE_DISPLAY)
    ax.set_ylabel(r'$\Delta_{\max}$  (Clustered vs Spread)')
    ax.set_title('Self-attention formation sensitivity per role')
    ax.legend()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate(per_seed_results: list) -> dict:
    routing_mean = np.mean([r['routing_mean'] for r in per_seed_results], axis=0)
    routing_std  = np.std ([r['routing_mean'] for r in per_seed_results], axis=0)
    delta_means = {role: float(np.mean([r['delta_max'][role] for r in per_seed_results]))
                   for role in ROLES}
    delta_stds  = {role: float(np.std ([r['delta_max'][role] for r in per_seed_results]))
                   for role in ROLES}
    return {
        'routing_mean': routing_mean,
        'routing_std':  routing_std,
        'delta_max_mean': delta_means,
        'delta_max_std':  delta_stds,
        'per_seed': per_seed_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=['3v3_balanced', '9v9_balanced'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--seed_map', type=str, default=None,
                    help='Per-config seeds, e.g. "3v3_balanced=0,1,2;9v9_balanced=0,1,3"')
    ap.add_argument('--step', type=int, default=None,
                    help='Specific checkpoint step (default: latest)')
    ap.add_argument('--config_file', type=str,
                    default=str(_ROOT / 'configs' / 'scaling.yaml'))
    ap.add_argument('--ckpt_root', type=str,
                    default=str(_ROOT / 'results_scaling' / 'checkpoints'))
    ap.add_argument('--out_dir', type=str,
                    default=str(_ROOT / 'results_scaling' / 'analysis'))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    config_file = Path(args.config_file)
    ckpt_root   = Path(args.ckpt_root)

    aggregated = {}
    csv_rows = [['config', 'seed', 'ckpt'] +
                [f'route_{ar}_{br}' for ar in ROLE_DISPLAY for br in ROLE_DISPLAY] +
                [f'delta_{r}' for r in ROLES]]

    seed_map = {}
    if args.seed_map:
        for part in args.seed_map.split(';'):
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            seed_map[k.strip()] = [int(s) for s in v.split(',') if s.strip()]

    for cfg_name in args.configs:
        print(f'\n=== {cfg_name} ===')
        per_seed = []
        seeds = seed_map.get(cfg_name, args.seeds)
        for seed in seeds:
            res = analyze_run(cfg_name, seed, args.step, config_file, ckpt_root, device)
            if res is None:
                continue
            per_seed.append(res)
            np.save(out_dir / f'routing_{cfg_name}_seed{seed}.npy',
                    res['routing_mean'])
            row = [cfg_name, seed, res['ckpt']]
            row += [f'{v:.4f}' for v in res['routing_mean'].flatten()]
            row += [f"{res['delta_max'][r]:.4f}" for r in ROLES]
            csv_rows.append(row)

        if not per_seed:
            print(f'  [warn] no results for {cfg_name}')
            continue
        agg = aggregate(per_seed)
        aggregated[cfg_name] = agg

        print(f'  routing matrix (seed-mean):')
        for i, r in enumerate(ROLE_DISPLAY):
            print('    ' + r + '  ' +
                  '  '.join(f'{v:.3f}' for v in agg['routing_mean'][i]))
        print(f'  delta_max: ' +
              ', '.join(f'{r}={agg["delta_max_mean"][r]:.3f}' for r in ROLES))
        # Strike→Support vs Strike→Vanguard inequality check
        ss = agg['routing_mean'][0, 2]
        sv = agg['routing_mean'][0, 1]
        ok = ss > sv
        print(f'  Strike→Support ({ss:.3f}) {">" if ok else "<="} Strike→Vanguard ({sv:.3f})  '
              f'[{"PASS" if ok else "FAIL"}]')

    if aggregated:
        plot_comparison(aggregated, out_dir / 'routing_3v3_vs_9v9.png',
                        title='Cross-attn role routing (Cond1)')
        plot_delta_bar(aggregated, out_dir / 'delta_max_3v3_vs_9v9.png')

    with open(out_dir / 'routing_summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)
    print(f'\nWrote: {out_dir}/routing_summary.csv, routing_3v3_vs_9v9.png, delta_max_3v3_vs_9v9.png')


if __name__ == '__main__':
    main()
