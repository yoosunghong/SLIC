"""
PLAN_V2 item A' — Unified routing metric (occlusion-based).

Reviewer #3 fix: Cond1 (attention) vs Cond3 (gradient) is not apples-to-apples.
Solution: a single metric applied identically to BOTH models — per-ally-slot
occlusion, measured by KL divergence between the unmasked policy distribution
and the policy distribution after the slot's tokens are zeroed (with mask=1).
Aggregate per (agent_role, ally_role) into a 3×3 sensitivity matrix.

Pearson-r between occlusion-routing (Cond1) and attention-routing (Cond1)
checks that occlusion is a faithful proxy for attention; the same metric on
Cond3 yields a fair comparison without measurement-unit mismatch.

Output:
  results_scaling/analysis/occlusion_<config>_seed<N>.npy
  results_scaling/analysis/occlusion_routing_summary.csv
  results_scaling/analysis/occlusion_routing_panel.png
  results_scaling/analysis/occlusion_vs_attention_corr.txt

Usage:
  python -m smacv2_poc.analysis.occlusion_routing \
      --configs 3v3_balanced 9v9_balanced \
      --seed_map "3v3_balanced=0,1,2;9v9_balanced=0,1,3"
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _ROOT.parent
for p in (_WORKSPACE, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from smacv2_poc.models.scaling_models import (
    build_scaling_policies, OBS_DIM,
    _SELF_END, _ALLY_END, _ZONE_END, _AMASK_END,
    ALLY_DIM, N_ALLY_SLOTS as MAX_ALLY_SLOTS,
)
from smacv2_poc.analysis.scenario_builder_scaling import (
    build_scenario_obs_scaling, build_random_scenarios,
    composition_to_roles, ROLES,
)

ROLE_DISPLAY = ['Strike', 'Vanguard', 'Support']


def _load_config(config_name: str, config_file: Path) -> dict:
    with open(config_file, encoding='utf-8') as f:
        return yaml.safe_load(f)['configs'][config_name]


def _step_num(p):
    try:
        return int(p.stem.replace('step', ''))
    except ValueError:
        return -1


def _find_checkpoint(ckpt_dir: Path, step: int = None) -> Path:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob('step*.pt'), key=_step_num)
    if not ckpts:
        return None
    if step is None:
        return ckpts[-1]
    for c in ckpts:
        if f'step{step}.pt' in c.name:
            return c
    return ckpts[-1]


def load_policies(condition: int, ckpt_path: Path, agent_roles: list,
                  device: torch.device):
    policies = build_scaling_policies(condition=condition, agent_roles=agent_roles)
    unique = list({id(p): p for p in policies}.values())
    for p in unique:
        p.to(device); p.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for p, sd in zip(unique, ckpt['policies']):
        p.load_state_dict(sd)
    return policies, unique


def _occlude_ally_slot(obs_np: np.ndarray, slot_idx: int) -> np.ndarray:
    """
    Zero ally slot tokens AND set its mask bit to 1 (padded).
    This is the cleanest occlusion: the model sees the slot as absent,
    consistent with how padded slots are treated during training.
    """
    o = obs_np.copy()
    base = _SELF_END + slot_idx * ALLY_DIM
    o[base:base + ALLY_DIM] = 0.0
    o[_ZONE_END + slot_idx] = 1.0     # mark as padded
    return o


def _logits(policy, obs_np: np.ndarray, device: torch.device) -> torch.Tensor:
    obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy(obs)
    return logits.squeeze(0)


def occlusion_routing_matrix(unique_policies, scenario_obs: dict,
                              device: torch.device) -> np.ndarray:
    """
    3×3 occlusion-sensitivity matrix.
    Entry (agent_role, ally_role) =
       average KL( π(·|obs) || π(·|obs with slot occluded) )
       over all ally slots whose ally has role==ally_role.
    """
    matrix = np.zeros((3, 3), dtype=np.float32)
    counts = np.zeros((3, 3), dtype=np.float32)

    slot_roles_by_role = scenario_obs['ally_slot_roles']

    for role_idx, role_name in enumerate(ROLES):
        policy = unique_policies[role_idx]
        base_obs = scenario_obs[role_name]
        base_logits = _logits(policy, base_obs, device)
        base_logp = F.log_softmax(base_logits, dim=-1)
        base_p = base_logp.exp()

        slot_roles = slot_roles_by_role[role_name]
        for slot_idx, ally_role in enumerate(slot_roles):
            if slot_idx >= MAX_ALLY_SLOTS:
                break
            occ_obs = _occlude_ally_slot(base_obs, slot_idx)
            occ_logits = _logits(policy, occ_obs, device)
            occ_logp = F.log_softmax(occ_logits, dim=-1)
            kl = float((base_p * (base_logp - occ_logp)).sum().item())
            kl = max(kl, 0.0)
            matrix[role_idx, ally_role] += kl
            counts[role_idx, ally_role] += 1

    counts = np.where(counts == 0, 1, counts)
    return matrix / counts


def attention_routing_matrix(unique_policies, scenario_obs: dict,
                              device: torch.device) -> np.ndarray:
    """For Cond1 only — duplicate of analyze_attention_scaling logic for in-script
    correlation against occlusion. Kept self-contained on purpose."""
    matrix = np.zeros((3, 3), dtype=np.float32)
    counts = np.zeros((3, 3), dtype=np.float32)
    slot_roles_by_role = scenario_obs['ally_slot_roles']
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
        slot_roles = slot_roles_by_role[role_name]
        for slot_idx, ally_role in enumerate(slot_roles):
            if slot_idx >= len(ca):
                break
            matrix[role_idx, ally_role] += ca[slot_idx]
            counts[role_idx, ally_role] += 1
    counts = np.where(counts == 0, 1, counts)
    return matrix / counts


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    av, bv = a.flatten(), b.flatten()
    if av.std() == 0 or bv.std() == 0:
        return float('nan')
    return float(np.corrcoef(av, bv)[0, 1])


def analyze_one(config_name: str, condition: int, seed: int,
                config_file: Path, ckpt_root: Path,
                device: torch.device,
                n_random_scenarios: int = 20,
                random_seed: int = 0) -> dict:
    cfg = _load_config(config_name, config_file)
    composition = cfg['composition']
    n_enemies = cfg.get('n_enemies', sum(composition.values()))
    agent_roles = composition_to_roles(composition)

    cond_tag = f'cond{condition}'
    ckpt_dir = ckpt_root / f'{config_name}_{cond_tag}_seed{seed}'
    ckpt = _find_checkpoint(ckpt_dir, step=None)
    if ckpt is None:
        return None

    policies, unique = load_policies(condition, ckpt, agent_roles, device)

    # Anchor scenarios (kept for backward compatibility / interpretability)
    clustered = build_scenario_obs_scaling('clustered', composition, n_enemies)
    spread    = build_scenario_obs_scaling('spread',    composition, n_enemies)
    occ_c = occlusion_routing_matrix(unique, clustered, device)
    occ_s = occlusion_routing_matrix(unique, spread,    device)

    # Multi-scenario averaging — stabilizes Strike row at scale.
    rand_scenarios = build_random_scenarios(
        n_random_scenarios, composition, n_enemies, seed=random_seed
    )
    occ_rand = np.mean(
        [occlusion_routing_matrix(unique, s, device) for s in rand_scenarios],
        axis=0,
    )
    # Combined: 2 anchors + N random, equally weighted within their groups.
    occ = (occ_c + occ_s) / 2.0 * 0.5 + occ_rand * 0.5

    out = {
        'config': config_name, 'condition': condition, 'seed': seed,
        'ckpt': ckpt.name,
        'occlusion_mean': occ,                  # combined (anchors + random)
        'occlusion_random_only': occ_rand,
        'occlusion_clustered': occ_c, 'occlusion_spread': occ_s,
    }
    if condition == 1:
        att_c = attention_routing_matrix(unique, clustered, device)
        att_s = attention_routing_matrix(unique, spread,    device)
        att_rand = np.mean(
            [attention_routing_matrix(unique, s, device) for s in rand_scenarios],
            axis=0,
        )
        att = (att_c + att_s) / 2.0 * 0.5 + att_rand * 0.5
        out['attention_mean'] = att
        out['attention_random_only'] = att_rand
        # Pearson r on raw matrix and on row-normalized matrix
        out['pearson_r_occ_vs_attn'] = _pearson_r(occ, att)
        out['pearson_r_occ_vs_attn_rownorm'] = _pearson_r(
            _row_normalize(occ), _row_normalize(att)
        )
    return out


def _row_normalize(m: np.ndarray) -> np.ndarray:
    """Normalize so each row sums to 1 (for visualization/comparison)."""
    s = m.sum(axis=1, keepdims=True)
    s = np.where(s <= 1e-9, 1.0, s)
    return m / s


def plot_panel(panel: dict, out_path: Path):
    """
    panel layout (rows × cols):
      rows = configs, cols = (Cond1 occ, Cond1 attn, Cond3 occ, |Cond1 occ - Cond3 occ|)
    """
    import matplotlib.pyplot as plt
    cfgs = list(panel.keys())
    col_titles = ['Cond1 occlusion', 'Cond1 attention',
                   'Cond3 occlusion', 'Δ(Cond1−Cond3) occ']
    fig, axes = plt.subplots(len(cfgs), 4,
                              figsize=(13.5, 3.4 * len(cfgs)),
                              constrained_layout=True)
    if len(cfgs) == 1:
        axes = np.array([axes])

    for ri, cfg_name in enumerate(cfgs):
        d = panel[cfg_name]
        c1_occ_n = _row_normalize(d['cond1_occ'])
        c1_att_n = _row_normalize(d['cond1_att'])
        c3_occ_n = _row_normalize(d['cond3_occ']) if d['cond3_occ'] is not None else np.zeros((3, 3))
        diff = c1_occ_n - c3_occ_n if d['cond3_occ'] is not None else np.zeros((3, 3))
        mats = [c1_occ_n, c1_att_n, c3_occ_n, np.abs(diff)]

        for ci, (m, ttl) in enumerate(zip(mats, col_titles)):
            ax = axes[ri, ci]
            im = ax.imshow(m, cmap='magma', vmin=0, vmax=max(0.6, m.max() * 1.05))
            ax.set_xticks(range(3)); ax.set_xticklabels(ROLE_DISPLAY, rotation=20, fontsize=8)
            ax.set_yticks(range(3)); ax.set_yticklabels(ROLE_DISPLAY, fontsize=8)
            ax.set_title(f'{cfg_name} | {ttl}', fontsize=9)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center',
                             color='white' if m[i, j] < m.max() * 0.55 else 'black',
                             fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle('Occlusion-based unified routing metric  (row-normalized)')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate(per_seed: list, key: str) -> tuple:
    arrs = [r[key] for r in per_seed if key in r and r[key] is not None]
    if not arrs:
        return None, None
    return np.mean(arrs, axis=0), np.std(arrs, axis=0)


def _structure_score(m: np.ndarray) -> float:
    """How 'sharp' is row-normalized routing? Use mean row entropy gap from uniform."""
    rn = _row_normalize(m)
    eps = 1e-9
    H = -np.sum(rn * np.log(rn + eps), axis=1)
    return float(np.log(3) - H.mean())  # higher = sharper (less uniform)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=['3v3_balanced', '9v9_balanced'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--seed_map', type=str, default=None)
    ap.add_argument('--config_file', type=str,
                    default=str(_ROOT / 'configs' / 'scaling.yaml'))
    ap.add_argument('--ckpt_root', type=str,
                    default=str(_ROOT / 'results_scaling' / 'checkpoints'))
    ap.add_argument('--out_dir', type=str,
                    default=str(_ROOT / 'results_scaling' / 'analysis'))
    ap.add_argument('--n_random_scenarios', type=int, default=20)
    ap.add_argument('--random_seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    config_file = Path(args.config_file); ckpt_root = Path(args.ckpt_root)

    seed_map = {}
    if args.seed_map:
        for part in args.seed_map.split(';'):
            if '=' not in part:
                continue
            k, v = part.split('=', 1)
            seed_map[k.strip()] = [int(s) for s in v.split(',') if s.strip()]

    panel = {}
    csv_rows = [['config', 'cond', 'seed', 'ckpt',
                  'pearson_r_occ_vs_attn',
                  'structure_score'] +
                 [f'occ_{ar}_{br}' for ar in ROLE_DISPLAY for br in ROLE_DISPLAY]]

    pearson_lines = []

    for cfg_name in args.configs:
        print(f'\n=== {cfg_name} ===')
        seeds = seed_map.get(cfg_name, args.seeds)

        cond1_seed_results, cond3_seed_results = [], []
        for cond, store in [(1, cond1_seed_results), (3, cond3_seed_results)]:
            for seed in seeds:
                r = analyze_one(cfg_name, cond, seed, config_file, ckpt_root, device,
                                n_random_scenarios=args.n_random_scenarios,
                                random_seed=args.random_seed)
                if r is None:
                    print(f'  [skip] cond{cond} seed{seed} - no checkpoint')
                    continue
                store.append(r)
                np.save(out_dir / f'occlusion_{cfg_name}_cond{cond}_seed{seed}.npy',
                         r['occlusion_mean'])
                row = [cfg_name, cond, seed, r['ckpt'],
                        f'{r.get("pearson_r_occ_vs_attn", float("nan")):.4f}',
                        f'{_structure_score(r["occlusion_mean"]):.4f}']
                row += [f'{v:.4f}' for v in r['occlusion_mean'].flatten()]
                csv_rows.append(row)
                if cond == 1 and 'pearson_r_occ_vs_attn' in r:
                    pearson_lines.append(
                        f'{cfg_name} seed{seed}: r(occ,attn)='
                        f'{r["pearson_r_occ_vs_attn"]:.3f}'
                    )

        c1_occ_mean, _ = aggregate(cond1_seed_results, 'occlusion_mean')
        c1_att_mean, _ = aggregate(cond1_seed_results, 'attention_mean')
        c3_occ_mean, _ = aggregate(cond3_seed_results, 'occlusion_mean')

        if c1_occ_mean is not None:
            print(f'  Cond1 occlusion (seed-mean):')
            for i, r in enumerate(ROLE_DISPLAY):
                print('    ' + r + '  ' +
                      '  '.join(f'{v:.4f}' for v in c1_occ_mean[i]))
            print(f'  Cond1 occ structure score: {_structure_score(c1_occ_mean):.3f}')

        if c1_att_mean is not None and c1_occ_mean is not None:
            r_seed_mean = _pearson_r(c1_occ_mean, c1_att_mean)
            r_rownorm = _pearson_r(_row_normalize(c1_occ_mean),
                                    _row_normalize(c1_att_mean))
            print(f'  Pearson r (occ, attn) on seed-mean : raw={r_seed_mean:.3f}, '
                   f'row-normalized={r_rownorm:.3f}')
            pearson_lines.append(
                f'{cfg_name} seed-mean: r(occ,attn) raw={r_seed_mean:.3f}, '
                f'row-norm={r_rownorm:.3f}'
            )

        if c3_occ_mean is not None:
            print(f'  Cond3 occlusion (seed-mean):')
            for i, r in enumerate(ROLE_DISPLAY):
                print('    ' + r + '  ' +
                      '  '.join(f'{v:.4f}' for v in c3_occ_mean[i]))
            print(f'  Cond3 occ structure score: {_structure_score(c3_occ_mean):.3f}')

            sharper = _structure_score(c1_occ_mean) > _structure_score(c3_occ_mean)
            print(f'  Cond1 sharper than Cond3? {"YES" if sharper else "NO"}')

        # Strike->Support vs Strike->Vanguard inequality (occlusion)
        if c1_occ_mean is not None:
            ss = c1_occ_mean[0, 2]; sv = c1_occ_mean[0, 1]
            print(f'  Cond1 occ raw     Strike->Support ({ss:.4f}) '
                   f'{">" if ss > sv else "<="} Strike->Vanguard ({sv:.4f}) '
                   f'[{"PASS" if ss > sv else "FAIL"}]')
            rn = _row_normalize(c1_occ_mean)
            ssn, svn = rn[0, 2], rn[0, 1]
            print(f'  Cond1 occ row-norm Strike->Support ({ssn:.4f}) '
                   f'{">" if ssn > svn else "<="} Strike->Vanguard ({svn:.4f}) '
                   f'[{"PASS" if ssn > svn else "FAIL"}]')

        panel[cfg_name] = {
            'cond1_occ': c1_occ_mean if c1_occ_mean is not None else np.zeros((3, 3)),
            'cond1_att': c1_att_mean if c1_att_mean is not None else np.zeros((3, 3)),
            'cond3_occ': c3_occ_mean,
        }

    if panel:
        plot_panel(panel, out_dir / 'occlusion_routing_panel.png')

    with open(out_dir / 'occlusion_routing_summary.csv', 'w', newline='') as f:
        csv.writer(f).writerows(csv_rows)
    with open(out_dir / 'occlusion_vs_attention_corr.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(pearson_lines) + '\n')
    print(f'\nWrote: {out_dir}/occlusion_routing_summary.csv, '
          f'occlusion_routing_panel.png, occlusion_vs_attention_corr.txt')


if __name__ == '__main__':
    main()
