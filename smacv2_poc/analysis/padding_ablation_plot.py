"""
PLAN_V2 item D — plot for padding_ablation results.

Two-panel figure:
  (left)  KL per (cond, strategy), averaged over roles & seeds
          - shows architectural invariance of Cond1 vs Cond3 sensitivity
  (right) win-rate per (cond, strategy), averaged over seeds with std
          - shows that even best fill does not close the 3v3 gap
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
KL_PATH = _ROOT / 'results_scaling' / 'analysis' / 'padding_ablation_kl.csv'
WR_PATH = _ROOT / 'results_scaling' / 'analysis' / 'padding_ablation_winrate.csv'
OUT_PNG = _ROOT / 'results_scaling' / 'analysis' / 'padding_ablation.png'
OUT_SUM = _ROOT / 'results_scaling' / 'analysis' / 'padding_ablation_summary.csv'


def main():
    kl_rows = list(csv.DictReader(open(KL_PATH, encoding='utf-8')))
    wr_rows = list(csv.DictReader(open(WR_PATH, encoding='utf-8')))

    strategies = ['zeros', 'noise_small', 'noise_large', 'mean_valid', 'role_repeat']
    conditions = [1, 3]
    cond_label = {1: 'Cond1 (Full / ISA)', 3: 'Cond3 (RoleMLP)'}
    colors = {1: '#1f77b4', 3: '#d62728'}

    # KL: avg over seeds and roles
    kl_grouped = defaultdict(list)
    for r in kl_rows:
        kl_grouped[(int(r['condition']), r['strategy'])].append(float(r['mean_kl']))
    kl_mean = {k: float(np.mean(v)) for k, v in kl_grouped.items()}
    kl_std  = {k: float(np.std(v))  for k, v in kl_grouped.items()}

    # WR: avg over seeds
    wr_grouped = defaultdict(list)
    for r in wr_rows:
        wr_grouped[(int(r['condition']), r['strategy'])].append(float(r['win_rate']))
    wr_mean = {k: float(np.mean(v)) for k, v in wr_grouped.items()}
    wr_std  = {k: float(np.std(v))  for k, v in wr_grouped.items()}

    # summary CSV
    with open(OUT_SUM, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'strategy', 'kl_mean', 'kl_std',
                    'win_rate_mean', 'win_rate_std', 'n_seeds'])
        for c in conditions:
            for s in strategies:
                w.writerow([c, s,
                            f'{kl_mean[(c,s)]:.6f}', f'{kl_std[(c,s)]:.6f}',
                            f'{wr_mean[(c,s)]:.4f}', f'{wr_std[(c,s)]:.4f}',
                            len(wr_grouped[(c,s)])])

    fig, (ax_kl, ax_wr) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(strategies))
    width = 0.38

    for i, c in enumerate(conditions):
        kl_vals = [kl_mean[(c, s)] for s in strategies]
        kl_err  = [kl_std [(c, s)] for s in strategies]
        ax_kl.bar(x + (i - 0.5) * width, kl_vals, width,
                  yerr=kl_err, capsize=3,
                  label=cond_label[c], color=colors[c], alpha=0.85)
    ax_kl.set_xticks(x); ax_kl.set_xticklabels(strategies, rotation=15)
    ax_kl.set_ylabel(r'mean KL$\,(\pi_{zeros}\Vert \pi_{fill})$  per role per obs')
    ax_kl.set_title('Policy stability under padded-slot perturbation\n(lower = more invariant)')
    ax_kl.grid(axis='y', alpha=0.3)
    ax_kl.legend(loc='upper left', fontsize=9)

    for i, c in enumerate(conditions):
        wr_vals = [wr_mean[(c, s)] for s in strategies]
        wr_err  = [wr_std [(c, s)] for s in strategies]
        ax_wr.bar(x + (i - 0.5) * width, wr_vals, width,
                  yerr=wr_err, capsize=3,
                  label=cond_label[c], color=colors[c], alpha=0.85)
        for j, (m, s) in enumerate(zip(wr_vals, wr_err)):
            ax_wr.text(x[j] + (i - 0.5) * width, m + s + 0.02,
                       f'{m:.2f}', ha='center', va='bottom', fontsize=8)
    ax_wr.set_xticks(x); ax_wr.set_xticklabels(strategies, rotation=15)
    ax_wr.set_ylabel('Win rate (30 ep / seed, 3 seeds)')
    ax_wr.set_title('Win-rate ablation per fill strategy\n(3v3, deterministic argmax)')
    ax_wr.set_ylim(0, 1.05)
    ax_wr.grid(axis='y', alpha=0.3)
    ax_wr.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f'wrote {OUT_PNG}')
    print(f'wrote {OUT_SUM}')

    print('\n=== KL stability (mean over roles & seeds) ===')
    for c in conditions:
        for s in strategies:
            print(f'  cond{c} {s:14s} KL={kl_mean[(c,s)]:.4f}')
        print()
    print('=== Win rate (mean ± std over seeds) ===')
    for c in conditions:
        for s in strategies:
            print(f'  cond{c} {s:14s} WR={wr_mean[(c,s)]:.3f} ± {wr_std[(c,s)]:.3f}')
        print()

    # Padding-contribution bound
    wr_c1_zeros = wr_mean[(1, 'zeros')]
    wr_c3_zeros = wr_mean[(3, 'zeros')]
    gap_zeros = wr_c1_zeros - wr_c3_zeros
    best_c3_fill = max(wr_mean[(3, s)] for s in strategies)
    best_c3_strat = max(strategies, key=lambda s: wr_mean[(3, s)])
    recoverable = best_c3_fill - wr_c3_zeros
    pct = 100.0 * recoverable / max(gap_zeros, 1e-9)
    print('=== Padding-contribution bound ===')
    print(f'  Cond1 zeros : {wr_c1_zeros:.3f}')
    print(f'  Cond3 zeros : {wr_c3_zeros:.3f}    (training-time padding)')
    print(f'  Gap (zeros) : {gap_zeros:.3f}')
    print(f'  Best Cond3 fill: {best_c3_strat} -> {best_c3_fill:.3f}')
    print(f'  Recoverable by ideal fill: {recoverable:+.3f} ({pct:+.1f}% of gap)')
    print(f'  Architectural residual: {gap_zeros - recoverable:.3f}')


if __name__ == '__main__':
    main()
