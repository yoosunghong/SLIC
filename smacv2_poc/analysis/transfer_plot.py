"""
PLAN_V2 item C — bar chart for transfer_table.csv.

Reads results_scaling/analysis/transfer_table.csv and renders a grouped bar
chart of win rates per (condition, target), with seed std as error bars.
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = _ROOT / 'results_scaling' / 'analysis' / 'transfer_table.csv'
OUT_PNG  = _ROOT / 'results_scaling' / 'analysis' / 'transfer_bar.png'
OUT_SUM  = _ROOT / 'results_scaling' / 'analysis' / 'transfer_summary.csv'


def main():
    rows = []
    with open(CSV_PATH, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # group: (cond, target) -> list of win_rates over seeds
    grouped = defaultdict(list)
    for r in rows:
        key = (int(r['condition']), r['target'])
        grouped[key].append(float(r['win_rate']))

    targets = ['3v3_balanced', '6v6_balanced', '9v9_balanced']
    conditions = [1, 3]
    cond_label = {1: 'Cond1 (Full / ISA)', 3: 'Cond3 (RoleMLP)'}

    means = {c: [np.mean(grouped[(c, t)]) for t in targets] for c in conditions}
    stds  = {c: [np.std(grouped[(c, t)])  for t in targets] for c in conditions}

    # write summary CSV
    with open(OUT_SUM, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'target', 'mean_win_rate', 'std_win_rate', 'n_seeds'])
        for c in conditions:
            for i, t in enumerate(targets):
                w.writerow([c, t, f'{means[c][i]:.4f}', f'{stds[c][i]:.4f}',
                            len(grouped[(c, t)])])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(targets))
    width = 0.35
    colors = {1: '#1f77b4', 3: '#d62728'}

    for i, c in enumerate(conditions):
        ax.bar(x + (i - 0.5) * width, means[c], width,
               yerr=stds[c], capsize=4,
               label=cond_label[c], color=colors[c], alpha=0.85)
        for j, (m, s) in enumerate(zip(means[c], stds[c])):
            ax.text(x[j] + (i - 0.5) * width, m + s + 0.02,
                    f'{m:.2f}', ha='center', va='bottom', fontsize=9)

    # reference lines: 9v9 from-scratch (paper background)
    ax.axhline(0.15, color='#1f77b4', linestyle=':', alpha=0.5,
               label='9v9 Cond1 from-scratch (~0.15)')
    ax.axhline(0.08, color='#d62728', linestyle=':', alpha=0.5,
               label='9v9 Cond3 from-scratch (~0.08)')

    ax.set_xticks(x)
    ax.set_xticklabels(['3v3 (ID)', '6v6', '9v9'])
    ax.set_ylabel('Win rate (50 ep / seed, 3 seeds)')
    ax.set_title('Cross-scale zero-shot transfer from 3v3 checkpoint')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f'wrote {OUT_PNG}')
    print(f'wrote {OUT_SUM}')

    print('\n=== Summary ===')
    for c in conditions:
        for i, t in enumerate(targets):
            print(f'  cond{c} -> {t:14s}  {means[c][i]:.3f} ± {stds[c][i]:.3f}')


if __name__ == '__main__':
    main()
