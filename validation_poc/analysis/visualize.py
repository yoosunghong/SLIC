"""All plot generation for Track 2 analysis."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_self_attn_heatmap(weights_clustered: np.ndarray, weights_spread: np.ndarray,
                            role_name: str, save_dir: str):
    """
    weights_*: shape (n_slots, n_slots) — averaged over heads.
    Saves side-by-side heatmap + difference map.
    """
    os.makedirs(save_dir, exist_ok=True)
    diff = weights_clustered - weights_spread

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Clustered', 'Spread', 'Difference (C-S)']
    data = [weights_clustered, weights_spread, diff]
    cmaps = ['Blues', 'Blues', 'RdBu_r']

    for ax, d, title, cmap in zip(axes, data, titles, cmaps):
        vmax = max(abs(d).max(), 0.01)
        vmin = -vmax if cmap == 'RdBu_r' else 0
        sns.heatmap(d, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, fmt='.3f', cbar=True,
                    xticklabels=[f'Ally{k}' for k in range(d.shape[1])],
                    yticklabels=[f'Ally{k}' for k in range(d.shape[0])])
        ax.set_title(f'{role_name} — {title}')
        ax.set_xlabel('Key (attended)')
        ax.set_ylabel('Query')

    plt.tight_layout()
    fname = os.path.join(save_dir, f'self_attn_heatmap_{role_name.lower()}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_role_routing_matrix(routing_matrix: np.ndarray, condition_name: str, save_dir: str):
    """
    routing_matrix: (3, 3) — rows=agent role, cols=attended ally role
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(routing_matrix, ax=ax, cmap='YlOrRd', vmin=0, vmax=routing_matrix.max(),
                annot=True, fmt='.3f', cbar=True,
                xticklabels=['Strike', 'Vanguard', 'Support'],
                yticklabels=['Strike', 'Vanguard', 'Support'])
    ax.set_title(f'Role Routing Matrix — {condition_name}')
    ax.set_xlabel('Attended Ally Role')
    ax.set_ylabel('Agent Role')
    plt.tight_layout()
    fname = os.path.join(save_dir, f'role_routing_matrix_{condition_name.lower().replace(" ", "_")}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_delta_max_comparison(delta_max_cond1: dict, delta_max_cond2: dict, save_dir: str):
    """
    delta_max_*: dict with keys 'strike', 'vanguard', 'support'
    """
    os.makedirs(save_dir, exist_ok=True)
    roles = ['strike', 'vanguard', 'support']
    x = np.arange(len(roles))
    width = 0.35

    vals1 = [delta_max_cond1.get(r, 0.0) for r in roles]
    vals2 = [delta_max_cond2.get(r, 0.0) for r in roles]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, vals1, width, label='Cond1 (Full)', color='steelblue')
    ax.bar(x + width / 2, vals2, width, label='Cond2 (No Self-Attn)', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in roles])
    ax.set_ylabel('Δmax (formation sensitivity)')
    ax.set_title('Formation Sensitivity by Role and Condition')
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(save_dir, 'delta_max_comparison.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_win_rate_curves(results_dir: str, save_dir: str):
    """Load CSVs for all conditions/seeds and plot mean win rate curves."""
    import csv
    import glob

    os.makedirs(save_dir, exist_ok=True)
    condition_names = {1: 'Full', 2: 'No Self-Attn', 3: 'Role+MLP', 4: 'No Role Label', 5: 'Flat MLP'}
    colors = ['steelblue', 'tomato', 'green', 'purple', 'orange']

    fig, ax = plt.subplots(figsize=(10, 6))

    for cond in range(1, 6):
        pattern = os.path.join(results_dir, f'cond{cond}_seed*.csv')
        files = glob.glob(pattern)
        if not files:
            continue
        all_data = {}
        for fpath in files:
            with open(fpath) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step = int(row['step'])
                    wr = float(row['win_rate'])
                    if step not in all_data:
                        all_data[step] = []
                    all_data[step].append(wr)
        if not all_data:
            continue
        steps = sorted(all_data.keys())
        means = [np.mean(all_data[s]) for s in steps]
        stds = [np.std(all_data[s]) for s in steps]
        means = np.array(means)
        stds = np.array(stds)
        color = colors[cond - 1]
        ax.plot(steps, means, label=f'Cond{cond}: {condition_names[cond]}', color=color)
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate Curves — All Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(save_dir, 'win_rate_curves.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')


def plot_gradient_attribution(role_name: str, spatial_grad: float, role_grad: float, save_dir: str):
    """Bar chart comparing spatial vs role gradient magnitude."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    categories = ['Spatial features', 'Role one-hot']
    values = [spatial_grad, role_grad]
    bars = ax.bar(categories, values, color=['steelblue', 'tomato'])
    ax.set_ylabel('Mean |gradient|')
    ax.set_title(f'Gradient Attribution — {role_name} (Cond3 MLP)')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    fname = os.path.join(save_dir, f'gradient_attribution_{role_name.lower()}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fname}')
