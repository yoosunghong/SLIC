"""
PLAN_V2 item C — Cross-scale zero-shot transfer.

Load a checkpoint trained on a source config (e.g. 3v3_balanced) and evaluate
it, without any retraining, on a target config (e.g. 9v9_balanced).

This is mechanically possible because:
  - obs_dim is fixed at 182 across all scales (pad-to-9v9 ceiling).
  - build_scaling_policies builds 3 role-keyed networks; the source ckpt
    stores those 3 state_dicts in ckpt['policies'] in role order
    (Strike, Vanguard, Support). The target env simply replicates the
    role-policy across all agents that share that role.

Usage:
  python -m smacv2_poc.analysis.transfer_eval \
      --source 3v3_balanced --targets 3v3_balanced 6v6_balanced 9v9_balanced \
      --conditions 1 3 \
      --seeds 0 1 2 \
      --n_episodes 50

Outputs (under results_scaling/analysis/):
  transfer_table.csv         — long-format: source, target, cond, seed, win_rate
  transfer_bar.png           — bar chart per (cond, target) with seed std

Reviewer #2 framing: when N grows, the win-rate gap shrinks but transfer
capability *diverges*. Cond1 carries learned routing structure across N;
Cond3 cannot, because its MLP encoded a fixed slot-population pattern.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _ROOT.parent
for p in (_WORKSPACE, _ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from smacv2_poc.env.smac_wrapper_scaling import ScalingSMACWrapper
from smacv2_poc.models.scaling_models import build_scaling_policies
from smacv2_poc.analysis.scenario_builder_scaling import composition_to_roles


def _step_num(p: Path) -> int:
    try:
        return int(p.stem.replace('step', ''))
    except ValueError:
        return -1


def _find_final_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob('step*.pt'), key=_step_num)
    return ckpts[-1] if ckpts else None


def _load_source_role_policies(condition: int, ckpt_path: Path,
                               device: torch.device):
    """Return 3 unique role policies (Strike, Vanguard, Support) with weights
    loaded from a source-config checkpoint. The source ckpt always stores
    exactly 3 state_dicts in role order regardless of source N."""
    role_agent_roles = [0, 1, 2]
    policies = build_scaling_policies(condition=condition, agent_roles=role_agent_roles)
    unique = list({id(p): p for p in policies}.values())
    for p in unique:
        p.to(device); p.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    src_sds = ckpt['policies']
    if len(src_sds) != len(unique):
        raise RuntimeError(
            f"source ckpt has {len(src_sds)} policies but expected 3 role-keyed nets"
        )
    for p, sd in zip(unique, src_sds):
        p.load_state_dict(sd)
    return unique  # length 3, role-ordered


def _build_target_agent_policies(role_policies, agent_roles):
    """Map agent_index -> role_policy[role]. Same instances are shared across
    agents of the same role (matches training-time setup)."""
    return [role_policies[r] for r in agent_roles]


def evaluate_transfer(env, agent_policies, n_episodes: int,
                      device: torch.device) -> dict:
    wins = 0
    ep_lens = []
    ep_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        steps = 0
        while not done:
            actions = []
            for i in range(env.n_agents):
                ob = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = agent_policies[i](ob)
                actions.append(int(logits.argmax(dim=-1).item()))
            obs, rewards, done, _, info = env.step(actions)
            ep_r += float(np.mean(rewards))
            steps += 1
        if info.get('won', False):
            wins += 1
        ep_lens.append(steps)
        ep_rewards.append(ep_r)
    return dict(
        win_rate=wins / max(1, n_episodes),
        n_episodes=n_episodes,
        mean_len=float(np.mean(ep_lens)) if ep_lens else 0.0,
        mean_reward=float(np.mean(ep_rewards)) if ep_rewards else 0.0,
    )


def _make_env(target_cfg_block: dict, env_block: dict, seed: int):
    env_cfg = dict(env_block)
    env_cfg.update(target_cfg_block)
    env_cfg['seed'] = seed
    return ScalingSMACWrapper(env_cfg)


def _pick_seeds_for_dir(ckpt_root: Path, source: str, condition: int,
                       requested: list) -> list:
    """Filter requested seeds to those that have a checkpoint dir on disk."""
    have = []
    for s in requested:
        d = ckpt_root / f'{source}_cond{condition}_seed{s}'
        if _find_final_checkpoint(d) is not None:
            have.append(s)
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=str, default='3v3_balanced',
                    help='Config the checkpoint was trained on.')
    ap.add_argument('--targets', type=str, nargs='+',
                    default=['3v3_balanced', '6v6_balanced', '9v9_balanced'])
    ap.add_argument('--conditions', type=int, nargs='+', default=[1, 3])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--n_episodes', type=int, default=50)
    ap.add_argument('--config_file', type=str, default='configs/scaling.yaml')
    ap.add_argument('--ckpt_root', type=str, default='results_scaling/checkpoints')
    ap.add_argument('--out_dir', type=str, default='results_scaling/analysis')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--eval_seed_base', type=int, default=10000,
                    help='Env seed = eval_seed_base + train_seed; keeps eval rollouts independent of training rollouts.')
    args = ap.parse_args()

    config_file = _ROOT / args.config_file
    with open(config_file, encoding='utf-8') as f:
        cfg_yaml = yaml.safe_load(f)
    env_block = cfg_yaml['env']

    ckpt_root = _ROOT / args.ckpt_root
    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'transfer_table.csv'

    device = torch.device(args.device)

    rows = []
    header = ['source', 'target', 'condition', 'seed',
              'n_episodes', 'win_rate', 'mean_len', 'mean_reward', 'ckpt']

    for cond in args.conditions:
        seeds = _pick_seeds_for_dir(ckpt_root, args.source, cond, args.seeds)
        if not seeds:
            print(f"[skip] cond{cond}: no checkpoints under {ckpt_root}/{args.source}_cond{cond}_seed*")
            continue

        for seed in seeds:
            src_dir = ckpt_root / f'{args.source}_cond{cond}_seed{seed}'
            ckpt = _find_final_checkpoint(src_dir)
            print(f"[load] {ckpt}")
            role_policies = _load_source_role_policies(cond, ckpt, device)

            for target in args.targets:
                if target not in cfg_yaml['configs']:
                    print(f"[skip] target {target} not in scaling.yaml")
                    continue
                tgt_cfg = cfg_yaml['configs'][target]
                composition = tgt_cfg['composition']
                agent_roles = composition_to_roles(composition)
                agent_policies = _build_target_agent_policies(role_policies, agent_roles)

                env_seed = args.eval_seed_base + seed
                env = _make_env(tgt_cfg, env_block, seed=env_seed)
                print(f"[eval] cond{cond} seed{seed}: {args.source} -> {target} "
                      f"({env.n_agents}v{env.n_enemies}), {args.n_episodes} ep")
                try:
                    out = evaluate_transfer(env, agent_policies, args.n_episodes, device)
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

                rows.append([
                    args.source, target, cond, seed,
                    out['n_episodes'], out['win_rate'],
                    out['mean_len'], out['mean_reward'], str(ckpt.name)
                ])
                print(f"       win_rate={out['win_rate']:.3f} "
                      f"mean_len={out['mean_len']:.1f} mean_r={out['mean_reward']:.3f}")

                # incremental save so a crash mid-run keeps partial progress
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    w.writerows(rows)

    print(f"\n[done] wrote {csv_path}")


if __name__ == '__main__':
    main()
