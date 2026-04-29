"""
PLAN_V2 item D — Padding-confound separation (Option 1, inference-only).

Question: Of the 3v3 Cond1-vs-Cond3 win-rate gap, how much is the padding
penalty (Cond3 MLP must process zero-padded ally slots 2-7) vs the true
architectural gap?

Two measurements on the same 3v3 checkpoints:

  (1) Policy-stability (KL):
      For a fixed sample of in-distribution observations, replace the
      contents of padded ally-slots with several fill strategies, keeping
      the mask bit = 1. Measure KL( pi_zeros || pi_alt ).
      Cond1 should be invariant (mask blocks attention).
      Cond3's KL bounds how much "signal" the MLP extracts from padding.

  (2) Win-rate ablation:
      Re-run env evaluation with on-the-fly post-processing of every obs
      to install the chosen fill into padded ally-slots. Compare win rates
      across {zeros, noise, mean_valid, role_repeat}.
      Delta(Cond3 fill - Cond3 zeros) bounds the recoverable padding penalty;
      delta(Cond1 fill - Cond1 zeros) confirms the architectural invariance.

Outputs (results_scaling/analysis/):
  padding_ablation_kl.csv         per (condition, seed, agent_role, fill) KL
  padding_ablation_winrate.csv    per (condition, seed, fill) win_rate
  padding_ablation.png            two-panel summary

Usage:
  python -m smacv2_poc.analysis.padding_ablation \
      --config 3v3_balanced --seeds 0 1 2 \
      --kl_obs 256 --n_episodes 30
"""
import argparse
import csv
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

from smacv2_poc.env.smac_wrapper_scaling import ScalingSMACWrapper
from smacv2_poc.models.scaling_models import (
    build_scaling_policies, OBS_DIM,
    _SELF_END, _ALLY_END, _ZONE_END, _AMASK_END,
    ALLY_DIM, N_ALLY_SLOTS,
)
from smacv2_poc.analysis.scenario_builder_scaling import composition_to_roles


# ---------------------------------------------------------------------------
# Fill strategies — mutate ally-slot tokens for slots flagged as padded.
# ---------------------------------------------------------------------------

def _ally_slot_view(obs: np.ndarray, slot_idx: int) -> slice:
    base = _SELF_END + slot_idx * ALLY_DIM
    return slice(base, base + ALLY_DIM)


def _ally_mask(obs: np.ndarray) -> np.ndarray:
    return obs[_ZONE_END:_AMASK_END]   # 8 entries, 1 = padded


def fill_obs(obs: np.ndarray, strategy: str, rng: np.random.Generator) -> np.ndarray:
    """Return a new obs with padded ally-slot tokens replaced per strategy.
    Mask bits are NOT touched — they remain 1 for padded slots so Cond1
    still treats those slots as absent (the architectural property under test).
    """
    o = obs.copy()
    mask = _ally_mask(o)                         # 1=padded
    padded = np.where(mask > 0.5)[0]
    valid  = np.where(mask < 0.5)[0]

    if strategy == 'zeros' or len(padded) == 0:
        return o

    if strategy == 'noise_small':
        for s in padded:
            o[_ally_slot_view(o, s)] = rng.normal(0.0, 0.05, size=ALLY_DIM).astype(np.float32)
        return o

    if strategy == 'noise_large':
        for s in padded:
            o[_ally_slot_view(o, s)] = rng.normal(0.0, 0.20, size=ALLY_DIM).astype(np.float32)
        return o

    if strategy == 'mean_valid':
        if len(valid) == 0:
            return o
        vsum = np.zeros(ALLY_DIM, dtype=np.float32)
        for s in valid:
            vsum += o[_ally_slot_view(o, s)]
        vmean = vsum / len(valid)
        for s in padded:
            o[_ally_slot_view(o, s)] = vmean
        return o

    if strategy == 'role_repeat':
        if len(valid) == 0:
            return o
        for s in padded:
            pick = int(rng.choice(valid))
            o[_ally_slot_view(o, s)] = o[_ally_slot_view(o, pick)]
        return o

    raise ValueError(f"unknown fill strategy: {strategy}")


FILL_STRATEGIES = ['zeros', 'noise_small', 'noise_large', 'mean_valid', 'role_repeat']


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

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


def _load_role_policies(condition: int, ckpt_path: Path, device: torch.device):
    role_agent_roles = [0, 1, 2]
    policies = build_scaling_policies(condition=condition, agent_roles=role_agent_roles)
    unique = list({id(p): p for p in policies}.values())
    for p in unique:
        p.to(device); p.eval()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for p, sd in zip(unique, ckpt['policies']):
        p.load_state_dict(sd)
    return unique  # role-ordered Strike, Vanguard, Support


# ---------------------------------------------------------------------------
# (1) KL stability over collected observations
# ---------------------------------------------------------------------------

def collect_observations(env, role_policies, agent_roles, n_obs: int,
                         device: torch.device) -> dict:
    """Roll out greedy policies until n_obs per agent_role observations are
    collected. Return {role_idx: list[np.ndarray]}."""
    bucket = {0: [], 1: [], 2: []}
    target = n_obs
    while min(len(v) for v in bucket.values()) < target:
        obs, _ = env.reset()
        done = False
        while not done:
            actions = []
            for i in range(env.n_agents):
                r = agent_roles[i]
                if len(bucket[r]) < target:
                    bucket[r].append(obs[i].copy())
                ob = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = role_policies[r](ob)
                actions.append(int(logits.argmax(dim=-1).item()))
            obs, _, done, _, _ = env.step(actions)
            if all(len(v) >= target for v in bucket.values()):
                break
    return {k: v[:target] for k, v in bucket.items()}


def kl_under_fill(role_policies, obs_bucket: dict, strategy: str,
                  rng: np.random.Generator, device: torch.device) -> dict:
    """Mean KL( pi(zeros) || pi(strategy) ) per role over the collected obs."""
    out = {}
    for r, obs_list in obs_bucket.items():
        policy = role_policies[r]
        # batch
        ob_z = np.stack(obs_list, axis=0)
        ob_a = np.stack([fill_obs(o, strategy, rng) for o in obs_list], axis=0)
        with torch.no_grad():
            lz, _ = policy(torch.tensor(ob_z, dtype=torch.float32, device=device))
            la, _ = policy(torch.tensor(ob_a, dtype=torch.float32, device=device))
        logp_z = F.log_softmax(lz, dim=-1)
        logp_a = F.log_softmax(la, dim=-1)
        p_z = logp_z.exp()
        kl = (p_z * (logp_z - logp_a)).sum(dim=-1)        # per-obs KL
        out[r] = float(kl.mean().item())
    return out


# ---------------------------------------------------------------------------
# (2) Win-rate ablation under fill
# ---------------------------------------------------------------------------

def evaluate_with_fill(env, role_policies, agent_roles, strategy: str,
                      n_episodes: int, rng: np.random.Generator,
                      device: torch.device) -> float:
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        if strategy != 'zeros':
            obs = [fill_obs(o, strategy, rng) for o in obs]
        done = False
        while not done:
            actions = []
            for i in range(env.n_agents):
                ob = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = role_policies[agent_roles[i]](ob)
                actions.append(int(logits.argmax(dim=-1).item()))
            obs, _, done, _, info = env.step(actions)
            if strategy != 'zeros':
                obs = [fill_obs(o, strategy, rng) for o in obs]
        if info.get('won', False):
            wins += 1
    return wins / max(1, n_episodes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='3v3_balanced')
    ap.add_argument('--conditions', type=int, nargs='+', default=[1, 3])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--strategies', type=str, nargs='+', default=FILL_STRATEGIES)
    ap.add_argument('--kl_obs', type=int, default=256,
                    help='observations per role for KL stability measurement')
    ap.add_argument('--n_episodes', type=int, default=30,
                    help='episodes per (cond, seed, strategy) for win-rate ablation')
    ap.add_argument('--config_file', type=str, default='configs/scaling.yaml')
    ap.add_argument('--ckpt_root', type=str, default='results_scaling/checkpoints')
    ap.add_argument('--out_dir', type=str, default='results_scaling/analysis')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--rng_seed', type=int, default=42)
    ap.add_argument('--eval_seed_base', type=int, default=20000)
    args = ap.parse_args()

    cfg_file = _ROOT / args.config_file
    with open(cfg_file, encoding='utf-8') as f:
        cfg_yaml = yaml.safe_load(f)
    env_block = cfg_yaml['env']
    tgt_cfg = cfg_yaml['configs'][args.config]
    composition = tgt_cfg['composition']
    agent_roles = composition_to_roles(composition)

    ckpt_root = _ROOT / args.ckpt_root
    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.rng_seed)

    kl_rows = []
    wr_rows = []
    kl_header = ['condition', 'seed', 'role', 'strategy', 'mean_kl']
    wr_header = ['condition', 'seed', 'strategy', 'n_episodes', 'win_rate']
    role_names = ['Strike', 'Vanguard', 'Support']

    for cond in args.conditions:
        for seed in args.seeds:
            ckpt_dir = ckpt_root / f'{args.config}_cond{cond}_seed{seed}'
            ckpt = _find_final_checkpoint(ckpt_dir)
            if ckpt is None:
                print(f"[skip] no checkpoint at {ckpt_dir}")
                continue
            print(f"\n[load] cond{cond} seed{seed}: {ckpt}")
            role_policies = _load_role_policies(cond, ckpt, device)

            # collect observations once per (cond, seed) for KL stability
            env_cfg = dict(env_block); env_cfg.update(tgt_cfg)
            env_cfg['seed'] = args.eval_seed_base + seed
            env = ScalingSMACWrapper(env_cfg)
            try:
                print(f"  collecting {args.kl_obs} obs/role for KL...")
                bucket = collect_observations(env, role_policies, agent_roles,
                                              args.kl_obs, device)

                for strat in args.strategies:
                    kl_per_role = kl_under_fill(role_policies, bucket, strat, rng, device)
                    for r in range(3):
                        kl_rows.append([cond, seed, role_names[r], strat,
                                        f'{kl_per_role[r]:.6f}'])
                    print(f"  KL[{strat:12s}] " + ' '.join(
                        f'{role_names[r]}={kl_per_role[r]:.4f}' for r in range(3)))

                for strat in args.strategies:
                    print(f"  win-rate eval [{strat:12s}] {args.n_episodes} ep ...", end=' ', flush=True)
                    wr = evaluate_with_fill(env, role_policies, agent_roles, strat,
                                            args.n_episodes, rng, device)
                    wr_rows.append([cond, seed, strat, args.n_episodes, f'{wr:.4f}'])
                    print(f"win_rate={wr:.3f}")
            finally:
                try: env.close()
                except Exception: pass

            # incremental save
            with open(out_dir / 'padding_ablation_kl.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(kl_header); w.writerows(kl_rows)
            with open(out_dir / 'padding_ablation_winrate.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f); w.writerow(wr_header); w.writerows(wr_rows)

    print(f"\n[done] wrote padding_ablation_kl.csv, padding_ablation_winrate.csv")


if __name__ == '__main__':
    main()
