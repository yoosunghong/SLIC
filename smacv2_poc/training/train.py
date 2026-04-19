"""
SMACv2 training entry point.

Usage:
  python train.py --condition 1 --seed 0
  python train.py --condition 6 --seed 0   # MAPPO Vanilla flat-obs baseline

Conditions 1-5 mirror the MiniGrid ablation exactly.
Condition 6: MAPPO with flat raw SMACv2 obs (no entity-centric reformatting).
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Make both repo packages importable when running this script directly.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKSPACE = os.path.dirname(_ROOT)
for path in (_WORKSPACE, _ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

_VP = os.path.join(_WORKSPACE, 'validation_poc')

from env.smac_wrapper import SMACWrapper
from models import build_policies, CentralizedCritic
from baselines.mappo_flat import FlatObsMLPModel, build_flat_policies
from validation_poc.training.rollout import collect_rollout
from validation_poc.training.mappo import MAPPOTrainer


def evaluate(env, policies, n_episodes: int, device: torch.device) -> float:
    wins = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = []
            for i in range(env.n_agents):
                ob = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = policies[i](ob)
                action = logits.argmax(dim=-1).item()
                actions.append(action)
            obs, _, done, _, info = env.step(actions)
        if info.get('won', False):
            wins += 1
    return wins / n_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='configs/smacv2.yaml')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pt file to resume training from')
    args = parser.parse_args()

    config_path = os.path.join(_ROOT, args.config)
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg['env']
    train_cfg = cfg['training']
    model_cfg = cfg['model']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    res_cfg = cfg.get('resources', {})
    n_threads = int(os.environ.get('CPU_THREADS', res_cfg.get('num_cpu_threads', 2)))
    torch.set_num_threads(n_threads)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        vram_fraction = float(res_cfg.get('vram_fraction', 1.0))
        torch.cuda.set_per_process_memory_fraction(vram_fraction)

    print(f'Condition {args.condition} | Seed {args.seed} | Device: {device} | '
          f'CPU threads: {n_threads}'
          + (f' | VRAM fraction: {vram_fraction:.0%}' if device.type == 'cuda' else ''))

    base = _ROOT
    ckpt_dir = os.path.join(base, args.results_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(base, args.results_dir, f'cond{args.condition}_seed{args.seed}.csv')

    env_cfg['seed'] = args.seed
    env = SMACWrapper(env_cfg)

    if args.condition == 6:
        # Flat obs baseline: raw SMACv2 obs, no entity-centric reformatting
        raw_obs_dim = cfg['env'].get('raw_obs_dim', None)
        if raw_obs_dim is None:
            # Auto-detect from env
            obs, _ = env.reset()
            raw_obs_dim = env._env.get_obs_agent(0).shape[0]
            print(f'  Auto-detected raw_obs_dim={raw_obs_dim}')
        policies = build_flat_policies(
            n_agents=env.n_agents,
            obs_dim=raw_obs_dim,
            action_dim=env_cfg['action_dim'],
        )
    else:
        policies = build_policies(
            args.condition,
            obs_dim=env_cfg['obs_dim'],
            action_dim=env_cfg['action_dim'],
            embed_dim=model_cfg['embed_dim'],
            num_heads=model_cfg['num_heads'],
        )

    unique_policies = list({id(p): p for p in policies}.values())
    for p in unique_policies:
        p.to(device)

    critic_obs_dim = env_cfg['obs_dim'] if args.condition != 6 else raw_obs_dim
    critic = CentralizedCritic(n_agents=1, obs_dim=critic_obs_dim).to(device)

    trainer = MAPPOTrainer(policies, critic, train_cfg, device)

    total_steps = train_cfg['total_steps']
    batch_size = train_cfg['batch_size']
    log_interval = train_cfg['log_interval']
    ckpt_interval = train_cfg['checkpoint_interval']
    n_eval = train_cfg.get('n_eval_episodes', 10)

    # Resume from checkpoint if requested
    if args.resume:
        step = trainer.load_checkpoint(args.resume)
        print(f'Resumed from {args.resume} at step {step}')
        csv_mode = 'a'  # append to existing CSV
    else:
        step = 0
        csv_mode = 'w'

    if csv_mode == 'w':
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'win_rate', 'rew_strike', 'rew_vanguard', 'rew_support', 'entropy'])

    early_stop_threshold = train_cfg.get('early_stop_threshold', 1.1)  # default: disabled
    early_stop_patience = int(train_cfg.get('early_stop_patience', 5))
    consecutive_above = 0

    pbar = tqdm(total=total_steps, initial=step, desc=f'Cond{args.condition} Seed{args.seed}')
    last_log = step
    last_ckpt = step

    while step < total_steps:
        # For condition 6, wrap env to return raw obs before our entity-centric transform
        if args.condition == 6:
            env_for_rollout = _RawObsEnvWrapper(env)
        else:
            env_for_rollout = env

        buffers, rollout_stats = collect_rollout(
            env_for_rollout, policies, critic, batch_size, device,
            gamma=train_cfg['gamma'], gae_lambda=train_cfg['gae_lambda']
        )
        update_stats = trainer.update(buffers, condition=args.condition if args.condition <= 5 else 1)
        step += batch_size
        pbar.update(batch_size)

        if step - last_log >= log_interval:
            last_log = step
            eval_env = _RawObsEnvWrapper(env) if args.condition == 6 else env
            win_rate = evaluate(eval_env, policies, n_eval, device)
            er = rollout_stats['episode_rewards']
            rew_s = er[0] if len(er) > 0 else 0.0
            rew_v = er[1] if len(er) > 1 else 0.0
            rew_sup = er[2] if len(er) > 2 else 0.0
            entropy = update_stats.get('entropy', 0.0)

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, win_rate, rew_s, rew_v, rew_sup, entropy])
            pbar.set_postfix({'win_rate': f'{win_rate:.2f}', 'ent': f'{entropy:.3f}'})

            # Early stopping check
            if win_rate >= early_stop_threshold:
                consecutive_above += 1
                if consecutive_above >= early_stop_patience:
                    pbar.close()
                    print(f'Early stop: win_rate={win_rate:.2f} >= {early_stop_threshold} '
                          f'for {early_stop_patience} consecutive evals at step {step}.')
                    ckpt_path = os.path.join(ckpt_dir, f'cond{args.condition}_seed{args.seed}_step{step}.pt')
                    trainer.save_checkpoint(ckpt_path, step)
                    env.close()
                    print(f'Done. Checkpoint: {ckpt_path}')
                    return
            else:
                consecutive_above = 0

        if step - last_ckpt >= ckpt_interval:
            last_ckpt = step
            ckpt_path = os.path.join(ckpt_dir, f'cond{args.condition}_seed{args.seed}_step{step}.pt')
            trainer.save_checkpoint(ckpt_path, step)

    pbar.close()
    ckpt_path = os.path.join(ckpt_dir, f'cond{args.condition}_seed{args.seed}_step{step}.pt')
    trainer.save_checkpoint(ckpt_path, step)
    env.close()
    print(f'Done. Checkpoint: {ckpt_path}')


class _RawObsEnvWrapper:
    """
    Thin wrapper that substitutes entity-centric obs with raw SMACv2 obs.
    Used only for Condition 6 (flat baseline).
    """
    def __init__(self, smac_wrapper: SMACWrapper):
        self._w = smac_wrapper
        self.n_agents = smac_wrapper.n_agents

    def reset(self, **kwargs):
        self._w.reset(**kwargs)
        return self._get_raw_obs(), {}

    def step(self, actions):
        _, rewards, done, trunc, info = self._w.step(actions)
        return self._get_raw_obs(), rewards, done, trunc, info

    def _get_raw_obs(self):
        return [self._w._env.get_obs_agent(i) for i in range(self.n_agents)]


if __name__ == '__main__':
    main()
