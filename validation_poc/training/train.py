"""Training entry point.
Usage: python train.py --condition 1 --seed 42
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.cooperative_grid import CooperativeGridEnv
from models import build_policies, CentralizedCritic
from training.rollout import collect_rollout
from training.mappo import MAPPOTrainer


def evaluate(env, policies, n_episodes: int, device: torch.device):
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
    parser.add_argument('--condition', type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg['env']
    train_cfg = cfg['training']
    model_cfg = cfg['model']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resource control: limit CPU threads per process (set via env var CPU_THREADS)
    n_threads = int(os.environ.get('CPU_THREADS', '2'))
    torch.set_num_threads(n_threads)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Setup dirs
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base, args.results_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(base, args.results_dir, f'cond{args.condition}_seed{args.seed}.csv')

    env = CooperativeGridEnv(env_cfg)

    policies = build_policies(
        args.condition,
        obs_dim=env_cfg['obs_dim'],
        action_dim=env_cfg['action_dim'],
        embed_dim=model_cfg['embed_dim'],
        num_heads=model_cfg['num_heads'],
    )
    for p in set(id(x) for x in policies):
        policies[[id(x) for x in policies].index(p)].to(device)
    unique_policies = list({id(p): p for p in policies}.values())
    for p in unique_policies:
        p.to(device)

    # Critic: uses per-agent obs (same dim) instead of true global for simplicity
    critic = CentralizedCritic(n_agents=1, obs_dim=env_cfg['obs_dim']).to(device)

    trainer = MAPPOTrainer(policies, critic, train_cfg, device)

    total_steps = train_cfg['total_steps']
    batch_size = train_cfg['batch_size']
    log_interval = train_cfg['log_interval']
    ckpt_interval = train_cfg['checkpoint_interval']
    n_eval = train_cfg.get('n_eval_episodes', 10)

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'win_rate', 'rew_strike', 'rew_vanguard', 'rew_support', 'entropy'])

    step = 0
    pbar = tqdm(total=total_steps, desc=f'Cond{args.condition} Seed{args.seed}')
    last_log = 0
    last_ckpt = 0

    while step < total_steps:
        buffers, rollout_stats = collect_rollout(
            env, policies, critic, batch_size, device,
            gamma=train_cfg['gamma'], gae_lambda=train_cfg['gae_lambda']
        )
        update_stats = trainer.update(buffers, condition=args.condition)
        step += batch_size
        pbar.update(batch_size)

        if step - last_log >= log_interval:
            last_log = step
            win_rate = evaluate(env, policies, n_eval, device)
            er = rollout_stats['episode_rewards']
            rew_s = er[0] if len(er) > 0 else 0.0
            rew_v = er[1] if len(er) > 1 else 0.0
            rew_sup = er[2] if len(er) > 2 else 0.0
            entropy = update_stats.get('entropy', 0.0)

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, win_rate, rew_s, rew_v, rew_sup, entropy])

            pbar.set_postfix({'win_rate': f'{win_rate:.2f}', 'ent': f'{entropy:.3f}'})

        if step - last_ckpt >= ckpt_interval:
            last_ckpt = step
            ckpt_path = os.path.join(ckpt_dir, f'cond{args.condition}_seed{args.seed}_step{step}.pt')
            trainer.save_checkpoint(ckpt_path, step)

    pbar.close()
    # Final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'cond{args.condition}_seed{args.seed}_step{step}.pt')
    trainer.save_checkpoint(ckpt_path, step)
    print(f'Done. Checkpoint saved: {ckpt_path}')


if __name__ == '__main__':
    main()
