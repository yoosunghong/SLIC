"""
Scaling-study training entry point.

Trains Condition 1 (Full) or Condition 3 (Role+MLP) on any of the scaling
configs defined in configs/scaling.yaml.

Outputs go to smacv2_poc/results_scaling/ — existing results/ is untouched.

Usage:
  python train_scaling.py --config_name 3v3_balanced --condition 1 --seed 0
  python train_scaling.py --config_name 6v6_support_heavy --condition 3 --seed 2
"""
import argparse
import csv
import os
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # smacv2_poc/
_WORKSPACE = os.path.dirname(_ROOT)
for p in (_WORKSPACE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from env.smac_wrapper_scaling import ScalingSMACWrapper, OBS_DIM as ENV_OBS_DIM
from models.scaling_models import build_scaling_policies, OBS_DIM as MODEL_OBS_DIM
from validation_poc.models.base_policy import CentralizedCritic
from validation_poc.training.rollout import collect_rollout, AgentBuffer
from validation_poc.training.mappo import MAPPOTrainer, _merge_buffers


def _group_buffers_by_policy(policies, buffers):
    """
    Same-role agents share one policy. Merge their buffers so the PPO update
    runs once per unique policy with all per-role experience pooled, instead
    of re-updating the same weights sequentially.

    Returns (unique_policies, merged_buffers) with len == # unique policies.
    """
    groups: dict = {}
    order: list = []
    for p, b in zip(policies, buffers):
        k = id(p)
        if k not in groups:
            groups[k] = (p, [])
            order.append(k)
        groups[k][1].append(b)
    unique_policies = [groups[k][0] for k in order]
    merged = [_merge_buffers(groups[k][1]) if len(groups[k][1]) > 1 else groups[k][1][0]
              for k in order]
    return unique_policies, merged


def evaluate(env, policies, n_episodes, device):
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
                actions.append(logits.argmax(dim=-1).item())
            obs, _, done, _, info = env.step(actions)
        if info.get('won', False):
            wins += 1
    return wins / n_episodes


def _truncate_log_after_step(log_path: str, step: int):
    if not os.path.exists(log_path):
        return
    with open(log_path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, data = rows[0], rows[1:]
    kept = [r for r in data if r and int(float(r[0])) <= step]
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_name', type=str, required=True,
                    help='Key from configs/scaling.yaml:configs (e.g. 3v3_balanced)')
    ap.add_argument('--condition', type=int, required=True, choices=[1, 3, 4, 5])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--config_file', type=str, default='configs/scaling.yaml')
    ap.add_argument('--results_dir', type=str, default='results_scaling')
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--vram_fraction', type=float, default=None,
                    help='Override resources.vram_fraction from yaml (e.g. 0.14 for 6 parallel jobs)')
    args = ap.parse_args()

    assert ENV_OBS_DIM == MODEL_OBS_DIM, \
        f"obs_dim mismatch: env={ENV_OBS_DIM} model={MODEL_OBS_DIM}"

    with open(os.path.join(_ROOT, args.config_file), encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if args.config_name not in cfg['configs']:
        raise KeyError(f"config_name {args.config_name} not in {list(cfg['configs'])}")

    run_cfg    = cfg['configs'][args.config_name]
    env_cfg    = dict(cfg['env'])
    env_cfg.update(run_cfg)
    env_cfg['seed'] = args.seed

    train_cfg = cfg['training']
    model_cfg = cfg['model']
    res_cfg   = cfg.get('resources', {})

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_threads = int(os.environ.get('CPU_THREADS', res_cfg.get('num_cpu_threads', 2)))
    torch.set_num_threads(n_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        vram = args.vram_fraction if args.vram_fraction is not None \
               else float(res_cfg.get('vram_fraction', 1.0))
        torch.cuda.set_per_process_memory_fraction(vram)

    tag = f'{args.config_name}_cond{args.condition}_seed{args.seed}'
    print(f'[{tag}] device={device} threads={n_threads} '
          f'composition={run_cfg["composition"]} n_enemies={run_cfg["n_enemies"]}')

    base_out = os.path.join(_ROOT, args.results_dir)
    ckpt_dir = os.path.join(base_out, 'checkpoints', tag)
    log_dir  = os.path.join(base_out, 'logs')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    log_path = os.path.join(log_dir, f'{tag}.csv')

    env = ScalingSMACWrapper(env_cfg)

    policies = build_scaling_policies(
        args.condition,
        agent_roles=env._agent_roles,
        action_dim=env_cfg['action_dim'],
        embed_dim=model_cfg['embed_dim'],
        num_heads=model_cfg['num_heads'],
    )
    unique_policies = list({id(p): p for p in policies}.values())
    for p in unique_policies:
        p.to(device)

    critic = CentralizedCritic(n_agents=env.n_agents, obs_dim=env_cfg['obs_dim']).to(device)
    trainer = MAPPOTrainer(policies, critic, train_cfg, device)

    total_steps   = train_cfg['total_steps']
    batch_size    = train_cfg['batch_size']
    log_interval  = train_cfg['log_interval']
    ckpt_interval = train_cfg['checkpoint_interval']
    n_eval        = train_cfg.get('n_eval_episodes', 10)

    if args.resume:
        step = trainer.load_checkpoint(args.resume)
        print(f'Resumed from {args.resume} at step {step}')
        _truncate_log_after_step(log_path, step)
        csv_mode = 'a'
    else:
        step = 0
        csv_mode = 'w'

    if csv_mode == 'w':
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'win_rate', 'mean_ep_reward', 'entropy'])

    es_thresh  = train_cfg.get('early_stop_threshold', 1.1)
    es_pat     = int(train_cfg.get('early_stop_patience', 5))
    above_cnt  = 0

    pbar = tqdm(total=total_steps, initial=step, desc=tag)
    last_log  = step
    last_ckpt = step

    while step < total_steps:
        buffers, rollout_stats = collect_rollout(
            env, policies, critic, batch_size, device,
            gamma=train_cfg['gamma'], gae_lambda=train_cfg['gae_lambda']
        )
        # Collapse same-role buffers so each unique policy is updated once per iter.
        group_policies, group_buffers = _group_buffers_by_policy(policies, buffers)
        trainer.policies = group_policies
        update_stats = trainer.update(group_buffers, condition=args.condition)
        trainer.policies = policies   # restore for rollout/eval paths
        step += batch_size
        pbar.update(batch_size)

        if step - last_log >= log_interval:
            last_log = step
            wr = evaluate(env, policies, n_eval, device)
            er = rollout_stats.get('episode_rewards', [])
            mean_r = float(np.mean(er)) if len(er) else 0.0
            ent = update_stats.get('entropy', 0.0)

            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, wr, mean_r, ent])
            pbar.set_postfix({'wr': f'{wr:.2f}', 'ent': f'{ent:.3f}'})

            if wr >= es_thresh:
                above_cnt += 1
                if above_cnt >= es_pat:
                    pbar.close()
                    ckpt = os.path.join(ckpt_dir, f'step{step}.pt')
                    trainer.save_checkpoint(ckpt, step)
                    env.close()
                    print(f'[{tag}] Early stop at step {step}. Checkpoint: {ckpt}')
                    return
            else:
                above_cnt = 0

        if step - last_ckpt >= ckpt_interval:
            last_ckpt = step
            trainer.save_checkpoint(os.path.join(ckpt_dir, f'step{step}.pt'), step)

    pbar.close()
    final = os.path.join(ckpt_dir, f'step{step}.pt')
    trainer.save_checkpoint(final, step)
    env.close()
    print(f'[{tag}] Done. Checkpoint: {final}')


if __name__ == '__main__':
    main()
