"""MAPPO trainer — PPO update for independent or shared policies."""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import List
from .rollout import AgentBuffer


class MAPPOTrainer:
    def __init__(self, policies: List[nn.Module], critic: nn.Module, config: dict, device: torch.device):
        self.policies = policies
        self.critic = critic
        self.device = device

        self.clip_eps = config.get('clip_eps', 0.2)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.minibatch_size = config.get('minibatch_size', 64)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.lr = config.get('lr', 3e-4)

        # Deduplicate for shared-policy conditions
        unique_policies = list({id(p): p for p in policies}.values())
        self.policy_optimizers = {
            id(p): Adam(p.parameters(), lr=self.lr) for p in unique_policies
        }
        self.critic_optimizer = Adam(critic.parameters(), lr=self.lr)

    def update(self, buffers: List[AgentBuffer], condition: int = 1):
        """Run PPO update. For condition 4 (shared policy), merge all buffers."""
        stats = {}

        if condition == 4:
            # Shared policy: pool all agents' data, update once
            merged = _merge_buffers(buffers)
            policy = self.policies[0]
            opt = self.policy_optimizers[id(policy)]
            p_loss, v_loss, ent = self._ppo_update(policy, opt, self.critic, self.critic_optimizer, merged)
            stats['policy_loss'] = p_loss
            stats['value_loss'] = v_loss
            stats['entropy'] = ent
        else:
            p_losses, v_losses, ents = [], [], []
            for i, (policy, buf) in enumerate(zip(self.policies, buffers)):
                opt = self.policy_optimizers[id(policy)]
                p_loss, v_loss, ent = self._ppo_update(policy, opt, self.critic, self.critic_optimizer, buf)
                p_losses.append(p_loss)
                v_losses.append(v_loss)
                ents.append(ent)
            stats['policy_loss'] = np.mean(p_losses)
            stats['value_loss'] = np.mean(v_losses)
            stats['entropy'] = np.mean(ents)

        return stats

    def _ppo_update(self, policy, policy_opt, critic, critic_opt, buf: AgentBuffer):
        obs = torch.tensor(np.array(buf.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(buf.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buf.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(buf.advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(buf.returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        n = len(obs)
        p_losses, v_losses, entropies = [], [], []

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = indices[start:start + self.minibatch_size]
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_lp = old_log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]

                logits, _ = policy(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                policy_opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                policy_opt.step()

                # Critic update (shared across all agents)
                # We use agent obs as proxy (in full MAPPO, use global obs)
                values_pred = critic(mb_obs).squeeze(-1)
                value_loss = nn.functional.mse_loss(values_pred, mb_ret)

                critic_opt.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                critic_opt.step()

                p_losses.append(policy_loss.item())
                v_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return np.mean(p_losses), np.mean(v_losses), np.mean(entropies)

    def save_checkpoint(self, path: str, step: int):
        unique_policies = list({id(p): p for p in self.policies}.values())
        torch.save({
            'step': step,
            'policies': [p.state_dict() for p in unique_policies],
            'critic': self.critic.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        unique_policies = list({id(p): p for p in self.policies}.values())
        for p, sd in zip(unique_policies, ckpt['policies']):
            p.load_state_dict(sd)
        self.critic.load_state_dict(ckpt['critic'])
        return ckpt['step']


def _merge_buffers(buffers: List[AgentBuffer]) -> AgentBuffer:
    merged = AgentBuffer()
    for buf in buffers:
        merged.obs.extend(buf.obs)
        merged.actions.extend(buf.actions)
        merged.log_probs.extend(buf.log_probs)
        merged.rewards.extend(buf.rewards)
        merged.values.extend(buf.values)
        merged.dones.extend(buf.dones)
    merged.advantages = np.concatenate([buf.advantages for buf in buffers])
    merged.returns = np.concatenate([buf.returns for buf in buffers])
    return merged
