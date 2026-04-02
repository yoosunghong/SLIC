"""Rollout buffer and trajectory collection for MAPPO."""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.actions)


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Compute GAE advantages and returns."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        next_done = 1.0 if (t == n - 1 and dones[-1]) else float(dones[t])
        delta = rewards[t] + gamma * next_val * (1 - next_done) - values[t]
        last_gae = delta + gamma * lam * (1 - next_done) * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def collect_rollout(env, policies, critic, n_steps: int, device: torch.device,
                    gamma: float = 0.99, gae_lambda: float = 0.95):
    """
    Collect n_steps of experience.
    Returns list of AgentBuffer (one per agent) with GAE-computed advantages and returns.
    """
    n_agents = env.n_agents
    buffers = [AgentBuffer() for _ in range(n_agents)]

    obs, _ = env.reset()
    done = False
    episode_rewards = np.zeros(n_agents, dtype=np.float32)
    episode_wins = []
    episode_reward_list = []

    for _ in range(n_steps):
        global_obs = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
        value = critic(global_obs).item()

        actions = []
        log_probs = []
        for i in range(n_agents):
            ob = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policies[i](ob)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            log_prob = dist.log_prob(action)
            actions.append(action.item())
            log_probs.append(log_prob.item())

        next_obs, rewards, done, _, info = env.step(actions)
        episode_rewards += rewards

        for i in range(n_agents):
            buffers[i].obs.append(obs[i].copy())
            buffers[i].actions.append(actions[i])
            buffers[i].log_probs.append(log_probs[i])
            buffers[i].rewards.append(float(rewards[i]))
            buffers[i].values.append(value)
            buffers[i].dones.append(done)

        obs = next_obs

        if done:
            episode_reward_list.append(episode_rewards.copy())
            episode_wins.append(info.get('won', False))
            episode_rewards = np.zeros(n_agents, dtype=np.float32)
            obs, _ = env.reset()
            done = False

    # Compute last value for bootstrap
    global_obs = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        last_value = critic(global_obs).item()

    # Compute GAE for each agent
    for i in range(n_agents):
        buf = buffers[i]
        adv, ret = compute_gae(
            buf.rewards, buf.values, buf.dones, last_value, gamma, gae_lambda
        )
        buf.advantages = adv
        buf.returns = ret

    stats = {
        'win_rate': float(np.mean(episode_wins)) if episode_wins else 0.0,
        'episode_rewards': np.mean(episode_reward_list, axis=0) if episode_reward_list else np.zeros(n_agents),
    }
    return buffers, stats
