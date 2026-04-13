import sys
sys.path.insert(0, '.')
import numpy as np
import torch

from validation_poc.env.cooperative_grid import CooperativeGridEnv
env = CooperativeGridEnv()
obs, _ = env.reset()
print('Obs shape:', obs.shape)
assert obs.shape == (3, 65), f"Expected (3,65) got {obs.shape}"

obs2, rewards, done, _, info = env.step([0, 1, 2])
print('Rewards:', rewards, 'done:', done)

from validation_poc.models import build_policies, CentralizedCritic
for cond in range(1, 6):
    policies = build_policies(cond)
    ob = torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0)
    logits, attn = policies[0](ob, return_attn=(cond in [1, 2]))
    print(f'Cond{cond}: logits={logits.shape}, attn={list(attn.keys()) if attn else None}')

critic = CentralizedCritic(n_agents=1, obs_dim=65)
val = critic(torch.tensor(obs[0], dtype=torch.float32).unsqueeze(0))
print(f'Critic value: {val.shape}')

from validation_poc.analysis.scenario_builder import build_scenario_obs
for sc in ['clustered', 'spread']:
    d = build_scenario_obs(sc)
    for role, o in d.items():
        assert o.shape == (65,), f"Wrong shape {o.shape}"
    print(f'Scenario {sc}: OK')

print('ALL SMOKE TESTS PASSED')
