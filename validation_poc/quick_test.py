import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml
from env.cooperative_grid import CooperativeGridEnv

with open(os.path.join(os.path.dirname(__file__), 'configs/default.yaml')) as f:
    cfg = yaml.safe_load(f)
env = CooperativeGridEnv(cfg['env'])
obs, _ = env.reset()
print('obs shape:', obs[0].shape)
obs2, rew, done, trunc, info = env.step([0,0,0])
print('step OK, rew:', rew)

import torch
print('CUDA:', torch.cuda.is_available())
from models import build_policies, CentralizedCritic
policies = build_policies(1, obs_dim=cfg['env']['obs_dim'], action_dim=cfg['env']['action_dim'],
                          embed_dim=cfg['model']['embed_dim'], num_heads=cfg['model']['num_heads'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for p in policies: p.to(device)
ob = torch.tensor(obs[0], dtype=torch.float32, device=device).unsqueeze(0)
logits, val = policies[0](ob)
print('forward OK, logits shape:', logits.shape)
print('Smoke test PASSED')
