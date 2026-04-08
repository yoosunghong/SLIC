# Prompt C — Review & Fix Phase 1 Environment (DONE — Phase 2 running)

```
We are in the SLIC project. Phase 1 training (all 5 conditions × 3 seeds × 2M steps) is complete.
Use the `paper` conda environment. Working directory: validation_poc/

### Context: What Training Showed

Training completed but results are noisy and the win rate ordering did NOT match the expected hypothesis.
**This is a diagnosis task, not a performance-tuning task.**
The goal of this project is NOT to prove the proposed method is best,
but to show that "this method is also possible" — that attention provides
structured, interpretable role-routing that MLP does not.

**Key observations from training:**
- All 5 conditions reached ~50% win rate within ~20-44k steps (very fast)
- Win rates oscillated chaotically between 0.0 and 1.0 throughout 2M steps
- No condition showed stable convergence — curves are noisy and non-monotonic
- High entropy persisted (~1.29 across most conditions), meaning policies never committed
- Rewards are heavily dominated by Vanguard (130-153) vs Support (59-66), which is suspicious
- Condition 4 (shared, no role label) showed lower entropy (0.95) — interesting signal

**Root cause hypotheses to investigate:**
1. The environment is too easy — 2-zone capture is achievable by random/near-random behavior,
   so there's no learning pressure to specialize
2. Reward shaping is imbalanced — Vanguard captures 2× more reward than Support,
   which may push all roles to vanguard-like behavior
3. The win condition (2 zones held for 20 steps) may be too loose —
   agents may win by accident without coordinated behavior
4. The evaluation (10 episodes) is too noisy to measure real learning signal

### Step 1 — Diagnose Environment Quality

Run the following checks:
cd validation_poc/

a) Test random policy baseline:
   conda run -n paper python -c "
import yaml, numpy as np
from env.cooperative_grid import CooperativeGridEnv
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
env = CooperativeGridEnv(cfg['env'])
wins = []
for _ in range(100):
    obs, _ = env.reset()
    done = False
    while not done:
        actions = [np.random.randint(0, 6) for _ in range(3)]
        obs, rew, done, _, info = env.step(actions)
    wins.append(info.get('won', False))
print(f'Random policy win rate: {np.mean(wins):.3f}')
"

b) Test stay-in-place policy (all zeros):
   Same as above but actions = [0,0,0] every step.

c) Inspect typical episode: run 5 episodes with random policy,
   print zone_progress at end of each episode.

### Step 2 — Review Environment Design

Read env/cooperative_grid.py carefully. Check:
- Is the capture radius (ZONE_RADIUS=1.5) too large relative to grid size?
- Is the win condition (hold_steps=20, 2/3 zones) achievable by 1 agent alone?
- Are enemies effective obstacles or irrelevant (do they actually impede agents)?
- Is role differentiation enforced by the environment or just by reward shaping?
- Does Support actually need to be near allies, or can it win by going to zones directly?

### Step 3 — Propose and Implement Fixes

Based on findings, propose targeted fixes. Likely candidates:
- Tighten win condition: require all 3 zones OR increase hold_steps to 50
- Reduce capture radius: ZONE_RADIUS from 1.5 to 1.0
- Rebalance rewards: normalize so all roles have similar reward scale
- Strengthen enemies: make them stationary blockers or give them HP
- Add role-enforcement: e.g., Support cannot capture zones directly (forces role dependency)

**Important constraint:** Do NOT over-engineer. The environment needs to be just
hard enough that random/undifferentiated policies fail ~80%+ of the time.
The goal is a learning signal that separates conditions — not a hard benchmark.

### Step 4 — Verify Fix Quality

After implementing fixes:
a) Confirm random policy win rate drops to <10%
b) Confirm at least 1 condition reaches >60% win rate after ~500k steps in a quick test run:
   CPU_THREADS=2 conda run -n paper python training/train.py --condition 1 --seed 0
   (let it run for ~100k steps, check results/cond1_seed0.csv)
c) Confirm reward scales are balanced across roles

### Step 5 — Re-run Training (if environment is fixed)

If the environment fixes pass the above checks, clear old results and re-run:
  rm results/cond*_seed*.csv
  rm results/checkpoints/*
Then launch training again using the same batch launch approach from last session.

### Reference Files
- env/cooperative_grid.py — environment to review
- configs/default.yaml — hyperparameters (do not change these)
- CLAUDE.md — full spec with original environment design intent
- PLAN.md — phase plan and success criteria
```
