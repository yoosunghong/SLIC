# SMACv2 Porting Guide

This document is a step-by-step guide for porting the SLIC validation codebase from the MiniGrid PoC environment to SMACv2. The goal is to replicate the 5-condition ablation (Track 1) and the role-routing analysis (Track 2) on a standard benchmark environment.

---

## 1. Role Mapping

SMACv2 does not have explicit Strike/Vanguard/Support roles. Use Protoss unit types as proxies:

| Paper Role | SMACv2 Unit | Justification |
| :--- | :--- | :--- |
| Strike | Stalker | Ranged DPS, ground-air attacks, positional play |
| Vanguard | Zealot | Melee, frontline, high HP |
| Support | Colossus | Area-effect, positioning-dependent, slower |

**Recommended map:** `protoss_5_vs_5` with a fixed team composition of Stalker×2, Zealot×2, Colossus×1 (or similar). The exact map should enforce unit-type heterogeneity — avoid homogeneous maps (e.g., marines-only).

The role label (unit-type one-hot) must appear in both `self_tok` and each `ally_tok` slot. This is the load-bearing structural requirement from the theory; verify it in the observation builder before any training.

---

## 2. Dependencies

```bash
# SMACv2 requires StarCraft II and SMAC
pip install git+https://github.com/oxwhirl/smacv2.git
pip install pysc2

# StarCraft II binary: download from Blizzard, extract to ~/StarCraftII/
# SMACv2 maps: copy smacv2/maps/ into ~/StarCraftII/Maps/
```

Add to `requirements.txt`:
```
smacv2>=1.0
pysc2>=3.0
```

The `paper` conda environment already has `torch` and the rest. Add SMACv2 packages there rather than creating a new environment.

---

## 3. Project Structure

Create a parallel directory `smacv2_poc/` at the same level as `validation_poc/`. Do not modify `validation_poc/` — keep the MiniGrid results intact.

```
smacv2_poc/
├── env/
│   └── smac_wrapper.py          # SMACv2 env adapter → same interface as CooperativeGridEnv
├── models/                      # Symlink or copy from validation_poc/models/
│   ├── base_policy.py
│   ├── full_model.py            # Needs obs_dim update only
│   ├── no_self_attn.py
│   ├── role_mlp.py
│   ├── no_role_label.py
│   └── flat_mlp.py
├── training/                    # Copy from validation_poc/training/; train.py needs env swap
│   ├── mappo.py                 # No changes needed
│   ├── rollout.py               # No changes needed
│   └── train.py                 # Swap env import, update obs_dim
├── analysis/
│   ├── analyze_attention.py     # Update scenario_builder import, obs_dim
│   ├── scenario_builder.py      # Rewrite for SMACv2 unit placement
│   └── visualize.py             # No changes needed
├── baselines/
│   ├── qmix.py                  # New: QMIX baseline
│   └── roma.py                  # New: ROMA baseline (or use existing impl)
├── configs/
│   └── smacv2.yaml              # New config with updated dims
└── results/
```

The models, MAPPO trainer, rollout collector, and visualizer require zero or minimal changes. The bulk of the porting work is in three places: `env/`, `analysis/scenario_builder.py`, and `configs/`.

---

## 4. Observation Space

### 4.1 SMACv2 Raw Observation

SMACv2 provides a flat observation per agent. The default content (per-unit features) varies by map. Use `env.get_obs_agent(agent_id)` to inspect. The raw obs typically includes:

- Own unit: health ratio, shield ratio, unit type one-hot (3 dims for Stalker/Zealot/Colossus), relative coordinates
- Ally units: health, shield, unit type, relative pos, alive flag — one block per ally
- Enemy units: health, shield, unit type, relative pos, alive flag — one block per enemy

### 4.2 Target Observation Format

Construct the same entity-centric token format as the MiniGrid PoC:

```
self_tok     8 dims   pos(2) + health(1) + shield(1) + unit_type_onehot(3) + padding(1)
ally_tok    16 dims   2 allies × 8: rel_pos(2) + health(1) + shield(1) + unit_type_onehot(3) + alive(1)
enemy_tok   16 dims   2 enemies × 8: rel_pos(2) + health(1) + shield(1) + type_conf(3) + alive(1)
zone_tok    18 dims   3 control points × 6: rel_pos(2) + owned(1) + contest(1) + value(1) + padding(1)
ally_mask    2 dims   per slot
enemy_mask   2 dims   per slot
zone_mask    3 dims   per slot
TOTAL:      65 dims
```

**Key decisions:**
- `unit_type_onehot` = [1,0,0] Stalker, [0,1,0] Zealot, [0,0,1] Colossus — directly analogous to Strike/Vanguard/Support
- `shield` replaces `vel` from MiniGrid; the semantic role (agent state) is equivalent
- `zone_tok` approximates SMACv2 control points or map landmarks; if the map has no explicit control points, use 3 fixed landmark positions (e.g., top-left, top-right, center of map)
- Keep `obs_dim = 65` to match existing model code — no model changes required

If the SMACv2 map has more than 2 allies or 2 enemies visible, cap to the 2 highest-priority slots (e.g., by distance) and set remaining slots to zero with mask=1.

### 4.3 Implementing the Wrapper

`smacv2_poc/env/smac_wrapper.py` must expose the same interface as `CooperativeGridEnv`:

```python
class SMACWrapper:
    def __init__(self, cfg: dict): ...
    def reset(self) -> tuple[list[np.ndarray], dict]: ...
    def step(self, actions: list[int]) -> tuple[list[np.ndarray], np.ndarray, bool, bool, dict]: ...
    @property
    def n_agents(self) -> int: ...
```

The `reset()` method must return `(obs_list, info)` where `obs_list[i]` is the 65-dim entity-centric vector for agent `i`. The `step()` method must return `(obs_list, rewards, done, truncated, info)` with `info['won']` set to True when the episode is won.

**Role assignment in the wrapper:** In SMACv2, unit IDs are assigned at map initialization. The wrapper must maintain a fixed mapping `{unit_id → role_index}` across resets so that `policies[i]` always corresponds to the same role.

---

## 5. Reward Shaping

SMACv2 provides a team win/loss reward by default. For the ablation to be valid, add role-specific shaping on top:

| Role (Unit) | Shaping |
| :--- | :--- |
| Stalker (Strike) | +approach bonus (scaled distance to nearest enemy) + attack bonus − penalty when distance < 2 to melee enemies |
| Zealot (Vanguard) | +approach bonus + attack bonus + melee bonus when in contact with enemy |
| Colossus (Support) | +proximity bonus to lowest-HP ally + splash coverage bonus (enemies within AoE range) |

Mix with team reward at α=0.2 (same as MiniGrid): `r_final = 0.8 × r_individual + 0.2 × mean(r_team)`.

The default SMACv2 sparse win reward is insufficient for the ablation to show condition separation — role-specific shaping is what makes role labels load-bearing.

---

## 6. Config File

`smacv2_poc/configs/smacv2.yaml`:

```yaml
env:
  map_name: protoss_5_vs_5
  n_agents: 3           # Stalker×1, Zealot×1, Colossus×1 (3-role subset)
  n_enemies: 2
  n_zones: 3
  max_steps: 200
  obs_dim: 65
  action_dim: 6         # SMACv2 discrete actions: no-op, move N/S/E/W, attack

training:
  total_steps: 2000000
  batch_size: 512
  minibatch_size: 64
  ppo_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  entropy_coeff: 0.01
  lr: 3.0e-4
  max_grad_norm: 0.5
  log_interval: 10000
  checkpoint_interval: 200000
  n_eval_episodes: 10
  n_seeds: 3

model:
  embed_dim: 64
  num_heads: 2
  action_dim: 6
```

All hyperparameters are identical to the MiniGrid run. This is required for valid cross-environment comparison.

---

## 7. Model Changes

**No architecture changes are needed.** The model files in `validation_poc/models/` can be copied or symlinked directly into `smacv2_poc/models/`. The only dependency is `obs_dim=65`, which is preserved.

The one line that changes in `train.py`:

```python
# Before (MiniGrid)
from env.cooperative_grid import CooperativeGridEnv
env = CooperativeGridEnv(env_cfg)

# After (SMACv2)
from env.smac_wrapper import SMACWrapper
env = SMACWrapper(env_cfg)
```

Everything else in `train.py`, `rollout.py`, and `mappo.py` is environment-agnostic.

---

## 8. Track 2 Analysis: Scenario Builder

`scenario_builder.py` must be rewritten for SMACv2. The Clustered and Spread scenarios cannot use MiniGrid's grid coordinates. Instead, construct synthetic observation vectors directly (bypassing the live environment) by manually populating the 65-dim entity-centric format:

**Clustered scenario:** All ally `rel_pos` values within radius 1 of (0, 0). Enemies placed at (+8, +8).

**Spread scenario:** Allies at (-6, 0), (0, 6), (6, 0) — one per nominal zone. Enemies at (0, 0).

The scenario builder produces a dictionary `{role_name: np.ndarray}` of shape `(65,)` per role — same interface as the MiniGrid version. The `analyze_attention.py` script imports this and needs no other changes.

Role dims to zero for Condition 4 (No-RoleLabel): `self_tok[5:8]` and per-ally `ally_tok[j*8+4 : j*8+7]` — identical slice positions as MiniGrid.

---

## 9. Additional Baselines (Phase 2)

Implement three baselines beyond the 5-condition ablation:

### 9.1 MAPPO Vanilla (Flat Obs)

Use Condition 5 (Flat-MLP, no labels) with SMACv2's default flat observation (no entity-centric reformatting). This establishes the benefit of the entity-centric observation format itself, independent of the attention architecture.

Implement as a sixth condition (`--condition 6`) in `train.py`. The policy is a standard 3-layer MLP over the raw SMACv2 observation vector.

### 9.2 QMIX

Use an existing QMIX implementation (e.g., PyMARL2 or epymarl). Run on the same SMACv2 map and seeds. Report win rate at 2M steps. Do not modify QMIX internals — treat it as an external baseline.

QMIX does not produce attention weights, so it is Track 1 only. No Track 2 analysis needed.

### 9.3 ROMA

ROMA (ICML 2020) performs automatic role discovery and is the closest related work. Use the original ROMA codebase or a reproduced implementation.

**Critical comparison point:** ROMA discovers roles dynamically; our system conditions on pre-defined labels. If ROMA achieves similar win rate, the framing is: "ROMA discovers, we explain — our framework makes the routing mechanism legible for a pre-specified role structure."

ROMA does not expose an interpretable routing matrix; the Track 2 comparison is one-sided (our system only).

---

## 10. Porting Checklist

Work through these in order. Each item is a discrete, testable step.

- [ ] **Install SMACv2** — confirm `from smacv2 import StarCraft2Env` imports without error
- [ ] **Implement `SMACWrapper`** — confirm `reset()` returns `(obs_list, info)`, `step()` returns correct signature
- [ ] **Verify observation format** — manually print a single obs vector; confirm `self_tok[5:8]` is the unit-type one-hot for each role
- [ ] **Verify role assignment stability** — reset the env 10 times; confirm `policies[0]` always corresponds to Stalker
- [ ] **Smoke test Condition 1** — run 10k steps, confirm no NaN in loss, win rate > 0 within 100k steps
- [ ] **Run all 5 conditions × 3 seeds** — 2M steps each; log to `smacv2_poc/results/`
- [ ] **Rewrite `scenario_builder.py`** — confirm synthetic obs vectors produce non-trivial attention weights
- [ ] **Run `analyze_attention.py`** — confirm routing matrix and Δmax are generated for SMACv2 checkpoints
- [ ] **Add MAPPO Vanilla baseline** (Condition 6) — flat obs, no entity-centric reformatting
- [ ] **Run QMIX** — external tool; collect win rate at 2M steps
- [ ] **Run ROMA** — external tool; collect win rate at 2M steps
- [ ] **Update paper** — add SMACv2 Track 1 table to experiments.tex; move MiniGrid to appendix or retain as PoC section

---

## 11. Key Risks

| Risk | Mitigation |
| :--- | :--- |
| SMACv2 map has no clean 3-role composition | Use a protoss map with `unit_type_bits` and fix the initial team composition programmatically via the map's YAML config |
| Sparse reward makes role shaping too weak | Increase shaping weight for role-specific bonuses early in training; reduce α from 0.2 to 0.1 if team reward dominates |
| Unit IDs shuffle across resets | Cache role→unit mapping on first reset; assert mapping stability in smoke test |
| SMACv2 obs dim differs from 65 | Pad or truncate in wrapper; do not change model obs_dim |
| ROMA codebase incompatible with current torch | Use ROMA as a reference performance number from the original paper if reproduction is blocked |
| Track 2 routing matrix is uniform in SMACv2 | Check if unit-type one-hots are correctly injected; run gradient probe on attention weights before concluding routing is absent |

---

## 12. Expected Outcomes

If the MiniGrid results generalize:

- **Track 1 ordering** should hold: Cond 1 > Cond 2 > Cond 3, Cond 4 worst among attention conditions.
- **Track 2 routing matrix** should show non-uniform, role-differentiated patterns. The specific dominant pair (e.g., Stalker→Zealot analogous to Strike→Vanguard) may differ from MiniGrid due to SMACv2 dynamics.
- **Δmax** ordering (Zealot highest, Stalker lowest or intermediate) may shift — Zealot as melee frontliner should remain most formation-sensitive.
- **Gradient attribution ratio** should remain > 1 for Condition 3 (spatial dims dominate role dims in MLP).

If the routing matrix is uniform or the condition ordering reverses, first check the observation builder (role one-hot injection), then reward shaping magnitudes, before concluding the claim does not generalize.
