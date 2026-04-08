# Prompt D — Monitor Phase 2 Training & Run Track 2 Analysis (DONE)

```
We are in the SLIC project. Phase 2 training is running (all 15 runs launched).
Use the `paper` conda environment. Working directory: validation_poc/

### Context: What Changed Between Phase 1 and Phase 2

Phase 1 training showed all conditions oscillating at ~50% win rate with no separation.
Root cause: task solvable WITHOUT role coordination — Support just followed allies, no dependency.

Phase 2 environment fixes (applied to env/cooperative_grid.py):
- WIN_HOLD_STEPS: 20 → 40
- Support boost: +0.05 capture rate when Support within radius 3.0 of occupied zone
- Support reward redesigned: boost_bonus=0.5 per nearby occupied zone (was rear-arc 0.1)
- Enemy damage: 0.05 → 0.10, range 1.5 → 2.0
- Vanguard melee_bonus: 0.3 → 0.1 per enemy
- Strike too_close_penalty: -0.3 → -0.1

### Step 1 — Check Training Progress

cd validation_poc/
Check if all 15 runs are still running:
  ps aux | grep "train.py" | grep -v grep | wc -l

Check current data (should be >> 100k steps by now):
  for f in results/cond{1,2,3,4,5}_seed0.csv; do echo "=== $f ==="; tail -5 $f; done

If any run crashed, relaunch it:
  CPU_THREADS=2 conda run -n paper python training/train.py --condition X --seed Y > results/logs/condX_seedY.log 2>&1 &

### Step 2 — Generate Win Rate Curves (when runs complete or near completion)

  conda run -n paper python analysis/analyze_attention.py --results_dir results
  (or) conda run -n paper python -c "
  from analysis.visualize import plot_win_rate_curves
  plot_win_rate_curves('results', 'results/plots')
  "

Check results/plots/ for win_rate_curves.png.
Expected order: Full > No_SelfAttn > Role+MLP > No_RoleLabel > Flat_MLP

### Step 3 — Run Track 2 Analysis (when 2M-step checkpoints exist)

  conda run -n paper python analysis/analyze_attention.py \
    --cond1_ckpt results/checkpoints/cond1_seed0_step2000384.pt \
    --cond2_ckpt results/checkpoints/cond2_seed0_step2000384.pt \
    --cond3_ckpt results/checkpoints/cond3_seed0_step2000384.pt

Check [PASS/FAIL] claim confirmation checklist.

### Reference
- Memory: phase 2 env changes are in ~/.claude/.../memory/project_env_fixes_phase2.md
- CLAUDE.md: full spec
- PLAN.md: phase plan
```
