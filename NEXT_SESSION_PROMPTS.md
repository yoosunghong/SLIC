# Next Session Prompts

---

## Prompt A — Start Training & Monitor Progress

```
We are in the de_paper project. The validation_poc/ codebase is fully implemented and smoke-tested.
Use the `paper` conda environment.

Task: Run training for all 5 ablation conditions (3 seeds each) and monitor progress.

Steps:
1. Launch training for all 5 conditions × 3 seeds in parallel background processes:
   - python training/train.py --condition 1 --seed 0
   - python training/train.py --condition 1 --seed 1
   - python training/train.py --condition 1 --seed 2
   - ... (repeat for conditions 2–5)
   Working directory: validation_poc/
   Use `conda run -n paper` for each.

2. After all runs complete, check results/cond*_seed*.csv files.

3. Generate win rate curves:
   - Run: python analysis/analyze_attention.py --results_dir results
     (no checkpoint args needed — this will call plot_win_rate_curves from the CSVs)
   - OR directly call visualize.plot_win_rate_curves('results', 'results/plots')

4. Build convergence_table.csv: for each condition, find the step where mean win_rate first exceeds 0.5.

5. Report:
   - Does the expected order hold? Full > No_SelfAttn > Role+MLP > No_RoleLabel > Flat_MLP
   - Which conditions converged to 50% win rate, and at what step?
   - Any training instabilities (NaN, reward collapse)?
```

---

## Prompt B — Track 2 Analysis & Paper Draft

```
We are in the de_paper project. Training for all 5 conditions is complete.
Use the `paper` conda environment. Checkpoints are in validation_poc/results/checkpoints/.

Task: Run Track 2 analysis and draft the paper sections.

### Step 1 — Run Analysis
cd validation_poc
conda run -n paper python analysis/analyze_attention.py \
  --cond1_ckpt results/checkpoints/cond1_seed0_step2000000.pt \
  --cond2_ckpt results/checkpoints/cond2_seed0_step2000000.pt \
  --cond3_ckpt results/checkpoints/cond3_seed0_step2000000.pt

Report the claim confirmation checklist output ([PASS/FAIL] lines).

### Step 2 — Interpret Results
Based on the Track 2 outputs:
- Report the 3×3 role-routing matrix values for Condition 1
- Report Δmax per role (Vanguard should be highest, Support lowest)
- Report gradient attribution ratio (role_grad / spatial_grad) for Condition 3
- Note any unexpected patterns (e.g. diagonal-dominant routing)

### Step 3 — Draft Paper Sections
Write the following sections in LaTeX, saved to de_paper/paper/:

1. experiments.tex — Track 1 table (5 conditions × win rate + convergence step) + Track 2 figures description
2. method.tex — Architecture description for all 5 conditions, observation layout table, reward functions
3. theory.tex — Q/K/V role-pair routing derivation, MLP opacity argument, testable predictions

Use the actual numerical results from the analysis output. Do not fabricate numbers.
Calibrate theory depth: if role-routing matrix shows clear off-diagonal patterns → full formal treatment.
If patterns are weak → keep at "inductive bias" level.

Reference files: IDEA.md (claims, theory), PLAN.md (paper structure), CLAUDE.md (full spec).
```
