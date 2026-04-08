# Prompt B — Track 2 Analysis & Paper Draft

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
