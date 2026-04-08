# Prompt A — Start Training & Monitor Progress

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
