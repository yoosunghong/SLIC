# Prompt E — Write Paper Sections from Phase 2 Results (CURRENT PRIORITY)

```
We are in the SLIC project. Phase 2 training (all 15 runs, 2M steps each) is complete and Track 2
analysis has been run. Use the `paper` conda environment. Working directory: validation_poc/

### Context: What the Results Show

Phase 2 Training Results (final win rate, mean of last 20 evals):
  Cond 1 — Full (Attn+Role):     0.920 ± 0.054
  Cond 2 — No-SelfAttn:          0.893 ± 0.050
  Cond 3 — Role+MLP:             0.830 ± 0.050
  Cond 4 — No-RoleLabel:         0.780 ± 0.113  (worst — 8.5pp below Cond1)
  Cond 5 — Flat-MLP:             0.843 ± 0.052

Track 2 Analysis — Claim Checklist:
  [PASS] Vanguard Δmax is highest (0.160 vs Strike 0.100, Support 0.133)
  [FAIL] Support Δmax is lowest — actually Strike is lowest (0.100)
  [FAIL] Strike→Support > Strike→Vanguard — Strike attends to Vanguard (0.961) not Support (0.039)
  [PASS] Attention routing more structured than MLP (std 0.323 vs 0.153)
  [PASS] No-RoleLabel win rate < Full by >5pp (8.5pp gap)

Role Routing Matrix — Cond 1 (rows=agent role, cols=attended ally role):
  Strike   → [Str=0.000, Van=0.961, Sup=0.039]
  Vanguard → [Str=0.574, Van=0.000, Sup=0.426]
  Support  → [Str=0.453, Van=0.547, Sup=0.000]

Gradient Attribution — Cond 3 MLP (spatial_grad / role_grad ratio):
  Strike:   3.7×  (spatial dominates)
  Vanguard: 2.4×
  Support:  3.5×

Δmax per role:
  Cond 1 — Vanguard: 0.160, Support: 0.133, Strike: 0.100
  Cond 2 — All: 0.000 (no self-attn, no formation sensitivity)

Key interpretations:
- Strike→Vanguard dominance is coherent: ranged Strike uses melee Vanguard as frontline positional anchor
- The core structural claim holds: attention produces 2× more structured role routing than MLP
- Role labels significantly matter: Cond 4 (no role label) is consistently worst, high variance
- MLP underutilizes role labels: spatial gradients 2.4–3.7× larger than role label gradients

Plots already generated in results/plots/:
  win_rate_curves.png
  self_attn_heatmap_{strike,vanguard,support}.png
  role_routing_matrix_cond1_full.png
  role_routing_matrix_cond3_mlp_gradient.png
  delta_max_comparison.png
  gradient_attribution_{strike,vanguard,support}.png

### Task: Write Paper Sections

Reference CLAUDE.md (full spec), IDEA.md (claims/theory), PLAN.md (paper structure).
Save LaTeX files to de_paper/paper/ (create directory if it doesn't exist).

Write the following sections using ACTUAL numbers from above — do not fabricate:

1. experiments.tex
   - Track 1 table: 5 conditions × (win rate mean±std, steps-to-50%)
   - Track 2 results: role routing matrix as a table, Δmax per role table
   - Claim checklist as a results paragraph (3 PASS, 2 FAIL — be honest, interpret failures)

2. method.tex
   - Full architecture description for all 5 conditions
   - Observation layout table (63 dims, entity-centric)
   - Reward functions per role
   - Training setup (MAPPO, hyperparameters from configs/default.yaml)

3. theory.tex
   - Q/K/V role-pair routing derivation showing how role labels condition cross-attention
   - Why MLP cannot do role-to-role routing (opacity argument)
   - Calibrated claim: since routing matrix shows Strike→Vanguard not Strike→Support,
     soften the specific prediction but maintain the structural claim (attention IS more structured)

Tone: this is a validation study, not a competitive benchmark paper. Be honest about failures.
The contribution is the analysis framework + interpretability, not SOTA performance.
```
