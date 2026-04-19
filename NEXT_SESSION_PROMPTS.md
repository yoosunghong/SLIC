# Next Session Prompts

---

## CURRENT STATUS (2026-04-19)

**Completed:** SMACv2 5-condition training (3 seeds × 2M steps), Track 2 analysis (routing matrix, Δ_max, gradient attribution, slot-ordering bias verification), full 7-section paper draft in `de_paper/paper/`.

**Gap assessment:** Reviewer feedback and `de_paper/MAIN_CONFERENCE_PLAN.md` agree on five open gaps before this can target ICML/NeurIPS. Priority order below.

---

## CURRENT PRIORITY ORDER

1. **Figure 1** — architecture diagram (removes most visible incompleteness)
2. **5+ seeds** — reduces routing matrix variance (std 0.165–0.170 is a credibility gap)
3. **Causal interventions** — converts descriptive routing analysis to a causal claim
4. **Second SMACv2 scenario** — addresses single-environment reviewer objection
5. **Baselines: QMIX, MAPPO vanilla, ROMA, RODE, DOP** — benchmark credibility + related-work positioning
6. **RODE / DOP related work** — explicit differentiation in paper

---

## Prompt 1 — Draw and Insert Figure 1 (Architecture Diagram)

```
We are in the SLIC project. Working directory: D:/Github/SLIC/

The paper at de_paper/paper/paper.md (and de_paper/paper/method.tex) currently has a
placeholder for Figure 1:

  > [FIGURE PLACEHOLDER]
  > Figure 1: SLIC architecture diagram (Condition 1 — Full)...

Task: Replace this placeholder with a real architecture figure.

Option A (preferred if LaTeX TikZ is available in the 'paper' conda env):
  Draw the figure in TikZ and save it as de_paper/paper/figures/slic_architecture.tex.
  Include it in method.tex via \input{figures/slic_architecture} inside a figure environment.

Option B (fallback):
  Generate a clean matplotlib diagram saved as de_paper/paper/figures/slic_architecture.png.
  Include in method.tex via \includegraphics.

What the figure must show:
  - Four entity encoders: self_tok (dim 8→64), ally_tok (8→64), enemy_tok (8→64), zone_tok (6→64)
  - Three Intra-Set Self-Attention blocks (2-head MHA + residual + LayerNorm) on ally, enemy, zone sets
  - Three Cross-Attention blocks (self_enc as Query, each set as K/V)
  - Output MLP: Linear(64×4, 128) → ReLU → Linear(128, action_dim)
  - Highlighted role one-hot positions in self_tok[4:7] and ally_tok[12:15]/[20:23]
    with a labeled arrow showing "role → Q" and "role → K" to make the routing path visible

After inserting: compile the LaTeX if possible (pdflatex de_paper/paper/main.tex) and
confirm Figure 1 renders. Update PLAN.md Completed Tasks.
```

---

## Prompt 2 — Increase to 5 Seeds

```
We are in the SLIC project. Working directory: D:/Github/SLIC/

Current state: All 5 ablation conditions are trained with 3 seeds each (seeds 0, 1, 2).
The routing matrix std for Strike and Vanguard is 0.165–0.170, which reviewers noted
as high. We need 5 seeds minimum (seeds 3 and 4) to reduce variance and add credibility.

Task:
1. Run seeds 3 and 4 for all 5 conditions using the same config:
     python -m smacv2_poc.training.train --condition {1..5} --seed {3,4} \
       --config smacv2_poc/configs/default.yaml
   Save checkpoints to smacv2_poc/results/checkpoints/ as usual.

2. After training, regenerate the multi-seed routing matrix for Cond1 using all 5 seeds:
     python -m smacv2_poc.analysis.analyze_attention \
       --multi_seed --seeds 0 1 2 3 4 --condition 1 \
       --results_dir smacv2_poc/results
   Report the 5-seed mean ± std routing matrix.

3. Update Table 1 (Track 1 performance) and Table 3 (routing matrix) in
   de_paper/paper/experiments.tex with 5-seed numbers and CIs (95% CI = 1.96 * std / √5).

4. Regenerate win_rate_curves.png to include all 5 seeds with shaded CI bands.

5. Update PLAN.md Completed Tasks.
```

---

## Prompt 3 — Causal Intervention Suite (Workstream B)

```
We are in the SLIC project. Working directory: D:/Github/SLIC/

The current paper's mechanism claim is descriptive: attention produces structured routing.
To make it causal, we need interventions that show *disrupting* the routing path
degrades both routing structure and task performance.

Use best Cond1 checkpoint: smacv2_poc/results/checkpoints/cond1_seed1_step1801728.pt

Implement the following in smacv2_poc/analysis/causal_interventions.py:

Intervention 1 — Test-time role-label shuffle:
  a. Shuffle role one-hot in self_tok only → measure win rate drop and routing matrix change
  b. Shuffle role one-hot in all ally_tok slots → measure win rate drop and routing collapse
  c. Shuffle both → full disruption baseline
  Run 100 evaluation episodes per intervention. Report win rate and routing matrix std.

Intervention 2 — Top-attended slot masking:
  For each agent, identify its top-attended ally slot (highest cross-attention weight).
  At evaluation time, zero out that slot's value vector before the output MLP.
  Compare win rate drop to: (a) masking a random slot, (b) masking the lowest-attended slot.
  Expected: masking top slot hurts more → confirms attention routing is load-bearing.

Intervention 3 — Counterfactual partner swap:
  Preserve ally positions but swap ally role identities (exchange Vanguard and Support labels).
  Check whether routing weights follow the new labels (role-conditioned) or original positions
  (position-conditioned). If routing follows new labels: confirms role-conditioned routing.

Output:
  - intervention_table.csv: [intervention, win_rate, routing_std, delta_vs_baseline]
  - causal_plot.png: x=routing disruption (routing_std drop), y=win_rate drop, one point per intervention
  Save both to smacv2_poc/results/plots/

Add a Section 5.5 "Causal Validation" to de_paper/paper/experiments.tex with the
intervention table and causal plot. Update PLAN.md Completed Tasks.
```

---

## Prompt 4 — Second SMACv2 Scenario (Workstream D)

```
We are in the SLIC project. Working directory: D:/Github/SLIC/

Current state: All experiments use a single custom SMACv2 scenario (10gen_terran map,
3 Terran units: Marine/Marauder/Medivac as Strike/Vanguard/Support).
Reviewers require at least two scenarios to rule out environment-specific artifacts.

Task: Add a second SMACv2 scenario using Protoss units as role proxies.

Role mapping:
  Strike   → Stalker  (ranged DPS, blink mobility)
  Vanguard → Zealot   (melee frontline, charge ability)
  Support  → Immortal (armored anchor, high armor — nearest analog to area-support)

Steps:
1. Configure the second scenario in smacv2_poc/configs/default.yaml as 'scenario_protoss':
   - Map: 10gen_protoss (or nearest 3-unit heterogeneous map available in SMACv2)
   - Same team size (3 agents), same episode structure
   - Inject role one-hots for Stalker/Zealot/Immortal into self_tok and ally_tok
     using the same injection logic as the Terran scenario

2. Train all 5 conditions on scenario_protoss with 3 seeds, 2M steps each:
     python -m smacv2_poc.training.train --condition {1..5} --scenario protoss \
       --config smacv2_poc/configs/default.yaml

3. Run Track 2 routing analysis on best Cond1 Protoss checkpoint. Extract:
   - Multi-seed routing matrix
   - Δ_max per role
   Compare with Terran results: do routing patterns share structural properties
   (off-diagonal dominance, unit with highest Δ_max is the repositioning unit)?

4. Add a "Generalization" subsection to de_paper/paper/experiments.tex comparing
   the two scenario routing matrices side by side. The claim does not require
   identical routing — it requires that structured routing (high std) reappears.

5. Update PLAN.md Completed Tasks.
```

---

## Prompt 5 — Baselines: QMIX, MAPPO vanilla, ROMA, RODE, DOP

```
We are in the SLIC project. Working directory: D:/Github/SLIC/

We need 5 external baselines to establish benchmark credibility and position against
prior work. Train all on the same 10gen_terran SMACv2 scenario, 3 seeds, 2M steps.

Baseline 1 — QMIX (vanilla):
  - Centralized monotonic value mixing, no attention, flat obs, no role labels
  - Implement in smacv2_poc/baselines/qmix.py using the same rollout infrastructure
  - Save CSVs to smacv2_poc/results/baseline_qmix_seed{N}.csv

Baseline 2 — MAPPO vanilla:
  - Flat 65-dim obs, shared policy for all agents, no role labels, no attention
  - Equivalent to Cond5 but with tuned MAPPO (confirm hyperparameters match default.yaml)
  - Save CSVs to smacv2_poc/results/baseline_mappo_vanilla_seed{N}.csv

Baseline 3 — ROMA (ICML 2020):
  - Load official ROMA implementation; adapt obs format to accept 65-dim entity-centric obs
  - Report win rate; document any adaptation choices in PLAN.md
  - Save CSVs to smacv2_poc/results/baseline_roma_seed{N}.csv

Baseline 4 — RODE (role decomposition):
  - RODE decomposes tasks via role-conditioned sub-policies selected by a role selector
  - Implement or adapt from open-source RODE; use the same env and reward
  - Key differentiation to verify: RODE discovers roles from trajectories; SLIC conditions
    on pre-defined labels. Check whether RODE's discovered roles align with unit types.
  - Save CSVs to smacv2_poc/results/baseline_rode_seed{N}.csv

Baseline 5 — DOP (directed optimization for role specialization):
  - DOP uses diversity objectives to enforce role specialization
  - Implement or adapt; report win rate and any specialization metric DOP produces
  - Save CSVs to smacv2_poc/results/baseline_dop_seed{N}.csv

After all baselines:
  - Regenerate win_rate_all.png including all 10 conditions (5 ablations + 5 baselines)
  - Add a baseline comparison table to de_paper/paper/experiments.tex
  - Update de_paper/paper/related_work.tex with explicit RODE/DOP differentiation:
    one paragraph each explaining the methodological distinction from SLIC.
  - Update PLAN.md Completed Tasks.
```

---

## Prompt 6 — Final Paper Assembly and LaTeX Compile

```
We are in the SLIC project. Working directory: D:/Github/SLIC/

All experiments are complete. Compile and finalize the paper.

1. Verify all figure references in paper sections resolve to actual files in
   de_paper/paper/figures/ or smacv2_poc/results/plots/. Fix any broken paths.

2. Compile: cd de_paper/paper && pdflatex main.tex && bibtex main && pdflatex main.tex
   Fix any LaTeX errors until main.pdf compiles cleanly.

3. Check:
   - Figure 1 (architecture): rendered, not placeholder
   - Table 1 (Track 1): shows 5-seed mean ± CI, includes baseline comparisons
   - Table 3 (routing matrix): shows 5-seed mean ± std
   - Section 5.5 (causal validation): intervention table and causal plot present
   - Related work: RODE and DOP paragraphs present with explicit differentiation
   - Limitations: mentions single-scenario limitation if second scenario not yet done,
     or removes that limitation if Prompt 4 is complete

4. Run the claim checklist from PLAN.md and print PASS/FAIL to console.

5. Save final main.pdf to de_paper/paper/main.pdf. Update PLAN.md.
```

---

## Notes for All Sessions

- Conda env: `paper` for LaTeX, base Python env for training/analysis
- All training results are in `smacv2_poc/results/`
- Best checkpoints (Terran): `cond1_seed1_step1801728.pt`, `cond2_seed1_step2000384.pt`, `cond3_seed2_step2000384.pt`
- Multi-seed routing matrix (3-seed): Strike→Vanguard 0.735 ± 0.165, Support→Strike 0.752 ± 0.087
- Slot-ordering bias: **verified None** (2026-04-19). Routing follows role content, not position.
- Do NOT use numbers from `prompts/prompt_F.md` — those are pre-experiment anticipated values
- Update `PLAN.md` Completed Tasks section after each prompt finishes
- MAIN_CONFERENCE_PLAN.md is the authoritative gap tracker — check it before starting each session
