# Prompt F — Write Remaining Paper Sections (Introduction, Related Work, Discussion, Abstract)

```
We are in the SLIC project. Phase 2 training is complete and three paper sections are already
written. Use the `paper` conda environment. Working directory: validation_poc/

### Context: What is Already Written

de_paper/paper/ contains:
  experiments.tex  — Track 1 table, routing matrix, Δmax table, claim checklist (3 PASS/2 FAIL)
  method.tex       — all 5 conditions, observation layout, rewards, MAPPO hyperparameters
  theory.tex       — Q/K/V derivation, MLP opacity argument, calibrated claim

These are complete and should NOT be modified.

### Context: Key Results to Draw From

Phase 2 Training Results (final win rate, mean of last 20 evals):
  Cond 1 — Full (Attn+Role):     0.920 ± 0.054
  Cond 2 — No-SelfAttn:          0.893 ± 0.050
  Cond 3 — Role+MLP:             0.830 ± 0.050
  Cond 4 — No-RoleLabel:         0.780 ± 0.113  (worst — 8.5pp below Cond1)
  Cond 5 — Flat-MLP:             0.843 ± 0.052

Track 2 Key Findings:
  - Vanguard Δmax highest (0.160), Strike lowest (0.100)
  - Strike→Vanguard dominates (0.961) — uses frontliner as spatial anchor
  - Attention routing 2× more structured than MLP (std 0.323 vs 0.153)
  - MLP spatial/role gradient ratio: 2.4–3.7× (role labels underutilized)
  - No-RoleLabel 8.5pp worse — labels are load-bearing

### Task: Write Remaining Paper Sections

Reference CLAUDE.md (full spec), IDEA.md (claims/theory), PLAN.md (paper structure).
Save LaTeX files to de_paper/paper/ (same directory as existing sections).

1. introduction.tex
   - Opening: reward engineering for heterogeneous cooperation is underspecified —
     role labels are easy to define but hard to exploit structurally
   - Motivate the routing problem: agents need to know not just their own role but
     which teammates are role-relevant to their decision
   - State the core claim: attention provides a direct structural inductive bias for
     role-to-role routing that MLP lacks
   - Preview contribution: not SOTA performance, but an analysis framework
     (routing matrix, Δmax, gradient attribution) that makes role routing legible
   - Brief preview of results: 3/5 predictions confirmed, 2 revised but informative

2. related_work.tex
   - ROMA (ICML 2020): automatic role discovery vs. our pre-defined label conditioning
   - R3DM (ICML 2025): attention-based role discovery + diversity vs. our mechanism analysis
   - Zambaldi (2018): single-agent relational reasoning — we extend to multi-agent heterogeneous roles
   - QMIX / MAPPO vanilla: cooperative optimization without interpretability
   - Entity-centric observations (prior work on structured obs): position this as building on that foundation
   - Keep tight — 1 paragraph per cluster, no exhaustive survey

3. discussion.tex
   - Interpret the two FAIL results constructively:
     (a) Strike→Vanguard (not Strike→Support): ranged agent anchoring on frontline tank
         is behaviorally rational — prediction was wrong but routing IS role-structured
     (b) Strike Δmax lowest (not Support): may reflect that Strike's ranged role requires
         less formation-sensitivity than either Vanguard (melee positioning) or Support (HP tracking)
   - The routing matrix as a behavioral fingerprint: each role has a distinct, non-uniform
     cross-attention pattern — this is the interpretability contribution
   - Limitations:
     (a) MiniGrid is a simplified environment — SMACv2 replication needed for benchmark credibility
     (b) Analysis uses a single seed's checkpoint for Track 2 — multi-seed averaging would strengthen claims
     (c) The theory argues for structural inductive bias, not impossibility — an MLP with explicit
         role-pair feature engineering could achieve similar routing
   - Future work: SMACv2 extension (Phase 2 of PLAN.md), UE5 appendix, multi-seed Track 2 analysis

4. abstract.tex
   - Write last, after all other sections exist
   - ~150 words
   - Cover: problem (routing in heterogeneous MARL), method (Intra-Set Self-Attention with
     role-label conditioning), experiments (5-condition ablation, 2M steps), key findings
     (structural routing confirmed, 2 specific predictions revised), contribution
     (analysis framework, not SOTA)
   - Be honest: 3/5 predictions pass; the routing analysis IS the contribution

Tone: validation study. Honest about failures. Contribution = analysis framework + interpretability.
Do NOT fabricate numbers. All numerical claims must come from the results above or the existing .tex files.
```
