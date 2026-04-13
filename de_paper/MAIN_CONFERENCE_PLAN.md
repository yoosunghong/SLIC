# Main-Conference Upgrade Plan

## Purpose

This document turns the current mechanism-validation study into a main-conference research program. The goal is to move from:

- "the mechanism appears in one controlled PoC environment"

to:

- "role-conditioned routing is a general and causally relevant principle for heterogeneous cooperative MARL"

The current MiniGrid validation remains the foundation, but main-conference readiness requires stronger external validity, stronger baselines, causal evidence, and a paper structure that makes the claim general rather than scenario-specific.

---

## Target Claim

### Current Claim

Label-conditioned attention appears to use role labels as routing signals more effectively than MLP policies in a custom 3-role cooperative environment.

### Main-Conference Claim

In heterogeneous cooperative MARL, role labels can function as relational routing keys. Architectures that expose this structure through label-conditioned set attention produce:

- better coordination performance,
- more stable sample efficiency,
- stronger and more interpretable role-to-role routing structure, and
- measurable performance degradation when the routing mechanism is causally disrupted.

This claim is broader, benchmark-backed, and falsifiable through ablation and intervention.

---

## Gaps Between Current State and Main-Conference Readiness

### Gap 1: External validity is too narrow

Current evidence comes from a single custom MiniGrid environment. This is good for mechanism discovery, but insufficient for a main-track reviewer who wants benchmark relevance.

### Gap 2: Analysis is stronger than performance evidence

The current package is rich in interpretability outputs, but main-conference papers still need competitive and statistically credible performance tables.

### Gap 3: Mechanism is not yet causally established

Attention heatmaps alone can be dismissed as descriptive byproducts. We need intervention studies showing that breaking the routing path changes both routing behavior and task performance.

### Gap 4: Baselines are not yet strong enough

The current ablations isolate architecture components well, but reviewers may still ask whether gains come from capacity, tokenization, or attention in general rather than role-conditioned routing specifically.

### Gap 5: The framing is still environment-story driven

Strike, Vanguard, and Support are useful for intuition, but the paper must ultimately argue for a general principle of heterogeneous coordination rather than one handcrafted role story.

---

## Success Criteria

The project is main-conference ready when all of the following are true:

1. The full method outperforms or matches strong baselines on at least one public heterogeneous benchmark with multi-seed confidence intervals.
2. The same routing structure found in the PoC appears again in at least one benchmark setting.
3. At least one causal intervention on role-conditioned routing produces a clear drop in both routing structure and task performance.
4. Parameter-matched and attention-based baselines fail to reproduce the same combination of performance and structured routing.
5. The paper can state one clean general claim, one clear mechanism, and one reproducible analysis protocol.

---

## Research Program

## Workstream A: Strengthen the Core PoC

### Goal

Convert the current PoC into a polished mechanism-identification environment that reviewers can trust as the causal microscope of the paper.

### Tasks

1. Expand reporting from point estimates to 5-seed or more runs.
2. Add confidence intervals and significance tests for win rate and convergence.
3. Add parameter-count reporting for all five conditions.
4. Add trajectory-level success/failure analysis tied to routing metrics.
5. Add seed-averaged Track 2 analysis rather than single-checkpoint analysis only.

### Deliverables

- seed-averaged `win_rate_curves.png`
- `performance_table.csv` with mean, std, CI, and parameter counts
- seed-averaged routing matrices and Delta-max metrics
- reproducibility appendix for environment construction and training protocol

---

## Workstream B: Add Causal Mechanism Tests

### Goal

Show that role-conditioned routing is not only correlated with behavior but causally relevant.

### Required interventions

1. Test-time role-label shuffle:
   - shuffle role one-hot in `self_tok`
   - shuffle role one-hot in `ally_tok`
   - shuffle both
   - compare performance and routing collapse

2. Train-time selective ablations:
   - remove role labels only from `self_tok`
   - remove role labels only from `ally_tok`
   - retain labels but disable intra-set self-attention
   - retain self-attention but replace ally cross-attention with pooling

3. Attention-to-performance intervention:
   - identify the top-attended ally slot
   - mask or perturb that slot at evaluation time
   - test whether performance drops more than masking a low-attended slot

4. Counterfactual partner swap:
   - preserve positions but swap ally role identities
   - test whether routing changes even when geometry stays fixed

### Key outputs

- intervention table with routing-metric change and win-rate change
- causal plot linking routing disruption to performance disruption
- qualitative counterfactual examples for appendix or main paper figure

---

## Workstream C: Add Stronger Baselines

### Goal

Ensure the claim survives against plausible reviewer alternatives.

### Required baselines

1. Parameter-matched flat MLP baseline
2. Role-embedding MLP baseline with similar parameter count
3. Set encoder without attention-based routing
4. Attention baseline without role labels but with extra capacity
5. Lightweight recurrent baseline if partial observability is important
6. Public benchmark baselines:
   - MAPPO vanilla
   - QMIX or equivalent value-factorization baseline
   - ROMA or closest role-discovery prior

### Reviewer questions these baselines answer

- Is the gain just model size?
- Is the gain just any attention?
- Is the gain just any entity-centric encoding?
- Is the gain redundant if a role-discovery method is used?

---

## Workstream D: Benchmark Generalization

### Goal

Show that the mechanism is not an artifact of one handcrafted task.

### Benchmark ladder

1. Controlled PoC:
   - current MiniGrid task
   - used for mechanism discovery and visual explanation

2. Mid-scale controlled benchmark:
   - a second grid or navigation-combat environment with different geometry
   - preserves role heterogeneity but changes map biases

3. Public benchmark:
   - SMACv2 heterogeneous unit scenario
   - use unit type as role proxy

### Minimum main-track requirement

At least one public benchmark with:

- full method,
- strongest ablations,
- standard baselines,
- multi-seed results,
- benchmark-level routing analysis

### Stretch goal

A second public benchmark or an out-of-distribution generalization setting where formations differ from training.

---

## Workstream E: Upgrade the Analysis Package

### Goal

Turn the current heatmap-heavy analysis into a compact but quantitative scientific toolkit.

### Add these metrics

1. Routing entropy per role
2. Off-diagonal routing preference score
3. Alignment score between routing and task-relevant role pairs
4. Formation sensitivity stability across seeds
5. Probe accuracy for role, formation, and ally priority from hidden states
6. Correlation between routing structure and episodic success

### Add these figures

1. Seed-averaged 3x3 routing matrix with confidence intervals
2. Causal ablation plot: routing collapse versus win-rate drop
3. Representation probe summary
4. Generalization figure comparing PoC and SMACv2 routing patterns

### Analysis rule

Every main claim should have both:

- a performance-facing statistic, and
- a mechanism-facing statistic

---

## Workstream F: Paper Framing Upgrade

### Goal

Present the paper as a general contribution to heterogeneous cooperative MARL, not as a one-off role toy study.

### Framing changes

1. Replace role-story-first framing with relational-routing-first framing.
2. Present Strike/Vanguard/Support as one concrete instantiation of heterogeneous coordination.
3. State clearly that the novelty is the combination of:
   - label-conditioned set-attention architecture,
   - mechanistic routing analysis protocol,
   - causal validation of routing usefulness.

### Suggested contribution statement

This paper contributes a structural hypothesis, an architecture that instantiates it, and an analysis toolkit that makes the hypothesis testable across environments.

---

## Experimental Matrix

| Layer | Environment | Purpose | Must Have |
| :--- | :--- | :--- | :--- |
| A | MiniGrid PoC | Mechanism discovery | Yes |
| B | Second controlled task | Robustness to geometry/task design | Recommended |
| C | SMACv2 | Benchmark credibility | Yes |

| Comparison Type | Methods |
| :--- | :--- |
| Core ablations | Conditions 1 to 5 |
| Capacity controls | Parameter-matched MLP, role-embedding MLP |
| Attention controls | No-role attention, pooled set encoder |
| Benchmark baselines | MAPPO, QMIX, ROMA |

| Evaluation Type | Outputs |
| :--- | :--- |
| Performance | Win rate, sample efficiency, variance |
| Mechanism | Routing matrix, Delta-max, entropy |
| Causal | Intervention drop, counterfactual shifts |
| Representation | Probing, attribution |

---

## Milestones

## Milestone 1: PoC becomes publication-grade

### Exit criteria

- 5-seed results complete
- seed-averaged Track 2 complete
- parameter-count table complete
- causal intervention suite implemented in PoC

## Milestone 2: Public benchmark replication

### Exit criteria

- SMACv2 experiments complete for full model and key ablations
- at least two benchmark baselines complete
- routing analysis ported to benchmark checkpoint format

## Milestone 3: Paper-ready package

### Exit criteria

- all core figures generated
- abstract and introduction can state benchmark-backed claims
- limitations and negative results are documented cleanly

---

## Risks and Responses

| Risk | Why It Matters | Response |
| :--- | :--- | :--- |
| Benchmark role proxies are noisy | Weakens role-conditioned interpretation | Use PoC for causal clarity and benchmark for external validity; be explicit that public-benchmark roles are proxies |
| Full model gains vanish against stronger baselines | Threatens novelty | Reframe around interpretability plus causal legibility if performance is comparable |
| Routing patterns differ by environment | Could look inconsistent | Present environment-dependent routing as a result, but test whether the structural mechanism remains legible and manipulable |
| Strike-to-Support prediction does not replicate | Threatens original story | Focus on "structured role-to-role routing" rather than one specific off-diagonal pattern unless replicated |
| Causal interventions are inconclusive | Mechanism claim weakens | Add more targeted interventions on self versus ally role labels and report null results honestly |

---

## Writing Strategy

Write the main paper around three linked questions:

1. Does the architecture improve heterogeneous coordination?
2. Does it produce structured role-to-role routing?
3. Does disrupting that routing reduce coordination quality?

If the answer is yes to all three, the paper reads as a full mechanism paper rather than a workshop-style exploratory analysis paper.

---

## Immediate Next Actions

1. Finalize the MiniGrid PoC reporting package with seed-averaged analysis.
2. Implement causal intervention hooks in the analysis code.
3. Define the SMACv2 role-proxy observation format and training protocol.
4. Add parameter-matched and role-embedding baselines.
5. Freeze the experimental matrix before large-scale training.
