# Main-Conference Paper Outline

## Working Title Options

1. Role-Conditioned Relational Routing for Heterogeneous Cooperative MARL
2. Making Role-to-Role Routing Legible in Cooperative MARL with Label-Conditioned Set Attention
3. When Do Role Labels Matter? Causal and Mechanistic Evidence from Heterogeneous Cooperative MARL

---

## One-Sentence Paper Positioning

We study whether role labels can serve as relational routing keys in heterogeneous cooperative MARL, and show that label-conditioned set attention improves coordination while producing interpretable and causally meaningful role-to-role routing patterns across controlled and benchmark environments.

---

## Abstract Draft Skeleton

### Version A: Balanced performance plus mechanism

Heterogeneous cooperative MARL requires agents to reason not only about what they should do, but also about which teammates matter for each decision. We hypothesize that role labels are most useful when they act as relational routing keys rather than passive input features. To test this, we introduce a label-conditioned set-attention policy that places role identity in both self and ally tokens, enabling role-pair-conditioned routing through intra-set self-attention and cross-attention. We evaluate the method across a controlled role-differentiated environment and a public heterogeneous benchmark using MAPPO, compare against parameter-matched MLP and attention baselines, and analyze learned coordination structure with a role-routing matrix, formation-sensitivity metrics, and causal routing interventions. The proposed architecture improves coordination performance, produces stronger and more stable role-to-role routing structure than MLP baselines, and shows significant degradation when routing cues are disrupted. These results suggest that role labels are most effective when embedded in architectures that expose relational selection structure explicitly.

### Version B: Mechanism-first

Why do role labels help heterogeneous cooperative MARL in some policies but remain underused in others? We argue that the answer lies in routing structure. In flat policies, role labels are weak features competing with many others; in label-conditioned attention, they directly shape which teammates are selected as relevant. We instantiate this idea with a set-attention architecture that conditions both query and key representations on role identity, then test it through performance evaluation, routing analysis, and causal interventions. Across controlled and benchmark settings, the architecture yields more structured role-to-role routing, stronger coordination, and measurable collapse under routing-specific perturbations. Our results position role-conditioned routing as a useful design principle for heterogeneous MARL and provide an analysis protocol for testing it.

---

## Contribution Framing

### Main contributions

1. A structural hypothesis:
   role labels are most useful when they operate as relational routing keys between agents.

2. A method:
   a label-conditioned set-attention policy that exposes this routing path explicitly.

3. An evaluation protocol:
   a mechanism analysis toolkit combining routing matrices, formation sensitivity, and causal interventions.

4. Empirical evidence:
   results across a controlled environment and at least one public benchmark showing that the mechanism is both behaviorally useful and empirically legible.

### Optional fourth contribution if benchmark results are strong

The same routing structure that appears in the causal microscope environment also reappears in a standard benchmark, supporting external validity.

---

## Section-by-Section Draft

## 1. Introduction

### Job of the section

Make the reader care about heterogeneous coordination as a routing problem, not just a reward-specification problem.

### Paragraph plan

1. Heterogeneous MARL commonly uses role labels, but policies often have no structural reason to use them well.
2. In many tasks, the main decision is relational selection: which teammate should matter to me right now?
3. Flat MLPs can represent this, but do not expose it directly; attention can.
4. We propose a label-conditioned set-attention policy and evaluate whether it creates useful role-to-role routing.
5. We contribute both a method and a mechanism-validation protocol with causal tests.

### Must state clearly

- The paper is not just "attention beats MLP."
- The novelty is role-conditioned routing plus a way to test whether it is real.
- Controlled and benchmark environments play different scientific roles.

### Candidate thesis sentence

We show that role labels become significantly more useful in heterogeneous MARL when policy architecture turns them into routing keys for relational selection rather than treating them as passive features.

---

## 2. Related Work

### Subsections

1. Heterogeneous and role-based cooperative MARL
2. Attention and entity-centric policies in MARL
3. Role discovery and specialization methods
4. Mechanistic and interpretability analyses in RL and MARL

### Positioning points

- Relative to ROMA and role-discovery papers:
  the question here is not how to discover roles, but how a known role label is structurally used.

- Relative to attention papers:
  the contribution is not attention alone, but role-conditioned routing and its causal analysis.

- Relative to interpretability papers:
  the contribution is not a post hoc visualization only, but a protocol tied to ablation and intervention.

---

## 3. Problem Setup

### Include

- Dec-POMDP or cooperative MARL notation
- heterogeneous agents with role labels
- entity-centric observations
- policy objective and centralized critic setup

### Key conceptual definition

Define role-conditioned routing:

For agent i, role-conditioned routing is the policy's learned preference over teammate entities conditioned jointly on the acting agent's role and the teammate role.

This sets up the role-routing matrix formally.

---

## 4. Method

### 4.1 Architecture

Describe the tokenization:

- `self_tok`
- `ally_tok`
- optional enemy and zone sets depending on the environment

Explain that role identity is injected into self and ally tokens so that both query and key/value pathways can be role-conditioned.

### 4.2 Label-Conditioned Set Attention

Present:

- intra-set self-attention
- self-to-set cross-attention
- optional formal expression for attention score decomposition

Core intuition:

If role identity is embedded in both self and ally tokens, attention scores can represent role-pair-dependent routing functions.

### 4.3 Ablations and Controls

List the five core conditions and the stronger parameter-matched controls.

This section must make it obvious that the paper distinguishes:

- role label presence,
- self-attention presence,
- cross-attention presence,
- shared versus role-specific policy,
- capacity confounds.

---

## 5. Analysis Protocol

This section is what separates the paper from a standard architecture paper.

### 5.1 Role-Routing Matrix

Rows:
acting role

Columns:
attended ally role

Values:
mean routing weight or gradient-based proxy

### 5.2 Formation Sensitivity

Measure how routing changes across controlled spatial configurations.

### 5.3 Causal Routing Interventions

Describe the intervention families:

- self-role perturbation
- ally-role perturbation
- slot masking based on attention rank
- counterfactual role swap at fixed geometry

### 5.4 Representation Diagnostics

Optional but useful:

- probe hidden states for role or priority information
- compare routing structure across seeds and environments

---

## 6. Experimental Setup

## 6.1 Controlled Environment

Purpose:
mechanism discovery and causal interpretability

Need to explain:

- why roles are behaviorally distinct
- why this environment is intentionally interpretable
- how scenarios are constructed for analysis

## 6.2 Benchmark Environment

Purpose:
external validity and standard-comparison credibility

Need to explain:

- role proxy mapping
- observation adaptation
- any known mismatch between benchmark unit types and idealized roles

## 6.3 Training Protocol

Include:

- MAPPO details
- seed count
- evaluation cadence
- hyperparameter sharing across conditions
- parameter counts and compute budget

---

## 7. Results

This section should be organized by scientific question, not by environment alone.

### 7.1 Does role-conditioned routing improve coordination?

Show:

- win-rate curves
- convergence speed
- benchmark summary table

### 7.2 Does the method produce structured role-to-role routing?

Show:

- role-routing matrices
- routing entropy
- formation sensitivity comparison

### 7.3 Is the routing mechanism causally relevant?

Show:

- intervention results
- routing collapse versus performance drop
- top-attended slot masking analysis

### 7.4 Does the mechanism generalize beyond the PoC?

Show:

- benchmark routing analysis
- comparison of structural trends between PoC and benchmark

### 7.5 Negative or revised findings

Important for credibility:

- if expected role pairs do not dominate, explain what the model actually routes to
- separate failed prior intuition from failed method claim

---

## 8. Discussion

### Themes to cover

1. Role labels matter when architecture exposes where they can act.
2. Attention is useful here because it creates relational selection structure, not because it is universally better.
3. Structured routing can still be environment-dependent; the mechanism can generalize even when the exact preferred pairs differ.
4. Benchmark role proxies and handcrafted observations are limitations, not hidden assumptions.

### Good discussion sentence

Our results suggest that the important question is not whether a policy contains role labels, but whether its computation graph gives those labels a short path to relational selection.

---

## 9. Limitations

Must include explicitly:

- role labels are assumed given rather than discovered
- benchmark roles may be proxies rather than ground-truth task roles
- analysis focuses on policy-level routing, not full credit assignment
- causal interventions are local and may not capture all coordination mechanisms

---

## 10. Conclusion

### Ending message

The paper should close on a general principle:

architectures for heterogeneous coordination should be evaluated not only by return, but by whether they expose and support the relational routing structure that role-differentiated behavior requires.

---

## Figure Plan

1. Architecture diagram:
   show self token, ally tokens, role labels entering both query and key paths

2. Core performance figure:
   win-rate curves across conditions

3. Role-routing matrix figure:
   full method versus MLP baseline

4. Formation sensitivity figure:
   Delta-max per role and condition

5. Causal intervention figure:
   routing disruption versus performance drop

6. Benchmark transfer figure:
   PoC routing pattern beside benchmark routing pattern

---

## Table Plan

1. Main results table:
   win rate, steps to threshold, variance, parameter count

2. Causal intervention table:
   intervention type, routing-metric change, performance change

3. Benchmark summary table:
   results against MAPPO, QMIX, ROMA, and ablations

4. Probe or attribution table:
   representation and routing diagnostics

---

## Reviewer-Facing Checklist

Before submission, make sure the paper can answer yes to these questions:

1. Is the main claim broader than a custom environment story?
2. Is there a public benchmark result?
3. Is there a causal test, not just a visualization?
4. Are the strongest alternative explanations covered by baselines?
5. Are the negative findings framed honestly rather than hidden?

If any answer is no, the paper is still likely to read as workshop-level.

---

## Short Abstract Template for Submission Iteration

We study how policy architecture affects the use of role labels in heterogeneous cooperative MARL. We hypothesize that role labels are most effective when they act as relational routing keys between agents rather than passive input features. To test this idea, we introduce a label-conditioned set-attention policy that embeds role identity in both self and ally tokens, enabling role-pair-conditioned routing through attention. Across a controlled role-differentiated environment and a public heterogeneous benchmark, the method improves coordination performance and produces stronger, more stable role-to-role routing structure than MLP and attention controls. Causal routing interventions further show that disrupting the learned routing cues degrades both routing structure and task performance. These results suggest that role-conditioned relational routing is a useful design principle for heterogeneous MARL and provide a practical analysis protocol for testing such mechanisms.
