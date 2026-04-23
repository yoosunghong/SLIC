# Routing Legibility in Cooperative MARL
### Measuring the Translation Gap Between Theory-Informed Roles and Learned Conventions

**Anonymous**

---

## Abstract

Classical game theory suggests which strategic dependencies should matter; cooperative multi-agent reinforcement learning instead discovers conventions through non-stationary learning dynamics. We study the gap between those two through **routing legibility**: whether a trained policy exposes clear, measurable role-to-role dependency structure.

We introduce a compact analysis framework built around three measurements: a role-routing matrix, a formation-sensitivity metric (`Delta_max`) for models that contain ally self-attention, and input-gradient attribution for MLP baselines. We evaluate five controlled policy conditions on a custom SMACv2 scenario under a fixed MAPPO training setup. The main result is structural: when role labels are present in both the self token and ally tokens, attention-based policies produce substantially more concentrated role-conditioned routing than MLP baselines. The full attention model yields a routing-matrix standard deviation of `0.275`, versus `0.055` for the MLP pseudo-routing matrix, and removing role labels from a shared attention policy reduces win rate by `17.8` percentage points (`0.652` to `0.474`).

Two theory-informed behavioral predictions did not hold in the expected direction. Support, not Vanguard, had the highest `Delta_max`, and Strike routed primarily to Vanguard rather than Support. We treat this not as a failure of the mechanism, but as the paper's main scientific observation: architecture makes strategic dependency legible, while learning dynamics determine which convention is selected. The same structural contrast between attention and MLP appears in a second grid-world environment. For the NExT-Game agenda, the contribution is a compact diagnostic for studying how learned strategic conventions depart from theory-informed prescriptions.

---

## 1. Introduction

As AI systems move from isolated prediction to strategic interaction, one of the central questions for both game theory and machine learning is no longer only *what equilibrium should exist*, but *what convention a learning system will actually settle on*. Classical game theory provides role-sensitive prescriptions under explicit assumptions about information, incentives, and rational response. Cooperative MARL agents, by contrast, learn through decentralized gradient updates in non-stationary environments. Even when role structure is known, the resulting coordination pattern need not resemble the theory-informed prescription. This mismatch is the **translation gap** we study.

Our focus is not performance alone. We ask whether architecture affects the **legibility** of learned strategic dependency: can we read off, from the policy itself, which agent roles condition on which other roles? In heterogeneous teams, this is a routing problem. A Strike agent may need to anchor on a Vanguard, a Support, or neither, depending on the convention learning has discovered. If those dependencies are encoded diffusely inside a flat MLP, they may be difficult to measure. If they are encoded through role-conditioned attention, they may become directly inspectable.

We make a scoped mechanism claim. We do **not** claim that attention guarantees the "correct" game-theoretic convention, nor that MLPs cannot use role labels at all. We claim that placing role labels in both the self token and ally tokens creates a direct architectural pathway for role-pair routing in attention, and that this pathway makes emergent strategic dependency more measurable than in flat MLP baselines.

To test that claim, we train five policy conditions under the same MAPPO setup on a custom SMACv2 scenario using three Terran unit types as role proxies: Marine as **Strike**, Marauder as **Vanguard**, and Medivac as **Support**. We then analyze learned policies with three tools:

1. a **role-routing matrix** that aggregates ally cross-attention by agent role and ally role,
2. a **formation-sensitivity** metric (`Delta_max`) derived from scenario-based ally self-attention changes, and
3. **input-gradient attribution** for MLP baselines.

The main findings are:

1. Attention produces much more structured role-conditioned routing than MLP.
2. Role labels are structurally important for the attention architecture under shared-policy training.
3. The specific convention selected by learning differs from our theory-informed behavioral prediction in a stable, tactically coherent way.

That third result is important for the NExT-Game workshop framing. The paper is not just about attention as an inductive bias; it is about how strategic conventions emerge under learning dynamics even when theory suggests a different routing profile. The analysis framework is our main contribution because it makes that divergence measurable.

### Contributions

1. **Mechanism study.** We instantiate a controlled label-conditioned attention architecture in which role identity appears in both the self token and ally tokens, making role-pair routing structurally explicit in Query and Key/Value interactions.
2. **Analysis framework.** We introduce a role-routing matrix, a formation-sensitivity metric for attention models, and gradient attribution for MLP baselines as a compact toolkit for measuring emergent strategic dependency.
3. **Empirical translation-gap example.** We show that architecture governs the legibility of routing, while the convention selected by learning can diverge from theory-informed expectations in an interpretable and environment-dependent way.

---

## 2. Related Work

**Role specialization in cooperative MARL.** Prior work on role-based MARL often treats roles as latent structure to be discovered from trajectories. ROMA [1], for example, learns latent role representations that encourage specialization. Our setting is different: role identity is known, injected into the observation, and used to study how architecture exploits that information. The question is not "which roles emerge?" but "how legibly does the policy encode role-conditioned strategic dependency?"

**Relational reasoning and attention.** Attention has been used as a relational inductive bias in reinforcement learning [2] and multi-agent learning [3, 4]. These works motivate attention as a way to focus on relevant entities, but they do not directly analyze role-to-role routing structure as a measurable object. Our use of attention is narrower and more diagnostic: we treat attention weights as a comparative readout of strategic dependency under controlled architectural ablations, not as a direct causal explanation of policy behavior.

**Attention faithfulness and interpretability.** The interpretability of attention weights has been debated extensively in NLP. Jain and Wallace [9] show that attention weights can diverge from gradient-based feature importance, while Serrano and Smith [10] and Bastings and Filippova [11] further analyze when attention tracks model sensitivity and when it does not. Wiegreffe and Pinter [12] argue that these critiques do not make attention useless, but instead narrow the conditions under which it should be interpreted. We adopt that more cautious position here: attention weights are useful as comparative and ordinal diagnostics of routing structure, and we cross-check them against gradient-based attribution rather than treating them as guaranteed measures of causal information flow.

**Cooperative MARL optimization.** MAPPO [5] and QMIX [6] are strong cooperative baselines, but neither is designed to make role-conditioned dependency legible. We hold the optimizer fixed and vary policy structure, so the differences we report are architectural rather than optimization-driven.

**Game-theoretic learning dynamics.** Classical work on correlated equilibrium [7] and learning in games [8] provides a language for how strategic dependencies can stabilize without requiring full rationality. Our contribution is empirical rather than formal: we provide a way to observe which role-conditioned dependency pattern a gradient-based MARL system actually learns, and where that pattern diverges from a theory-informed expectation.

---

## 3. Framing the Hypothesis

### 3.1 Routing as Strategic Dependency

In a heterogeneous cooperative team, an agent must decide which teammate matters most for its next action. This is more than perception. It is an internal statement about strategic dependency: which ally role the policy conditions on most strongly at decision time.

We use **routing legibility** to mean the degree to which this dependency structure is observable in the learned policy. A highly legible policy has concentrated, role-sensitive routing patterns. A weakly legible policy has diffuse or near-uniform routing, even if the agent still behaves competently.

### 3.2 Why Attention Should Make Routing More Legible

The architectural argument is simple. In our observation design, role labels appear in:

- the agent's own token (`self_tok`), and
- each ally token (`ally_tok`).

In the attention policy, the self token contributes to the Query while each ally token contributes to the Key and Value. This gives attention a direct path to compute role-pair-sensitive scores:

```text
score(i, j) = <W_Q x_i, W_K x_j> / sqrt(d_k)
```

where `x_i` contains the agent's role and `x_j` contains the ally's role. The model can therefore express "given that I am role A, ally role B is strategically relevant" through a short and inspectable computation.

An MLP can still use role labels, but only through implicit feature interactions inside dense weight products. Our claim is therefore comparative, not absolute: attention provides a more direct and more measurable pathway for role-pair routing than a flat MLP.

### 3.3 Testable Predictions

We separate **structural** predictions from **behavioral** predictions.

**Structural predictions**

1. Attention will produce more structured role-routing than MLP.
2. Removing role labels from a shared-policy attention architecture will substantially reduce performance.

**Behavioral predictions**

3. Vanguard will show the highest formation sensitivity (`Delta_max`).
4. Strike will route primarily to Support.

The structural predictions test the architectural claim directly. The behavioral predictions test a stronger theory-informed expectation about which specific convention learning will select.

---

## 4. Method

### 4.1 Environment

We evaluate on a custom SMACv2 scenario built around three Terran unit types used as role proxies:

- **Strike**: Marine,
- **Vanguard**: Marauder,
- **Support**: Medivac.

The controlled team has one agent of each role. Win rate is the primary task metric. We use the same training setup across all conditions and report mean results over five seeds.

### 4.2 Entity-Centric Observation

Each agent receives a `65`-dimensional observation with four token groups:

| Token | Dimensions | Content |
|---|---:|---|
| `self_tok` | 8 | position(2), health(1), shield(1), role_onehot(3), pad(1) |
| `ally_tok` | 16 | 2 allies x 8: rel_pos(2), health(1), shield(1), role_onehot(3), alive(1) |
| `enemy_tok` | 16 | 2 enemies x 8: rel_pos(2), health(1), shield(1), type_conf(3), alive(1) |
| `zone_tok` | 18 | 3 zones x 6: rel_pos(2), owned(1), contest(1), value(1), pad(1) |
| `masks` | 7 | validity flags for ally, enemy, and zone slots |

Role one-hot vectors are present in both `self_tok` and `ally_tok`. This is the load-bearing part of the design: the self role conditions the Query, and ally roles condition the Key/Value tensors.

### 4.3 Policy Conditions

All policies output a discrete action distribution over `{stop, move-N, move-S, move-E, move-W, attack}`. For Medivac, `attack` is remapped to `stop`.

![Figure 1: SLIC Architecture](figures/slic_architecture.png)

*Figure 1: Condition 1 architecture. Separate encoders process self, ally, enemy, and zone tokens. Intra-set self-attention contextualizes each entity set, and cross-attention uses the self embedding as the query. The critical structural path is that role labels appear in both self and ally tokens, enabling role-conditioned routing.*

**Condition 1: Full attention model.**  
Dedicated encoders map each token type to a `64`-dimensional embedding. We apply intra-set self-attention to ally, enemy, and zone tokens, followed by cross-attention from the self embedding to each set.

**Condition 2: No self-attention.**  
We remove the three intra-set self-attention blocks but keep cross-attention. This isolates the value of relational contextualization before routing.

**Condition 3: Role+MLP.**  
We remove all attention and process the flat `65`-dimensional observation with an MLP. Role labels remain present in the input.

**Condition 4: No-role-label shared attention.**  
We keep the full attention architecture but zero the role-onehot dimensions in both `self_tok` and `ally_tok` before encoding. A single shared policy is trained across all roles.

**Condition 5: Flat MLP without labels.**  
Same as Condition 3, but role-onehot inputs are zeroed.

### 4.4 Training Setup

All conditions use MAPPO with a centralized critic and identical hyperparameters.

| Hyperparameter | Value |
|---|---|
| Total environment steps | 2M per seed |
| Seeds | 5 |
| Discount `gamma` | 0.99 |
| GAE `lambda` | 0.95 |
| PPO clip | 0.2 |
| Entropy coefficient | 0.03 |
| Optimizer | Adam, `lr = 1e-4` |
| Max grad norm | 0.5 |
| Batch size | 512 |
| Mini-batch size | 64 |
| PPO epochs/update | 3 |
| Checkpoint interval | every 200k steps |

Conditions 1, 2, 3, and 5 use independent role-specific policies. Condition 4 uses one shared policy.

### 4.5 Analysis Measurements

We use three analysis tools.

**Role-routing matrix.**  
For each role, we extract ally cross-attention weights and aggregate them by the occupied ally role, producing a `3 x 3` matrix.

**Formation sensitivity (`Delta_max`).**  
For architectures that contain ally self-attention, we construct two fixed scenarios:

- **Clustered**: allies grouped near one landmark,
- **Spread**: allies distributed across separate landmarks.

For each role, `Delta_max` is the maximum change in ally self-attention weight between the two scenarios. This metric is therefore defined for Condition 1 and not applicable to Condition 2, which removes ally self-attention entirely.

**Gradient attribution.**  
For MLP baselines, we compute input gradients with respect to the maximum action logit and compare average gradient magnitude on spatial dimensions versus the role-label coordinates. For no-label conditions, we do not interpret gradients on zeroed role coordinates as evidence of role use; they are reported only to show that removing explicit labels eliminates a meaningful role signal in the input.

---

## 5. Experiments

### 5.1 Performance Evidence

Table 1 reports the main task metric on the SMACv2 scenario. The full attention model performs best overall.

**Table 1: Win rate over five seeds (mean of final evaluation checkpoints).**

| Condition | Win Rate | Std |
|---|---:|---:|
| Cond 1: Full | 0.652 | 0.079 |
| Cond 2: No Self-Attn | 0.551 | 0.210 |
| Cond 3: Role+MLP | 0.440 | 0.143 |
| Cond 4: No Role Label (shared) | 0.474 | 0.204 |
| Cond 5: Flat MLP | 0.494 | 0.064 |

![Win rate learning curves for all five conditions over 2M training steps](../../smacv2_poc/results/plots/new/win_rate_curves.png)

*Figure 2: Learning curves across five seeds. The full attention condition converges highest. Removing intra-set self-attention or removing role labels under a shared policy both reduce performance substantially.*

Three observations matter for the paper's core claim.

1. **Attention helps.** Condition 1 outperforms Condition 3 by `21.2` percentage points (`0.652` vs. `0.440`).
2. **Intra-set context helps.** Condition 2 trails Condition 1 by `10.1` points, indicating that contextualizing ally entities before routing adds value.
3. **Labels matter for attention.** Condition 4 is `17.8` points below Condition 1, supporting the claim that explicit role labels are structurally useful in the attention architecture.

Notably, Condition 5 slightly outperforms Condition 3 (`0.494` vs. `0.440`). That does not weaken the main thesis; it sharpens the narrower interpretation that adding role labels to an MLP is not enough to produce useful role-conditioned routing.

### 5.2 Routing Legibility

#### 5.2.1 Formation Sensitivity

Table 2 reports `Delta_max` for the attention model that contains ally self-attention.

**Table 2: `Delta_max` by role for Condition 1.**

| Role | `Delta_max` |
|---|---:|
| Strike | 0.028 |
| Vanguard | 0.035 |
| Support | **0.103** |

![Delta_max comparison bar chart across roles for Conditions 1 and 2](../../smacv2_poc/results/plots/new/delta_max_comparison.png)

*Figure 3: Support shows the highest formation sensitivity in the full model. Condition 2 is omitted from the table because `Delta_max` is not defined once ally self-attention is removed.*

This result does **not** support our original behavioral prediction that Vanguard would be most formation-sensitive. It nevertheless supports a narrower architectural point: the full attention model exposes a role-specific sensitivity profile that can be measured directly.

#### 5.2.2 Role-Routing Matrix

We next aggregate ally cross-attention weights by agent role and ally role. We verified that reversing ally-slot order leaves the routing matrix unchanged, ruling out a slot-position artifact.

**Table 3: Multi-seed role-routing matrix for Condition 1 (mean +/- std).**

| Agent role \\ Ally role | Strike | Vanguard | Support |
|---|---:|---:|---:|
| Strike | 0.000 | **0.713 +/- 0.147** | 0.287 +/- 0.147 |
| Vanguard | 0.666 +/- 0.171 | 0.000 | 0.334 +/- 0.171 |
| Support | **0.628 +/- 0.169** | 0.372 +/- 0.169 | 0.000 |

![Role-routing matrix heatmap for Condition 1 (multi-seed)](../../smacv2_poc/results/plots/new/role_routing_matrix_cond1_multiseed.png)

*Figure 4: Multi-seed role-routing matrix for Condition 1. The matrix is clearly off-diagonal and non-uniform.*

This is the clearest evidence of routing legibility. Strike routes primarily to Vanguard rather than Support. Support routes primarily to Strike. The matrix is structured, role-specific, and stable in direction across seeds even though magnitude varies.

#### 5.2.3 MLP Comparison via Gradient Attribution

Because Condition 3 has no attention weights, we build an MLP pseudo-routing view through input gradients. We do not treat these two quantities as identical. Attention weights need not faithfully measure causal feature use [9-12], so our goal is not to claim exact agreement between attention and gradients. Instead, we use them as complementary diagnostics: attention reveals whether routing is structurally concentrated, while gradients provide a second check on whether the MLP baseline is using role information in a comparably legible way.

**Table 4: Gradient attribution for the Role+MLP baseline (Condition 3).**

| Role | Spatial grad | Role grad | Spatial/Role ratio |
|---|---:|---:|---:|
| Strike | 0.517 | 0.089 | 5.8 |
| Vanguard | 0.290 | 0.126 | 2.3 |
| Support | 0.433 | 0.116 | 3.7 |

![Role-routing matrix for Condition 3 (MLP gradient attribution version)](../../smacv2_poc/results/plots/new/role_routing_matrix_cond3_mlp_gradient_smacv2.png)

*Figure 5: Pseudo-routing matrix for the MLP baseline. The matrix is much flatter than the attention-based routing matrix.*

The main takeaway is ordinal rather than absolute: the MLP baseline does register some role-label signal when labels are present, but the resulting role-conditioned structure is much weaker. The routing-matrix standard deviation is `0.275` for Condition 1 and `0.055` for the Condition 3 pseudo-routing matrix.

### 5.3 Claim Summary

**Table 5: Outcome of the four predictions.**

| Claim | Type | Prediction | Outcome | Interpretation |
|---|---|---|---|---|
| C1 | Structural | Attention produces more structured routing than MLP | PASS | Main claim supported directly |
| C2 | Structural | Removing role labels in shared attention hurts performance | PASS | Labels are load-bearing in the attention model |
| C3 | Behavioral | Vanguard has highest `Delta_max` | Not supported | Support has highest formation sensitivity |
| C4 | Behavioral | Strike routes primarily to Support | Not supported | Strike routes primarily to Vanguard |

The paper's contribution does not depend on C3 and C4 being correct. Those failed behavioral predictions are informative precisely because the structural mechanism still works: the architecture makes strategic dependency legible, while learning dynamics decide which convention emerges.

### 5.4 Cross-Environment Replication

We also ran the analysis module in a second, simpler environment: a custom grid-world built on MiniGrid with the same three-role setup.

The cross-environment result is important because it separates **structural** replication from **behavioral** replication.

- The structural contrast between attention and MLP replicates.
- Strike-to-Vanguard routing remains dominant in the attention model.
- The exact behavioral ranking of roles changes across environments.

**Table 6: Structural comparison between grid-world and SMACv2.**

| Measurement | Grid Env | SMACv2 | Interpretation |
|---|---:|---:|---|
| Strike -> Vanguard routing weight | 0.961 | 0.713 | Dominant in both |
| MLP Strike routing max off-diagonal difference | 0.011 | 0.055 std | Weak structure in both |
| MLP Strike spatial/role gradient ratio | 3.7 | 5.8 | Same direction |
| Highest `Delta_max` role | Vanguard | Support | Behavioral convention differs |

This strengthens the workshop-facing message. The architectural pathway governs legibility; the environment and learning dynamics govern which convention becomes stable.

---

## 6. Discussion

### 6.1 What the Results Support

The cleanest supported statement is:

> When role labels are injected into both self and ally tokens, attention-based policies learn much more legible role-conditioned routing structure than flat MLP baselines.

That statement is supported by:

1. the strong gap in routing concentration between attention and MLP,
2. the performance drop when role labels are removed from the shared attention model, and
3. the replication of the structural contrast in a second environment.

### 6.2 The Translation Gap

The more interesting scientific result is that our theory-informed behavioral prediction was not the convention selected by learning.

- We expected Vanguard to be most formation-sensitive; Support was.
- We expected Strike to route primarily to Support; Strike routed primarily to Vanguard.

This is not a contradiction of the mechanism claim. It is the concrete empirical instance of the translation gap. The architecture made the learned convention visible. The convention itself was selected by the learning process, not prescribed by the theory.

For the NExT-Game workshop, this is the key bridge: classical strategic analysis helps specify candidate dependency structures, but modern learning systems may settle into different, locally stable conventions. A useful research program therefore needs both theory and diagnostics.

### 6.3 Why the Emergent Convention Is Still Coherent

The emergent convention is not random noise. In SMACv2, Strike routing to Vanguard is tactically sensible because the frontliner anchors engagement geometry for the ranged unit. Support's high formation sensitivity is also plausible because healer positioning depends strongly on ally clustering. The point is not that the original prediction was meaningless; it is that the actual learned convention reflects environmental incentives and gradient dynamics more than our initial prescription.

### 6.4 Limitations

The paper has several clear limitations.

1. **Environment scope.** The main evidence comes from one custom SMACv2 setup plus one simpler grid-world replication.
2. **Seed sensitivity.** The routing direction is reasonably stable, but magnitude varies across seeds.
3. **Cross-metric comparison.** Attention routing and MLP gradient pseudo-routing are not identical quantities, and prior work has shown that attention weights need not coincide with causal feature importance [9-12]. We therefore interpret their variance gap as an ordinal concentration comparison rather than a literal apples-to-apples metric.
4. **Injected labels.** Role identity is manually inserted into observations, which is appropriate for a controlled mechanism study but limits external validity.

### 6.5 Why This Fits NExT-Game

The workshop description emphasizes the gap between classical strategic abstractions and modern learning dynamics in strategic multi-agent AI systems. This paper contributes a compact empirical diagnostic for that agenda. Instead of asking only whether a policy performs well, we ask:

- which strategic dependency pattern did learning settle on,
- how legible is that pattern in the policy, and
- how does that pattern differ from the theory-informed expectation?

Those questions apply not only to cooperative MARL, but more broadly to learned strategic systems in which roles, capabilities, or incentives are explicit but the selected convention is not.

---

## References

[1] Tonghan Wang, Heng Dong, Victor Lesser, and Chongjie Zhang. ROMA: Multi-Agent Reinforcement Learning with Emergent Roles. In *Proceedings of the International Conference on Machine Learning (ICML)*, 2020.

[2] Vinicius Zambaldi, David Raposo, Adam Santoro, Victor Bapst, Yujia Li, Igor Babuschkin, Karen Simonyan, Rajesh Hadsell, and Timothy Lillicrap. Relational Deep Reinforcement Learning. *arXiv preprint arXiv:1806.01830*, 2018.

[3] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. In *Proceedings of NeurIPS*, 2017.

[4] Shariq Iqbal and Fei Sha. Actor-Attention-Critic for Multi-Agent Reinforcement Learning. In *Proceedings of ICML*, 2019.

[5] Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre M. Bayen, and Yi Wu. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. In *Proceedings of NeurIPS*, 2022.

[6] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder de Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning. In *Proceedings of ICML*, 2018.

[7] Robert J. Aumann. Correlated Equilibrium as an Expression of Bayesian Rationality. *Econometrica*, 55(1):1-18, 1987.

[8] Drew Fudenberg and David K. Levine. *The Theory of Learning in Games*. MIT Press, 1998.

[9] Sarthak Jain and Byron C. Wallace. Attention is not Explanation. In *Proceedings of NAACL-HLT*, 2019.

[10] Sofia Serrano and Noah A. Smith. Is Attention Interpretable? In *Proceedings of ACL*, 2019.

[11] Jasmijn Bastings and Katja Filippova. The Elephant in the Interpretability Room: Why Use Attention as Explanation When We Have Saliency Methods? In *Proceedings of the Third BlackboxNLP Workshop*, 2020.

[12] Sarah Wiegreffe and Yuval Pinter. Attention is not not Explanation. In *Proceedings of EMNLP-IJCNLP*, 2019.
