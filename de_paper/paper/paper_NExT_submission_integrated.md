# Routing Legibility in Cooperative MARL
### Measuring the Translation Gap Between Theory-Informed Roles and Learned Conventions

**Anonymous**

---

## Abstract

Classical game theory suggests which strategic dependencies should matter; cooperative multi-agent reinforcement learning instead discovers conventions through non-stationary learning dynamics. We study the gap between those two through *routing legibility*: whether a trained policy exposes clear, measurable role-to-role dependency structure. We introduce a compact analysis framework built around three measurements: a role-routing matrix, a formation-sensitivity metric (`Delta_max`) for models that contain ally self-attention, and input-gradient attribution for MLP baselines. We evaluate five controlled policy conditions on a custom SMACv2 scenario under a fixed MAPPO training setup, and we use a simpler custom grid-world as a secondary structural replication. The main result is structural: when role labels are present in both the self token and ally tokens, attention-based policies produce substantially more concentrated role-conditioned routing than MLP baselines. The full attention model yields a routing-matrix standard deviation of `0.275`, versus `0.055` for the MLP pseudo-routing matrix, and removing role labels from a shared attention policy reduces win rate by `17.8` percentage points (`0.652` to `0.474`). Two theory-informed behavioral predictions did not hold in the expected direction. Support, not Vanguard, had the highest `Delta_max`, and Strike routed primarily to Vanguard rather than Support. We treat this not as a failure of the mechanism, but as the paper's main scientific observation: architecture makes strategic dependency legible, while learning dynamics determine which convention is selected. The same structural contrast between attention and MLP also appears in the grid-world replication. The contribution for the NExT-Game workshop is a compact diagnostic for studying how learned strategic conventions depart from theory-informed prescriptions.

---

## 1. Introduction

As AI systems move from isolated prediction to strategic interaction, one of the central questions for both game theory and machine learning is no longer only *what equilibrium should exist*, but *what convention a learning system will actually settle on*. Classical game theory provides role-sensitive prescriptions under explicit assumptions about information, incentives, and rational response. Cooperative MARL agents, by contrast, learn through decentralized gradient updates in non-stationary environments. Even when role structure is known, the resulting coordination pattern need not resemble the theory-informed prescription. This mismatch is the *translation gap* we study.

Our focus is not performance alone. We ask whether architecture affects the *legibility* of learned strategic dependency: can we read off, from the policy itself, which agent roles condition on which other roles? In heterogeneous teams, this is a routing problem. A Strike agent may need to anchor on a Vanguard, a Support, or neither, depending on the convention learning has discovered. If those dependencies are encoded diffusely inside a flat MLP, they may be difficult to measure. If they are encoded through role-conditioned attention, they may become directly inspectable.

We make a scoped mechanism claim. We do **not** claim that attention guarantees the "correct" game-theoretic convention, nor that MLPs cannot use role labels at all. We claim that placing role labels in both `self_tok` and `ally_tok` creates a direct architectural pathway for role-pair routing in attention, and that this pathway makes emergent strategic dependency more measurable than in flat MLP baselines.

We evaluate this claim on a custom SMACv2 scenario using three Terran unit types (Marine, Marauder, Medivac) as role proxies. Five ablation conditions are trained for 2M steps using MAPPO under a fixed setup, and we use a simpler custom grid-world as a secondary replication of the analysis pipeline. The paper's contribution is not state-of-the-art performance; it is an **analysis framework** — the role-routing matrix, the formation-sensitivity metric `Delta_max`, and gradient attribution comparison — that makes routing legible and comparable across architectures. This framework reveals:

1. Attention produces much more structured role-conditioned routing than MLP.
2. Removing role labels from a shared-policy attention system substantially reduces performance.
3. The specific convention selected by learning differs from our theory-informed behavioral prediction in a stable, tactically coherent way.

That third result is especially important for the NExT-Game workshop framing. The paper is not just about attention as an inductive bias; it is about how strategic conventions emerge under learning dynamics even when theory suggests a different routing profile. The analysis framework is our main contribution because it makes that divergence measurable.

---

## 2. Related Work

**Role discovery in cooperative MARL.** ROMA learns latent role representations automatically from trajectories using variational inference, enabling emergent specialization without pre-defined role semantics. Our work takes the complementary view: given pre-defined role labels, we analyze the *mechanism* by which different policy architectures exploit those labels. This is a conditioning problem, not a discovery problem. The question is not "what roles emerge?" but "does the architecture make role labels structurally load-bearing for entity selection?"

**Relational reasoning in reinforcement learning.** Prior work introduced self-attention as a relational inductive bias for reinforcement learning and multi-agent learning. We extend this to the heterogeneous multi-agent setting, where the relevant entity is not simply the spatially proximate one but the role-compatible one. Our setting is narrower and more diagnostic: we use attention as a comparative tool for measuring role-conditioned routing structure.

**Cooperative optimization without role interpretability.** QMIX and MAPPO are strong cooperative baselines, but neither is designed to make role-conditioned dependency legible. We hold the optimizer fixed and vary policy structure, so the differences we report are architectural rather than optimization-driven.

**Attention faithfulness.** We do not treat attention weights as guaranteed causal explanations. Instead, we use attention and gradients as complementary diagnostics: attention reveals whether routing is structurally concentrated, while gradients provide a second check on whether the MLP baseline uses role information in a comparably legible way.

---

## 3. Theory: Role-Label-Conditioned Routing

### 3.1 The Routing Problem

In cooperative MARL with heterogeneous roles, agents must not only follow their own behavioral mandate but also identify *which teammates* are most relevant to their decision at each timestep. This is a routing problem: given a set of ally entities, which subset should a given agent condition on, and how strongly?

Role labels provide a natural signal for this selection. A ranged attacker may benefit most from the frontline tank. A healer prioritizes the most at-risk ally. These routing preferences are determined by role-pair relationships, not spatial proximity alone. The central question of this paper is: *do different policy architectures differ in how effectively they exploit role labels as routing signals?*

### 3.2 Why Attention Makes Routing More Legible

In an MLP policy, the role label is concatenated with all other observation features and processed by shared weight matrices. The computation "which entity should matter given my role" is not structurally encoded. It must be learned implicitly through dense feature interactions.

In the proposed attention architecture, role labels appear in two places:

- `self_tok`: the agent's own role one-hot, encoded into the query
- `ally_tok`: each ally's role one-hot, encoded into the corresponding key and value

The cross-attention score between agent `i` and ally `j` is:

```text
attn(i, j) = softmax((W_Q x_i)(W_K x_j)^T / sqrt(d_k))
```

Because `x_i` contains the agent role and `x_j` contains the ally role, the score is a direct function of the role pair. This gives attention a shorter and more inspectable path for role-pair routing than a flat MLP.

### 3.3 Intra-Set Self-Attention

Before cross-attention, each entity group passes through an intra-set self-attention block. This provides relational context within the ally set before routing occurs. It is the main architectural difference between Condition 1 and Condition 2.

### 3.4 Testable Predictions

We evaluate four predictions:

1. Attention produces more structured role routing than MLP.
2. Removing role labels degrades performance by more than 5 percentage points when a shared policy is used.
3. Formation sensitivity (`Delta_max`) is highest for Vanguard.
4. Strike attends preferentially to Support.

Predictions 3 and 4 are revised by the empirical results. That revision is the paper's central translation-gap result rather than a failure case.

---

## 4. Method

### 4.1 Environment

We evaluate on a custom SMACv2 scenario using three Terran unit types as heterogeneous role proxies:

- **Strike**: Marine
- **Vanguard**: Marauder
- **Support**: Medivac

The team of three agents operates against opponents on a generation map with 10 total agents per side. Episodes terminate at the environment's default step limit. Win rate is the primary task metric.

SMACv2 does not natively support pre-defined role labels in observations. We inject role identity as a 3-dimensional unit-type one-hot into both the agent's own token and each ally token, preserving the structural condition required for role-conditioned query/key routing.

### 4.2 Entity-Centric Observation

Each agent receives a 65-dimensional entity-centric observation with four named token groups:

- `self_tok` in `R^8`: position(2), health(1), shield(1), role_onehot(3), pad(1)
- `ally_tok` in `R^16`: 2 allies × 8 dims with rel_pos(2), health(1), shield(1), role_onehot(3), alive(1)
- `enemy_tok` in `R^16`: 2 enemies × 8 dims with rel_pos(2), health(1), shield(1), type_conf(3), alive(1)
- `zone_tok` in `R^18`: 3 zones × 6 dims with rel_pos(2), owned(1), contest(1), value(1), pad(1)
- `masks` in `R^7`: validity flags for ally, enemy, and zone slots

Role one-hots occupy 3 dimensions in `self_tok[4:7]` and in each ally slot. This placement is structural: the attention mechanism encodes these into Q (from self) and K/V (from ally), creating a direct role-pair routing path.

![Architecture](<../../smacv2_poc/results/plots/new/SLIC (1).png>)

*Figure 1. SLIC architecture (Condition 1). Four entity token groups are encoded by dedicated linear layers. Intra-Set Self-Attention contextualizes each entity set independently. Cross-Attention uses the self-embedding as query and each set as key/value. The critical path is that the role one-hot in `self_tok` is projected into Q, while the role one-hots in `ally_tok` are projected into K/V.*

### 4.3 Policy Conditions

We implement five ablation conditions as swappable policy classes sharing the same interface. All take the 65-dimensional observation and produce a 6-dimensional action distribution.

**Condition 1 — Full.** Dedicated encoders map each entity group to 64-dimensional embeddings. We apply intra-set self-attention to ally, enemy, and zone tokens, then cross-attention from the self embedding to each set. The output MLP consumes the concatenation of ally, enemy, zone, and self context.

**Condition 2 — No Self-Attn.** Removes the three intra-set self-attention blocks. Encoded entity tensors feed directly into cross-attention.

**Condition 3 — Role+MLP.** Removes all attention. A flat MLP processes the full 65-dimensional observation. Role one-hots remain in the input.

**Condition 4 — No-RoleLabel (shared).** Same architecture as Condition 1, but role one-hot dimensions are zeroed in both `self_tok` and all `ally_tok` role dimensions before encoding. A single shared policy handles all three roles.

**Condition 5 — Flat-MLP.** Same as Condition 3 with role one-hots zeroed. No labels, no attention.

All attention computations apply padding masks. A safety function force-unmasks slot 0 if all slots are masked, preventing NaNs in softmax.

### 4.4 Training Setup

All conditions use MAPPO with constant hyperparameters:

| Hyperparameter | Value |
|---|---|
| Algorithm | MAPPO |
| Total steps | 2M per seed |
| Seeds | 3 |
| GAE lambda | 0.95 |
| Discount gamma | 0.99 |
| PPO clip epsilon | 0.2 |
| Entropy coefficient | 0.01 |
| Optimizer | Adam, lr = 3e-4 |
| Batch size | 512 |
| Mini-batch size | 64 |
| PPO epochs per update | 4 |
| Centralized critic | MLP on concatenated observations (195 dims) |
| Checkpoint interval | every 200k steps |

Independent policies (Conditions 1–3 and 5) are updated only on trajectories from their respective role. The shared policy (Condition 4) is updated on all trajectories.

### 4.5 Reward Structure

Role-specific rewards encode the behavioral mandate of each unit type:

- **Strike**: approach bonus toward enemies, attack bonus, too-close penalty
- **Vanguard**: approach bonus, melee engagement bonus when in close range
- **Support**: approach-to-lowest-HP-ally bonus, heal event bonus, rear-arc positioning bonus

Team reward is mixed with individual reward:

```text
r_final = 0.8 * r_individual + 0.2 * r_team_avg
```

---

## 5. Experiments

### 5.1 Track 1: Performance Comparison

Table 1 reports the final win rate for all five conditions.

| Condition | Win Rate | Std |
|---|---:|---:|
| Cond 1: Full | 0.652 | 0.079 |
| Cond 2: No Self-Attn | 0.551 | 0.210 |
| Cond 3: Role+MLP | 0.440 | 0.143 |
| Cond 4: No Role Label (shared) | 0.474 | 0.204 |
| Cond 5: Flat MLP | 0.494 | 0.064 |

![Win Rate Curves](../../smacv2_poc/results/plots/new/win_rate_curves.png)

*Figure 2. Learning curves for all five conditions over 2M training steps. The full attention condition converges highest, while removing intra-set self-attention or removing role labels lowers performance.*

Three observations matter for the paper's core claim. First, attention helps: Condition 1 outperforms Condition 3 by 21.2 percentage points (`0.652` vs. `0.440`). Second, intra-set context helps: Condition 2 trails Condition 1 by 10.1 points, indicating that contextualizing ally entities before routing adds value. Third, labels matter for attention: Condition 4 is 17.8 points below Condition 1, supporting the claim that explicit role labels are structurally useful in the attention architecture.

Notably, Condition 5 slightly outperforms Condition 3 (`0.494` vs. `0.440`). This does not weaken the main thesis; it sharpens the narrower interpretation that adding role labels to an MLP is not enough to produce useful role-conditioned routing.

### 5.2 Track 2: Role-to-Role Routing Analysis

#### 5.2.1 Self-Attention Formation Sensitivity

We construct two fixed evaluation scenarios without running a live environment:

- **Clustered**: all allies positioned near one landmark
- **Spread**: each ally placed at a distinct landmark

For each role, `Delta_max` is the maximum difference in ally self-attention weight between the two scenarios, averaged across heads. This metric is defined only for architectures that contain ally self-attention.

| Role | `Delta_max` |
|---|---:|
| Strike | 0.028 |
| Vanguard | 0.035 |
| Support | **0.103** |

Support shows the highest `Delta_max`, indicating that the healer's self-attention weights are most sensitive to ally formation. This does **not** support our original behavioral prediction that Vanguard would be most formation-sensitive. It nevertheless supports a narrower architectural point: the full attention model exposes a role-specific sensitivity profile that can be measured directly.

#### 5.2.2 Cross-Role Routing Matrix

For each role and each scenario, we extract the cross-attention weight that the agent's `self_tok` query assigns to each ally slot, then tag each slot with the occupying ally's role. Slot-ordering bias is ruled out by verifying that reversing the ally slot order produces identical per-role routing weights.

| Agent \ Ally | Strike | Vanguard | Support |
|---|---:|---:|---:|
| Strike | --- | **0.713 ± 0.147** | 0.287 ± 0.147 |
| Vanguard | 0.666 ± 0.171 | --- | 0.334 ± 0.171 |
| Support | **0.628 ± 0.169** | 0.372 ± 0.169 | --- |

This is the clearest evidence of routing legibility. Strike routes primarily to Vanguard rather than Support. Support routes primarily to Strike. The matrix is structured, role-specific, and stably off-diagonal rather than diffuse or uniform.

#### 5.2.3 MLP Gradient Attribution

To assess how the MLP baseline (Condition 3) utilizes role labels, we compute input gradients with respect to the maximum action logit and measure the mean gradient magnitude separately for spatial features and role one-hot dimensions. We do not claim that attention weights and gradients are identical quantities; we use them as complementary diagnostics of structural concentration.

| Role | Spatial grad | Role grad | Ratio |
|---|---:|---:|---:|
| Strike | 0.517 | 0.089 | 5.8x |
| Vanguard | 0.290 | 0.126 | 2.3x |
| Support | 0.433 | 0.116 | 3.7x |

In Condition 3, spatial features account for 2.3–5.8x more gradient signal than role-label dimensions. The MLP baseline does register some role-label signal when labels are present, but the resulting role-conditioned structure is much weaker than in the attention model.

The routing structure difference between attention and MLP is quantified by the standard deviation of the routing matrix. Condition 1 achieves `std = 0.275` versus `std = 0.055` for the Condition 3 gradient-based pseudo-routing matrix, confirming that attention produces substantially more structured role-to-role routing than MLP.

#### 5.2.4 Secondary Grid-World Replication

We also ran the analysis module in a second, simpler custom grid-world with the same three-role setup. We treat this environment as a structural replication rather than a co-equal benchmark: it reuses the same routing analysis to test whether the contrast between attention and MLP survives outside the SMACv2 dynamics. The same qualitative pattern appears. Attention again produces concentrated role-conditioned routing, while the MLP baseline remains much weaker and more spatially dominated. The exact behavioral convention is not identical across environments, which reinforces the paper's core interpretation: architecture governs routing legibility, while the environment and learning dynamics influence which convention becomes stable.

### 5.3 Claim Verification Summary

1. **[PASS]** Role routing is more structured in attention than MLP: Condition 1 routing std = `0.275` vs. Condition 3 gradient std = `0.055`.
2. **[PASS]** No-RoleLabel (Condition 4) win rate is more than 5 percentage points below Full (Condition 1): `0.474` vs. `0.652`.
3. **[REVISED]** Vanguard `Delta_max` is not highest — Support is highest.
4. **[REVISED]** Strike→Support routing weight is not dominant — Strike→Vanguard dominates.

Two of four a priori behavioral predictions are revised. The mechanism claim — that attention produces structured role-to-role routing where MLP does not — is strongly supported. The specific routing pattern that emerged is tactically coherent with the Terran unit roles in SMACv2.

---

## 6. Discussion

### 6.1 What the Results Support

The cleanest supported statement is: when role labels are injected into both self and ally tokens, attention-based policies learn much more legible role-conditioned routing structure than flat MLP baselines.

That statement is supported by three pieces of evidence. First, the role-routing matrix is much more structured in the attention model than in the MLP pseudo-routing baseline. Second, removing role labels from a shared attention policy causes a substantial performance drop. Third, the learned convention is not diffuse: the routing matrix is stably off-diagonal and role-specific.

### 6.2 The Translation Gap

The more interesting scientific result is that our theory-informed behavioral prediction was not the convention selected by learning. We expected Vanguard to be most formation-sensitive; Support was. We expected Strike to route primarily to Support; Strike routed primarily to Vanguard.

This is not a contradiction of the mechanism claim. It is the concrete empirical instance of the translation gap. The architecture made the learned convention visible. The convention itself was selected by the learning process, not prescribed by the theory.

For the NExT-Game workshop, this is the key bridge: classical strategic analysis helps specify candidate dependency structures, but modern learning systems may settle into different, locally stable conventions. A useful research program therefore needs both theory and diagnostics.

### 6.3 Why the Emergent Convention Is Still Coherent

The emergent convention is not random noise. In SMACv2, Strike routing to Vanguard is tactically sensible because the frontliner anchors engagement geometry for the ranged unit. Support's high formation sensitivity is also plausible because healer positioning depends strongly on ally clustering. The point is not that the original prediction was meaningless; it is that the actual learned convention reflects environmental incentives and gradient dynamics more than our initial prescription.

### 6.4 Limitations

1. **Environment scope.** The main evidence comes from one custom SMACv2 setup, with a simpler custom grid-world serving as a secondary structural replication. We still cannot claim the routing structure generalizes to other maps, team sizes, or factions without additional experiments.
2. **Seed sensitivity.** The routing direction is reasonably stable, but magnitude varies across seeds.
3. **Cross-metric comparison.** Attention routing and MLP gradient pseudo-routing are not identical quantities. We therefore interpret their variance gap as an ordinal concentration comparison rather than a literal apples-to-apples metric.
4. **Injected labels.** Role labels are injected manually into SMACv2 observations, which is appropriate for a controlled mechanism study but limits external validity.

### 6.5 Why This Fits NExT-Game

The workshop description emphasizes the gap between classical strategic abstractions and modern learning dynamics in strategic multi-agent AI systems. This paper contributes a compact empirical diagnostic for that agenda. Instead of asking only whether a policy performs well, we ask which strategic dependency pattern learning settled on, how legible that pattern is in the policy, and how that pattern differs from the theory-informed expectation.

---

## References

1. Tonghan Wang, Heng Dong, Victor Lesser, and Chongjie Zhang. *ROMA: Multi-Agent Reinforcement Learning with Emergent Roles*. ICML, 2020.
2. Vinicius Zambaldi et al. *Relational Deep Reinforcement Learning*. arXiv:1806.01830, 2018.
3. Filippos Christianos, Lukas Schäfer, and Stefano V. Albrecht. *Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning*. NeurIPS, 2020.
4. Tabish Rashid et al. *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. ICML, 2018.
5. Chao Yu et al. *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*. NeurIPS, 2022.
6. Ryan Lowe et al. *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*. NeurIPS, 2017.
7. Shariq Iqbal and Fei Sha. *Actor-Attention-Critic for Multi-Agent Reinforcement Learning*. ICML, 2019.
