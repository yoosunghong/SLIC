# SLIC — Structured Label-Conditioned Routing in Cooperative MARL

Code and paper for **"Learned Coordination Conventions in Cooperative MARL: Measuring the Translation Gap Between Theory-Informed Roles and Learned Routing."**

Role-semantic assignments supply *priors* over how heterogeneous agents may coordinate. Cooperative MARL agents instead settle on *conventions* through decentralized, non-stationary learning — with no guarantee the resulting structure matches those priors. This repository contains the diagnostic we use to measure that gap, plus the training code for the five ablation conditions it compares.

**The contribution is a measurement procedure, not an equilibrium or causal claim.** Attention weights are treated as architecture-exposed patterns, corroborated by gradient and occlusion attribution — not as explanations.

---

## Headline results

| Claim | Evidence |
| :--- | :--- |
| Label-conditioned attention routes more concentrated than flat MLP | routing entry std **0.246** vs. **0.055** (4.5×), 5 seeds |
| Win rate, SMACv2 3v3 | C1 **0.794 ± .033** vs. C3 0.480 ± .155 (+31.4 pp) |
| Zero-shot 3v3 → 9v9 transfer | C1 **0.224 ± .088**, above from-scratch 9v9 C1 (0.110) |
| Removing labels under shared attention | −14.0 pp (bundled with policy sharing — an *upper bound*) |
| Theory-informed predictions | 3 of 4 hold, 1 tied, under 5-seed re-evaluation |

At 6v6/9v9 the win-rate gaps fall inside seed std; **we do not claim significance there.** The structural signature persists where win-rate does not — that is the point of the paper.

---

## The five conditions

Two axes: *architecture* (attention vs. MLP) and *role-label + policy sharing*.

| Cond. | Architecture | Role labels | Policy |
| :--- | :--- | :---: | :--- |
| **C1 Full** | Intra-Set Self-Attn + Cross-Attn | ✓ | per-role |
| **C2** | Cross-Attn only (no intra-set) | ✓ | per-role |
| **C3 Role+MLP** | Flat MLP | ✓ | per-role |
| **C4 No-Role** | Intra-Set Self-Attn + Cross-Attn | ✗ | **shared** |
| **C5 Flat MLP** | Flat MLP | ✗ | per-role |

`smacv2_poc` additionally exposes **C6** (`--condition 6`): vanilla MAPPO on the raw flat SMACv2 observation, with no entity-centric reformatting. It is a sanity baseline, not part of the 2×2 ablation.

**Read the ablations carefully.** C1 vs. C4 bundles label removal with a per-role → shared-policy change, so it upper-bounds rather than isolates the label effect. C3 vs. C5 is the clean label-in-input test for MLPs (+1.6 pp, within noise). C4 vs. C5 additionally confounds a 3× data-per-parameter advantage from sharing.

---

## Layout

```
validation_poc/     Domain 1 — custom 15x15 MiniGrid zone-capture task, 3 roles
smacv2_poc/         Domain 2 — SMACv2 (Terran) + the 3v3/6v6/9v9 scaling study
llm_micro/          Cross-domain probe: routing legibility in an LLM planner
de_paper/           Manuscript, figures, reviews, and the A0 conference poster
train.py            Convenience wrapper -> smacv2_poc.training.train
prompts/            Prompt scaffolding for the LLM probe
```

Role labels are injected as a 3-dim one-hot into **both** `self_tok` and every `ally_tok` slot. That placement is load-bearing: it is what makes the cross-attention score a direct function of the role pair `(r_i, r_j)`.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r smacv2_poc/requirements.txt          # or validation_poc/requirements.txt
```

SMACv2 needs a StarCraft II installation and the SMAC maps; see the [SMACv2 repo](https://github.com/oxwhirl/smacv2) for platform specifics. MiniGrid (`validation_poc`) has no such dependency and is the faster way to reproduce the qualitative result.

---

## Training

All conditions share identical hyperparameters. **Do not tune per condition** — the comparison is structural, not performance maximization.

```bash
# Domain 2 (SMACv2), from the repo root
python -u train.py --condition 1 --seed 0

# Domain 2, explicit
python -u -m smacv2_poc.training.train --condition 3 --seed 0 --config configs/smacv2.yaml

# Domain 1 (MiniGrid)
python -u -m validation_poc.training.train --condition 1 --seed 0 --config configs/default.yaml

# Scaling study (3v3 / 6v6 / 9v9)
python -u -m smacv2_poc.training.train_scaling --condition 1 --seed 0 --config configs/scaling.yaml
```

Flags: `--condition {1..5}` (`{1..6}` for `smacv2_poc`), `--seed`, `--config`, `--results_dir`, and `--resume` (SMACv2 only).

MAPPO with a centralized critic. 2M environment steps; GAE λ=0.95, γ=0.99, clip 0.2, entropy 0.01, Adam 3e-4, batch 512. Reward mixing `r = 0.8·r_ind + 0.2·r̄_team`.

Reported runs use **5 seeds** for all 3v3 conditions and 9v9 C1/C3; **3 seeds** for 6v6 C1/C3 and the MiniGrid Track-2 replication. Those 3-seed runs are exploratory. Two of the original small-*n* routing findings turned out to be seed artifacts — the diagnostic is itself seed-sensitive, so treat any 3-seed routing claim as provisional.

---

## Analysis

```bash
# Role-routing matrix, formation sensitivity, gradient attribution
python -m smacv2_poc.analysis.analyze_attention

# Occlusion routing — the unified, architecture-agnostic metric
python -m smacv2_poc.analysis.occlusion_routing

# Zero-shot cross-scale transfer, and the padding confound ablation
python -m smacv2_poc.analysis.transfer_eval
python -m smacv2_poc.analysis.padding_ablation
```

Three measurements, in increasing order of interpretation-independence:

1. **Role-routing matrix** — mean cross-attention weight from role *i* to ally slots occupied by role *j*. Concentration is reported as the std of the off-diagonal entries (zero under uniform routing).
2. **Formation sensitivity (Δmax)** — max shift in ally self-attention between two fixed probe scenarios, *Clustered* vs. *Spread*.
3. **Occlusion sensitivity** — zero each ally token (mask bit held at 1) and measure policy KL from baseline. Well-defined for *both* attention and MLP architectures, so it is the only strictly apples-to-apples comparison. On C1 it correlates with attention at row-normalized Pearson r = 0.976.

---

## Poster

An A0 (841×1189 mm) conference poster is at `de_paper/paper/next_game_workshop/poster/`. Single-file Typst, no packages:

```bash
cd de_paper/paper/next_game_workshop/poster
typst compile poster.typ poster.pdf
```

---

## Citation

```bibtex
@inproceedings{hong2026learned,
  title     = {Learned Coordination Conventions in Cooperative {MARL}:
               Measuring the Translation Gap Between Theory-Informed Roles
               and Learned Routing},
  author    = {Hong, Yoosung},
  booktitle = {NeXT Game Workshop},
  year      = {2026}
}
```

## Acknowledgements

Thanks to **Prof. Jinseok Seo** (Dong-Eui University) and **Gyuyeol Jeong** (Netmarble) for insightful discussions and valuable comments that greatly improved this work.

## Contact

Yoosung Hong · `yoosunghong.main@gmail.com` · <https://github.com/yoosunghong/SLIC>
