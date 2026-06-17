# Review

## Review 1

### Pros:

The explicit correction of a small-n (3-seed) artifact through a broader 5-seed re-evaluation is a fantastic display of transparent research.
Shifting focus from raw reward-based performance to structural evaluation and "coordination legibility" is a strong contribution to MARL interpretability.
The attention-based architecture shows strong generalization, transferring zero-shot from 3v3 to 9v9 (
) better than from-scratch baselines.

### Cons:
The C1 vs. C4 comparison mixes policy sharing with label removal, diluting the precision of the 14.0 pp drop claim.
The evaluation is somewhat narrow, restricted strictly to an entity-centric schema with exactly three roles.
The ablation conditions and scaling results are presented very densely in the text; the paper would benefit from restructuring Section 4 for better visual clarity.


## Review 2

### Strengths 
*Interesting research question: The paper tackles an important and underexplored issue in MARL: how learned coordination conventions relate to theory-informed role semantics and equilibrium selection. *Good empirical diagnostic framework: The proposed routing diagnostics, including role-routing matrices and formation sensitivity metrics. *Strong experimental analysis: The authors conduct multiple experiments: scaling studies, zero-shot transfer, padding confound ablations, seed sensitivity analyses.

### Weaknesses 
*Limited theoretical grounding: While motivated by equilibrium selection, the work remains largely empirical and does not provide formal analysis connecting routing behavior to equilibrium concepts. *Interpretability claims should remain cautious: Although the paper is relatively careful, some readers may still interpret routing concentration as evidence of semantic coordination, which is not fully established. *Limited diversity of environments: The environments remain relatively controlled and role-structured. More open-ended cooperative tasks would improve generalization claims.


## Review 3

### Summary
The authors ask a question whether cooperative MARL agents learn specified behavioural patterns that match roles pre-specified by the game designer or if they actually learn to coordinate in a way consistent with those roles and did not settle on some opaque convention that has nothing to do with the design intent. Their solution is architectural: to inject role labels into an attention mechanism so that cross-attention scores between agents. They compare this to a flat MLP baseline where role information is just concatenated into the input vector and is implicitly reflected in the model's weights. Thus, the contribution is a measurement toolkit (attention-based routing matrix + occlusion sensitivity + formation sensitivity) for inspecting what coordination convention the learned policy converged towards.

### Strengths
The core idea may serve as a conceptual contribution for cooperative MARL. The attention-based routing matrix to distinguish behavioural differences is a nice diagnostic tool.
Good (including the ablations) thorough experimental design.
Cross-scale zero-shot transfer is clean.

### Weaknesses
It is an experiment report presented as a scientific paper. There is no formal model, no derived prediction, no equilibrium analysis, no optimisation framework, the author's proposition is just an intuition, not deductions from a formal game model. It does not really address any method gap. Moreover, the abstract and keywords imply a problem of "equilibrium selection", but the paper explicitly avoids any equilibrium or solution concept, lying towards general interpretability.
The framework is not causal. The paper measures what convention formed, but says nothing about why: no credit assignment analysis, no discussion of non-stationarity or curriculum effects. This makes it feel like a pure inspection rather than a learning-dynamics study. Also, what would be the consequences of the agents choosing "a wrong role" if they performed the same.

### Verdict
The paper is a competent empirical study with a measurement framework, however it is not a game-theoretic and not even a MARL paper, with the authors attempting to frame it as one. It would be stronger if it owned its identity as an XRL/MARL/mechanical interpretability paper rather than borrowing game-theoretic language it does not use. And it would be better if the study followed a more or less formal model (for example, IMRAD structure), but it is a subjective comment.