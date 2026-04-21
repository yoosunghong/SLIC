# ICML 2026 Workshop Submission Checklist

## Positioning

- Confirm the workshop title and scope are referenced correctly in the submission form.
- Use the main paper message consistently: this is a mechanism-and-analysis paper about routing legibility and the translation gap, not a state-of-the-art MARL paper.
- Keep the contribution statement aligned across title, abstract, intro, and submission summary.

## Manuscript

- Verify the title matches the final PDF metadata and submission form.
- Re-read the abstract for word-limit compliance and consistency with final numbers.
- Check that all section numbers, table numbers, and figure numbers render correctly in the PDF.
- Confirm all numbers in the text match the final tables and plots:
  - `Cond 1 = 0.652`
  - `Cond 3 = 0.440`
  - `Cond 4 = 0.474`
  - routing std `0.275` vs `0.055`
  - `Delta_max`: Strike `0.028`, Vanguard `0.035`, Support `0.103`
- Make sure the paper never overclaims the failed behavioral predictions.
- Keep Appendix A clearly framed as secondary and non-core evidence.

## Figures

- Open the PDF and confirm every image path resolves.
- Check that figures with spaces in filenames render correctly.
- Make sure figure captions are readable at workshop page scale.
- Ensure the architecture figure is legible in grayscale or low-resolution preview.

## References

- Verify every citation is real and complete.
- Double-check that no placeholder or uncertain citation remains.
- Ensure reference style matches the workshop template.

## Reproducibility

- Confirm the description of seeds, metrics, and environments matches the codebase.
- Verify whether `lr = 1e-4` and entropy `0.03` are the intended final values.
- Make sure the text about grid-world replication matches the actual analysis outputs.

## Anonymity

- Remove or check any accidental author-identifying metadata in the PDF.
- Confirm there are no repository names, personal paths, or hidden comments in the submission files.
- Check PDF properties for author name leakage.

## Submission Form

- Prepare a 2-3 sentence workshop-fit summary:
  - The paper studies the gap between theory-informed strategic prescriptions and conventions selected by gradient-based multi-agent learning.
  - It contributes a measurement framework for routing legibility and shows that attention makes role-conditioned dependency more observable than MLP baselines.
- Prepare 3-5 keywords:
  - multi-agent reinforcement learning
  - game-theoretic learning
  - attention
  - interpretability
  - strategic dependency

## Nice-to-Have Before Upload

- Compile the final PDF once more from a clean state.
- Do one cold read of only the abstract, intro, and discussion to check narrative clarity.
- If page pressure exists, keep Appendix A but consider moving detail to supplement.
