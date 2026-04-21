# LLM Micro-Experiment: Routing Legibility in LLM Orchestration

This is a cross-domain probe for the paper's discussion section, not a new main
result. The goal is to test whether the routing-legibility framework can be
translated to an LLM planner that dispatches one of three specialists.

The probe mirrors the paper's distinction between:
- `explicit`: canonical role/type labels are visible
- `implicit`: the same specialists are exposed only through randomized aliases

## Design

- **3 scenario families** x **2 formations** x **2 conditions** = 12 cells, with
  `N` samples per cell.
- Each scenario describes a tactical need without naming the answer directly.
- The planner sees a roster plus the tactical state and must dispatch exactly one
  specialist.
- In `explicit`, the roster uses `Strike`, `Vanguard`, `Support`.
- In `implicit`, the same specialists are renamed to `Agent_X`, `Agent_Y`,
  `Agent_Z`; the alias-to-role map is randomized per sample and stored in the
  raw output for later decoding.
- The runner requests JSON output from Gemini and records both the visible
  dispatch name and the hidden alias map.

## Main Measurements

- **Surface routing matrix**: rows = scenario family, cols = visible dispatch name.
- **Latent routing matrix**: for implicit runs, decode aliases back to the
  underlying role and measure whether the same structure still appears.
- **Surface-name accuracy**: dispatch matches the visible expected name.
- **Latent-role accuracy**: dispatch maps back to the intended underlying role.
- **Surface legibility ratio**: routing-matrix std for explicit visible labels
  divided by the std for anonymized visible aliases.
- **Delta_max**: shift in decoded dispatch distribution between Clustered and Spread.

## Run

```bash
pip install -r llm_micro/requirements.txt
# GEMINI_API_KEY must be in project-root .env
python -m llm_micro.runner --samples 15 --model gemini-2.5-flash
python -m llm_micro.analyze
```

## Outputs (`llm_micro/results/`)

- `dispatches.json`: raw records, including alias maps
- `llm_summary.json`: matrices, accuracies, stds, Delta_max
- `llm_routing_explicit_surface.png`
- `llm_routing_implicit_surface.png`
- `llm_routing_implicit_latent.png`
- `llm_accuracy.png`
- `llm_delta_max_latent.png`

## Interpretation

This probe is useful if it shows a clean separation between:
- **legibility**: explicit visible labels yield a much more structured surface routing matrix
- **competence**: the planner can still often recover the correct latent specialist even when labels are hidden

That pattern would support the paper's discussion claim that the routing
framework may transfer to LLM-based orchestration as a diagnostic tool, while
still keeping the main evidence in cooperative MARL.
