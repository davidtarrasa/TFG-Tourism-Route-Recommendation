# Experiments / Baselines

This folder stores immutable config snapshots used in experiments.

Goal:

- keep `configs/recommender.toml` and `configs/recommender_<qid>.toml` as active configs
- preserve known-good states for reproducibility and rollback

Recommended workflow:

1. copy current active config to this folder before major tuning
2. run tuning/evaluation
3. if new setup improves results, promote it to active config
4. keep previous snapshot here as historical baseline

Suggested naming:

- `recommender_baseline_YYYY-MM-DD.toml`
- `recommender_q35765_tuned_YYYY-MM-DD.toml`

Active configs (auto-loaded por ciudad):

- `configs/recommender.toml` — global fallback
- `configs/recommender_q35765.toml` — Osaka: als.factors=64, markov.backoff=0.3, trail_current optimizado
- `configs/recommender_q406.toml` — Istanbul: als.factors=128, markov.backoff=0.25, context_n=1
- `configs/recommender_q864965.toml` — Petaling Jaya: als.factors=64, markov.backoff=0.25, context_n=3

Snapshot de referencia:

- `recommender_baseline_2026-02-07.toml` — estado previo al tuning sistematico
