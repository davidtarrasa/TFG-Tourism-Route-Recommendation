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

Current snapshot example:

- `recommender_baseline_2026-02-07.toml`
