# Evaluation

Open **Model Validation** (`/validation`) → **Evaluation**. This tab reads saved offline reports. Changing controls does not fit models, rerun a benchmark, unlock a holdout or obtain missing outcomes.

## Controls and metrics

The current active comparison appears first. **Include earlier research evaluations** exposes preserved reports. **Benchmark size** selects 5, 10, 50 or 100 nested chronological test races: latest five inside ten, fifty and hundred. This changes test cohort size, not a learning curve's training size. Candidates share the chosen test races; walk-forward folds train only on earlier outcomes. Qualifying is a deliberate simple reference that predicts qualifying order, not a retrospectively corrected starting grid.

Choose all models or an **Evaluation model**. Whole-order Spearman correlation is better higher; mean position error is better lower. Podium ranking/recall, Top 10 overlap and winner accuracy measure different objectives. Every metric shows its own race count. The 95% intervals resample whole races; small samples remain uncertain. Ranking-only outputs cannot establish Brier score, log loss or calibrated winner probability. **Show metrics this model cannot yet provide** exposes unavailable quantities without converting them to zeros.

**Data readiness & evaluation protocol** records cohort, temporal split, preprocessing, entrant/outcome handling and scoring. A zero-race coverage audit is not an accuracy benchmark. Historical reconstruction uses hindsight-downloaded prior sessions with assumed original availability. It supports research, not verified live performance. Foundation-model pretraining overlap is unknown. The historical development benchmark excludes the locked 2026 holdout.

## Example 1: completed 2025 United States Grand Prix

Use the active historical evaluation, then open Historical Race Studies for the United States 2025 case and inspect predictions/actuals. Return to Evaluation to compare qualifying with a challenger on the same cohort. A cohort correlation is not that single race's score. A completed race allows independently joined labels but does not prove the forecast existed before it.

## Example 2: Mexico 2026, race dated 1 November

The pending panel may show eight models and 22 drivers, but those prepared forecasts remain unscored. Only FP1/FP2/FP3/Q are published locally because R collection failed. Mexico is not an extra completed test merely because it has a calendar date or prepared ranking. Inspect awaiting outcomes in Performance Tracking and preserve the separate 2026 research holdout.

## Implementation

Sources: [validation UI](../../streamlit_app/screens.py), [report selection](../../streamlit_app/services.py), [active study](../../ml/active_study.py), [protocols](../../ml/protocols.py), [evaluation](../../ml/evaluation.py). See [historical evaluation](../historical-model-evaluation.md), [foundation models](../foundation-models.md), and [Historical Race Studies](historical-race-studies.md).
