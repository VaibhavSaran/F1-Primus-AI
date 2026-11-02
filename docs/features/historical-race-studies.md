# Historical Race Studies

Open **Model Validation** (`/validation`) → **Historical Race Studies** to inspect a full race-level comparison. These are saved offline studies, not experiments launched by the page.

## Using a study

Choose Historical study, Historical race and Case-study model. Read the saved earlier input sessions and UTC cutoff. The table shows driver, predicted rank, actual finishing position, signed error and recorded race status. Error is predicted minus actual: positive means the driver finished better than predicted. Retirements remain real outcomes rather than missing telemetry.

Model summaries show race counts, order correlation, mean position error and winner accuracy. Different candidate cohorts trigger a warning: their overall scores are not direct head-to-head evidence. Matched qualifying comparisons use the same events for each challenger and baseline, with correlation changes and intervals. Inspect paired sample sizes.

The loader verifies study/prediction references and digests. Download historical study exports the saved report. Historical reconstruction has unverified original publication timing. TabFM uses prior labelled context, while foundation pretraining overlap is unknown. Actual target-race weather and pit losses are not learned v1 predictors. TimesFM remains only in earlier research records.

## Example 1: completed 2025 United States Grand Prix

Select the active study's United States 2025 case. Compare qualifying and a learned candidate using the full predicted/actual field, including retired drivers. Interpret signed errors, then inspect matched comparisons to assess whether a single good result persists across races. This is a completed historical case with independently joined labels, not verified live forecasting evidence.

## Example 2: Mexico 2026, race dated 1 November

Mexico 2026 is a pending forecast case, not a scored historical study. Only FP1/FP2/FP3/Q are saved locally because R collection failed; eight prepared models cover 22 drivers and remain unscored. Use Predictions and Performance Tracking. Do not add actual-position labels using Q or forecast ranks, or unlock the 2026 benchmark to populate a case study.

## Implementation

Sources: [study UI](../../streamlit_app/screens.py), [study loading](../../streamlit_app/services.py), [historical study](../../ml/historical_study.py), [active study](../../ml/active_study.py), [foundation study](../../ml/foundation_study.py). See [historical evaluation](../historical-model-evaluation.md).
