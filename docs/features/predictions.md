# Predictions

**Predictions** (`/predictions`) estimates final Grand Prix finishing order from earlier information. Sprint, where actually held, is an input rather than the target. Choose a saved earlier-weekend session, **Saved input snapshot**, mode and exact UTC cutoff. Event, mode and cutoff must match the immutable snapshot before submission.

## Models and controls

The prepared active comparison has eight candidates: qualifying, rolling form, weekend form, Ridge, Random Forest, Gradient Boosting, XGBoost and TabFM. The UI also retains a grid baseline and historical references; these are not additional members of that active eight-model comparison.

| Candidate | Inputs and interpretation |
|---|---|
| Qualifying | Qualifying order unchanged, without retrospectively applying final-grid penalties. |
| Rolling form | Earlier driver/team race results and explicit priors when history is absent. |
| Weekend form | Qualifying 60%, practice pace rank 30%, Sprint finish 10%; available weights renormalize. |
| Ridge, Random Forest, Gradient Boosting, XGBoost | Qualifying position/gap, clean practice pace/variation/lap count, available tyre age/compound, prior driver/team/circuit form and missingness; earlier-only fitted preprocessing. |
| TabFM | The same tabular fields with earlier labelled races as context. |

No active candidate supplies calibrated winner probabilities. Actual target-race weather, telemetry, pits, results and future DNFs are excluded. Weather needs an actually issued pre-race forecast; penalties/pit-loss estimates require sourced evidence known by cutoff. Missing fields remain missing.

Learned models need an eligible earlier-trained artifact. TabFM needs matching completed inference. **Generate forecast** opens an exact saved run when available; it does not silently substitute another model. Training and large-model inference preparation are operator operations outside rendering. **Use saved cutoff & mode** restores the request; **Use forecast-ready inputs** switches from an older historical snapshot to prepared inputs.

The all-model panel reads immutable prepared results and shows predicted first picks plus a full driver-by-model ranking table. Reruns do not reload large weights. Individual results provide Finishing order, Evidence, Briefing, driver focus and JSON/CSV/Markdown exports. A previous result remains labelled with its original request after controls change.

## Example 1: completed 2025 United States Grand Prix

Select the saved historical pre-Grand-Prix snapshot for United States 2025 and its matching cutoff. Inspect qualifying and an eligible fitted candidate or saved study result. Actual Race results permit retrospective comparison in Model Validation; they must not enter pre-race features. Historical reconstruction lacks verified original publication timing, so the score is research evidence rather than proof of a forecast saved before that race.

## Example 2: Mexico 2026, race dated 1 November

Choose Mexico and an earlier saved session, then **Prepared pre-race inputs · forecasts ready**. Expand **Forecast this upcoming Grand Prix with all models** and click **Forecast with all active models**. The prepared comparison has eight models and 22 drivers. **Generate forecast** opens an individual matching ranking. Only FP1/FP2/FP3/Q are published locally; R collection failed and all prepared forecasts remain unscored. A past calendar date neither supplies actual results nor permits freezing a new prospective forecast after race start. Read the already frozen run.

## Implementation and references

Sources: [prediction screen](../../streamlit_app/screens.py), [artifact selection/reuse](../../streamlit_app/services.py), [forecast service](../../flows/prediction_service.py), [pending preparation](../../ml/pending_forecast.py), [foundation receipts](../../flows/foundation_prediction.py), [features](../../ml/features.py). See [model workflow](../archive-model-workflow.md), [foundation models](../foundation-models.md) and [historical evaluation](../historical-model-evaluation.md).
