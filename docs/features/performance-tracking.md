# Performance Tracking

Open **Model Validation** (`/validation`) → **Performance Tracking** for immutable forecasts and later outcome evidence. Prediction quality and service request health are separate.

## How tracking works

**Performance evidence** selects Saved pre-race forecasts or Historical reconstructions; these cohorts never mix. Choose a model and All outcomes, Scored, Awaiting outcome or Incomplete matching field. Cards count model/race pairs, not unique races across models. Tables show metric-specific denominators and race history.

Summaries select the latest saved cutoff per event/model/forecast mode. Older runs remain in **Saved forecast runs** but are labelled outside the current summary instead of inheriting a newer score. A genuine outcome revision makes old scores stale until rescoring. A result with a different driver set is incomplete matching evidence, not a future race or a zero-error score.

`TrackingQueries` enforces server-established owner/role and public visibility. Outcome/score history is accessible only for visible forecasts. Forecasts, outcome revisions and protocol-specific scores are separate records; appending an outcome does not rewrite the ranking or cutoff.

**Request health** shows the current owner's aggregate failures/runtime. Fast requests do not imply accurate forecasts. Raw operational events and costs remain operator concerns.

## Example 1: completed 2025 United States Grand Prix

Choose Historical reconstructions and the United States 2025 saved run where available. Inspect its selected cutoff, outcome revision and matching score; open the original ranking in Reports. It remains retrospective research because original publication timing is unverified. An older run must not inherit the latest run's score.

## Example 2: Mexico 2026, race dated 1 November

Choose Saved pre-race forecasts and Awaiting outcome. The prepared eight-model comparison has 22-driver rankings frozen before the Grand Prix. Only FP1/FP2/FP3/Q are published locally because R collection failed. No Mexico actual Race result or score is available in this state. Pairs remain unscored rather than counting as misses, zero error or historical benchmark additions. Later genuine outcomes require explicit append/scoring without rewriting frozen inputs.

## Implementation and references

Sources: [tracking UI](../../streamlit_app/screens.py), [read boundaries/summaries](../../monitoring/queries.py), [records/scoring](../../monitoring/store.py), [historical projection](../../flows/archive_tracking.py). See [model workflow](../archive-model-workflow.md) and [new-race plan](../new-race-update-plan.md); the latter is planning, not an installed scheduler.
