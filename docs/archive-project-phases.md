# Archive-connected project checkpoint

Implemented locally on `ver_2`, November 1, 2026. All project Python commands
use `primusEnv` from the project root. Mexico collection remains with the user.
No deployment, external inference, Kaggle upload or new subscription is included.

## Data readiness

The pinned cohort contains 743 sessions and 456 strict review sessions. Its ID is
`87cf0ea1a028ed7d738409699e472741e251a9a4b3c952176fb73fab02535c83`.
The original 207 partial flags remain preserved. Entrant-aware review explains
93 as available entrant channels plus supplemental unmatched raw-stream flags;
other gaps and identities remain explicit. This is not gap-free certification.
2026 scope is through November 2 inclusive, with 94 sessions and 18 weather
weekends collected; Mexico race/full-weekend weather are not yet collected.

## Predictions

All ten Streamlit routes are enabled locally. Predictions reads only the selected
event's active frozen input snapshot. Select Qualifying, Rolling form, Weekend
form, or an eligible fitted Ridge, Random Forest or Gradient Boosting version.
The verified-grid baseline remains separate and can be unavailable. Switching
models never trains on a page render. The finishing-order table, focus driver,
input limitations and JSON/CSV/Markdown downloads use the saved result.

The workflow produced 150 snapshots and 3,032 event-driver rows. Qualifying
rosters exclude practice-only reserves and metadata placeholders; target-race
results never supply features or replace a qualifying entrant. Twenty-four
fitted research artifacts cover the first eligible event of each season and
Mexico 2026. Earlier events with insufficient training history retain baselines.

Tyre/practice/qualifying/sprint inputs use earlier sessions. Missing weather
forecasts, verified grid penalties and sourced pit-loss estimates remain missing.
ERA5 and target-race measured weather are not substituted for pre-race forecasts.
Foundation adapters now execute real local research inference: TabFM finishing-position regression and TimesFM 2.5 pace ordering. The UI enables only completed inference pinned to the exact snapshot/cutoff and does not load weights on reruns. See [historical-model-evaluation.md](historical-model-evaluation.md) and [foundation-models.md](foundation-models.md) for evaluation, coverage, resource and runtime restrictions.

## Evaluation and performance tracking

The original evaluation compares five candidates on 109 common chronological development
races; 2026 is locked. Every fold fits preprocessing and models on earlier
races only. Historical input publication times remain reconstructed assumptions,
so these are retrospective research comparisons, not verified live backtests.
No model is promoted. Ranking, position error, winner/top-finish diagnostics and
whole-race confidence intervals are displayed with their actual race counts.
Win probabilities/calibration scores remain unavailable for ranking-only models.

Performance Tracking contains 30 scored retrospective runs across the last five
common 2025 development races. UI-generated 2026 runs remain unscored; the holdout is
not automatically unlocked. Saved runs, outcome revisions and score histories
are immutable. Owner-scoped request health includes numeric forecast and
pseudonymous chat events, without exposing raw prompts or other users' events.
The sanitized evaluation projection is also saved in local MLflow SQLite.

## Primus AI and profiles

Primus AI is a guarded deterministic local evidence assistant. It answers saved
classification, lap, tyre, weather, seasonal driver/team form and curated incident
questions with citations. It uses no external LLM or search, requires no API key
and does not claim unrestricted generative conversation. Profiles are seasonal;
observed affiliation is not a current-contract claim. Recent form and season-long
status counts are separately described.

Authorization, shared admission limits, tool/token bounds, F1 scope and read-only
tools remain enforced at the service boundary. Source text is untrusted. There
is no arbitrary SQL, shell, filesystem, URL fetch or hidden forecast action.
Unknown winners/incident causes abstain; future winner requests hand off to
Predictions. Knowledge cutoffs include local retrieval, not only old event dates.
Private provider keys would be needed for a separately configured generative
provider; none were requested, stored or used in this phase.

## Reports and local release packages

Reports exports a small selected-session ZIP of classification, laps, stints,
pit visits, status and weather with a pinned manifest and member hashes. Forecast
exports retain exact models/cutoffs without showing code or JSON blocks onscreen.
Views use full driver names and readable race/model labels where available.

All seven yearly local packages are prepared under `.local/release_packages`.
They contain 30,063 normalized dense Parquet members and portable relational
catalogues, measured/reanalysis weather, explicit partial/cancelled states,
documentation, checksums and standalone example notebooks. Total size is about
40.1 GB decimal. Every year's example ran offline. Hashed context-availability
sidecars must accompany each package. 2026 is labelled in progress.
Packages are not published; redistribution/licence and dataset metadata need
review before a future Kaggle upload. Raw caches, pickles and credentials are
excluded. See [dataset-release.md](dataset-release.md).

## Operator entry points

```powershell
& '.\primusEnv\Scripts\python.exe' -m ml.archive_workflow --benchmark
& '.\primusEnv\Scripts\python.exe' -m flows.archive_tracking
& '.\primusEnv\Scripts\python.exe' -m streamlit run streamlit_app/app.py --server.port=8502
& '.\primusEnv\Scripts\python.exe' -m data.archive.release
& '.\primusEnv\Scripts\python.exe' -m pytest tests
```

The default workflow pins the current cohort; future collection does not rewrite
it. A new cohort requires a new versioned benchmark output and explicit activation
of its snapshots/artifacts, rather than overwriting past experiment evidence.
Prepared releases are immutable and require a new release version for new data.

## Evidence

Full tests and actual-data Streamlit checks are recorded in ignored `.scratch/`.
The actual-data matrix covered all ten pages across 2020 Spain, 2025 Abu Dhabi
and 2026 Mexico qualifying, plus fitted forecast, pending-race and chat actions.
Browser testing verified fitted-model submission, full names, weather citations,
profile tabs and model-validation navigation. All seven release validators check
hashes, ZIP CRC, rows, exact clocks/IDs, catalogue joins and metadata leakage.
These tests do not establish live model accuracy, production readiness or absence
of provider-side data gaps. Mexico and the declared experimental prerequisites
remain the outstanding collection/runtime gates.
