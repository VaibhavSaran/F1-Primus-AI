# Primus F1 documentation

Start with the [walkthrough](project-walkthrough.md) for the complete user journey and [architecture](architecture.md) for implemented data, provider, ownership and forecast flows. Feature guides describe current Streamlit behavior; specialist references retain deeper collection, research and migration records.

## Feature guides

| Page or tab | Guide |
|---|---|
| Primus AI, home and `/ask` | [Questions, binding, evidence and conversations](features/primus-ai.md) |
| Race Overview | [Classification, practice times and participants](features/race-overview.md) |
| Predictions | [Cutoff-safe inputs, active candidates and saved results](features/predictions.md) |
| Compare Drivers | [Clean laps, sectors and car telemetry](features/compare-drivers.md) |
| Race Replay | [Clock, observed timing, freshness and fallbacks](features/race-replay.md) |
| Strategy Lab | [Observed stints, tyre age and visits](features/strategy-lab.md) |
| Weather & Context | [Conditions, track status and sourced context](features/weather-context.md) |
| Model Validation → Evaluation | [Protocols, cohorts and metrics](features/evaluation.md) |
| Model Validation → Performance Tracking | [Forecasts, outcome revisions and scores](features/performance-tracking.md) |
| Model Validation → Historical Race Studies | [Full predicted/actual race comparisons](features/historical-race-studies.md) |
| Reports | [Conversations, analysis ZIPs and forecasts](features/reports.md) |
| Settings & Data | [Inventory, coverage and readiness](features/settings-data.md) |
| About | [Getting started, page guide and credits](features/about-credits.md) |

Each guide includes completed Spain 2020/United States 2025 examples and Mexico
2026 examples. Some feature-guide examples describe the earlier November 2
collection checkpoint. The updated walkthrough and data tracker reflect the
November 8 release, which includes all Mexico and Sao Paulo sessions. Saved
pre-race forecasts retain their original cutoffs and require explicit outcome
scoring. The hosted Mexico prepared snapshot exposes saved TabFM and XGBoost
runs. Other candidates depend on their eligible fitted artifacts and inputs.

## Data and collection

- [Dated data tracker](data-tracker.md) and [upcoming collection calendar](data-refresh-calendar.md).
- [Archive design/contracts](data-archive.md): schema, source retention, coverage and publication.
- [Collection guide](data-collection-2020-2026.md): notebooks, resumability and offline status refresh.
- [Quality and recovery](data-quality-recovery.md): local repair versus provider gaps.
- [Event context](event-context.md): outcomes, incidents and availability.
- [Dataset exporter](dataset-release.md): implemented curated release structure and validation.
- [Kaggle release design](kaggle-release.md): year-based packaging/data cards.
- [Release status and remaining work](release-status.md): published weather, local F1 packages and outstanding milestones.
- [New-race update plan](new-race-update-plan.md): proposed lifecycle; no installed scheduler implied.

## Models, evaluation and assistant

- [Current model/test results](evaluation-results.md): measured outcomes, denominators and limitations.
- [Archive model workflow](archive-model-workflow.md): pins, features, separate labels, fitting and tracking.
- [Historical model evaluation](historical-model-evaluation.md): temporal tests and interpretation.
- [Foundation models](foundation-models.md): verified checkpoints, inference and limits.
- [Archive-grounded chat](archive-grounded-chat.md): evidence, cutoff and private-record contracts.
- [Primus provider](primus-provider.md): optional Claude, bounded tools, spending and explicit evaluations.
- [Primus evaluation](primus-evaluation.md): assistant evidence/answer-quality evaluation.

## Runtime and project history

- [Environment](../ENVIRONMENT.md): pinned runtime, setup/repair and rollback.
- [Python 3.12 migration](python312-migration.md) and [dependency matrix](python312-dependency-matrix.md): compatibility decisions and evidence.
- [Archive project phases](archive-project-phases.md): progression and earlier scope.
- [Streamlit phased restart](streamlit-phased-restart.md): staged implementation history.
- [Project instructions](../AGENTS.md): environment and working rules.
- [License](../LICENSE.md): repository licensing, separate from upstream data rights.

Earlier references can describe previous phases or experiments. Read their dates/scope alongside current feature guides and actual evidence. Optional integration packages, readiness flags and historical reports do not establish a functioning hosted deployment. Keep credentials, private databases, local evidence identifiers and workstation details out of public releases.
