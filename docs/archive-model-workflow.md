# Archive-connected model research

Run offline from the repository root on `ver_2`:

```powershell
& '.\primusEnv\Scripts\python.exe' -m ml.archive_workflow --benchmark --event 2026:19
```

This reads only the version pins in the existing review cohort, opens the F1 SQLite database in read-only mode, and writes derived evidence into ignored `.local/`. It never authenticates, downloads data, loads raw pickles, installs packages, or mutates the source archive. Repeat execution reuses identical snapshots and matching event model artifacts.

`prepare_archive_workflow()` writes a current manifest, separate labels and model rows under `.local/archive_ml/`, plus immutable snapshots under `.local/snapshots/`. The manifest is the authoritative snapshot selection: older derivation versions can coexist on disk. Each snapshot uses only completed FP/Q/SQ/S sessions before the scheduled final race. Its roster is the evidenced qualifying roster, retaining untimed entrants when qualifying classification identifies them. Practice-only reserves and metadata-only identities are excluded. No final-race grid, roster, weather, laps or results enter the target event's features. Missing inputs remain missing; qualifying position is never relabelled as a verified race grid. Completed earlier races supply rolling driver/team/circuit form. Their results remain outside the target event feature payload.

The data was collected retrospectively. Session completion is an explicit research assumption for availability, not evidence of original publication. Saved source fetch timestamps remain intact; `historical_publication_verified=false` is carried through rows, models and reports. These are historical reconstructions, not a live forecasting record.

The benchmark freezes an evaluation protocol and uses expanding chronological development folds, fitting preprocessing and Ridge/Random Forest/Gradient Boosting separately on each fold's earlier events. Qualifying and rolling-form baselines use the identical held-out field. If the qualifying roster lacks race labels, the whole event is excluded with an explicit reason; the race roster never repairs the forecast field. The report includes race-macro metrics, per-race rows, denominators, whole-race bootstrap intervals and fold event IDs. It is published both under `.local/archive_ml/benchmark.json` and `.local/evaluations/` for the UI. All 2026 events remain locked out of evaluation and model fitting. No model promotion or probability calibration occurs.

`artifact_for_event(event_id, candidate)` fits only earlier, labelled, pre-2026 rows, requiring at least twelve development events. Models are versioned in `.local/models/`; JSON sidecars contain metadata and the joblib file SHA-256. Listing models should read sidecars; loading is restricted to the trusted local artifact store. The UI must select an artifact whose training event and availability bounds precede the selected snapshot, then explicitly pass `research_reconstruction=True` for historical reconstruction. Artifacts marked `research_reconstruction_only` cannot enter the live forecast path. Model fitting is a CLI/offline operation, never a page-render operation.

Mexico 2026 qualifying can produce an unscored research forecast while the final race remains pending. This does not unlock the 2026 accuracy holdout.

TabFM and TimesFM adapters are now implemented with real pinned-checkpoint CPU inference in primusEnv. TabFM uses an explicitly unsupported Python 3.10 source compatibility path; TimesFM 2.5 supplies an experimental pace-ranking heuristic. Dependency/checkpoint/source hashes, memory admission and immutable per-event inference receipts are required. See [foundation-models.md](foundation-models.md) and [historical-model-evaluation.md](historical-model-evaluation.md) for the subsequent research evaluation; the conventional benchmark below remains its original frozen evidence.

The frozen run on November 1, 2026 produced 150 snapshots and 3,032 event-driver rows from 743 pinned sessions. The final common development evaluation contains 109 races; 2026 remains locked. Ten otherwise eligible development events are excluded for missing qualifying positions or unequal qualifying/result identity fields. Unlabelled qualifying entrants remain in the proposed test field until that whole-event check; they are never silently dropped. The exact two-way roster check also rejects an otherwise numerically complete field whose identities are unresolved. The first twelve events form the initial training history. Three models were also fitted for the first eligible event of each season (2020 round 13, then round 1 for 2021â€“2026) and for Mexico 2026, giving 24 local artifacts with metadata sidecars.

| Candidate | Spearman | Position MAE | Winner accuracy |
|---|---:|---:|---:|
| Qualifying baseline | 0.6611 | 3.2911 | 56.88% |
| Rolling form | 0.5711 | 3.8939 | 45.87% |
| Ridge | 0.6743 | 3.2787 | 45.87% |
| Random Forest | 0.6639 | 3.3017 | 53.21% |
| Gradient Boosting | 0.6636 | 3.3270 | 48.62% |

Ridge's Spearman improvement is below the existing 0.02 practical-improvement threshold, and it has lower winner accuracy than qualifying. No candidate is promoted. These figures describe reconstructed development folds and must not be presented as live performance or 2026 accuracy.
