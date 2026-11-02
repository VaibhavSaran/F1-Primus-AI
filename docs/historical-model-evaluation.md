# Historical model and Primus evaluation

The current evaluation plan is implemented as local, reproducible research on
the version-pinned archive. Historical session completion is used as an
availability assumption; original publication timing is unverified. These are
retrospective experiments, not a record of predictions made live in 2025.

The active comparison now contains qualifying, rolling form, weekend form,
Ridge, Random Forest, Gradient Boosting, XGBoost and TabFM. TimesFM has been
removed from active choices; its earlier immutable studies remain historical
evidence. The eight-model section below records the earlier run, before this
change.

## Active comparison and standard cohorts — November 1, 2026

The current report is selected by `.local/archive_ml/active_study.json`.
It retains all 109 complete-field development races and adds deterministic
nested benchmarks: the latest 100 eligible chronological races, then the last
50, 10 and 5 of that same set. Every active candidate uses the same event IDs
and complete driver fields in each benchmark. Counts are not sample-size knobs
that select a different cohort for each model. Qualifying order is the reference.

The saved 109-race TabFM receipts are reused without rerunning the model.
XGBoost 3.2.0 completed 109 actual expanding chronological folds, with median
imputation, scaling and categorical encoding fitted only to each training
fold. Its fixed configuration uses 100 histogram trees, depth 3, learning rate
0.1, squared-error regression and seed 42. This follows the boosted-tree
candidate idea in the [2025 reference project](https://github.com/mar-antaya/2025_f1_predictions)
and [2026 reference project](https://github.com/mar-antaya/2026_f1_predictions),
using this project's cutoff-safe features and full-field protocol.

XGBoost's full 109-race Spearman correlation is 0.6559, position MAE 3.3720
and winner accuracy 57.80%. The winner advantage over qualifying is 0.92
percentage points with a paired 95% whole-race interval of -9.17 to +11.01
points. These results do not justify promotion. Each standard cohort includes
whole-race bootstrap intervals and paired differences against qualifying.
Real XGBoost fit/inference times are saved per fold. Missing timings for older
cached candidates remain explicitly unavailable; no timings are inferred.
Unavailable probability and pace metrics stay null.

The active report is
`.local/model_studies/19fe53e32551d33c1850b4c951763d243274cd709926d2bf0c3e14ff9a5eb804/`.
Its protocol, exact event IDs, source hashes, XGBoost metadata and original
study hashes are retained. The earlier TimesFM report is unchanged.

## Pending Mexico 2026 forecast

All eight active models have actual, unscored 22-driver forecasts frozen at
**1 November 2026, 15:33:28.735100 UTC**, before the 20:00 UTC Grand Prix.
The saved input snapshot contains FP1, FP2, FP3 and qualifying only; Mexico is
a standard weekend, so no sprint input is expected. The original archive
retrievals and completed earlier sessions precede the frozen cutoff. The old
future race-minus-one-second research cutoff was replaced by an immutable
snapshot frozen at the actual preparation time.

Conventional artifacts and TabFM context use labelled pre-2026 training rows.
The target race's labels and telemetry never enter inference. TabFM executed
real CPU inference once for this complete field. The other models produced
their own numeric scores. These runs have no actual race score yet and do not
unlock the 2026 benchmark holdout. When an official outcome becomes available,
the saved forecasts can be scored as prospective tracking evidence separately
from that locked development-evaluation protocol.

The owner-scoped summary is `.local/archive_ml/pending-2026-19.json`.
`ml.pending_forecast.cached_pending_forecast` reads a saved numeric run without
fitting, importing Torch or downloading anything. The operator-only CLI prepares
inference; UI and Primus consume the saved immutable results:

```powershell
& '.\primusEnv\Scripts\python.exe' -m ml.active_study
& '.\primusEnv\Scripts\python.exe' -m ml.pending_forecast --event 2026:19
```

## Protocol

The saved conventional benchmark contains 109 complete-field development races
from 2020–2025 after an initial twelve-race training window. Ten otherwise
eligible races are excluded for missing qualifying positions, unequal entrant
fields or unresolved identities. Never reduce the field to the intersection
after observing outcomes. All 2026 races remain locked from fitting/evaluation.
Practice-only reserve drivers do not define the final-race forecast field.

Ridge, Random Forest and Gradient Boosting are fitted inside expanding temporal
folds. Imputation, scaling and encoding see only earlier training rows. Their
v1 inputs include qualifying order/gap, practice pace/variation/lap count, tyre
age/compound and prior driver/team/circuit form, where available. Missing
pre-race weather forecasts, verified grid penalties and historical pit-loss
estimates remain missing. The separately evaluated weekend heuristic combines
qualifying, practice pace rank and sprint results with fixed 60/30/10 weights.
Sprint qualifying is retained as source context but is not a v1 model feature.

The study runner independently joins model scores with the saved race-label
file and reproduces every cached benchmark metric before accepting it.
It adds the weekend heuristic on the same 109 races. Reports include full
per-driver predicted/actual positions, signed errors, recorded race status,
ranking correlation, position MAE, podium NDCG/recall, top-ten overlap, winner
accuracy and winner-in-top-three. Whole-race bootstrap intervals and paired
model-minus-qualifying intervals use identical race pairs. A negative MAE
difference is favourable; a positive correlation difference is favourable.
Probability calibration and pace-error metrics are unavailable without their
required model outputs and independently defined targets.

## United States Grand Prix 2025 case study

Event `2025:19` uses a cutoff of **19 October 2025, 18:59:59 UTC**. Recorded
earlier sessions are Free Practice 1, Sprint Qualifying, Sprint and Qualifying.
The final Grand Prix's results, laps, weather and grid are not model inputs.
Only after predictions are produced are actual results joined for comparison.

| Model | Predicted first | Predicted second | Predicted third |
|---|---|---|---|
| Qualifying baseline | Max Verstappen | Lando Norris | Charles Leclerc |
| Ridge | Max Verstappen | Lando Norris | George Russell |
| Random Forest | Max Verstappen | George Russell | Lando Norris |
| Gradient Boosting | Max Verstappen | George Russell | Lando Norris |
| Rolling form | Max Verstappen | George Russell | Oscar Piastri |
| Weekend heuristic | Charles Leclerc | George Russell | Lando Norris |

The saved actual winner is Max Verstappen. Correctly choosing one winner does
not establish model quality: the full field and repeated historical races are
evaluated. Across the 109-race conventional comparison, qualifying winner
accuracy is 56.88%, Ridge 45.87%, Random Forest 53.21%, Gradient Boosting 48.62%,
rolling form 45.87% and weekend heuristic 28.44%. Ridge has the best conventional
order correlation (0.6743) but its improvement over qualifying (0.6611) is below
the existing practical-improvement threshold. No candidate is promoted.

## Foundation challengers and reproducibility

TabFM performs actual in-context finishing-position regression using at most
100 most recent labelled prior-race rows and the held-out pre-race feature
table. Its pinned official PyTorch source is an explicitly unsupported Python
3.10 compatibility experiment; official packaging requests Python 3.11+.
The regression safetensors checkpoint is about 6.6 GB. Memory-mapped loading
and resource gates avoid allocating a second full FP32 heap model.

TimesFM uses the official 2.5 200M PyTorch checkpoint (about 925 MB) on CPU.
It receives each driver's latest eligible contiguous clean run, with longest
run breaking same-session ties. Pit visits, compound changes, inaccurate laps
and gaps break sequences; different sessions are never concatenated.
A minimum three clean laps is fixed before reviewing challenger results;
short contexts are padded upstream and explicitly flagged. The median next
five forecast lap times gives a **pace-ordering heuristic**. It does not model
traffic, tyre/compound comparability, pit strategy, overtaking, safety cars or
reliability, and is not a calibrated winner forecast. TimesFM 3 is a separate
future multivariate candidate; installing the package does not mean its weights
were used. See [foundation-models.md](foundation-models.md) for pinned sources,
versions, licences, resource requirements and setup evidence.

Foundation experiments freeze implementation/checkpoint/dataset hashes before
inference and write a per-event receipt. They are resumable: completed immutable
receipts are reused. Missing full-field inputs are reported as unavailable,
never replaced with baseline scores. The matched-cohort study compares each
challenger against the same qualifying races and reports its own coverage.
Potential overlap of foundation pretraining data with historical races remains
unknown; these experiments do not prove uncontaminated zero-shot performance.

```powershell
& '.\primusEnv\Scripts\python.exe' -m ml.historical_study
& '.\primusEnv\Scripts\python.exe' -m ml.foundation_study timesfm
& '.\primusEnv\Scripts\python.exe' -m ml.foundation_study tabfm
```

Outputs live in ignored `.local/model_studies/` and
`.local/foundation_evaluations/`. Each foundation run prints its output folder.
Pass its `predictions.json` and `metadata.json` to the historical runner's
`--additional` and `--metadata` options. Combine candidate maps when comparing
multiple foundation runs; preserve the original per-event receipts and hashes.
Streamlit's **Historical Race Studies** tab exposes the model table and a race
selector for all 109 historical events, defaulting to the US example. Full
prediction files are digest-checked before display. Predictions enables a foundation candidate only when completed
inference exists for the exact selected snapshot, cutoff, cohort and field.
Rendering never loads weights or downloads data. Ordinary forecast permission,
admission and owner-scoped immutable tracking apply to receipt-backed results.

## Primus AI

Primus is configured for bounded, cited local Q&A. It uses read-only tools and
no external generative provider. Independent direct-SQL expected answers test
facts, source citations, historical cutoffs, unknowns, seasonal profiles and
F1-only scope. The executed evaluation passed 86/86 cases, with 95 focused
regression tests. Mixed-topic requests, instruction overrides, secret/file/code
requests and injected source instructions are refused before unsafe actions.
There is no arbitrary SQL/shell/URL execution capability.

```powershell
& '.\primusEnv\Scripts\python.exe' -m agents.chat_evaluation
```

See [primus-evaluation.md](primus-evaluation.md) for actual questions, responses,
oracle design, guardrails and limits. Passing a finite attack suite is not proof
against all jailbreaks or factual errors. Any later external LLM requires its
own provider configuration, budget and adversarial re-evaluation.

## Completed eight-model run — November 1, 2026

| Model | Evaluated races | Spearman | Position MAE | Winner accuracy |
|---|---:|---:|---:|---:|
| Qualifying baseline | 109 | 0.6611 | 3.2911 | 56.88% |
| Rolling form | 109 | 0.5711 | 3.8939 | 45.87% |
| Weekend heuristic | 109 | 0.6157 | 3.6592 | 28.44% |
| Ridge | 109 | 0.6743 | 3.2787 | 45.87% |
| Random Forest | 109 | 0.6639 | 3.3017 | 53.21% |
| Gradient Boosting | 109 | 0.6636 | 3.3270 | 48.62% |
| TabFM | 109 | 0.6578 | 3.3260 | 58.72% |
| TimesFM 2.5 pace heuristic | 106 | 0.1752 | 5.8025 | 16.98% |

TimesFM lacks a complete three-clean-lap field for Hungary 2023 (Alonso),
Azerbaijan 2024 (Ocon) and Azerbaijan 2025 (Antonelli); those races are excluded
only from its coverage. Its matched 106-race qualifying winner baseline is
56.60%, so its lower score is not an artifact of the missing three races.

TabFM's winner accuracy exceeds qualifying by 1.83 percentage points, but the
paired 95% whole-race bootstrap interval is **-7.34 to +10.09 points**. Its
whole-field correlation is slightly lower than qualifying. Ridge's correlation
lift is +0.0132 with interval -0.0040 to +0.0304. These results do not establish
reliable improvement or justify promotion. TimesFM's direct pace ordering is
substantially worse on the matched cohort; retain it as an experimental pace
component, not the default winner model.

For US2025, TabFM predicted Verstappen/Leclerc/Russell and TimesFM predicted
Verstappen/Hamilton/Sainz. The actual podium was Verstappen/Norris/Leclerc.
Ridge's field position MAE was 2.7, TabFM's 3.1 and TimesFM's 4.6. All three
picked the winner, while their overall field accuracy differed considerably.

TabFM finished every development race. Its initial memory-deferred results and
retry attempts remain preserved. Own-process Windows working-set trimming keeps
file-backed weights reclaimable between events; the 350 MB admission floor was
not lowered. The inference-class AST was identical across the memory-only
change before combining receipts. No outcome-based choice among retries occurs.

Final local study: `.local/model_studies/691db355d012e9a7e386d556347a736b80fb724cbf155d88de08e99dbc2f628a/`.
It contains report.json/report.md, full predictions.json with a verified digest,
and case-comparison.csv. Every event's source/cutoff and each candidate's actual
coverage, fold results, paired intervals and checkpoint/source provenance are
retained. Three additional pre-US fitted artifacts reproduce the saved
conventional US fold predictions numerically; 27 fitted artifacts now exist.
Thirteen additional runs were scored in local tracking (three exact-US fitted
models plus five recent races for each foundation candidate). The full
sanitized comparison is also projected to local MLflow.

Verification: **863 tests passed, one skipped**; the 33-case actual-data
Streamlit matrix passed with provider networking disabled. Both real foundation
CPU adapters passed offline smoke tests. Primus passed 86 independent factual,
cutoff, profile and adversarial cases. The finite tests and retrospective data
limitations remain explicit.
