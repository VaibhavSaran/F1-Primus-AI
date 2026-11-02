# Evaluation and verification results

This report separates **model research**, **application tests** and **assistant
checks**. A passing software test is not a prediction-accuracy metric. Results
below describe recorded local experiments; a clean checkout does not include
their private input archives/checkpoints or reproduce the tables automatically.

## Active model comparison

The canonical active study contains 109 matched, eligible development races
before 2026. Training uses earlier outcomes and fold-local preprocessing. The UI
also exposes nested latest-5/10/50/100 cohorts with identical race sets across
models. These are evaluation sample sizes, not training-size learning curves.

| Model | Scored races | Order correlation (Spearman) | Mean position error | Winner accuracy |
|---|---:|---:|---:|---:|
| Qualifying baseline | 109 | 0.6611 | 3.2911 | 56.88% |
| Rolling form | 109 | 0.5711 | 3.8939 | 45.87% |
| Weekend form | 109 | 0.6157 | 3.6592 | 28.44% |
| Ridge | 109 | 0.6743 | 3.2787 | 45.87% |
| Random Forest | 109 | 0.6639 | 3.3017 | 53.21% |
| Gradient Boosting | 109 | 0.6636 | 3.3270 | 48.62% |
| XGBoost | 109 | 0.6559 | 3.3720 | 57.80% |
| TabFM | 109 | 0.6578 | 3.3260 | 58.72% |

Higher correlation/winner accuracy and lower position error are better, but
different metrics answer different questions. The baseline assumes qualifying
order equals finishing order and does not retrospectively apply grid penalties.
The active report is selected by a checksum-verified local pointer; previous
studies remain immutable rather than being overwritten.

### Interpretation and uncertainty

TabFM's winner-accuracy difference from qualifying is +1.83 percentage points;
the paired whole-race bootstrap 95% interval is approximately **−7.34 to +10.09
points**. XGBoost's interval also includes zero. There is no established reliable
winner-prediction improvement and no promotion claim. Ridge has the highest
whole-order correlation in this comparison, not the highest winner accuracy.

Original historical publication timing was not recorded. These are chronological
**historical reconstructions**, not verified as-published pre-race backtests.
Future-data leakage controls within reconstruction do not repair that missing
historical evidence. The 2026 evaluation holdout stays locked. Mexico's eight
saved 22-driver forecasts are unscored until a genuine race outcome is ingested
and operational scoring is explicitly performed; do not insert them into the
locked development comparison.

TimesFM's earlier direct-pace heuristic scored 106 races, with poor results and
three missing complete clean-run fields; it is archived and excluded from the
active eight. No verified official TabFM v3.0 checkpoint was used. Pace RMSE,
Brier score and log loss remain unavailable when the model supplies neither
race-pace targets nor calibrated probabilities. They are not silently zero.

## Application verification

The repository-preparation pass on **November 2, 2026** recorded **926 passed,
1 skipped in 69.39 seconds** under Python 3.12.10. It includes release hygiene,
public-tracker, About-page and local-report preservation regressions. The prior
milestone recorded 908 passed, 1 skipped. Ordinary tests default to offline mode
and do not download races or spend provider credits.

The final publishable-tree check inspected 258 present files from 265 candidate
paths (including absent tracked deletions) with zero pattern/artifact findings.
A separate history audit covered 18 reachable commits and 42 unique blobs with
no credential-pattern findings. Exact values from locally configured secrets
also had no matches in the candidate files. These are bounded checks, not proof
that every possible secret format is detectable. All 40 public Markdown documents
checked had valid local links. Remote CI and a Docker/cloud build were not run.

Recorded actual-data acceptance includes a 38-case Streamlit route/action matrix
under Python 3.12, eight separate Mexico model submissions with 22-driver fields,
and browser checks of replay, cited chat, Markdown chat download and TabFM ranks.
Ridge, Random Forest, Gradient Boosting, XGBoost and actual TabFM inference
reproduced their saved Mexico scores exactly across the runtime migration.
These checks require local data and are distinct from clean-checkout CI.

## Assistant evaluation

- Earlier deterministic grounded mode passed 86 independent SQL-oracle factual,
  cutoff, profile and adversarial cases. It is not the same system as live Claude.
- A recorded live Claude suite passed 7/7 integration cases: engineering with web
  sources, internal race results, context mismatch, weekend-format question,
  saved forecast, injection and mixed-domain refusal. An earlier citation failure
  was retained, repaired and retested rather than counted as a success.
- Bound Barcelona qualifying/race comparisons and the exact Kimi question were
  additionally exercised. The browser qualifying answer's P3/22 and 0.319-second
  gap were checked against stored results. Full-field tool output replaced an
  overly narrow six-driver projection.
- Independent public-passage checks examined answer completion, reference
  relevance, authority, event/session fit and factual support. Earlier unknown
  or failed judgments remain evidence, not universal source-quality passes.

Finite cases cannot prove universal factual correctness or jailbreak immunity.
Owner isolation, read-only tools, citation validation, input/output/tool/time
limits, shared budgets and refusal checks are separate defense layers.

## Reproduce the relevant check

```powershell
& '.\primusEnv\Scripts\python.exe' -m pytest tests -q --basetemp=.scratch/pytest-verification
& '.\primusEnv\Scripts\python.exe' scripts/check_repository.py
& '.\primusEnv\Scripts\python.exe' scripts/update_data_tracker.py
```

For model execution see [historical evaluation protocol](historical-model-evaluation.md),
[model workflow](archive-model-workflow.md) and [foundation setup](foundation-models.md).
For explicitly paid checks see [provider evaluation](primus-provider.md).
Record the cohort, cutoff, model version, command and denominator with every new
result. Never turn unavailable artifacts into a claimed successful benchmark.
