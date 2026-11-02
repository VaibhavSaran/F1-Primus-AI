# Local foundation-model research

These optional adapters execute real pretrained weights on CPU. They do not
download during page rendering or prediction. Streamlit consumes a saved,
snapshot-matched operator result; model installation is separate from app startup.
No foundation model is automatically promoted or credited with calibrated win odds.

Use the existing `primusEnv` only:

```powershell
& '.\primusEnv\Scripts\python.exe' -m pip install -c requirements.txt -r requirements-foundation.txt
& '.\primusEnv\Scripts\python.exe' -m pip check
```

The optional dependency set is additive to the pinned application environment.
It installs TimesFM3.0.2 software, which retains the TimesFM2.5 implementation.
The runnable TimesFM checkpoint below is **version2.5**, not version3.

## Pinned artifacts and trust boundary

All artifacts live in ignored `.local/foundation_models/`. Only operator-managed,
official public source and safetensors are allowed; no uploaded pickle/joblib,
credentials or remote executable model code is accepted by these adapters.
Install/checkpoint preparation is an explicit operator action. Each checkpoint
directory contains a `manifest.json` with `model_id`, official repository,
exact `revision`, and `files` entries (`path`, `bytes`, `sha256`). Inference checks
the exact expected file set and sizes; model load also verifies every hash.

| Candidate | Repository and pinned revision | Local files |
| --- | --- | --- |
| TabFM1.0 regression | `google/tabfm-1.0.0-pytorch` at `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83` | `tabfm/LICENSE`, `tabfm/regression/config.json`, `tabfm/regression/model.safetensors` |
| TimesFM2.5 | `google/timesfm-2.5-200m-pytorch` at `1d952420fba87f3c6dee4f240de0f1a0fbc790e3` | `timesfm/config.json`, `timesfm/model.safetensors` |

TabFM source is pinned separately to official
`google-research/tabfm@fbb665569425fd2f490c6576b3af967876fe11ff`, extracted under
`source/tabfm-fbb665569425fd2f490c6576b3af967876fe11ff/`. Keep its Apache license.
The downloaded source archive's hash is recorded in `source-provenance.json`.
Its packaging declares Python>=3.11. The pinned PyTorch implementation imports,
constructs and executes real CPU regression on the project's Python3.10.20; this is an **unsupported upstream
runtime experiment**, not a claim that the official packaging supports3.10.
The package is not installed with a Python-requirement override.

## TabFM contract

```python
from ml.foundation_runtime import TabFMAdapter

adapter = TabFMAdapter()
try:
    rows = adapter.predict(
        historical_rows, target_rows,
        feature_columns=feature_columns,
        cutoff_utc=cutoff_utc,
        target_event_id="2025:19",
    )
finally:
    adapter.close()
```

Context rows require `event_id`, `driver_id`, `actual_position` and aware
`outcome_available_at_utc`. A context label must be strictly earlier than cutoff
and from another race. Feature columns must belong to the ML feature whitelist;
labels never enter X. The adapter uses the most recent100 eligible context rows,
one deterministic ensemble member, and at most32 target drivers. Missing features
are left missing for the upstream preprocessing. No network fallback is possible.

TabFM has1,647,782,989 parameters; the official FP32 regression checkpoint is
6,591,243,724bytes. Meta initialization plus `load_state_dict(assign=True)` retains
read-only safetensor mappings instead of allocating a duplicate FP32 model. The
OS can reclaim file-backed weight pages, but paging can make inference slow. A
1.5GB available-RAM admission gate protects working memory; this is not a promise
of low latency or a total memory cap. The adapter does not mutate pretrained
weights. Default TabFM weights permit **non-commercial, non-production research
only**; this local setup is not a production deployment route.

## TimesFM contract

```python
from ml.foundation_runtime import PaceSeries, TimesFMAdapter

series = [PaceSeries(driver_id, event_id, session_code, completed_at_utc,
                     tuple(clean_contiguous_lap_seconds))]
adapter = TimesFMAdapter()
try:
    rows = adapter.predict(series, cutoff_utc=cutoff_utc,
                           target_event_id="2025:19", driver_ids=field_ids,
                           horizon=5)
finally:
    adapter.close()
```

The caller must provide clean, contiguous laps from one stint/compound per
PaceSeries. Never join laps across pit stops, gaps or tyre regimes. Each session
must finish strictly before forecast cutoff; the target Grand Prix's race laps
are prohibited. Full-field driver coverage is required. The adapter selects each
driver's latest session record, with longer runs winning ties, and uses at most256
laps from that record. It does not concatenate records. Minimum3 laps is our
declared short-context heuristic, not an upstream model requirement: TimesFM pads
short contexts, and results need separate evaluation.32+ observed laps are
preferable when they genuinely form an eligible contiguous run.

TimesFM2.5's checkpoint is925,181,104bytes with SHA256
`1cdd295a562dc4af186e2f9742a462008d591ceb22223ff0fa2baaf92c7d9fac`.
It uses Apache-2.0 weights. The adapter uses CPU, FP32 mapped weights, batch1,
two Torch threads, no Torch compilation, and a1.2GB available-RAM load gate.
The median of5 future lap-pace estimates gives an explicitly labelled **pace-based
ranking heuristic**. This excludes explicit future pit loss, caution events,
retirement probability and tyre strategy; it is not a race simulator or calibrated
winner model. TimesFM3 is a distinct multivariate research candidate with separately
restricted weights and has not been installed here.

Both adapters return `event_id`, `driver_id`, `prediction_score`, `predicted_rank`,
`model_id`, `score_units`, and `interpretation`. They never manufacture a probability
or substitute another estimator. Cutoffs accept aware datetime objects or ISO8601
strings. `checkpoint_info()` gives checkpoint hash/revision, adapter source hash,
dependency versions and license; `.close()` releases adapter references. Run one
large model per operator process on a memory-constrained machine.

On Windows, a sequential operator can call `trim_cpu_working_set()` **between**
inference calls when free headroom is low. It invokes `EmptyWorkingSet` on only
`GetCurrentProcess()`; it cannot target a PID or another application. The call
releases resident pages without changing tensor values or unmapping safetensors.
Future accesses can fault pages back in, so this trades speed for headroom. The
study runner must recheck its existing memory admission gate after a trim and
still defer a case if headroom is insufficient. Non-Windows platforms and native
failures are reported without bypassing that gate. Do not call it concurrently
with inference as an attempt to enforce an active-model memory cap.

## Verification and sources

Unit tests cover prior-only labels, whole-field identity, timestamp awareness,
feature allowlisting, no target-race pace, finite outputs and unavailable artifacts.
Both real CPU smokes completed offline with socket connections forbidden:
TimesFM returned finite forecasts for two synthetic pace sequences in4.7s including
load/hash; TabFM returned two finite regression scores in19.7s including load/hash.
The TabFM bounded child process recorded a minimum525MB host available RAM and
exited normally. These are execution checks, never predictive accuracy or a
multi-user performance claim. Actual historical scores are maintained separately
in the operator study reports.

- [Official TabFM source, requirements and license notice](https://github.com/google-research/tabfm)
- [TabFM regression weights and terms](https://huggingface.co/google/tabfm-1.0.0-pytorch)
- [Official TimesFM implementation](https://github.com/google-research/timesfm)
- [TimesFM2.5 weights and license](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [TimesFM3 weights and separate license](https://huggingface.co/google/timesfm-3.0-pytorch)
- [Windows EmptyWorkingSet API](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset)
