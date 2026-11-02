# Streamlit restart: one feature at a time

## Current archive-connected checkpoint — November 1, 2026

The local phases from data readiness through Predictions, Evaluation/Tracking,
Primus AI/profiles and Reports are now connected to the bulk archive. Local
yearly release packages are also prepared; no upload occurred. See the current
[phase checkpoint](archive-project-phases.md) for capabilities, actual evidence
and limitations. Historical model comparisons remain research reconstructions,
2026 evaluation stays locked, and Mexico collection is reserved for the user.
The dated checkpoints below are historical context.

## Historical checkpoint — October 27, 2026

The telemetry UI and replay timing integration are implemented. The shared
selectors, Overview, Compare Drivers, Replay, Strategy Lab, Weather & Context
and data coverage views use the published local archive. Compare/replay support
the full recorded field. Replay shows recorded race/sprint positions, leader
gaps, car-ahead intervals and lap deficits, synchronized with playback; weather
and event markers follow the same clock.

Timing comes from verified, version-linked retained FastF1 state snapshots.
Snapshots can carry earlier reported field values; these are historical provider
states, not proof of exact broadcast refresh times. Missing values clear earlier
values, and snapshots older than two minutes become unknown. A driver subset
retains absolute race ranks and intervals to the car ahead in the full field.
Practice/qualifying retain completion analysis. Without a usable timing stream,
recorded lap-end positions are labelled checkpoints and gaps remain unknown.
Final classification and map coordinates never supply missing running standings.

Verified with full-field 2020 Spanish and 2026 Japanese races, Python/browser
clock parity, rewind, selected-driver consistency, and browser playback/seek.
The telemetry features do not establish complete 2020–2026 data coverage.
Next: audit remaining data completeness, then develop Predictions, Validation,
Primus AI/profiles and Reports in that order. Forecasting/chat remain deferred.

## Historical pilot checkpoint — October 25, 2026

The single-session walkthrough was implemented on October 25, 2026. The user
approved using the current pilot to explore the phases before forecasting/chat.
The verified pilot is Australian Grand Prix 2025 Practice 1: 20 drivers, 421 laps,
362,700 car samples, 370,640 positions and 81 weather observations. This is not a
complete race or a complete 2025/2026 collection. No further downloads are running.

The shared selector, Overview, Compare Drivers, Replay, Strategy Lab and Weather
& Context now read this SQLite/Parquet archive. Try them at localhost:8502.
Comparison telemetry is loaded explicitly for the selected driver/lap. Replay
plays the entire saved session for the selected recorded drivers, using sampled observed
coordinates; all 20 drivers remain available for comparison. Practice pit visit
durations include lane/garage time, and tyre-age plots are descriptive.
Forecasting, model validation development and Primus AI remain deferred.

October 26 update: collection scope is now 2020–2026. Separate F1 and weather
notebooks for each year are prepared in the
[collection guide](data-collection-2020-2026.md); the user starts them locally.
The older two-year pilot notebook remains a reference, not the bulk-run entrypoint.

Keep Streamlit, `ver_2`, and `primusEnv`. Reuse the existing screens and components;
connect and verify each feature before moving to the next. No cloud deployment,
frontend migration, background collector or paid API purchase is part of this
restart. The old source scripts remain for reference but are not automatically run.

## Working agreement

For subsequent phases: show the small scope, implement it, run focused tests, open the
actual Streamlit page with real saved data, and show the result for review. Move
to the next phase after that review. A missing-data prerequisite holds that
feature; it does not justify invented values or a larger implementation batch.

The new notebook output is `.local/f1_2025_2026/`: SQLite reference entities and
manifests, immutable Parquet, parser source evidence, SDK HTTP cache and a visible
progress file. The new read-only archive adapter connects downloaded sessions to
the existing screens. Legacy JSON readers remain a fallback when no published
notebook archive is present. Page rendering never fetches provider data.

## Phase 0 — collect and verify a small dataset

**Deliverable:** the notebook, then a one-session pilot, then the selected
completed 2025/2026 sessions. Start with the first 2025 session, or set
`SESSION_CODES={'R'}` for a Grand Prix replay pilot. Set `SESSION_LIMIT=None`
only after reviewing the pilot's coverage.

**Steps:** select the primusEnv kernel; install a verified FastF1 release with
archive authentication; restart the kernel; sign in locally; preview the exact
calendar/session list; download the pilot; inspect SQLite, Parquet and coverage.
No separate Pro API key is required; account entitlement to the archive is what
matters. Download may remain blocked without an eligible F1 TV subscription.

**Acceptance:** names/dates/full session labels match the provider; keys and
foreign keys are valid; files reopen and hashes match; driver/lap/sample counts
are nonzero where coverage is available; missing channels are explicit; a
restart reuses verified published sessions; authentication/rate-limit failure
stops cleanly. Replays require actual timestamped positions, not merely lap times.
The notebook's disabled-download run must make no HTTP requests or create data.

**Gate:** do not implement Phase 1 until the pilot's real data and coverage have
been reviewed. Full-season completion need not block an initial UI feature if its
selected event/session already has verified data.

## Phase 1 — data selection and Race Overview

**UI:** year, race name plus date, full session name, followed by event details,
saved classification, driver/team names and an honest dataset-availability panel.
Only offer downloaded sessions for analysis; future calendar entries are clearly
labelled and cannot start an analysis. Unknown results stay unknown.

**Implementation:** add a read-only archive adapter to `streamlit_app/services.py`
and the shared selection context. Read event/session/manifest tables from SQLite,
and only the Parquet partitions the selected view requires. Cache immutable reads
by published version ID. Bind the existing Overview screen to this adapter; do
not regenerate legacy JSON or fetch providers while rendering a page.

**Tests:** two events with different sessions; year/race switching resets dependent
selection; race names/dates and full session names display correctly; actual
classification rows appear; unavailable/empty/corrupt files are handled; page
rendering makes no provider requests. Verify in a real browser, not just unit tests.

## Phase 2 — Compare Drivers

**UI:** select two or more drivers; compare lap-time trends and sector splits;
filter clean laps, pit laps and compounds. Show the definition of a clean lap.

**Implementation:** query one chosen lap version with `v_current_laps`; use the
existing comparison screen and Plotly. Load dense telemetry only when explicitly
requested for a selected driver/lap. Never treat missing sectors as zero.

**Tests:** plotted values agree with saved records; lap/driver/compound filters
work; deleted, generated and inaccurate laps are identified; unavailable sectors
render a clear state. Test the actual interactive plot in Streamlit.

## Phase 3 — full-race Replay

**UI:** complete-session playback, play/pause, seek, speed, driver selection, lap
counter, timing tower and selected-driver telemetry when available.

**Implementation:** use the existing embedded replay component inside Streamlit.
Build a versioned replay bundle from each driver's position and car partitions;
one browser clock drives the track, tower and telemetry. Bound interpolation gaps
and mark stale/absent samples. Load the selected race, not all seasons into memory.
Use real positions; results or lap order cannot stand in for track coordinates.
Keep original streams separate from any lower-resolution derived playback bundle.

**Tests:** start-to-finish playback, seek/pause/resume, all laps, multiple speeds,
retirements and missing intervals; synchronized displays at sampled times; no
animation-rate Python reruns. Measure frame rate, initial load and memory on this
machine and report those measurements before accepting the feature.

## Phase 4 — tyres, pit visits and Strategy Lab

**UI:** stint timeline, compound and tyre age, observed pit visits, and clean-lap
pace/degradation summaries. Add one clearly labelled what-if control only after
the descriptive data is correct.

**Implementation:** derive stints and pit visits from saved lap boundaries and
source-linked pit records. Keep pit-lane duration distinct from a stationary tyre
change. Separate observed data from simulations. Do not show measured tyre
temperature: normal FastF1 feeds do not provide it.

**Tests:** stint/compound changes match source laps, unmatched pit entries/exits
are flagged, degradation excludes inappropriate laps, and every what-if output
is labelled with its assumptions. Review the screen against one known race.

## Phase 5 — weather and race context

**UI:** air/track temperature, wind, humidity, pressure and rain indication over
the selected session, aligned with laps/track-status events when their clocks
are available. Show provider, units and gaps.

**Implementation:** use FastF1 track-weather observations first. Optional regional
weather enrichment is a later explicit addition. Do not equate a rain flag to
millimetres of rain, or retrospective weather to a forecast known before the race.

**Tests:** correct units/timestamps, bounded as-of joins, honest gaps, wet/dry race
examples where available, and no invented track or tyre temperature.

## Phase 6 — Predictions

**UI:** select a model and generate the final Grand Prix forecast using earlier
practice, qualifying and sprint inputs. Display probabilities/uncertainty only
when supported, plus the inputs, cutoff and source coverage used.

**Implementation:** start with a grid baseline and one weekend-form model; compare
them before adding more. Weather, tyres, pit loss and driver/team form are usable
only when available at the forecast cutoff. TabFM remains an experimental tabular
candidate; TimesFM remains a pace-forecasting candidate whose output may feed a
race simulator. Neither is selectable as working until installed, tested and
benchmarked. An LLM may explain supported outputs, not invent a calibrated winner.

**Tests:** race results and future laps cannot leak into inputs; switching models
changes the requested model; insufficient inputs stop or disclose fallback;
repeated runs record the same dataset/model/configuration. Historical hindsight
downloads alone do not prove a valid pre-race backtest.

## Phase 7 — Model Validation: evaluation and tracking

**UI:** separate Evaluation and Performance Tracking tabs. Compare fixed model
runs across events; inspect present and past behavior, errors, coverage and drift.

**Implementation:** version datasets, feature cutoffs, models and outcomes; score
only when outcomes exist. Use chronological event splits. Choose metrics by task:
log loss/Brier score and calibration for winner probabilities; rank correlation
or pairwise ordering for finishing order; MAE/RMSE and interval coverage for pace.
Keep reliability/runtime/failed-request telemetry separate from predictive skill.
Reuse existing tracking/MLflow integration after connecting it to the new archive.

**Tests:** independent metric examples, chronological splits, delayed outcomes,
reproducible run comparison and correct empty states. Do not display old experiment
history as validation of the newly collected dataset.

## Phase 8 — Primus AI and driver/team profiles

**UI:** a clean grounded F1 chat with source links and selected race context;
driver/team profiles expose sourced facts and available performance history.

**Implementation:** use the existing guarded chat path. Add allowlisted read-only
archive tools, F1 scope limits, request/input/output budgets and rate limits.
Treat retrieved text as untrusted input. Keep credentials out of prompts/logs;
never permit arbitrary SQL, shell commands or filesystem access from chat.
Version profile facts by season so team changes do not rewrite history.

**Tests:** grounded answers, absent-data abstention, prompt injection, non-F1
requests, oversized inputs, quota exhaustion and race/context isolation. Review
the UI and guardrails before enabling any external LLM calls.

## Phase 9 — Reports, data settings and final polish

**UI:** export the selected analysis, show source/coverage information and data
refresh state, and provide consistent navigation/empty/error/loading states.

**Implementation:** reuse existing Reports and Settings screens; reports refer to
the exact dataset/model version. Keep long data collection in the visible notebook,
not inside an incidental Streamlit rerun. No deployment in this phase.

**Tests:** exports match the selected analysis, navigation preserves/reset context
correctly, every page works with the actual new dataset, and a full regression plus
browser pass succeeds. A feature remains unavailable when its prerequisite data
is unavailable.
