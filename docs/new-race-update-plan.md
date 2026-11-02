# New-race ingestion and UI update plan — not implemented

This is a proposed operational lifecycle. No new collector, scheduler, timer,
automatic publication or data-fetch job is created by this task.

## 1. Discover and gate events

Read the provider schedule into a year/event/session catalogue with full names,
round identities, UTC times, local venue dates and actual session formats.
Distinguish postponed, cancelled, not-yet-held and completed sessions. A weekend
without a Sprint must not gain Sprint dropdown entries. Schedule changes are
versioned; race names and dates must not be guessed from round numbers.

Mark a session eligible only after scheduled/observed completion and a
conservative provider publication margin. Wait for a full race weekend before
declaring the independent weather archive complete. Do not query future results
until the gate is satisfied. An operator-triggered run should precede any later
scheduled automation; deployment/scheduling remains a separate decision.

## 2. Collect with resumable provider-aware jobs

Reuse the existing authenticated provider client, shared pacing/limit ledger,
cache and progress state. Fetch missing or explicitly selected repair targets,
not every historical race. Respect Retry-After, exponential backoff and provider
access pauses. Stop on authentication/access blocks rather than retrying through
mirrors or bypassing controls. Bound each run's calls, elapsed time and disk use.

Track F1 and independent weather jobs separately. Keep measured session weather
distinct from later reanalysis and from forecasts issued before the race. Store
source retrieval/issued/valid timestamps and provenance for every source object.
Weather lag must not prevent already verified telemetry becoming available.

## 3. Normalize and validate before publication

Stage incoming objects under a run identifier. Retain raw source hashes and
source version, normalize SQLite rows and Parquet partitions with explicit clock
units and exact driver/team/session identities, and record channel completeness.
Validate hashes, row counts, schema, clocks, joins, duplicate identities and
requested session coverage. Do not treat DNS, DNF, DSQ or a cancelled race as
missing API data. Never infer incident causes solely from missing telemetry.

Keep timing/position/car/lap/weather channels independently reportable. Preserve
provider placeholders, unmatched identities and sample gaps as declared quality
flags. When evidence changes, publish a new immutable version instead of editing
an old one. Use the existing writer lock and an atomic catalogue transaction;
the UI continues reading the previous publication until the new one passes.

## 4. Publish to the local read layer and refresh UI caches

Use a lightweight catalogue revision/dataset hash as the cache key. On explicit
Refresh, invalidate only affected event/session/profile/coverage projections;
avoid rescanning every dense Parquet stream. Update available years, named races
with dates, actual sessions and coverage. Preserve the user's active selection
if still valid; clear dependent race/session selections when their parent changes.

Newly published sessions should enable their plots and replay channels. Missing
channels stay visibly unavailable. Active replay/analysis views retain their
current version until the user refreshes, preventing data from changing halfway
through a session. Chat answers and report exports retain their cited versions.

## 5. Freeze forecasts before the Grand Prix

After eligible practice/qualifying/sprint sessions are published, create a new
immutable forecast snapshot at the actual current UTC time before race start.
Every feature/source must be known by that cutoff. Do not use scheduled race
start as a future “observed now” timestamp. Evaluate only models eligible for
that mode and field; show missing artifact/input reasons without fallbacks.

Save each numeric ranking, model/checkpoint, preprocessing/context, snapshot and
cutoff before results arrive. Keep it pending and never regenerate it using
post-race data. Optional issued weather forecasts need their own historical
availability evidence. Target-race measurements/reanalysis cannot substitute
for an unavailable pre-race forecast.

## 6. Append outcomes and measure prospective performance

When a genuine result is published, validate its event/driver field and append
an outcome revision with retrieval/publication timestamps. Score existing frozen
forecasts using the exact saved field and the configured sporting outcome rules.
Keep pending, incomplete-field and completed states separate; no fake zero
accuracy. Later penalties or disqualifications append new outcome/score
revisions while retaining earlier history.

Operational prospective performance belongs in its own evidence cohort, separate
from retrospective reconstruction and the locked 2026 research benchmark.
Appending a new race does not silently unlock/retrain a holdout or rewrite the
5/10/50/100 historical study. New development datasets, studies and model
promotion decisions require explicit versions and evaluation.

## 7. Maintain profiles, context and public dataset releases

Refresh season-specific driver/team form from eligible completed results while
preserving old cutoff views. Add trustworthy F1/FIA/Pirelli/team context for
cancellations, race format changes and documented incident causes with event,
publication/knowledge time and source URL. Unknown causes remain unknown.

Kaggle/release packages are immutable. Produce a new reviewed release version
with updated coverage, hashes, attribution, quality notes and standalone reader
tests; do not modify a previously prepared package in place. Public upload is a
separate authorized step.

## 8. Operational acceptance and monitoring

Before enabling automation, prove: interrupted downloads resume; rate/access
pauses are respected; duplicate publication is idempotent; a failed write leaves
the previous UI version usable; weather can lag independently; cancellations and
DNFs retain correct meaning; post-race labels never enter frozen forecasts;
new results score exactly once per outcome revision; stale caches refresh with
the new catalogue version; and existing reports remain reproducible.

The eventual status view should show completed/missing/partial sessions, next
eligible fetch time, last successful publication, latest UI revision and
actionable failures. Notify only on meaningful completion/failure/user action,
not on every unchanged poll. No such monitor is started by this planning work.
