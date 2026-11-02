# Local historical data archive

**Historical runbook:** the previous collection was stopped and its downloaded
archive/caches/session exports were deleted on October 25, 2026. Use
[`notebooks/f1_2025_2026_download.ipynb`](../notebooks/f1_2025_2026_download.ipynb)
for the new visible 2025/2026 workflow and
[the phased Streamlit plan](streamlit-phased-restart.md) for subsequent UI work.
The CLI examples below are retained as reference; no background restart is intended.

This is a resumable local collection workflow for personal research. The presence
of the schema or a completed task does not establish full 2018–2026 coverage.
Read the reported dataset coverage before using an event in analysis or replay.
Collection does not train, validate or promote a prediction model.

## Run and resume

Run from `.` on `ver_2`, using the existing environment:

```powershell
& '.\primusEnv\Scripts\python.exe' -m data.archive.collect run --from-year 2018 --through 2026-10-23 --timezone America/Chicago
& '.\primusEnv\Scripts\python.exe' -m data.archive.collect status
```

The default archive root is `.local/f1_archive`. The cutoff is interpreted in the
specified timezone; session timestamps retain their UTC meaning. Future sessions
are excluded. Rerunning uses persisted task state and checksums to reuse completed
evidence. An unattended Python worker handles requests and records; there is no
LLM call for each race, driver, lap or sample.

Start with a small pilot and inspect its coverage, request counts, runtime and
storage before allowing the full range to proceed. One session load can cause
many HTTP requests. Counts and estimates must use the persisted HTTP attempts,
rather than treating one `Session.load()` as one request.

## Stop and retry

Create the stop file to request a graceful stop between collection work items:

```powershell
New-Item -ItemType File -Path '.local/f1_archive/STOP' -Force
```

An in-flight request or task may finish first. Wait for the worker to stop before
backing up or changing its configuration. Remove only the stop file when ready
to resume, then run the collection command again:

```powershell
Remove-Item -LiteralPath '.local/f1_archive/STOP'
```

Inspect failures before deliberately requeuing the corresponding tasks:

```powershell
& '.\primusEnv\Scripts\python.exe' -m data.archive.collect retry --status failed
& '.\primusEnv\Scripts\python.exe' -m data.archive.collect retry --status blocked
& '.\primusEnv\Scripts\python.exe' -m data.archive.collect retry --status unavailable
```

Requeuing does not repair missing data or authentication. A persisted provider
denial must remain in effect until legitimate access has changed and the operator
deliberately resets that route. Do not repeatedly retry blocked tasks on a timer.
Timed rate-limit deferrals retain their next eligible retry time across restarts.

After legitimate public access has been restored, an explicit retry can clear
the corresponding denied-host circuit while retaining all quota counters:

```powershell
& '.\primusEnv\Scripts\python.exe' -m data.archive.collect retry --status blocked --reset-provider fastf1
```

For a scoped pilot after calendars have been collected, use `run --event 2018:1
--kinds results,publish --idle-exit`. Published evidence is versioned; a later
sparse classification cannot replace richer cached laps in `v_current_laps`.
Counts in the physical tables include historical revisions. The current-laps
view selects one whole version using observed-field coverage, lap count and
publication order, rather than mixing providers within a version.

The cutoff is midnight following the requested date in the chosen timezone.
Finalized-session imports require a conservative completion margin of four hours
for races and two hours for other sessions; sessions near the boundary remain
blocked until completion before the cutoff can be established. Recent ERA5
windows remain pending for publication rather than being filled with forecasts.
An initial accessible backfill is estimated at 18–30 hours at the configured
pace; blocked timing and weather publication delays are outside that estimate.

## Provider access and pacing

FastF1's installed request limiters remain enabled. Additional local safety
policies initially reserve at least 11 seconds between FastF1 and shared SDK
requests, 21 seconds for Jolpica, and 2.5 seconds for Open-Meteo. Hourly safety
budgets are 400 for FastF1/shared SDK and 180 for Jolpica. Open-Meteo uses weighted
variable, date-range, model and location accounting, with local limits of 450 per
minute, 4,000 per hour and 8,000 per rolling day. These are conservative local
settings, not claims about provider-required minimum intervals. Counters and
policies are persisted in `http.sqlite3` and shared by archive workers using the
same root. Independent programs outside this workflow can still consume the
provider's IP-wide or account-wide quota.

Cache hits bypass the network policy hook and consume no request budget. Each
retry or redirect that is sent reserves another request. HTTP 429 respects
`Retry-After`, including HTTP dates; timeouts and server failures have bounded
backoff. Longer delays are saved for a later resume. HTTP 401/403 quarantines the
denied host route, rather than repeatedly probing it. FastF1 can still execute its
existing public mirror fallback; a denied official host does not declare a
different host blocked. A mirror 404 is unavailable evidence for that object,
not proof that every mirror dataset is absent.

The FastF1 maintainer reports authentication is now required for the official
historical timing archive. An installed SDK's ability to describe a dataset does
not prove the current endpoint is accessible. No authentication workaround,
proxy rotation, paid service or SDK limiter removal is part of this collector.
See [FastF1 issue 978](https://github.com/theOehrly/Fast-F1/issues/978),
[Jolpica rate limits](https://github.com/jolpica/jolpica-f1/blob/main/docs/rate_limits.md),
[Open-Meteo terms](https://open-meteo.com/en/terms) and
[weighted-call accounting](https://open-meteo.com/en/pricing).

## Evidence and coverage

`f1.sqlite3` holds reference entities, normalized laps and event/weather records,
source provenance, dataset manifests and task/coverage state. Original payloads
are retained under `raw/`; full car and position streams are separate immutable
Parquet files under `parquet/`. Files are staged, validated and atomically
published before SQL points to a published version. A crash can leave an
unreferenced staged object; it must not publish an incomplete file as valid data.
Revised source data produces a new version rather than rewriting old evidence.

Coverage is typed: available, partial, unavailable, unsupported or blocked.
Missing timing, telemetry, positions or weather must not become synthetic laps,
invented coordinates, final-result substitutes for live positions, or a false
complete flag. Position timestamps and car timestamps remain separate; duplicate
timestamps keep their original sample indices. Provider coordinate scale and
clock origins remain explicit.

FastF1's processed session car/position streams are retained alongside cached
parser frames when available. Parser frames are earlier SDK representations,
not original wire packets. Parser-generated missing-time zero fills are flagged
and excluded from valid observed-sample counts. SDK defaults (including DRS
values when the provider omits that channel) must not be presented as verified
measurements. The provider may not supply all advertised channels in every year.

FastF1 weather is track-observation evidence when supplied. External Open-Meteo
historical weather uses explicitly requested ERA5 regional reanalysis at
versioned circuit reference coordinates from Jolpica. The returned grid
coordinate can differ from the circuit coordinate. This is a regional estimate,
not a physical track sensor. Its air temperature never substitutes for track or
tyre temperature. Rainfall flags, millimetres of rain/precipitation and forecast
probabilities remain separate, with precipitation intervals and pressure units
recorded. See the [historical weather documentation](https://open-meteo.com/en/docs/historical-weather-api).

Historical forecast publication and run availability remain unverified unless
separately evidenced. Retrospective reanalysis is not a forecast known before a
past race and must not enter a pre-race validation set as one. The normal FastF1
channels do not supply measured tyre temperature; that dataset is explicitly
unsupported, rather than a table of fabricated tyre readings.

## Backup and restore

Stop the worker first. Preserve the complete archive root together: SQLite
metadata, `http.sqlite3` quota/circuit state, raw payloads, Parquet files, manifests
and any provider cache used to reproduce collection. A database-only backup
does not contain the payload files referenced by its manifests. `Archive.backup`
uses SQLite's backup API for a consistent metadata snapshot; file copies must
match that snapshot. Use a new backup destination and retain the original until
restoration and checksum checks succeed.

Restore into a separate local directory, verify source and dataset hashes, and
confirm existing published sessions can be read offline before resuming writes.
Keep quota state when restarting the collector; deleting it to obtain a fresh
request budget is not a valid resume procedure. Local archives, virtual
environments, secrets and temporary plans stay Git-ignored.
