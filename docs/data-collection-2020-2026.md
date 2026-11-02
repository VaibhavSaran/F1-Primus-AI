# Collect F1 and weather data: 2020–2026

Start with the [notebook index](../notebooks/README.md). Actual aggregate coverage
is in the [public data tracker](data-tracker.md); the
[November 2 acquisition calendar](data-refresh-calendar.md) plans remaining races
and later project updates. Planning does not start collection or schedule jobs.

The 2026 F1 and weather notebooks use a November 2, 2026 inclusive cutoff. Creating these notebooks has not started
collection. Run locally in VS Code using **Python (primusEnv)**. No package
installation, upgrade, background collector, cloud service or upload is started.

## Start here

1. Open [00_setup.ipynb](../notebooks/collection_2020_2026/00_setup.ipynb), select
   **Python (primusEnv)** and **Run All**. Enter F1 account details only into the
   local prompts; the password prompt is hidden. FastF1 manages its cached login
   locally. Weather notebooks need no F1 sign-in or API key.
2. Open an F1 year notebook and **Run All**. Defaults collect all eligible
   sessions, with no pilot limit. Existing validated Australian GP 2025 Practice 1
   data are reused. You may start several F1 year notebooks; waiting kernels
   print that another notebook owns the shared archive.
3. Open a weather year notebook and **Run All** alongside F1. Several weather
   notebooks may also be started; they wait for the independent weather lock.
4. Keep the computer awake and kernels running. Interrupt a kernel to stop
   safely. **Run All again to resume**; retain progress files and caches.
5. After workers finish or are interrupted, run
   [99_status_validate.ipynb](../notebooks/collection_2020_2026/99_status_validate.ipynb).
   It checks SQLite integrity, foreign keys, raw source hashes and manifest
   Parquet hashes/row counts, and reports sessions and weather task states.

No configuration edits are required for full-year collection. `DRY_RUN=True`
provides an offline preview. Optional round/item limits allow a smaller run.

The 2020 notebook now starts with a read-only remaining-work audit. **Restart its
primusEnv kernel before Run All** to load the corrected driver-ID normalizer.
It targets missing sessions and explicitly identified legacy identity repairs,
reusing cached provider data where available. Three cancelled practices are
excluded: Styrian FP3 and Eifel FP1/FP2. Other saved partial sessions are shown
separately and revisited only with `RETRY_PARTIAL=True`. The audit is recomputed
each Run All, so the pending list shrinks as collection finishes.

For targeted recovery, inspect the actual notebook's available configuration
before running it. Do not assume that an optional recovery cell exists in every
saved copy. Missing sessions and saved partials remain separate; deliberate
partial recovery can retain historical sample gaps after retry. Keep provider
evidence and use the shared rate limits rather than clearing caches.

## Individual notebooks

| Year | F1 sessions | Independent hourly weather |
|---|---|---|
| 2020 | [F1 2020](../notebooks/collection_2020_2026/f1_2020.ipynb) | [Weather 2020](../notebooks/collection_2020_2026/weather_2020.ipynb) |
| 2021 | [F1 2021](../notebooks/collection_2020_2026/f1_2021.ipynb) | [Weather 2021](../notebooks/collection_2020_2026/weather_2021.ipynb) |
| 2022 | [F1 2022](../notebooks/collection_2020_2026/f1_2022.ipynb) | [Weather 2022](../notebooks/collection_2020_2026/weather_2022.ipynb) |
| 2023 | [F1 2023](../notebooks/collection_2020_2026/f1_2023.ipynb) | [Weather 2023](../notebooks/collection_2020_2026/weather_2023.ipynb) |
| 2024 | [F1 2024](../notebooks/collection_2020_2026/f1_2024.ipynb) | [Weather 2024](../notebooks/collection_2020_2026/weather_2024.ipynb) |
| 2025 | [F1 2025](../notebooks/collection_2020_2026/f1_2025.ipynb) | [Weather 2025](../notebooks/collection_2020_2026/weather_2025.ipynb) |
| 2026 | [F1 2026](../notebooks/collection_2020_2026/f1_2026.ipynb) | [Weather 2026](../notebooks/collection_2020_2026/weather_2026.ipynb) |

The notebooks expose settings, execution and output. Shared runners in
`data/archive/notebook_f1.py` and `notebook_weather.py` keep retry, timing and
storage rules identical across all years.

## Scope and cutoff

Collect available normal-weekend sessions: Grand Prix races, sprints,
qualifying/sprint qualifying and practice. Within each round, F1 prioritises
races and sprints. Pre-season testing and 2018/2019 are excluded.

F1 retains entries, driver/team affiliations, results, laps, sectors, tyre age
and compound, car telemetry, positions, event/circuit schedule metadata,
session/track status, race control and recorded session weather. Original parser
evidence remains separate from normalized streams. Missing provider channels
stay explicit; tyre temperatures and fuel loads are not supplied by this feed.
Corner-number circuit annotations are not fetched by this runner; replay uses
recorded positions. Pit entry/exit observations do not measure stationary pit
stop duration or race pit loss.

Each run fetches the calendar rather than assuming which 2026 rounds exist.
The F1 and weather 2026 notebooks are explicitly limited to **November 2, 2026
(inclusive)**, covering eligible race weekends through that date. Other year notebooks
retain their October 26 configuration, which does not change those past seasons.
The effective cutoff is the earlier of current UTC time and the end of
the notebook's configured inclusive date in America/Chicago. For F1 2026,
November 2 ends at the exclusive boundary November 3 at 06:00 UTC.
Conservative completion margins—four hours for races, two for other
sessions—exclude unfinished sessions. Actual observations beyond the effective
cutoff are not published. Calendar presence does not prove a session ran;
missing/cancelled/unavailable data remain visible as partial or failed.

## Weather

FastF1 track observations are collected with each F1 session. Independent
weather notebooks use Jolpica circuit coordinates and request one ERA5 hourly
series per eligible event window. Each window covers its completed sessions
with a one-hour buffer on either side, bounded by the cutoff. Units, UTC times,
source, requested/returned coordinates and event session IDs are retained.

External weather includes air temperature, humidity, dew point, surface pressure,
wind speed/direction/gusts, rain, precipitation, cloud cover and weather code.
This is retrospective grid-based reanalysis, not measured track temperature or
a forecast available before the race. Rainfall flags and measured/estimated
precipitation amounts remain distinct.

ERA5 has a publication delay; the runner uses a six-day safety margin. Recent
2026 events remain **pending**, with a retry date, while older events continue.
Rerun on/after that date to complete coverage; no forecast or zero substitution.
See [Open-Meteo historical data documentation](https://open-meteo.com/en/docs/historical-weather-api).

## Parallel runs and recovery

One F1 writer and one independent weather writer can run together. Additional
year kernels wait using OS locks that release on interruption or process exit.
Seven concurrent SDK downloads do not multiply quota. Do not run the legacy
pilot notebook or CLI collector concurrently: they use a different HTTP ledger.

All new notebooks share `.local/f1_collection_control/http.sqlite3`. FastF1
logical sends are capped at 200/hour, at least 11 seconds apart, leaving headroom
for an internal SDK authentication retry. The ledger counts logical SDK sends,
not every underlying authentication request. Jolpica/Open-Meteo have separate
conservative policies; shared accounting, caching and provider `Retry-After`
remain effective across kernels.

Provider pauses resume automatically within a bounded wait budget. The notebook
prints its retry time. Repeated failures stop the run or record an event failure
rather than retrying forever. For access denial, rerun local setup, verify your
account access, then resume. Sign-in expiry may require another local sign-in
during a long batch.

F1's `RETRY_PARTIAL=False` default preserves published partial data; set it to
`True` to deliberately retry missing channels. Weather's `RETRY_FAILED=True`
retries failed events on a new Run All with a fresh bounded attempt budget,
retaining previous response evidence.

## Output

| Directory | Contents |
|---|---|
| `.local/f1_2025_2026/` | Shared F1 SQLite catalogue, immutable Parquet, original evidence, SDK cache and `progress-f1-YEAR.json` |
| `.local/f1_weather/` | Independent weather SQLite catalogue, raw responses, event Parquet and task checkpoints |
| `.local/f1_collection_control/` | Shared HTTP accounting and local sign-in coordination |

The historical F1 folder name keeps the verified pilot and current Streamlit
reader working; its catalogue now accepts 2020–2026. Independent reanalysis is
stored for later integration and does not replace measured FastF1 weather in
the UI. Weather task checkpoints hold session links, windows, export paths and
hashes. The weather runner verifies/rebuilds exports from valid retained sources
on resume; status validation checks the SQL/source/manifest archive.

Allow roughly 4–10 continuous days as an initial full-scope planning budget,
subject to provider access, file size and cooldowns. This is not a measured
guarantee; the new conservative retry headroom can extend it. Use a representative
weekend to measure throughput and storage before extrapolating. Keep ample disk
headroom and all downloaded data/caches ignored by Git.

## Maintained collection status

The local status ledger covers **every year from 2020 to 2026**, with 2026 limited
to race weekends dated **November 2, 2026 or earlier**. Cancelled sessions are
excluded. F1 session coverage, partial-quality flags and separate weather
coverage are tracked independently; a published calendar is not downloaded data.

Refresh counts after a notebook run from the project folder:

```powershell
& '.\primusEnv\Scripts\python.exe' -m data.archive.status
```

For a full refresh that also verifies SHA-256 hashes and Parquet row counts:

```powershell
& '.\primusEnv\Scripts\python.exe' -m data.archive.status --verify-files
```

Both commands read the archives without downloading, signing in or changing
collection checkpoints. Results go to ignored `.local/data_status/current.md`
and `current.json`; timestamped JSON snapshots are retained in `history/`.
The report records when it was checked and whether hashes were verified.
It lists missing sessions, partial coverage, deferred/blocked checkpoints and
weather pending publication. Refresh it after collecting more data.

To regenerate the public aggregate snapshot without copying private diagnostics:

```powershell
& '.\primusEnv\Scripts\python.exe' -m scripts.update_data_tracker
# Optional full file validation before writing the snapshot:
& '.\primusEnv\Scripts\python.exe' -m scripts.update_data_tracker --verify-files
```

Both read the archives offline. The public writer projects numeric year summaries
only into `docs/data-tracker.md`; it never copies local error text, absolute paths,
driver identities, source objects or private run/version identifiers. Set
`--through-2026 YYYY-MM-DD` deliberately for later coverage. Defaults stay
November 2 until changed; future acquisition dates are in the separate plan.

## Kaggle later

After collection: verify event/session/channel completeness, document gaps and
units, record source versions, check redistribution terms and attribution, then
prepare a curated Parquet/CSV bundle with a data card and reproducible settings.
Exclude passwords, tokens, notebook outputs, SDK caches, HTTP cache databases,
control logs and private environment configuration. No publication or uploader
has been executed or scheduled.

The client library's software license does not itself authorize redistribution
of the underlying F1 feeds. Follow the [release guide](kaggle-release.md) and its
source-by-source licensing decision before including any data in a public bundle.

Keep these as later comparison references; the notebooks collect our own data:

- [Formula 1 — Season 2023](https://www.kaggle.com/datasets/juampimgi/formula-1-season-2023)
- [Formula One — All Laps & Telemetry](https://www.kaggle.com/datasets/stevefza/f1-2026-bahrain-testing-day-2)
- [F1 Weather Dataset — 2018–2023](https://www.kaggle.com/datasets/quantumkaze/f1-weather-dataset-2018-2023)
- [F1 World Championship — 1950–2026](https://www.kaggle.com/datasets/atharvranjan/formula-1-world-championship-1950-present)
