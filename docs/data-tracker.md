# Data tracker

## November 11 Kaggle publication

F1 Kaggle Version 4 is published, using the `v4-20261111` 2026 release. It
contains the repaired Mexico and refreshed Sao Paulo race streams, with 100
sessions and 59,224 lap records through November 8. The earlier 2020–2025
releases are retained. Remote manifest/catalogue/index hashes, SQLite integrity
and foreign keys, measured weather counts (Mexico 132; Sao Paulo 160), and
position samples from both races passed verification.

The separate weather dataset is verified unchanged at 8,660 hourly rows across
150 event windows. Its downloaded CSV and all 150 local source hashes match
the published release. Sao Paulo ERA5 remains pending until provider availability.

## November 10 review

The [latest quality review](data-quality-2026-11-10.md) supersedes the earlier
Mexico coverage claim. A fresh-cache download recovered full-duration Mexico
race streams: 1,103 lap records, 521,136 positions and 132 measured weather
observations. All 100 eligible 2026 sessions are saved through November 10.
The [review CSV](data-coverage-review.csv) enumerates partial channels and known
cancelled practices across all seven seasons. Sao Paulo independent ERA5 is
still pending; measured session weather is available.
The refreshed Mexico and Sao Paulo race versions are active on GCP in
`20261110-v1`; the rest of the serving archive is reused. Kaggle Version 4 now
also includes this repair.

## November 8, 2026 refresh

Mexico City and Sao Paulo each have all five sessions published, including both
Grand Prix races. Renewed local authentication resolved the initial HTTP 401.
The 2026 archive has 100/100 eligible sessions, with 89 complete and 11 partial
under the required-channel checks. Across 2020–2026, the new cohort pins 749
sessions. Sao Paulo's race retains four parser-generated car samples for Rafael
Camara alongside 36,146 timestamped samples. The partial flag is preserved.
Measured session weather is present for every collected weekend session.

Independent Mexico weather is complete with 57 hourly observations for the
window ending November 2 at 01:00 UTC. Sao Paulo ERA5 remains pending until
November 14 UTC under the collector's publication delay policy. The public
weather Kaggle dataset now contains 8,660 rows across 150 event windows. A
downloaded CSV matches the staged hash and Mexico row count.

The 2026 notebooks now use November 8 as their inclusive cutoff, bounded by
the actual execution time and session completion margins. The hosted F1 archive
uses the verified November 8 release. Mexico classification and Sao Paulo
classification, replay playback and weather were checked through the custom
domain. F1 Kaggle Version 3 is published. Its selected-year index, 2026 manifest,
catalogue and a Sao Paulo position file were downloaded and verified against
local hashes. The catalogue has 100 sessions, with both races present, and
passes SQLite integrity and foreign-key checks. Verification completed November 9.
See [GCP update instructions](gcp-updates.md).

## Previous full archive audit

Checked (UTC): 2026-11-02T16:14:19.667556+00:00
2026 scope: race weekends through **2026-11-02 inclusive**.

| Year | Saved / eligible sessions | Complete | Partial | Repair targets | Missing | Cancelled excluded | Independent weather weekends |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 80/80 | 48 | 32 | 0 | 0 | 3 | 17/17 |
| 2021 | 109/109 | 89 | 20 | 0 | 0 | 1 | 22/22 |
| 2022 | 110/110 | 86 | 24 | 0 | 0 | 0 | 22/22 |
| 2023 | 110/110 | 84 | 26 | 0 | 0 | 0 | 22/22 |
| 2024 | 120/120 | 86 | 34 | 0 | 0 | 0 | 24/24 |
| 2025 | 120/120 | 68 | 52 | 0 | 0 | 0 | 24/24 |
| 2026 | 94/95 | 84 | 10 | 0 | 1 | 0 | 18/19 |

Saved includes partial publications; complete counts follow the current archive quality rules.
A complete session count does not prove gap-free telemetry. Independent weather is delayed
retrospective reanalysis, separate from measured session weather and pre-race forecasts.
Eligible totals follow the retained local calendar and known cancelled sessions, not a fresh
official-calendar reconciliation. Unknown scope is shown as `?`.

File hashes and Parquet row counts: **not checked** in this quick refresh.
This snapshot contains aggregate counts only; private diagnostics remain local.

## Refresh locally

```powershell
& '.\primusEnv\Scripts\python.exe' -m scripts.update_data_tracker
# Optional full file validation:
& '.\primusEnv\Scripts\python.exe' -m scripts.update_data_tracker --verify-files
```

These commands read local archives offline and replace this public Markdown snapshot.
They do not collect data, sign in, schedule work, upload, commit or publish anything.
For private session-level diagnostics use `-m data.archive.status`.

See the [collection guide](data-collection-2020-2026.md),
[notebook index](../notebooks/README.md), and
[dated refresh plan](data-refresh-calendar.md).
Future public datasets require source-by-source redistribution review; local access
and an open-source client license do not establish rights to redistribute feed data.
