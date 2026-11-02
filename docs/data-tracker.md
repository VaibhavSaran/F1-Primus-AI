# Data tracker

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
