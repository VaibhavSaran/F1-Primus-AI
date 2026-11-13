# Data and replay review — November 10, 2026

## Latest weekends

Both Mexico City (November 1) and Sao Paulo (November 8) have all five saved
sessions. The local calendar has 100 eligible 2026 sessions through November 10,
and all 100 are published. A hosted inventory check also found all five Sao
Paulo sessions; select 2026, Sao Paulo, then Grand Prix Race.

The previous Mexico Race publication combined full lap records with a stale,
short telemetry/weather cache and misaligned race-control clocks. Its ten
weather observations and roughly eleven minutes of position data did not
establish full race coverage, despite passing the old required-channel checks.
The earlier completeness statement was therefore too broad.

A fresh, isolated provider download followed by local promotion recovered:

| Race | Lap records | Position rows | Car rows | Measured weather rows |
|---|---:|---:|---:|---:|
| Mexico 2026 | 1,103 | 521,136 | 502,920 | 132 |
| Sao Paulo 2026 | 1,435 | 794,904 | 795,300 | 160 |

Mexico's race-control clocks now fall within the replay duration. The replay
adapter additionally skips out-of-range provider messages with an explicit
warning instead of failing the entire replay. It retains the original archive
records and never shifts timestamps to manufacture an alignment.

Sao Paulo retains four parser-generated car samples for Rafael Camara. Those
remain flagged; generated fills are excluded from observed replay samples.
Placeholder driver names fall back to recorded broadcast names/abbreviations.
Entries with no observed laps or positions are excluded from replay selection.

## Whole-archive inventory

The offline audit found 749 saved eligible sessions across 2020–2026, zero
entirely missing eligible sessions, and 199 partial-channel review sessions.
This is an inventory/channel audit, not a new checksum scan of every historical
file. New serving-delta payloads are hashed separately.

| Year | Saved / eligible | Required-channel complete | Partial review |
|---|---:|---:|---:|
| 2020 | 80 / 80 | 48 | 32 |
| 2021 | 109 / 109 | 89 | 20 |
| 2022 | 110 / 110 | 86 | 24 |
| 2023 | 110 / 110 | 84 | 26 |
| 2024 | 120 / 120 | 86 | 34 |
| 2025 | 120 / 120 | 68 | 52 |
| 2026 | 100 / 100 | 89 | 11 |

The [session-level review CSV](data-coverage-review.csv) lists every partial
session and all four known cancelled practices, with retained references.
Partial-channel categories: 117 position-only, 29 car-only, 52 car-and-position,
and one session missing positions, measured weather and race-control messages.
Many per-driver flags identify parser-generated fills or missing timestamps;
they must not be equated with a cancelled race or a driver's retirement.

## Cancellation versus missing provider evidence

The retained cancellation register identifies Styrian 2020 FP3, Eifel 2020 FP1
and FP2, and Russian 2021 FP3 as not held because of weather. These are excluded
from eligible-session denominators. Their reference URLs are in the CSV.

Austria 2023 FP1 has laps and car telemetry but no positions, measured weather
or race-control messages. A fresh authenticated, empty-cache attempt on
November 10 reproduced the same unavailable channels. This supports an
unavailable-provider-channel classification for that attempt; it does not
establish a cancellation, DNF, or permanent absence from every possible source.
The fresh attempt is retained locally rather than replacing the existing
publication with equivalent partial evidence.

Independent Open-Meteo/ERA5 covers 19 of the 20 collected 2026 weekend windows.
Sao Paulo remains pending under the collector's publication-delay policy until
November 14 UTC (November 13 at 6 p.m. Chicago). Measured session weather for
Sao Paulo is already available; reanalysis is a separate source.

## Repair procedure

1. Inspect the affected driver, channel and timestamp span, not just row count.
2. Compare observed laps with driver-specific positions. Use recorded sporting
   status and trusted event context to explain a DNF/cancellation; absence alone
   is not evidence of either.
3. Retry recoverable gaps with the authenticated SDK and shared request limiter,
   using an isolated empty cache. Reusing a partial parser cache is not a fresh
   provider retry.
4. Compare the returned channels, preserve old versions, and promote improved
   evidence only. Keep genuine unavailable channels and parser fills labelled.
5. Validate new hashes and SQLite integrity, publish a serving delta, and verify
   the actual UI. Kaggle updates require a separate versioned publication;
   local/GCP repairs do not change an already published Kaggle version.

No blanket re-download of all 199 partial sessions was performed. The targeted
fresh checks cover Mexico Race, Sao Paulo Race and Austria 2023 FP1.

## Publication follow-up — November 11

Kaggle Version 4 (`v4-20261111`) now contains the repaired 2026 race streams.
The uploaded package passed all 4,400 dense-member checks locally; downloaded
manifest/catalogue/index hashes, SQLite checks and position samples from both
races passed independent verification. Earlier 2020–2025 files were retained.
The separate weather dataset matches the current local export and has no new
independent observations to publish; Sao Paulo ERA5 remains pending.
