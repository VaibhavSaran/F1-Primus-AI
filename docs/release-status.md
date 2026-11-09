# Release status and remaining work

Data scope through November 8, 2026. Publication verified November 9, 2026,
America/Chicago.

## Hosted application

The private application is available at https://f1primus.vaibhavsaran.com/ with
Google IAP owner access. The active serving release is `20261108-v1` on the
persistent disk of `primus-personal`. The old release is retained for rollback.
The archive reader selects session and driver partitions on demand.

The refreshed cohort pins 749 sessions across 2020–2026. For 2026, all 100
eligible sessions through the November 8 Sao Paulo race are saved. Required
channel checks classify 89 as complete and 11 as partial. Sao Paulo's race
retains four parser-generated car samples for Rafael Camara, alongside 36,146
timestamped samples. Partial flags and unavailable circuit metadata remain
explicit. These counts do not establish gap-free telemetry.

GCP received a 937 MB delta. Upload checksum comparison, VM file hashes, SQLite
integrity and foreign keys passed. The running container is healthy. Browser
checks verified Mexico classification and Sao Paulo classification, replay
playback, session weather charts and race-control messages on the custom domain.

## Kaggle

The [F1 Weather Dataset](https://www.kaggle.com/datasets/vaibhavsaran/circuit-window-hourly-weather-2020-2026)
is ready with 8,660 hourly observations across 150 event windows. Mexico adds
57 observations. The downloaded published CSV matches the staged SHA256 and
row counts. Sao Paulo independent ERA5 is pending publication, with the next
collector retry boundary at November 14 UTC. FastF1 session weather is already
present for both weekends.

The [Primus F1 Race and Telemetry Dataset](https://www.kaggle.com/datasets/vaibhavsaran/primus-f1-race-and-telemetry-2020-2026)
has Version 3 published and verified. The 2026 bundle is `v3-20261108`, with
100 sessions and 59,224 lap records. The web uploader retained 2020–2025 files.
The downloaded publication index still selects all seven years. The 2026
manifest, catalogue and a Sao Paulo position file match local hashes. SQLite
integrity and foreign keys pass, both races are present, and the downloaded
position sample has 36,132 rows with its pinned version ID. The dataset card
now describes November 8 coverage and correctly explains that filename numeric
prefixes are archive indices. Existing licensing settings are preserved.

## Source and deployment workflow

Local work is on main and remains uncommitted. The edited verification workflow
runs on main pushes. Its successful completion triggers immutable image deployment
to the existing VM through IAP, with health checks and image rollback. Source
must reach GitHub before this workflow can run. The CI service account's VM
permissions need verification in the first actual GitHub deployment.

The focused release, weather and notebook suite passed 59 tests. Repository
hygiene scanning reported zero findings after setup notebook execution outputs
were removed. Detailed update steps are in [gcp-updates.md](gcp-updates.md), and
coverage is recorded in [data-tracker.md](data-tracker.md).

## Remaining work

- Review, commit and push the local source changes, then verify the first CI deployment.
- Retry Sao Paulo independent ERA5 after its publication delay.
- Walk through every page together and practise the interview demonstration.
- Append genuine Mexico outcomes to the saved prospective forecast evaluation
  without changing the original predictions or input cutoffs.

The 2026 research holdout remains separate from prospective forecast scoring.
Fresh TabFM inference is an offline operator task. Advanced strategy research,
optional Jev/ensemble experiments and Kaggle notebook publication remain later
work. Dataset file-description refinement is on the back burner at the owner's
request.
