# Release status and remaining work

Data scope through November 8, 2026. Public-access and data-repair review November 10, 2026;
Kaggle publication verified November 11, 2026,
America/Chicago.

## Hosted application

The application is available at https://f1primus.vaibhavsaran.com/ without
visitor sign-in. Public visitor conversations are session-isolated and shared
AI spending caps remain enforced. The active serving release is `20261110-v1` on the
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
has Version 4 published and verified. The 2026 bundle is `v4-20261111`, with
100 sessions and 59,224 lap records through the November 8 weekend. It includes
the repaired Mexico Race streams and refreshed Sao Paulo Race. The web uploader
retained the unchanged 2020–2025 files and existing cover image.

The downloaded manifest, catalogue and publication index match local hashes.
SQLite integrity and foreign keys pass. Mexico has 132 measured weather
observations and Sao Paulo has 160. Downloaded Albon position samples match
their hashes, version IDs and row counts: 23,688 Mexico rows spanning 7,916.341
seconds, and 36,132 Sao Paulo rows spanning 9,598.001 seconds. The index retains
all seven years and the earlier six selections are unchanged. The dataset card
now describes the repairs. Existing licensing settings are preserved.

The standalone weather CSV was downloaded again on November 11 and matches
the local export exactly. All 150 completed weather task hashes match the
published provenance. No duplicate weather version was created because no new
independent weather observations were downloaded.

## Source and deployment workflow

The previous main push and GitHub deployment rerun succeeded. New November 10
replay, public-access and documentation changes are on main and remain uncommitted.
The verification workflow
runs on main pushes. Its successful completion triggers immutable image deployment
to the existing VM through IAP, with health checks and image rollback. Source
must reach GitHub before this workflow can run. The CI identity's required VM
permissions were verified by the successful deployment rerun.

The focused release, weather and notebook suite passed 59 tests. Repository
hygiene scanning reported zero findings after setup notebook execution outputs
were removed. Detailed update steps are in [gcp-updates.md](gcp-updates.md), and
coverage is recorded in [data-tracker.md](data-tracker.md).

## Remaining work

- Review, commit and push the November 10 source changes, then check CI deployment.
- Retry Sao Paulo independent ERA5 after its publication delay.
- Walk through every page together and practise the interview demonstration.
- Append genuine Mexico outcomes to the saved prospective forecast evaluation
  without changing the original predictions or input cutoffs.

The 2026 research holdout remains separate from prospective forecast scoring.
Fresh TabFM inference is an offline operator task. Advanced strategy research,
optional Jev/ensemble experiments remain later work. The owner completed Kaggle
example notebook publication. Dataset file-description refinement is on the back burner at the owner's
request.

The [November 10 quality review](data-quality-2026-11-10.md) documents the stale
Mexico parser-cache repair and all historical partial/cancelled-session review
targets. Kaggle Version 4 now includes the local/GCP race repairs; prior Kaggle
versions remain historical snapshots.
The November 10 replay/public-mode regression suite passed 67 tests, and the
repository hygiene scan reported zero findings. The 713,765,222-byte serving
delta passed upload checksum comparison, all 195 payload SHA-256 checks and
SQLite integrity/foreign-key checks before activation. Both race timing source
objects are included so the new replay can retain recorded ranks and gaps.
The public browser check selected all observed Mexico drivers, verified
Play/Pause, and sought to one hour into the race with a measured weather
observation at 00:59:39. The former replay load error did not recur.
