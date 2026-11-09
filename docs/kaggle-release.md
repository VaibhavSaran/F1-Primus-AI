# Curated Kaggle dataset release guide

This guide describes the yearly F1 publication design. The weather-only dataset
has been published; see [release status](release-status.md). The yearly F1
packages remain local. The canonical local
archive stays unchanged. Publication requires a reviewed export, appropriate
redistribution rights and the owner's explicit upload action. This code repository
is not the dataset and must not contain downloaded telemetry or model weights.

## Recommended distribution

Publish **one versioned dataset per year** (2020–2026), with a small index linking
years. Mark 2026 as incomplete/in progress. Use the existing exporter described
in [dataset-release.md](dataset-release.md), rather than copying the working
archive. Keep its implemented structure:

```text
f1-YYYY/vN/
├── README.md                         # Data card, coverage and examples
├── catalogue.sqlite                  # Reference/results/laps/weather tables
├── bundles/
│   └── round-NN.zip                  # Normalized car/position Parquet members
├── schema.json                      # Types, nullable fields, units and keys
├── manifest.json                    # Pinned file/session inventory
├── source_versions.json             # Sanitized source/version lineage
├── quality.json                     # Missing/partial/unsupported coverage
├── examples.ipynb                   # Clean, runnable loading examples
└── ...                              # Checksums, validation and attribution
```

ZIP member paths retain year-specific archive round identity, session code,
dataset kind and source filename. The authoritative manifest maps members to
driver/session identities; never infer identities solely from filenames. Store
source event names and dates as well as stable IDs because calendars can change.

Dense telemetry stays Parquet; small query tables stay in SQLite. This avoids
duplicating all small tables in two formats and works with the existing builder.
The export does not include SDK caches, raw HTTP responses, parser pickles,
login state, chat/forecast databases, secrets or model checkpoints.

## Data card requirements

- Scope: collected years, completed-session cutoff, refresh timestamp, missing
  sessions, partial coverage and known cancellations. Do not advertise a complete
  2026 season while Mexico's race or future weekends are absent.
- Sources and attribution: FastF1/upstream origins, Jolpica, Open-Meteo and specific
  curated context references; distinguish library licence from upstream data rights.
- Schema: primary/foreign keys, event/session/driver joins, original identifiers,
  nullable fields, clock origin, units and measured versus derived values.
- Timing: UTC publication/retrieval times differ from event time. Canonical integer
  nanoseconds must be converted explicitly. Track coordinates are recorded local
  coordinates, not guaranteed GPS or an official circuit map.
- Weather: track observations and independent reanalysis have separate provenance.
  Neither should be relabelled as a pre-race weather forecast.
- Limitations: missing/stale telemetry, generated samples, unknown tyre temperatures,
  pit-lane durations rather than stationary stop time, DNF classification and
  session cancellation/withdrawal context. Missing is not zero.
- Use cases and examples: SQL results, joining laps to drivers, reading one
  driver's Parquet, complete-session replay alignment and source-aware weather joins.
- Reproducibility: package versions, manifest/checksum verification, per-file row
  counts and release version. Separate datasets from model evaluation claims.
- Terms: selected dataset licence only after source-by-source review. Do not
  default third-party F1 data to the repository's GPLv3 licence.

## Release checks and updates

1. Pin a fresh reviewed cohort; the existing exporter defaults to a historical
   cohort, so new downloads are not automatically included.
2. Measure normalized export size and free staging space before building.
3. Build to a new ignored release directory; validate checksums, row counts, schema,
   relational joins, reference IDs and clean sample loading.
4. Inspect every exported metadata field for local paths, emails, source query
   tokens and private operational state. Check the generated data card and notebooks.
5. Review the dataset privately on Kaggle, inspect the uploaded file tree and run
   its examples before making it public. Credentials never live in the release folder.
6. Future releases append dated versions and changelogs; preserve old manifests.
   Replacing a corrected result must record the change rather than rewrite history.

See [the dated collection calendar](data-refresh-calendar.md),
[public tracker](data-tracker.md) and
[official Kaggle dataset CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).
No uploader or recurring publication task is enabled here.
