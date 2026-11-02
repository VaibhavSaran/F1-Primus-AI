# Offline yearly dataset preparation

The exporter prepares local packages for review; it has no uploader, provider,
authentication or Kaggle dependency. Work on `ver_2` using `primusEnv`:

```powershell
# Measurement only: counts, exact source sizes, available disk.
& '.\primusEnv\Scripts\python.exe' -m data.archive.release
# Prepare all 2020–2026 years, with per-year verification before promotion.
& '.\primusEnv\Scripts\python.exe' -m data.archive.release --execute
# Recheck an existing release without changing it.
& '.\primusEnv\Scripts\python.exe' -m data.archive.release --validate '.local/release_packages/f1-2020/v1-local'
```

Inputs are read-only F1/weather catalogues, the immutable Parquet publications and
the curated event-context catalogue. The default cohort is
`87cf0ea1a028ed7d738409699e472741e251a9a4b3c952176fb73fab02535c83`:
743 published sessions, including 456 strict review sessions. It pins exact F1
version IDs and weather series independently. No "latest" publication is selected
while building. The 2026 event-date scope ends November 2, 2026 America/Chicago;
the season is in progress. The pinned cohort lacks Mexico's race and independent
weekend weather. Later collection does not silently change this release.

Each `.local/release_packages/f1-YYYY/v1-local` contains:

- A standalone `catalogue.sqlite` with small normalized reference/analytical tables,
  explicit source metadata, observed-lap stint/pit views and sanitized context.
- Stored `bundles/round-NN.zip` files holding only normalized car/position Parquet.
  Every original byte is preserved and checked against its pinned SHA256.
- `manifest.json`, `source_versions.json`, `schema.json`, `quality.json`, checksums,
  validation and build receipts; README, dictionary, attribution and runnable
  `examples.ipynb`. The SQLite small tables replace duplicate small Parquet copies.
- An incomplete `dataset-metadata.template.json`: owner/slug/licence decisions remain
  unresolved. It is deliberately unsuitable for an unattended upload.

Dense files are streamed directly into per-round ZIPs: no hardlinks, no second
unpacked telemetry copy, no compression pass over already compressed Parquet.
The recorded schema includes original provider columns alongside canonical columns.
Use the canonical nullable integer nanosecond clocks for joins. No raw `raw_*`
Parquet, parser pickle, HTTP cache, source URL, private path, worker state or
credential store is exported. Source objects use an explicit metadata allowlist;
context uses a separate sanitized field projection and content hash.

Before starting a year, the exporter checks disk headroom. It creates a new
`.VERSION.building` directory and refuses existing final or incomplete outputs.
Do not delete an incomplete output blindly: inspect the build log and choose a new
version for a retry, retaining evidence. Prior completed releases are immutable.
No partial directory is presented as a completed release.

Validation rereads SHA256 and sizes, every ZIP member CRC/hash and Parquet row count,
SQLite integrity and publication/driver relationships, and scans actual exported
text metadata and catalogue text for private path/account/credential markers.
Manifest/checksums cover the stable payload. The separately written build/validation
receipts are not included in that stable payload; build time is intentionally
separate from data-through, observation and source fetch/publication times.

Examples read the release only, compare observed lap times and load recorded
positions with earlier measured weather using a 120-second maximum age. The
reanalysis tables are hindsight products and must never be described as archived
forecasts. Laps' observed position checkpoints are available; XY/final results
are not converted into live running-order or gap estimates.

Local preparation does not settle redistribution rights. Review Formula 1/FastF1,
weather/model acknowledgments and curated-context rights, then choose a compatible
licence and Kaggle owner/visibility before a separately authorized publication.
No dataset is created or uploaded by this command.

The cross-year `release-index.json` links a separately hashed context availability
sidecar for every year/version, including early immutable packages whose sanitized
context projection omitted the store's separate `known_at` column. Copy the paired
`context-availability/f1-YYYY-VERSION.json` with an independent yearly package.
Join its `record_id` to `curated_context.record_id` and check the exported record
hash and manifest hash. `known_at` is exact local knowledge time from a consistent
context-store read, never event date or proof of historical pre-race availability.
Unknown knowledge times remain null. Sidecars preserve the promoted yearly
packages and are independently indexed by SHA256; examples do not use context
as historical predictive features. Future builds also retain `known_at` directly
in the context projection and still produce sidecars for a consistent reader API.
