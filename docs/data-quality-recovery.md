# Local data-quality recovery

Recovery is separate from collection completeness. A saved session can have
missing channels, parser-generated samples, unknown source identities or an
unknown timing reference even when file hashes and database checks pass.

## Local-first workflow

1. Pin the publication selected by `ArchiveReader.available_sessions()`. The
   reader prefers stronger lap evidence, not simply the newest timestamp.
2. Inspect the saved participation, results, raw parser streams and source
   identity evidence. Cancellation, DNS, DNF, unused reserve entries and missing
   samples are different states. Similar stream lengths or complementary values
   do not prove two driver numbers identify the same driver.
3. Recover normalized data from existing raw evidence before making API requests.
   Preserve every original publication and source file; publish additive versions.
4. For genuine unresolved acquisition gaps, try one representative source through
   the existing quota-aware transport. Keep the receipt and original response;
   stop on denied access, backoff or unchanged data. Never erase the shared cache
   or bypass a 403. A fresh local sign-in may be needed before later collection.
5. Validate changed versions, compare coverage and check the actual UI. Repeating
   the same historical download is not a repair unless observed coverage improves.

## Retained-stream recovery API

`data/archive/recovery.py` supplies `preview_raw_stream_recovery()` and
`execute_raw_stream_recovery()`. Use the existing archive with explicit session
and publication IDs. The preview checks retained file/source hashes, exact driver
membership, clock types and stream validity. It refuses to replace an existing
nonempty processed stream, so it is not an arbitrary merge or overwrite tool.

Publication requires `execute=True` and the exact preview `plan_hash`. The caller
must acquire the existing collector lock and create a catalogue backup first.
Source links, relational records, weather, old publications and other datasets
remain preserved. Repeating the same reviewed operation reuses its immutable
result instead of adding duplicate versions.

Clock rules:

- Keep original UTC `Date`, original parser `Time` and derived `SessionTime`
  distinct. Parser `Time` can repeat and is not automatically an accurate lap clock.
- Derive session time only from a pinned, known UTC origin, using explicit
  nanosecond units. If an original `SessionTime` conflicts, refuse recovery.
- With an unknown origin, recover UTC measurements while keeping session clocks
  null and quality flags explicit. Do not substitute scheduled race start or a
  guessed clock offset.
- Exclude parser missing-Time fills from observed processed samples; retain the
  original raw frames and report raw/observed/excluded counts separately.

Compare and Replay disclose when car telemetry is saved but cannot be aligned to
laps. They must not draw invented traces or make missing positions look observed.

## Qatar recovery outcomes

The local 2023 Qatar Sprint pilot recovered **459,340 observed car samples for
20 drivers** from retained raw Parquet. It preserved the original publication and
all other data. That first version remained partial because its session-clock
origin and position streams were unavailable.

After the account subscription was linked and the SDK login refreshed on
November 1, 2026, authenticated archive access succeeded. A targeted Qatar
position download supplied **468,840 observed position samples for 20 matched
entrants**. An offline reconstruction used the installed FastF1 SDK's own
telemetry/clock calculation across retained car and new position frames. The
new publication has a verified session origin, aligned car/position samples and
available required lap, car, position, weather and race-control groups. This
reduces the current partial-session inventory from 208 to 207; optional circuit
markers remain unavailable. Available channels do not prove gap-free driving.

Three additional position numbers (241, 242, 243) were retained as unidentified
raw vehicle streams, without assigning them to drivers or adding race entrants.
Car channel values were compared for every driver; only the SDK's documented
millisecond timestamp rounding and derived session clocks changed. Existing
source facts and all previous archive versions remain preserved. Eight real
Streamlit page/telemetry checks and 44 focused regressions passed.

The before/after pilot checks verified original file hashes, unchanged SQL tables,
unchanged existing dataset contents, source links, reader selection, new hashes
and row counts, database integrity and idempotence. A catalogue backup and detailed
evidence are kept locally under ignored `.scratch/partial-recovery/`.

Several other cases have retained measurements under unresolved alternate source
numbers. Those remain quarantined from guessed reassignment. A representative
Austrian 2023 FP1 position request still returned 403 with the active subscription,
while its authenticated session metadata returned 200. The specific position
resource remains unavailable; the result does not establish a general account
failure or prove that the historical source never held positions. Do not repeat
it indefinitely or promise a higher subscription will repair it.

## Entrant-aware review of remaining partials

The ignored local `.local/data_quality/current.json` and `current.md` reports
separate recorded result/lap participants from unmatched supplemental raw streams.
They pin exact selected publication IDs and retain recorded outcomes and relevant
curated citations. This review does not rewrite published coverage or clear gaps.

The November 1 snapshot retains 207 published partial sessions. Of these, 93
have available recorded-entrant channels and supplemental raw-stream flags;
93 have partial entrant channels, 17 have participant identity uncertainty as
their primary review class, and four have missing entrant channels as their
primary class. These primary classes are not independent axes: identity and
channel problems can coexist. The report preserves the underlying gap counts.

`add_original_cached_streams()` now keeps unmatched raw-vehicle fills out of the
processed entrant-channel denominator. Their original raw datasets, provisional
identities and separate partial coverage remain intact. An actual processed
entrant's missing-Time fills still downgrade that driver's and aggregate channel
coverage. Existing publications are not silently relabelled or re-exported.

Cancellation, DNS, withdrawal, retirement and missing telemetry remain separate.
Recorded retirement text is not automatically an independently verified cause.
No alternate driver-number mapping is accepted merely because sample counts or
time ranges complement another stream. Repeated unchanged 403 responses are a
reason to stop a retry batch, not evidence that samples never existed.

## Frozen local review cohort

Local cohort manifests live in ignored `.local/review_cohorts/<content-hash>/`.
They reference exact publication IDs, file paths/hashes and weather source IDs;
they do not duplicate the dense archive or publish it externally.

The strict review subset requires all required saved channels, a known session
clock and no unresolved identity flags. These checks do **not** establish
point-in-time feature availability, leakage-free labels or forecasting readiness.
ERA5 is retrospective weather. A modelling protocol must separately freeze
features, labels, cutoff policy, folds and evaluation criteria.

Later source repairs require a new cohort hash. Consumers must use the pinned
versions and paths rather than silently substituting the newest reader selection.
Keep earlier referenced immutable files so old cohorts remain reproducible.

## Verification

Use only the project environment:

```powershell
& '.\primusEnv\Scripts\python.exe' -m pytest tests/test_archive_recovery.py tests/test_recovery_ui.py
```

Use a writable ignored test temporary directory where Windows sandbox permissions
require it. The saved `99_status_validate.ipynb` evidence covers the pre-recovery
whole archive; validate each new publication separately rather than claiming that
the older notebook execution verified files created later.
