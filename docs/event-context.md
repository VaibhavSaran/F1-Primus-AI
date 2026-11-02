# Sourced event context

`data/archive/event_context.py` maintains a separate SQLite repository at
`.local/event_context/context.sqlite3` (ignored local data). Its own schema is
version 1 (`PRAGMA user_version`); it does not migrate or write the F1 archive.
This supplies evidence to a future Primus AI consumer without changing chat,
models, collection, or UI routes. Importing the module and constructing a store
do not create a database. No import happens automatically.

```python
import json
from pathlib import Path
from data.archive.event_context import EventContextStore

store = EventContextStore()
# Explicit curated local JSON, after reviewing its citations:
records = json.loads(Path('.scratch/official-context-seed.json').read_text(encoding='utf-8'))
inserted = store.import_records(records)

# Explicit published versions, pinned from ArchiveReader.available_sessions():
inserted = store.import_archive(
    '.local/f1_2025_2026',
    versions={'2020:11:R': '<reviewed published version_id>'},
    cancellation_session_ids=['2020:11:FP1', '2020:11:FP2'],
    retrieved_at='2026-11-01T04:00:00Z',
)
facts = store.read(session_id='2020:11:R', version_id='<same version_id>')
available = store.read(event_id='2020:11', as_of='2026-11-01T04:00:00Z')
```

The archive importer uses SQLite `mode=ro` and `query_only=ON`, rejects missing,
unpublished or mismatched version IDs, and imports only the requested results.
It copies raw status, recorded lap count, original normalized status, classified
position, canonical driver/session/version identifiers, archive publication time,
archive path, and every version-linked source's ID, publisher, URL, payload hash,
original publication time and first fetched time. A linked source is version
provenance; it is not proof that each field was individually verified there.
No file cache, authentication, network request or source archive write occurs.

Cancellation import is separately explicit. Each requested ID must occur both in
`notebook_audit.NOT_HELD` and the selected archive's calendar. It never cancels
all practice sessions because one was cancelled, or fills a missing result with
a cancellation. Those legacy curated references have unknown source-page hash
and publication time; richer researched statements remain separate records.

`classify_outcome()` distinguishes `finished`, `lapped`, `retired`, `dns`, `dnq`,
`withdrawn`, `disqualified`, and `unknown`. Explicit statuses and recorded D/W
classification codes are used. A small explicit vocabulary of retirement
statuses is recognized only for race/sprint sessions. Unrecognized or empty
statuses remain unknown, even at zero recorded laps. Practice and qualifying
mechanical statuses do not establish DNFs. Raw provider text such as `Engine`
is preserved, but independently verified retirement reason remains unknown.
It is never inferred from absent telemetry or lap counts.

Disqualification evidence takes precedence. Explicit DNS/DNQ status then takes
precedence over a generic `W` classification; `W` remains a withdrawal fallback
for less specific raw statuses, including illness or mechanical descriptions.
The raw status and classified position remain preserved for review. Classifier
changes require a reviewed rebuild of derived context with its old database
backed up; ordinary idempotent imports never rewrite existing evidence.

Curated JSON is a list of objects. Each object requires:

- Exactly one `event_id` or `external_event_ref`. Use an `external:` identifier
  for a whole cancelled event absent from the calendar. Do not attach it to an
  unrelated event that inherited its round number. External events cannot have
  an archive `session_id`.
- `summary`, `status`, `publisher`, `url`, `confidence`, and UTC `retrieved_at`.
- `evidence_kind`: `official_statement`, `reported_context`, `cancellation`, or
  `archive_result`.
- `assertion_kind`: `fact` or `report`; `cause_qualification`: `stated`,
  `reported`, or `unknown`.

Optional `session_id`, `version_id`, `driver_id`, `author`, `published_at`,
`published_at_precision`, `first_verified_at`, `source_sha256`, and `sources`
retain specificity and provenance. Extra JSON fields are retained, including
titles, source verification limits and original time precision. Session IDs
must belong to their supplied event; versions require a session. A null author,
publication date or page SHA means unknown. A provided source SHA must be a
lowercase SHA256 of retained source bytes, not a hash of an invented summary.
Each stored canonical JSON record has its own separately labelled
`record_sha256`; this does not verify source-page bytes.

Imports validate the whole batch before inserting, use a SQLite transaction,
and ignore exact evidence duplicates. A later retrieval timestamp alone does
not create a second record; the first recorded retrieval and verification are
retained. Changed content or source hashes create additive evidence records.
No evidence is overwritten and conflicting statements are returned separately.
Existing foreign databases and unknown schema versions are refused.

`read()` accepts event, external event, session, version and driver filters and
returns JSON-compatible dictionaries with citations, canonical record text,
record ID/hash and `known_at`. Pin `version_id` when comparing archive results;
the reader never silently selects the newest result. Event-only driver records
may apply to a withdrawal before the weekend rather than a specific session.

`as_of` is a conservative **local knowledge** cutoff: `known_at` is the latest
of retrieval, first verification, publication, archive publication, and the
linked sources' publication/verification timestamps. Date-only publications
are treated as the end of their UTC day. Historical reports first verified now
are therefore excluded from earlier cutoffs even if their article date is old.
This does not establish point-in-time historical public availability or a
leakage-free forecasting corpus. A consumer must not bypass the cutoff by
filtering on event date alone.

All source text, summaries and extra JSON fields are untrusted data. The module
does not execute instructions from them, interpret them as prompts, synthesize
causes, or label all cited reports as established facts. A future agent should
cite returned sources, keep facts and reports separate, and state unknown
reasons explicitly.

Focused verification (synthetic databases only):

```powershell
& '.\primusEnv\Scripts\python.exe' -m pytest tests/test_event_context.py -q --basetemp=.scratch/event-context-tests
```
