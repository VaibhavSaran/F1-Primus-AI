# Archive grounded chat and profiles

Primus AI can answer bounded local questions about the selected saved event and
session without an LLM, API key, provider call or network request. It reads the
existing archive and optional event-context store; neither reader initializes or
updates a source archive. Forecast and scenario creation still use the explicit
application workflows and their existing access controls.

`agents.archive_context.ArchiveContext(reader, context_store=None)` provides:

- `profiles(season, as_of_utc)`: the existing `build_profiles` bundle, with the
  selected season, full display names and available vetted source URLs. Only race
  outcomes from that season contribute. Affiliation is the latest observed team
  at an event, rather than a current contract claim. Form uses the latest five
  saved races; incomplete archives are not full-season statistics.
- `history(season, as_of_utc)`: trusted outcome rows for the existing profile
  builder, with publication, first retrieval and source IDs.
- An internal, finite question allowlist for results, observed lap summaries,
  compound counts, observed weather ranges, curated context and profiles.

Pass the adapter as `ChatService(..., archive_context=adapter)`. The existing
`ask` signature is unchanged: supply `event_id`, `session_id`, and a separate
`as_of_utc` knowledge cutoff. Driver/team IDs are optional; saved names and
abbreviations can resolve them. Examples include “Who won the race?”, “Which
tyres did Hamilton use?”, “Who had the fastest lap?”, “What was the race
weather?”, “Mercedes team form” and “Why was practice cancelled?”.

The cutoff is **local knowledge availability**, not just sporting event time.
Archive evidence must have been both published locally and first retrieved by
the cutoff. Context additionally respects its conservative `known_at` timestamp.
An article dated 2020 but first retrieved in 2026 cannot support a 2020 knowledge
cutoff. Current-cutoff Q&A can still discuss that historical event. Unknown
availability never gets a fabricated historical timestamp.

Answers distinguish provider result labels from independently supported causes.
Missing telemetry does not establish a retirement or DNS. An unsupported “why”
question can cite a recorded status while remaining `insufficient_data`. Source
IDs and exact version references remain available when the retained publication
has no original public URL. Displayed URLs are restricted to vetted FIA,
Formula 1 and Pirelli publisher domains; local paths and URL query credentials
are not returned as links.

Answers are intentionally compact (up to six entries unless asking for the
winner/fastest recorded lap). Laps include observed unfiltered laps; best recorded
lap is not a claim of official fastest-lap eligibility. Compound lap counts do
not establish tyre degradation or tyre life. Weather summarizes saved observed
measurements rather than predicting weather. Cross-event/session requests need
the matching selector; unsupported general explanations return insufficient
evidence. Existing authorization, leases, quotas, token bounds, duplicate
request handling and zero-cost operational accounting apply at the service
boundary, including direct tool calls.

Regression coverage: `tests/test_archive_chat.py`, `tests/test_chat.py`,
`tests/test_profiles.py`, and `tests/test_event_context.py`.
