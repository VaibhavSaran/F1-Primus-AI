# Acquisition and project update plan — November 2, 2026

This is an operator plan, checked as of November 2 in America/Chicago. No
collection, authentication, scheduler, automatic refresh, release or upload was
started. Existing notebooks retain their November 2 cutoff until an operator
deliberately changes it. The [tracker](data-tracker.md) reports actual saved
coverage; dates below are proposed acquisition windows, subject to provider access.

## Current backlog

The current local archive has 94 of 95 eligible 2026 sessions: Mexico FP1/FP2/FP3/Q
are published, but its race is absent after a caught collection failure. A notebook
finishing is not proof of successful race publication. On November 2, review the
private failure evidence and plan one bounded targeted Mexico race retry after
the provider has published usable data. Do not delete caches or raw evidence,
re-authenticate blindly, or turn a missing race into a partial success.
This documentation task does not run that retry.

Mexico independent weather is 18/19 coverage, with the saved task next eligible
on **November 6, 2026 at 00:00 UTC** (November 5 at 18:00 Chicago). That is the
current stored task's gate. Once the race is published, recheck its full event
window: adding a later session can shift the six-day weather gate. Existing
practice-only weather must not be labelled full-weekend coverage without checking
the newly extended window and session links.

Historical 2020–2025 sessions are all present but include declared partials.
Review those quality flags as a separate, explicit recovery batch; do not refetch
six seasons merely to acquire one upcoming weekend.

## Remaining official weekends and collection windows

F1's [current calendar](https://www.formula1.com/en/racing/2026) and individual
event pages were checked on November 2. Remaining venue dates are São Paulo
November 6–8, Las Vegas November 19–21, Qatar November 27–29, and Abu Dhabi
December 4–6. Their current event pages list FP1/FP2/FP3/Q/R; recheck the live
format before running and collect Sprint/Sprint Qualifying only if actually
scheduled and held. Earlier seasons' Sprint formats do not establish 2026 format.

| Event and official schedule | Practice acquisition (UTC date) | Final practice / qualifying (UTC date) | Post-race acquisition (UTC date) | Earliest planned full-window reanalysis check (UTC) |
|---|---|---|---|---|
| [São Paulo](https://www.formula1.com/en/racing/2026/brazil) | November 6: FP1, FP2 | November 7: FP3, Q | November 8, or November 9 if delayed | November 14, 00:00 |
| [Las Vegas](https://www.formula1.com/en/racing/2026/las-vegas) | November 20: FP1, FP2 | November 21: FP3, Q | November 22 | November 28, 00:00 |
| [Qatar](https://www.formula1.com/en/racing/2026/qatar) | November 27: FP1, FP2 | November 28: FP3, Q | November 29, or November 30 if delayed | December 5, 00:00 |
| [Abu Dhabi](https://www.formula1.com/en/racing/2026/united-arab-emirates) | December 4: FP1, FP2 | December 5: FP3, Q | December 6, or December 7 if delayed | December 12, 00:00 |

These are dates to check eligibility, not guaranteed feed-availability times.
Las Vegas runs on Saturday November 21 at the venue but Sunday November 22 UTC
and Chicago; use an inclusive Chicago cutoff of at least November 22 for its race.
After the November 1 daylight-saving transition Chicago is UTC−6, so a 00:00 UTC
weather check is 18:00 on the preceding Chicago date. Preserve timezone-aware
timestamps instead of comparing bare dates.

For each practice/qualifying check, wait for actual session completion and provider
availability; the existing runner applies its conservative two-hour eligibility
window from the scheduled session start. Races use four hours. If a red flag,
postponement or late finish exceeds that allowance, delay the operator run rather
than treating elapsed scheduled time as proof of completion. Independent weather
uses the final eligible event-window date plus six days at 00:00 UTC; buffered
windows crossing midnight or later schedule changes can move the dates above.
The saved `next_retry_utc` and recalculated full-window coverage take precedence.
This is the project's safety margin, not a promise from the provider; see
[Open-Meteo's historical API](https://open-meteo.com/en/docs/historical-weather-api).

The [original FIA announcement](https://www.fia.com/news/fia-and-formula-1-announce-2026-calendar)
describes a 24-round calendar. F1 subsequently confirmed that Bahrain and Saudi
Arabia [would not take place in April](https://www.formula1.com/en/latest/article/bahrain-and-saudi-arabian-grands-prix-will-not-take-place-in-april.1hnqllVG85RSt8pbFc5Ivx.1hnqllVG85RSt8pbFc5Ivx).
The current F1 calendar has a different historical sequence, including an October
entry, and labels the remaining races rounds 20–23. The local archive's existing
event/session identities and observations remain its own provenance. Reconcile
calendar revisions against retained evidence before adding new events; never
renumber or silently rewrite historical local data from a newer web schedule.

## Repeatable operator sequence

1. Before each listed date, check the current official event schedule, actual
   session format and any cancellation/postponement. Preview the annual F1 and
   weather notebooks offline with `DRY_RUN=True`. Adjust only the explicit 2026
   cutoff and target event selection for the intended local run, matching the
   retained catalogue instead of guessing a round from this table.
2. When collection is separately authorized, run the existing annual F1 notebook
   with the existing shared lock, cache and checkpoint. Keep one F1 writer and one
   independent weather writer. Additional year notebooks wait; their count does
   not multiply quota. Preserve defaults that skip saved partials unless the
   operator deliberately selects a repair batch.
3. Keep FastF1 logical sends at the existing 200/hour cap and at least 11 seconds
   apart. Retain independent provider budgets and `Retry-After`/cooldown state;
   never bypass access blocks by switching mirrors or deleting the ledger. Bound
   each retry batch and stop on access denial for an explicit local account
   review. Interrupt safely; Run All resumes from retained progress.
4. After each batch, use `99_status_validate.ipynb` or the offline status command
   with `--verify-files`. Validate SQLite integrity, foreign keys, manifest hashes,
   row counts and channel quality. Keep failed, partial, missing and cancelled
   distinct. Reanalysis can remain pending without blocking F1 publication.
5. Refresh the private ledger and public aggregate tracker with the two commands
   in the [collection guide](data-collection-2020-2026.md#maintained-collection-status).
   Read every unresolved session and weather state before calling the batch done.

## How a new publication should reach the project

The existing immutable archive publications remain the source of truth. The
following coordinated refresh is the desired lifecycle; this plan does not add
an automatic end-to-end refresh or timer. See the existing
[new-race update plan](new-race-update-plan.md) for acceptance conditions.

| Stage | Required result before moving on |
|---|---|
| Local publication | Stage and validate new normalized files/source links, publish atomically as a new immutable version, retain the previous valid publication on failure. |
| UI and cache | Refresh available years/events/actual sessions from the catalogue; include the publication revision in affected caches, preserve valid selections, and keep an active replay pinned until explicit refresh. Confirm new channels render and unavailable ones remain labelled. |
| Context and profiles | Update eligible completed-result form and authoritative dated F1/FIA/Pirelli context with knowledge times. Preserve older cutoff views and source references. |
| Pre-race forecast | After available practices/Q and before actual race start, freeze a new current-time input snapshot and model ranking with exact versions/field/cutoff. Use no later race observations or retrospective weather as pre-race features. Save the receipt before outcomes exist. |
| Post-race scoring | Once the real result is validated, score only the already frozen ranking. Preserve pending/incomplete-field states; append outcome and score revisions for later penalties. Do not overwrite original forecasts or silently retrain/unlock the 2026 holdout. |
| Public dataset candidate | Prepare a new immutable, allowlisted release from eligible licensed data; update coverage, schema/units, quality notes, attribution, hashes and changelog. Test its standalone reader and inspect the exact staged files. |

Review Mexico's existing frozen pre-race receipts after its result is successfully
published; scoring is pending until then. Subsequent forecasts and scores belong
to prospective operational evidence, separate from historical benchmark studies.

## Proposed Kaggle release checkpoints

Prepare a baseline candidate after the Mexico retry and a full integrity check;
weather may be explicitly pending. Then review candidates November 9/14 (São
Paulo F1/weather), November 23/28 (Las Vegas), November 30/December 5 (Qatar), and
December 7/12 (Abu Dhabi). December 13 is a proposed season-close review after
the final weather check. These are local preparation/review dates, not scheduled
uploads. Provider delay or unresolved terms can postpone any release.

At every checkpoint compare with the previous candidate: new event/session
coverage, repaired channels, changed result revisions, source versions and file
hashes. Publish a new version only after source-by-source redistribution terms,
attribution and Kaggle license compatibility are established. An open-source
FastF1 client license does not grant redistribution rights to its underlying
feeds. Keep raw timing responses, SDK/HTTP caches, authentication state, private
IDs, model receipts, chat history and notebook outputs out of public bundles.
Use [the release guide](kaggle-release.md) for the separate staging procedure.
Public upload and deployment require their own explicit user authorization.
