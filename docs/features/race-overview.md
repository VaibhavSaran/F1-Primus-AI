# Race Overview

**Race Overview** (`/overview`) introduces a selected saved session. Choose Season, Event/round and Session. Only published downloaded sessions are offered. Changing season clears dependent event/session selections; changing event clears the session. A changed publication also resets dependent analysis selections. Event names, dates and session choices come from the saved archive calendar.

## What the page shows

The archive heading becomes **event · session**. Cards count drivers, recorded laps, weather observations and observed stints. Race/Sprint classification shows recorded finish position, driver/team at the event, grid, laps, raw sporting status and points. Qualifying shows classification with available Q1/Q2/Q3 times. Practice lists fastest recorded non-deleted, non-generated timed laps and recorded lap counts. Practice times are not a Race classification: fuel loads and engine modes are unknown.

Expand **Drivers and teams in this session** for event-specific identities. Follow the pointers to Compare, Replay, Strategy and Weather for deeper views. Classification and coverage answer different questions: a genuine retirement can coexist with a complete publication, while missing telemetry can coexist with valid results. Participant filtering excludes empty provider placeholders from the observed field.

If only a legacy saved input snapshot is available, the fallback shows saved entries, source details and historical session inventory. It does not fabricate archive tables. Missing context asks for a selection; unreadable data is reported.

## Example 1: completed 2020 Spanish Grand Prix

Choose 2020 → Spanish Grand Prix → Race. Read Race classification alongside the cards. A recorded retirement is a sporting outcome rather than evidence of failed collection. Switch to FP1: the heading and table change to practice, and a best timed lap no longer describes Grand Prix finish order.

## Example 2: Mexico 2026, race dated 1 November

Choose Mexico Qualifying or FP1/FP2/FP3. Only those four sessions are saved because R collection failed. Q may show its own classification, but the page must not offer a synthesized Mexico Race classification. Eight prepared models with 22 drivers remain unscored elsewhere; their estimated ranks do not fill this missing outcome.

## Implementation

Sources: [context selectors](../../streamlit_app/components/context.py), [routing](../../streamlit_app/screens.py), [archive overview](../../streamlit_app/archive_views.py), [archive cache](../../streamlit_app/archive_services.py), [ArchiveReader](../../data/archive/reader.py). See [archive storage](../data-archive.md) for publication and coverage semantics.
