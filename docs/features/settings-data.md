# Settings & Data

**Settings & Data** (`/settings`) explains what is saved and why an option/chart may be unavailable. Select an existing race/session to inspect coverage. Rendering never starts collection, repairs data, trains a model or publishes the application.

## What to check

**AI and integration readiness** reports configuration/dependency presence, without displaying keys. Presence is not a live authentication check, successful provider request or proof that Prefect/MLflow services are running.

**Local data library** lists published seasons, events, dates and full session names. Selectors use this saved inventory rather than every future/scheduled slot. A standard weekend does not acquire Sprint because another event held one.

**Selected session coverage** shows status/counts for car, positions, laps, classification, entries, weather, track status and race control. For R/S, Replay standings availability comes from version-pinned retained timing: Recorded timing, Lap-end checkpoints or Unavailable. It does not blindly reuse an old ingestion timing flag.

Partial coverage means missing samples/metadata. Charts omit identified parser fills and preserve gaps. Retirements and unmatched pit observations are not automatically collection defects. These streams do not supply tyre temperatures or stationary service duration. With no published sessions the page asks an operator to collect and refresh; it does not fetch automatically.

## Example 1: completed 2020 Spanish Grand Prix

Choose Spain 2020 Race. Compare lap, position and weather coverage with Replay standings. A usable timing stream can support standings while another channel is partial. Investigate the specific missing channel rather than assuming the whole race was lost. Switch to FP1 to distinguish practice from race-order semantics.

## Example 2: Mexico 2026, race dated 1 November

Mexico inventory contains FP1/FP2/FP3/Q; R collection failed and no Race publication exists locally. Inspect Q coverage without turning the missing Race into a cancellation claim. Eight prepared models with 22 drivers remain unscored. Readiness does not establish an actual outcome or functioning deployment.

## Implementation and references

Sources: [Settings UI](../../streamlit_app/screens.py), [readiness](../../streamlit_app/services.py), [inventory/cache](../../streamlit_app/archive_services.py), [version-pinned timing](../../data/archive/reader.py). Operator references: [collection](../data-collection-2020-2026.md), [recovery](../data-quality-recovery.md), [environment](../../ENVIRONMENT.md).
