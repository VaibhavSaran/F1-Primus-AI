# Compare Drivers

**Compare Drivers** (`/compare`) compares recorded performance and opens a driver's actual car telemetry for a chosen lap. Shared selectors establish the saved session and archive publication; dense data loading checks the selected version.

## Controls and interpretation

Select one through all recorded participants in **Drivers**. **Clean laps only** defaults on: after lap 1, explicitly accurate, non-deleted/non-generated, green track, no pit entry/exit. Unknown quality is excluded. Compound choices come from all recorded laps of selected drivers, including laps removed by the clean filter. Empty chart results mean the filters have no eligible observations.

**Lap times** shows pace by lap. **Sector Times** uses box plots: middle 50%, median line, whiskers within 1.5 times the box height, and recorded points. These are observations rather than controlled equal-fuel comparisons.

**Car Telemetry** has independent driver/lap controls within the selected participants. Recorded lap boundaries are required, but official lap duration may be unknown. Available tabs group Speed, Throttle & brake, Engine & gear, and DRS. Brake is binary applied/released displayed at 100/0, not pressure. DRS remains a provider code. An out-lap window starts at recorded pit exit to exclude preceding garage time; its elapsed time is not an invented official lap time.

Missing-Time parser fills are excluded. Gaps over two seconds and missing samples break chart lines; real zero speed remains valid. Missing channels, unaligned clocks and unavailable boundaries yield messages. Tyre temperatures are not supplied.

## Example 1: completed 2020 Spanish Grand Prix

Choose Spain 2020 Race and two drivers, inspect clean-lap trends and change compounds. Inspect a recorded telemetry lap. For missing-time behavior, choose Spanish FP1, Roy Nissany, lap 2: samples after recorded pit exit remain inspectable when the official lap time is unavailable. Do not convert the sample interval into a certified lap duration.

## Example 2: Mexico 2026, race dated 1 November

Choose Mexico FP2 or Q and compare recorded participants/compounds. Only FP1/FP2/FP3/Q are published locally because R collection failed. These charts describe the selected session, not Grand Prix race pace. Eight-model, 22-driver prepared forecasts remain unscored separately; predicted ranks are never actual telemetry or results.

## Implementation

Sources: [comparison/telemetry](../../streamlit_app/archive_views.py), [archive cache](../../streamlit_app/archive_services.py), [shared selection](../../streamlit_app/components/context.py). Snapshot-only fallback can show sourced profile evidence; the archive comparison follows recorded session data. See [archive contracts](../data-archive.md).
