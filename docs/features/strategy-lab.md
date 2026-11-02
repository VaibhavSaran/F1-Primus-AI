# Strategy Lab

**Strategy Lab** (`/strategy`) explores observed tyre use and pit visits for a saved session. A compound change or visit does not establish optimal strategy, stationary service duration, measured tyre temperature or why a team stopped.

## What to inspect

**Observed tyre stints** shows drivers, compound, observed boundaries and tyre-age evidence. **Observed pit visits** pairs recorded entry/exit clocks where possible. Labels distinguish initial pit departure, paired visits and unmatched entry/exit observations. A lap-1 departure need not have an earlier entry; final garage returns may lack a later departure. Unpaired records are incomplete evidence rather than automatic proof of an API failure.

**Lap time and tyre age** focuses on one driver's observed pace and compound/stint progression. Fuel, traffic, flags, circuit evolution and pace targets are not controlled. An entry-to-exit duration includes pit-lane travel where bounded; it must not be called stationary tyre-change time. Missing durations stay unknown.

The repository also contains strategy/simulation helpers. Their existence does not make the archive page's observed chart a newly run optimizer. Viewing a stint timeline does not establish measured counterfactual improvement.

## Example 1: completed 2020 Spanish Grand Prix

Select Spain 2020 Race, read a driver's recorded compounds/stints, then match visits and the lap-time/tyre-age chart. Describe what the clocks support. An unbounded visit keeps an unknown duration. Faster laps after a stop do not isolate degradation or prove the stop was optimal.

## Example 2: Mexico 2026, race dated 1 November

Select Mexico FP2 to inspect practice stints and departures. Only FP1/FP2/FP3/Q are published because R collection failed. Practice tyre use is not the Grand Prix's actual strategy. Eight prepared models cover 22 drivers and remain unscored; finishing-order estimates contain no evidence of actual Mexico Race pit stops.

## Implementation

Sources: [archive strategy](../../streamlit_app/archive_views.py), [stint logic](../../ml/stints.py), [strategy component](../../streamlit_app/components/strategy_ui.py), [schema](../../data/archive/schema.py). See [model workflow](../archive-model-workflow.md) for modelling and missing-input requirements.
