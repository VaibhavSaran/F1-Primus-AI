# Race Replay

**Race Replay** (`/replay`) projects recorded observations onto a playback clock. Select participants, then use Play/pause, speed, seek and native jump-to-time/lap/event controls. Reduced motion disables automatic animation while retaining seek. This is saved evidence analysis, not a live feed or replacement for missing data.

## Reading the replay

The circuit uses observed session geometry with stable bounds. Driver markers follow recorded positions and disappear after five seconds without a fresh position; missing coordinates do not prove retirement. Weather and event highlights follow the same clock. Air/track temperature, humidity, pressure, wind speed/direction and rain status remain unknown when absent or older than two minutes. Session time uses hours:minutes:seconds.

Race/Sprint standings distinguish three evidence modes:

- **Race standings** uses version-linked retained timing snapshots. Position, gap and interval are recorded observations. Carried snapshots expire after two minutes. Interval refers to the car ahead in the full field even when a subset is selected. Lap deficits stay lap labels rather than invented seconds.
- **Lap-end classification checkpoints** uses recorded lap positions when usable timing is absent. These are checkpoints, with gaps/intervals unavailable between them.
- **Session lap completion** orders positive timed-lap observations. Its **Order** column is not race position; this is the practice/qualifying or missing-classification fallback.

Final classification never substitutes for historical running order. Native and browser playback share bounded replay contracts. Causal mode limits observations to the play clock; hindsight controls do not rewrite recorded timing. Tyres and lap completion honor selected participants. Missing samples, parser fills and untimed laps remain explicit gaps.

## Example 1: completed 2020 Spanish Grand Prix

Choose Spain 2020 Race and the desired field. Play, pause and jump to a later recorded lap/event. Compare standings with changing markers and weather. Retained timing updates with the play clock, and need not match final classification until the end. Selecting two drivers must preserve their absolute full-field ranks and intervals rather than relabelling them first/second.

## Example 2: Mexico 2026, race dated 1 November

Choose Mexico FP1/FP2/FP3/Q and replay available coordinates and lap completion. Only those sessions are published locally because R collection failed. The application cannot replay the missing Grand Prix or derive its running order from Q. Eight prepared models with 22 drivers remain unscored; their estimated ranks are not a simulated actual Race.

## Implementation and limits

Sources: [archive adapter](../../data/archive/replay_adapter.py), [timing reader](../../data/archive/timing.py), [replay contracts](../../data/replay.py), [native UI](../../streamlit_app/components/replay_ui.py), [browser clock](../../streamlit_app/components/replay_component/clock.js). Timing uses retained sources linked to the selected publication, not an unrelated raw HTTP cache. See [archive reference](../data-archive.md).
