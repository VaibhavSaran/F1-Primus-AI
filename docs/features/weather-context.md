# Weather & Context

**Weather & Context** (`/weather`) displays saved conditions, track status and race-control messages. Observations, sourced causal explanations and genuinely issued pre-race forecasts have different evidence requirements.

## Reading the panels

**Session conditions** groups supplied measurements into Temperature, Wind, Humidity & pressure, and Rain tabs. Air/track temperature, wind direction/speed, humidity, pressure and rain flags preserve recorded units and missingness. A rain flag is not rainfall depth, forecast probability or proof that every part of the circuit was wet. Session clocks use hours:minutes:seconds.

**Track conditions** shows track-status events. **Race control** lists time-stamped provider messages and available associated fields. A flag, pit visit or retirement describes an observation. Its cause needs independent supporting evidence. The event-context subsystem keeps normalized sporting outcomes and sourced incident context separately; a valid status can coexist with an unknown cause.

Independent Open-Meteo historical weather has a separate collection path and provenance. Historical observations/reanalysis describe the past; they are not automatically forecasts known at a model cutoff. Forecast features require issued-forecast provenance. This page does not start downloads or synthesize missing measurements.

## Example 1: completed 2020 Spanish Grand Prix

Choose Spain 2020 Race. Inspect temperature/wind, then track status and race control. Compare an event's clock with Replay. A recorded temperature or status change is evidence of that condition; attributing a driver's pace change to it remains an inference unless evidence isolates the cause.

## Example 2: Mexico 2026, race dated 1 November

Choose Mexico FP3 or Q and inspect weather recorded during that session. Only FP1/FP2/FP3/Q are published locally because R collection failed. Do not relabel those observations as Mexico Race weather or add later observations to frozen pre-race inputs. Eight prepared models with 22 drivers remain unscored; missing weather/result evidence does not supply an outcome.

## Implementation and references

Sources: [weather view](../../streamlit_app/archive_views.py), [weather reader](../../data/archive/reader.py), [weather extraction](../../data/archive/weather_extract.py), [event-context store](../../data/archive/event_context.py), [weather notebooks](../../data/archive/notebook_weather.py). See [event context](../event-context.md), [collection](../data-collection-2020-2026.md) and [archive](../data-archive.md).
