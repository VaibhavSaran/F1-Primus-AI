# Notebook index

Use the project `primusEnv` kernel. Collection is an explicit local operator action;
opening these notebooks or reading the tracker starts no download or sign-in.

| Purpose | Entry point |
|---|---|
| First-time local setup and explicit sign-in | [00_setup](collection_2020_2026/00_setup.ipynb) |
| F1 telemetry/results: one notebook per year | `collection_2020_2026/f1_YEAR.ipynb`, 2020–2026 |
| Independent hourly reanalysis: one notebook per year | `collection_2020_2026/weather_YEAR.ipynb`, 2020–2026 |
| Full local integrity and collection validation | [99_status_validate](collection_2020_2026/99_status_validate.ipynb) |
| Original pilot, retained for provenance | [f1_2025_2026_download](f1_2025_2026_download.ipynb) |

The [collection guide](../docs/data-collection-2020-2026.md#individual-notebooks)
links every annual notebook and documents parameters, resume and rate limits.
Use the annual notebooks for maintained collection. Do not run the original pilot
concurrently with them: its HTTP accounting differs. Paths remain stable so saved
workflows and notebook links continue working.

Shared implementation lives in `data/archive/notebook_f1.py` and
`data/archive/notebook_weather.py`; notebooks provide configuration and execution.
The offline CLI `python -m data.archive.status` writes private diagnostics, and
`python -m scripts.update_data_tracker` refreshes the aggregate public
[data tracker](../docs/data-tracker.md). Run both with the project interpreter
as shown in the collection guide. Detailed source payloads and outputs are private;
the repository's notebooks should be saved without outputs before publication.

For the remaining season, follow the [dated acquisition and update plan](../docs/data-refresh-calendar.md).
Changing a future cutoff and running a notebook can make provider calls; the plan
does not execute, schedule or authorize those calls by itself.
