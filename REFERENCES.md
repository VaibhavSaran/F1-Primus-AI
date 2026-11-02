# References and acknowledgements

References credit inspiration and tooling, not endorsement or equal prediction
performance. Preserve upstream licences/notices when incorporating code; citation
alone is not a licence.

## Reference projects and authors

| Author / GitHub profile | Repository | Relevance |
|---|---|---|
| [Tom Shaw (@IAmTomShaw)](https://github.com/IAmTomShaw) | [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) | Race replay, telemetry and visualization inspiration |
| [Mariana Antaya (@mar-antaya)](https://github.com/mar-antaya) | [2025_f1_predictions](https://github.com/mar-antaya/2025_f1_predictions) | Weekend inputs and historical prediction experiments |
| [Mariana Antaya (@mar-antaya)](https://github.com/mar-antaya) | [2026_f1_predictions](https://github.com/mar-antaya/2026_f1_predictions) | Subsequent prediction research and XGBoost exploration |

## Data sources

- [FastF1 maintainers](https://docs.fastf1.dev/): timing, laps, telemetry, track
  conditions and source-provided weather. Library availability and upstream rights differ.
- [Jolpica F1 maintainers](https://github.com/jolpica/jolpica-f1): historical results
  and schedule/reference data, continuing the Ergast-compatible interface.
- [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api): independent
  reanalysis, kept distinct from observed track weather.
- [Formula 1](https://www.formula1.com/), [FIA](https://www.fia.com/) and
  [Pirelli Motorsport](https://press.pirelli.com/): individually cited race,
  regulation and tyre context. Source articles are not redistributed wholesale.

## Models and software

- [Google Research TabFM](https://github.com/google-research/tabfm) and
  [TabFM introduction](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/): research adapter/checkpoint attribution.
- [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/)
  and [PyTorch](https://pytorch.org/): model implementation and inference.
- [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/python/),
  [pandas](https://pandas.pydata.org/) and [Apache Arrow](https://arrow.apache.org/):
  interface, visualization, analysis and Parquet interoperability.
- [Anthropic documentation](https://docs.anthropic.com/): bounded Claude tools
  and web citations. 
- [MLflow](https://mlflow.org/): optional experiment projection;
  canonical application tracking remains in the local store.

