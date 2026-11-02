# About and credits

**About** (`/about`) describes the project, acknowledges sources and distinguishes research from verified future forecasting. It does not require a selected session. Primus AI is an independent personal F1 data-science project, not an official Formula 1 service.

## Getting started, Page guide and Credits

**Getting started** explains how to ask a question, optionally bind context, choose a downloaded weekend and move from analysis to forecasts, validation and exports. It includes a completed United States 2025 example and the Mexico 2026 missing-Race example. **Page guide** has an expander for each of the eleven visible pages, including the three Model Validation tabs. It explains the controls and evidence a user should expect.

**Credits** links FastF1 for timing/telemetry, Jolpica for historical results, Open-Meteo for independent historical weather, Streamlit/Plotly/pandas, Google Research TabFM, XGBoost/scikit-learn/PyTorch, and Anthropic Claude. It names Tom Shaw (@IAmTomShaw) for replay visualization inspiration and Mariana Antaya (@mar-antaya) for the linked 2025/2026 prediction research. Attribution is not a data redistribution license; review upstream terms before sharing data. Claude supplies reasoning/cited search; saved numeric forecasts come from local tools.

Historical reconstruction measures archived-data research. Future forecasting performance needs frozen pre-race records and later genuine results. Sources and uncertainty still need checking. About does not install services or demonstrate hosted deployment.

## Example 1: completed 2025 United States Grand Prix

After viewing the United States 2025 historical case, use About to explain its research status and acknowledge the data/model libraries. A completed archived race supports retrospective scoring; unverified original publication timing prevents presenting that score as proof of a live pre-race forecast.

## Example 2: Mexico 2026, race dated 1 November

Describe Mexico as eight prepared models, 22 drivers, frozen inputs and unscored forecasts. Only FP1/FP2/FP3/Q are published because R collection failed. Credit providers without implying a local actual Race result or an operating deployed service.

## Implementation and references

Sources: [About](../../streamlit_app/screens.py), [foundation registry](../../ml/foundation_models.py), [Claude provider](../../agents/claude_provider.py). See [foundation models](../foundation-models.md), [historical evaluation](../historical-model-evaluation.md), [license](../../LICENSE.md) and [documentation index](../README.md).
