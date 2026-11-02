# Primus AI — Formula 1 analytics

Primus AI is a personal data science and software engineering project for exploring
Formula 1 weekends, replaying recorded telemetry, comparing pre-race models and
asking evidence-based F1 questions. A Streamlit interface connects local SQLite
catalogues, versioned Parquet data, model artifacts and sourced answers.

## Features

- Ask **Primus AI** an F1 question with optional race/session context; inspect
  internal-data references or cited web sources.
- Browse results and compare drivers, laps, sectors, throttle and braking.
- Replay recorded positions alongside timing observations, weather and race events.
- Explore observed tyre stints, pit visits and weather with explicit missing data.
- Compare qualifying/form baselines, Ridge, Random Forest, Gradient Boosting,
  XGBoost and TabFM using chronological evaluation and saved forecasts.
- Track forecast outcomes and download session analysis or saved chat transcripts.

This is an independent research project, not an official Formula 1 service.
Ranking scores are not calibrated winning odds. The
[evaluation report](docs/evaluation-results.md) explains results and limitations;
retrospective research scores do not establish live accuracy.

## Start locally

Use Python **3.12.10** and a project environment named `primusEnv`.
For an existing installation:

```powershell
& '.\primusEnv\Scripts\python.exe' -m pip install -r requirements-dev.txt
& '.\primusEnv\Scripts\python.exe' -m streamlit run streamlit_app/app.py
```

For a fresh checkout, create `primusEnv` with an installed Python 3.12.10
interpreter first. See [environment setup](ENVIRONMENT.md) for Windows setup and
optional foundation dependencies. Do not copy another machine's environment.

The app binds to localhost. The repository **does not include downloaded archives,
model checkpoints or private chat/forecast stores**. Follow the
[notebook guide](notebooks/README.md) to collect data locally. An empty checkout
shows unavailable data rather than fabricated results. Telemetry browsing needs
no LLM key. Claude and provider downloads require separate local configuration;
never commit account details or `.env` files.

## Project structure

```text
F1-Primus-AI/
├── .github/workflows/          # Automated verification
├── .streamlit/                # Local Streamlit configuration
├── agents/                    # F1 chat, evidence tools and conversation history
├── data/
│   ├── archive/               # Collection, normalized storage, audits and exports
│   └── ...                    # Session, snapshot, replay and weather contracts
├── flows/                     # Forecast orchestration, reports and tracking
├── ml/                        # Features, baselines, models and evaluation
├── monitoring/                # Forecast/outcome history and query projections
├── security/                  # Access control and request/spending limits
├── streamlit_app/
│   ├── components/            # Selectors, tables and replay component
│   ├── app.py                 # Navigation and application entry point
│   ├── screens.py             # User-facing pages
│   └── services.py            # UI-to-data/model service boundary
├── notebooks/
│   └── collection_2020_2026/  # Setup, yearly F1/weather and validation notebooks
├── scripts/                   # Repository checks and tracker generation
├── tests/                     # Offline regression and UI tests
├── docs/
│   ├── features/              # One guide per feature
│   ├── architecture.md        # Mermaid architecture and request flows
│   ├── project-walkthrough.md # Completed-race and Mexico examples
│   ├── evaluation-results.md  # Results, evidence and limitations
│   └── data-tracker.md        # Dated aggregate collection status
├── docker/                    # Container scaffolding; remote release gates remain
├── CONTRIBUTING.md            # Issues, branches, checks and pull requests
├── REFERENCES.md              # Sources, authors and acknowledgements
├── SECURITY.md                # Secret handling and safe publication
├── ENVIRONMENT.md             # Runtime setup and recovery
├── LICENSE.md                 # Project code licence
└── requirements*.txt          # Runtime, development and optional packages
```

Private working directories, environments, caches, downloaded data and credentials
are intentionally excluded from this tree and from version control.

## Documentation and evidence

- [Documentation index](docs/README.md), [full walkthrough](docs/project-walkthrough.md)
  and [architecture](docs/architecture.md).
- [Model/test results](docs/evaluation-results.md), [data tracker](docs/data-tracker.md)
  and [collection/update calendar](docs/data-refresh-calendar.md).
- [Dataset export format](docs/dataset-release.md) and
  [Kaggle release guide](docs/kaggle-release.md); publication is a separate review step.

## Contributing

Open an issue with reproduction steps, expected/actual behaviour and a sanitized
screenshot or error. Do not attach secrets, session tokens, private chat logs or
an entire archive. Create a separate branch for each change, run relevant checks,
and open a PR against the maintainer-designated development branch. Include the
linked issue, behaviour change, tests and UI evidence. Do not push directly to
`main`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete workflow.

## References and acknowledgements

- **[Tom Shaw (@IAmTomShaw)](https://github.com/IAmTomShaw)** —
  [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay), for visualization ideas.
- **[Mariana Antaya (@mar-antaya)](https://github.com/mar-antaya)** —
  [2025_f1_predictions](https://github.com/mar-antaya/2025_f1_predictions) and
  [2026_f1_predictions](https://github.com/mar-antaya/2026_f1_predictions), for
  weekend-data prediction research.

Data and tooling include [FastF1](https://docs.fastf1.dev/),
[Jolpica F1](https://github.com/jolpica/jolpica-f1), [Open-Meteo](https://open-meteo.com/),
[Google Research TabFM](https://github.com/google-research/tabfm),
[XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/),
[PyTorch](https://pytorch.org/), [Streamlit](https://streamlit.io/),
[Plotly](https://plotly.com/python/) and [Anthropic](https://docs.anthropic.com/).
See [REFERENCES.md](REFERENCES.md) for attribution and research context.

Project code follows [LICENSE.md](LICENSE.md). Upstream code, checkpoints, datasets
and articles retain their own terms; the project licence does not grant permission
to redistribute third-party data.
