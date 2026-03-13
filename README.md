# 🏎️ F1 Primus AI

Autonomous pre-race prediction pipeline for the 2026 F1 season.
Built with LangGraph, LangChain, Claude AI, MLflow & Streamlit.

---

## What It Does

For any 2026 F1 race, the pipeline automatically:
1. Fetches race weekend weather (historical or forecast)
2. Analyses FP2 practice session data via FastF1
3. Searches for grid penalties and team news via Tavily
4. Runs a Gradient Boosting ML model to predict race times
5. Generates a full pre-race report using Claude as the AI brain

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| LLM | Claude (claude-sonnet-4-5) |
| F1 Data | FastF1 |
| Weather | Open-Meteo |
| News | Tavily |
| ML Model | Gradient Boosting (scikit-learn) |
| ML Tracking | MLflow |
| Pipeline Scheduling | Prefect |
| UI | Streamlit |
| Deployment | GCP Cloud Run |

---

## Project Structure
```
f1-primus-ai/
├── agents/
│   ├── graph.py                  # LangGraph orchestrator
│   ├── prompts/race_analyst.py   # System prompts
│   └── tools/                    # 4 LangChain tools
├── ml/model.py                   # ML model + MLflow
├── flows/pre_race_pipeline.py    # Prefect flow
├── streamlit_app/                # UI
├── docker/                       # Dockerfile + Compose
├── .github/workflows/deploy.yml  # CI/CD
└── config.py                     # Season config + calendar
```

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/f1-primus-ai.git
cd f1-primus-ai
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Start MLflow server
```bash
mlflow server --port 5001 --workers 1
```

### 4. Run the Streamlit UI
```bash
streamlit run streamlit_app/app.py
```

### 5. Or run via Prefect flow
```bash
# Specific round
python -m flows.pre_race_pipeline --round 2

# Auto-detect next race
python -m flows.pre_race_pipeline
```

---

## API Keys Required

| Key | Where to get |
|---|---|
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `TAVILY_API_KEY` | app.tavily.com |
| `GEMINI_API_KEY` | aistudio.google.com |

---

## Run with Docker
```bash
# Build and start all services
docker-compose -f docker/docker-compose.yml up --build

# App runs at http://localhost:8501
# MLflow runs at http://localhost:5001
```

---

## Deploy to GCP Cloud Run

### GitHub Secrets Required
| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Service account JSON key |
| `ANTHROPIC_API_KEY` | Your Anthropic key |
| `TAVILY_API_KEY` | Your Tavily key |
| `GEMINI_API_KEY` | Your Gemini key |

Once secrets are set, every push to `main` auto-deploys.

---

## Known Limitations

- ML model trained on a single 2025 race — predictions need improvement
- Weather forecasts limited to 16 days ahead (Open-Meteo limit)
- STR and PER missing from 2026 FP2 data — estimated from slowest driver pace

---

*Generated reports powered by LangGraph & Claude AI*
*ML Tracking: MLflow | Data: FastF1 & Open-Meteo | News: Tavily*
*Idea and Credits: https://github.com/mar-antaya/2025_f1_predictions*