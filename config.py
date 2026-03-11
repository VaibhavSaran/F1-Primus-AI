"""
Central configuration for F1 Primus AI
Loads environment variables and defines shared constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys 
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")

# MLflow 
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT   = "f1-primus-ai"

# Prefect
PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")

# F1 Season 
CURRENT_SEASON = 2026

# 2026 F1 Calendar  {round: (name, location, country, race_date)}
F1_CALENDAR = {
    1:  ("Australian GP",          "Melbourne",    "Australia",   "2026-03-08"),
    2:  ("Chinese GP",             "Shanghai",     "China",       "2026-03-15"),
    3:  ("Japanese GP",            "Suzuka",       "Japan",       "2026-03-29"),
    4:  ("Bahrain GP",             "Sakhir",       "Bahrain",     "2026-04-12"),
    5:  ("Saudi Arabian GP",       "Jeddah",       "Saudi Arabia","2026-04-19"),
    6:  ("Miami GP",               "Miami",        "USA",         "2026-05-03"),
    7:  ("Canadian GP",            "Montreal",     "Canada",      "2026-05-24"),
    8:  ("Monaco GP",              "Monaco",       "Monaco",      "2026-06-07"),
    9:  ("Barcelona-Catalunya GP", "Barcelona",    "Spain",       "2026-06-14"),
    10: ("Austrian GP",            "Spielberg",    "Austria",     "2026-06-28"),
    11: ("British GP",             "Silverstone",  "UK",          "2026-07-05"),
    12: ("Belgian GP",             "Spa",          "Belgium",     "2026-07-19"),
    13: ("Hungarian GP",           "Budapest",     "Hungary",     "2026-07-26"),
    14: ("Dutch GP",               "Zandvoort",    "Netherlands", "2026-08-30"),
    15: ("Italian GP",             "Monza",        "Italy",       "2026-09-06"),
    16: ("Madrid GP",              "Madrid",       "Spain",       "2026-09-13"),
    17: ("Azerbaijan GP",          "Baku",         "Azerbaijan",  "2026-09-26"),
    18: ("Singapore GP",           "Singapore",    "Singapore",   "2026-10-11"),
    19: ("US GP",                  "Austin",       "USA",         "2026-10-25"),
    20: ("Mexico City GP",         "Mexico City",  "Mexico",      "2026-11-01"),
    21: ("São Paulo GP",           "São Paulo",    "Brazil",      "2026-11-08"),
    22: ("Las Vegas GP",           "Las Vegas",    "USA",         "2026-11-21"),
    23: ("Qatar GP",               "Lusail",       "Qatar",       "2026-11-29"),
    24: ("Abu Dhabi GP",           "Abu Dhabi",    "UAE",         "2026-12-06"),
}

# Paths 
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
CACHE_DIR       = os.path.join(BASE_DIR, ".fastf1_cache")

# ensure dirs exist at import time
for _dir in [DATA_RAW_DIR, DATA_PROC_DIR, CACHE_DIR]:
    os.makedirs(_dir, exist_ok=True)

# 2026 Driver Lineup 
DRIVERS_2026 = [
    # McLaren
    "NOR", "PIA",
    # Mercedes
    "RUS", "ANT",
    # Ferrari
    "LEC", "HAM",
    # Red Bull
    "VER", "HAD",
    # Williams
    "ALB", "SAI",
    # Aston Martin
    "ALO", "STR",
    # Alpine
    "GAS", "COL",
    # Haas
    "OCO", "BEA",
    # Audi (formerly Sauber)
    "HUL", "BOR",
    # Racing Bulls
    "LAW", "LIN",
    # Cadillac (NEW)
    "BOT", "PER",
]