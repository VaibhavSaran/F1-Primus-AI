# F1 Primus AI â€” Streamlit app container
FROM python:3.12.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install â€” excluding Windows-only packages
COPY requirements.txt .
RUN grep -v "pywin32" requirements.txt > requirements_linux.txt && \
    pip install --no-cache-dir -r requirements_linux.txt

# Copy application code
COPY . .
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p data/reports data/raw data/processed .fastf1_cache .local

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
