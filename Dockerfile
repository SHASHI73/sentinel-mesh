FROM python:3.12-slim

WORKDIR /app

# System deps needed to build a couple of the heavier Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY sentinel_mesh/ sentinel_mesh/
COPY scripts/ scripts/
COPY data/ data/

# Chroma persistence and audit log live here; mounted as a volume in compose
# so they survive container restarts and rebuilds.
VOLUME ["/app/data"]

ENTRYPOINT ["python", "scripts/run_agent.py"]
CMD ["--help"]
