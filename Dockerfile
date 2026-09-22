FROM python:3.11-slim

LABEL maintainer="PRISM Team" \
      description="Streaming Live RAG Engine — Samsung PRISM GenAI Hackathon 3.0 Theme 4"

# Non-root user for security
RUN groupadd -r raguser && useradd -r -g raguser raguser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (pinned)
COPY requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock

# Pre-download embedding model during build (so startup is fast offline)
# BGE-small-en-v1.5 (~130MB) — no external knowledge, just the embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY bench/ ./bench/
COPY .env.example ./

# Create directories
RUN mkdir -p data/corpus logs && \
    chown -R raguser:raguser /app

# Switch to non-root user
USER raguser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
