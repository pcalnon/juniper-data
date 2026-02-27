# =============================================================================
# JuniperData - Dataset Generation Service
# Multi-stage Dockerfile for production deployment
# =============================================================================
# Build: docker build -t juniper-data:latest .
# Run:   docker run -p 8100:8100 juniper-data:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and project
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install pinned dependencies from lockfile (best layer caching)
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Copy project files and install without deps (already installed above)
COPY pyproject.toml README.md LICENSE ./
COPY juniper_data/ ./juniper_data/
RUN pip install --no-cache-dir --no-deps .

# -----------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="JuniperData"
LABEL org.opencontainers.image.description="Dataset generation service for the Juniper ecosystem"
LABEL org.opencontainers.image.version="0.4.0"
LABEL org.opencontainers.image.authors="Paul Calnon"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/pcalnon/juniper-data"

# Create non-root user for security
RUN groupadd --gid 1000 juniper && \
    useradd --uid 1000 --gid juniper --shell /bin/bash --create-home juniper

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create data directory with proper ownership
RUN mkdir -p /app/data/datasets && \
    chown -R juniper:juniper /app

# Switch to non-root user
USER juniper

# Environment configuration
ENV JUNIPER_DATA_HOST=0.0.0.0
ENV JUNIPER_DATA_PORT=8100
ENV JUNIPER_DATA_STORAGE_PATH=/app/data/datasets
ENV JUNIPER_DATA_LOG_LEVEL=INFO

# Expose the API port
EXPOSE 8100

# Health check for container orchestration (liveness + readiness)
# start-period=5s: FastAPI with lightweight deps starts in <2s on typical hardware
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/v1/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "-m", "juniper_data"]
