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
FROM python:3.14-slim AS builder

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
FROM python:3.14-slim AS runtime

# Build provenance (juniper-ml notes/BUILD_PROVENANCE_DESIGN_2026-06-14.md):
# the deploy Makefile passes this repo's own git SHA, an ISO-8601 build
# timestamp, and the package version at build time. They are stamped as OCI
# labels and exported as env vars (below) so the running service reports them
# on /v1/health and `make doctor` can detect stale-image drift. Default empty
# when the image is built bare (read back as None by the app).
ARG GIT_SHA=""
ARG BUILD_DATE=""
ARG APP_VERSION=""

# Labels for container metadata
LABEL org.opencontainers.image.title="JuniperData"
LABEL org.opencontainers.image.description="Dataset generation service for the Juniper ecosystem"
LABEL org.opencontainers.image.version="${APP_VERSION}"
LABEL org.opencontainers.image.authors="Paul Calnon"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/pcalnon/juniper-data"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

# Create non-root user for security
RUN groupadd --gid 1000 juniper && \
    useradd --uid 1000 --gid juniper --shell /bin/bash --create-home juniper

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
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

# Build provenance (see the ARG block in the runtime stage above): exported so
# the app process can read its own source revision / build date and report them
# on /v1/health. Empty when built bare (read back as None).
ENV JUNIPER_DATA_GIT_SHA=${GIT_SHA}
ENV JUNIPER_DATA_BUILD_DATE=${BUILD_DATE}

# Expose the API port
EXPOSE 8100

# Health check for container orchestration (liveness + readiness)
# start-period=5s: FastAPI with lightweight deps starts in <2s on typical hardware
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/v1/health', timeout=5)" || exit 1

# Run the application
CMD ["python", "-m", "juniper_data"]
