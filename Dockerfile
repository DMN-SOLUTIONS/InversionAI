# =============================================================================
# InversionAI - Multi-stage Docker Build
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build Tomofast-x from source
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS tomofast-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gfortran \
    make \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone https://github.com/TOMOFAST/Tomofast-x.git && \
    cd Tomofast-x && \
    make -j1

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# System dependencies for GDAL, Fortran runtime, and OpenMPI
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgfortran5 \
    libopenmpi-dev \
    openmpi-bin \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Tomofast-x binary from builder stage
COPY --from=tomofast-builder /build/Tomofast-x /app/Tomofast-x

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
