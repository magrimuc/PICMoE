# ==============================================================================
# Q.ANT Native Computing Toolkit & Phi-mini-MoE Inference Dockerfile
# ==============================================================================

FROM python:3.12-slim

# Build-Argument: 'cpu' (Standard: Consumer-Hardware ohne Treiber) 
# oder 'hardware' (Server-Rack mit physischer Q.ANT NPU PCIe-Karte)
ARG BACKEND=cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/app/cache/huggingface

WORKDIR /app

# System-Abhaengigkeiten:
# - libgomp1: Erforderlich fuer OpenMP (Parallelisierung von PyTorch & Q.ANT)
# - curl & ca-certificates: Fuer sichere Verbindungen zu Hugging Face
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Layer-Caching: Erst Requirements und Wheels kopieren
COPY requirements.txt .
COPY qant_wheels_cpu/ ./qant_wheels_cpu/
COPY qant_wheels_hardware/ ./qant_wheels_hardware/

# 1. Pip aktualisieren
# 2. PyTorch CPU-Version installieren
# 3. Passendes Q.ANT Toolkit installieren (CPU-Simulations-Backend oder Hardware-Treiber-Backend)
# 4. Restliche Abhaengigkeiten installieren
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    if [ "$BACKEND" = "hardware" ]; then \
        echo "=== Baue Image fuer physische Q.ANT NPU Hardware ===" && \
        pip install ./qant_wheels_hardware/qant_native_computing_toolkit-2.4.0-cp312-cp312-linux_x86_64.whl; \
    else \
        echo "=== Baue Image fuer Consumer-Hardware (CPU-Simulation) ===" && \
        pip install ./qant_wheels_cpu/qant_native_computing_toolkit-2.4.0-cp312-cp312-linux_x86_64.whl; \
    fi && \
    pip install -r requirements.txt

# Quellcode kopieren
COPY . .

# Cache-Verzeichnis anlegen
RUN mkdir -p /app/cache/huggingface

# Einstiegspunkt: Standardmaessig wird phi3_inference.py ausgefuehrt.
# Argumente wie `--full "..."` koennen direkt an `docker run` uebergeben werden.
ENTRYPOINT ["python", "phi3_inference.py"]
CMD []
