# ==============================================================================
# Q.ANT Native Computing Toolkit & Phi-mini-MoE Inference Dockerfile
# ==============================================================================

# Python 3.12 Slim als Basis-Image (abgestimmt auf die cp312 Wheels)
FROM python:3.12-slim

# Umgebungsvariablen fuer Python, Hugging Face Hub und Cache
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/app/cache/huggingface

# Arbeitsverzeichnis festlegen
WORKDIR /app

# System-Abhaengigkeiten:
# - libgomp1: Erforderlich fuer OpenMP (CPU-Parallelisierung von PyTorch & Q.ANT)
# - curl & ca-certificates: Fuer HTTPS-Verbindungen zu Hugging Face
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Layer-Caching: Erst Requirements und Wheel-Dateien kopieren
COPY requirements.txt .
COPY qant_wheels_cpu/ ./qant_wheels_cpu/

# 1. Pip aktualisieren
# 2. PyTorch CPU gezielt ueber das offizielle PyTorch-Wheel-Repo installieren
#    (reduziert das Image um > 2 GB im Vergleich zum CUDA-Default)
# 3. Projekt-Abhaengigkeiten inkl. lokalem Linux-Wheel fuer Q.ANT installieren
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Quellcode kopieren
COPY . .

# Cache-Verzeichnis fuer Hugging Face anlegen
RUN mkdir -p /app/cache/huggingface

# Einstiegspunkt: Standardmaessig wird phi3_inference.py ausgefuehrt.
# - Ohne Argumente: Schnelle Q.ANT NPU Token-Generierungs-Demo
# - Mit Argumenten: z. B. `docker run <image> --full "Deine Frage"`
ENTRYPOINT ["python", "phi3_inference.py"]
CMD []
