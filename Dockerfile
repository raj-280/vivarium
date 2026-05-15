# ============================================================
# Vivarium Monitoring Pipeline — Dockerfile
# Python 3.11 slim base
# ============================================================

FROM python:3.11-slim

# --- System dependencies ---
# libgl1 and libglib2.0-0 are required by opencv-python at runtime
# git is required to clone YOLOX from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Install Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Install YOLOX from source ---
# PyPI yolox package is incomplete — must install from official repo
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git /tmp/YOLOX \
    && cd /tmp/YOLOX \
    && pip install -e . --no-deps \
    && rm -rf /tmp/YOLOX/.git

# --- Copy project files ---
COPY api/        ./api/
COPY core/       ./core/
COPY pipeline/   ./pipeline/
COPY ml_models/  ./ml_models/
COPY config/     ./config/
COPY weights/    ./weights/

# --- Outputs and logs directories ---
RUN mkdir -p /app/outputs/annotated /app/logs

# --- Expose API port ---
EXPOSE 8000

# --- Start the API ---
# No --reload in production. Workers=1 because models are loaded in memory.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
