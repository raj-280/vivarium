# ============================================================
# Vivarium Monitoring Pipeline — Dockerfile
# Python 3.11 slim base
# ============================================================

FROM python:3.11-slim

# --- System dependencies ---
# libgl1 and libglib2.0-0 are required by opencv-python at runtime
# without these, cv2 import fails silently inside the container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Install Python dependencies ---
# Copy requirements first so this layer is cached separately from code.
# Docker only re-runs pip install when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY api/        ./api/
COPY core/       ./core/
COPY pipeline/   ./pipeline/
COPY ml_models/  ./ml_models/
COPY config/     ./config/
COPY weights/    ./weights/

# --- Outputs directory ---
# Annotated images are saved here at runtime.
# Mount this as a volume in docker-compose to persist outside the container.
RUN mkdir -p /app/outputs/annotated

# --- Expose API port ---
EXPOSE 8000

# --- Start the API ---
# No --reload in production. Workers=1 because models are loaded in memory.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
