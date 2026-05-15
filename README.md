# Vivarium Monitoring Pipeline

A complete, production-ready computer vision pipeline built with FastAPI and the Factory Pattern. It analyzes images of a small animal vivarium to detect water levels, food levels, and mouse presence.

## Features

- **Modular Architecture**: Every component (Detectors, Measurers, Notifiers, Storage) is swappable via `config.yaml` with zero code changes.
- **YOLOX Detection**: Uses [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (installed from source) for object detection.
- **Multiple Measurers**: Estimate levels using OpenCV edge detection or custom ONNX classifiers.
- **Asynchronous**: Built on FastAPI, asyncpg, and SQLAlchemy Async for high concurrency.
- **Alerting Engine**: Configurable thresholds and cooldowns to trigger Telegram, Email, or Webhook notifications.
- **Storage Options**: Save pipeline results to PostgreSQL, and images to Local Disk or AWS S3.

## Setup Instructions

### 1. Requirements

- Python 3.11+
- PostgreSQL 14+
- Git (to install YOLOX from source)

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

#### Install YOLOX from source

YOLOX is not available on PyPI and must be installed manually:

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX && pip install -e . --no-deps
```

### 3. Environment Variables

Copy the example environment file and fill in your secrets:

```bash
cp .env.example .env
```
Ensure you provide real values for database credentials, your API key, and notification secrets.

### 4. Database Setup

Run the initial migration against your PostgreSQL database:

```bash
psql -U your_postgres_user -d your_database -f migrations/001_initial.sql
```

## How to Run Locally

You can override settings for local development by editing `config/config.local.yaml`.
Make sure you have downloaded the required YOLOX model weights into `weights/`.

Run the application using `uvicorn`:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## How to Use

### Analyze an Image

Send a POST request to `/analyze` with your image and API key:

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "X-API-Key: your-secret-api-key-here" \
     -F "image=@/path/to/vivarium.jpg"
```

### Query Historical Results

Fetch past results from the database:

```bash
curl -X GET "http://localhost:8000/results?limit=10" \
     -H "X-API-Key: your-secret-api-key-here"
```

## Swapping Engines via Config

Because of the strict factory pattern, you can switch the backend engine for any module just by changing a string in `config/config.yaml` and restarting the server.

## Adding a New Engine Implementation

The application is designed to be infinitely extensible without modifying core logic.
To add a new Image Store (for example, Azure Blob Storage):

1. **Create the file**: `pipeline/storage/image_store/azure.py`
2. **Subclass the base**: Inherit from `BaseImageStore` defined in `base.py`.
3. **Implement methods**: Implement the `save()` and `get_url()` abstract methods.
4. **Register in Factory**: Open `pipeline/storage/factory.py` and add your class to the `_IMAGE_STORE_REGISTRY`:
   ```python
   _IMAGE_STORE_REGISTRY = {
       "local": "pipeline.storage.image_store.local.LocalImageStore",
       "s3": "pipeline.storage.image_store.s3.S3ImageStore",
       "azure": "pipeline.storage.image_store.azure.AzureImageStore", # <-- New!
   }
   ```
5. **Update Config**: Set `storage.image_store: azure` in your `config.yaml`.

Nothing else needs to change. The orchestrator will automatically instantiate your new class and call its methods.
