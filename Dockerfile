# Base stage with common dependencies
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/
COPY data/ data/

# Install the package
RUN pip install . --no-deps --no-cache-dir


# API stage
FROM base AS api
RUN mkdir -p /root/.cache/wandb
EXPOSE 8080
ENTRYPOINT ["uvicorn", "ehr_classification.api:app", "--host", "0.0.0.0", "--port", "8080"]

# Training stage
FROM base AS train
ENTRYPOINT ["python", "-u", "src/ehr_classification/train.py"]

# Evaluation stage
FROM base AS evaluate
COPY models/ models/
ENTRYPOINT ["python", "-u", "src/ehr_classification/evaluate.py"]

# Inference stage
FROM base AS infer
COPY models/ models/
ENTRYPOINT ["python", "-u", "src/ehr_classification/inference.py"]
