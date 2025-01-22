# Base image
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# Copy project files
COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/
COPY data/ data/
COPY models/ models/

# Install the package
RUN pip install . --no-deps --no-cache-dir

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run inference
ENTRYPOINT ["python", "-u", "src/ehr_classification/inference.py"]
