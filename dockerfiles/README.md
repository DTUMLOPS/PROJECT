# Docker Setup Instructions

This project provides several Docker containers for training, evaluation, inference, and API serving of the EHR classification model.

## Prerequisites

- Docker installed on your machine
- Docker Compose installed on your machine
- At least 8GB of RAM recommended
- GPU support (optional but recommended for training)

## Running Services with Docker Compose

First, build all services:
```bash
docker-compose build
```

### API Service
```bash
docker-compose up api
```
The API will be available at `http://localhost:8000`

### Training
```bash
docker-compose run --rm train
```

### Evaluation
```bash
docker-compose run --rm evaluate
```

### Inference
```bash
docker-compose run --rm infer
```

## Running Services with Invoke

The project includes an `invoke` task runner for convenience. Here are the available commands:

### Docker-Related Tasks
```bash
invoke docker-build         # Build all docker images
invoke docker-train        # Run training in Docker container
invoke docker-evaluate     # Run evaluation in Docker container
invoke docker-infer        # Run inference in Docker container
invoke docker-api          # Run API server in Docker container
invoke docker-down         # Stop all docker containers
```

## Project Structure Requirements

The project expects certain directories to be mounted as volumes. Make sure you have the following structure:

```
your_project/
├── models/
│   └── split_1/
│       └── model checkpoints (.ckpt files)
├── data/
│   └── your data files
└── docker-compose.yaml
```
