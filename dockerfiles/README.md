# Docker Setup and Usage

This document explains how to use Docker for training and evaluating the EHR classification model without using the invoke commands.

## Prerequisites

- Docker installed on your system
- Project code and data ready in your local directory

## Directory Structure

Ensure you have the following directory structure:
```
.
├── data/
│   └── processed/
├── models/
├── outputs/
├── src/
├── configs/
├── dockerfiles/
│   ├── train.dockerfile
│   └── evaluate.dockerfile
├── requirements.txt
└── requirements_dev.txt
```

## Building the Docker Images

Build both the training and evaluation images:

```bash
# Build training image
docker build -t train_ehr:latest . -f dockerfiles/train.dockerfile

# Build evaluation image
docker build -t evaluate_ehr:latest . -f dockerfiles/evaluate.dockerfile
```

## Running Training

To run training, mount the necessary volumes for data, models, and outputs:

```bash
docker run --rm train_ehr:latest
```

The `--rm` flag ensures the container is removed after completion.

## Running Evaluation

To evaluate a trained model, run:

```bash
docker run --rm evaluate_ehr:latest
```

## GPU Support (Optional)

If you want to use GPU support, add the `--gpus all` flag:

```bash
docker run --rm --gpus all train_ehr:latest
```

## Customizing Training/Evaluation

You can override any configuration parameters by adding them at the end of the docker run command:

```bash
# Example: Change batch size and learning rate
docker run --rm \
  train_ehr:latest \
  training.batch_size=32 \
  training.learning_rate=0.0001
```
