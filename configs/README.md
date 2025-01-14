# EHR Classification Configuration Guide

This document explains how to use and configure the EHR Classification project's configuration system.

## Overview

The project uses [Hydra](https://hydra.cc/) for configuration management, which provides:
- Hierarchical configuration
- Config file composition
- Command-line overrides
- Multiple config file formats

## Directory Structure

```
configs/
├── model/
│   └── dssm.yaml         # Model architecture configuration
├── train.yaml            # Training configuration
└── evaluate.yaml         # Evaluation configuration
```

## Configuration Files

### Model Configuration (`model/dssm.yaml`)

Basic model architecture parameters:

```yaml
input_size: 37            # Number of temporal features
hidden_size: 32           # Size of hidden layers
static_input_size: 8      # Number of static features
num_classes: 2            # Number of output classes
num_layers: 2             # Number of LSTM layers
dropout_rate: 0.2         # Dropout probability
bidirectional: true       # Use bidirectional LSTM
```

### Training Configuration (`train.yaml`)

```yaml
defaults:
  - _self_
  - model: dssm

paths:
  root_dir: ${hydra:runtime.cwd}/../../
  data_dir: ${paths.root_dir}/data/processed
  model_dir: ${paths.root_dir}/models
  output_dir: ${paths.root_dir}/outputs

data:
  base_dir: ${paths.data_dir}
  splits: [1]            # Cross-validation splits to use, provide a list of splits like [1,2,3] or null to run all of them.

training:
  max_epochs: 100        # Maximum training epochs
  batch_size: 64         # Batch size
  learning_rate: 0.0001  # Learning rate
  class_weights: [1.0, 7.143]  # Class weights for imbalanced data
  use_gpu: false         # Use GPU for training
  checkpoint_dir: ${paths.model_dir}
  patience: 10           # Early stopping patience
```

### Evaluation Configuration (`evaluate.yaml`)

```yaml
defaults:
  - _self_
  - model: dssm

paths:
  root_dir: ${hydra:runtime.cwd}/../../
  data_dir: ${paths.root_dir}/data/processed
  model_dir: ${paths.root_dir}/models
  output_dir: ${paths.root_dir}/outputs

data:
  base_dir: ${paths.data_dir}
  split_number: 1

training:
  batch_size: 64
  use_gpu: false

evaluation:
  checkpoint_path: null  # Set to null to use mode-based selection
  mode: random           # Options: 'random', 'best', 'last'
```

## Usage Examples

### Training

1. Basic training with default configuration:
```bash
python src/ehr_classification/train.py
```

2. Override specific parameters:
```bash
python src/ehr_classification/train.py training.batch_size=32 training.learning_rate=0.001
```

3. Train on multiple splits:
```bash
python src/ehr_classification/train.py data.splits=[1,2,3]
```

4. Enable GPU training:
```bash
python src/ehr_classification/train.py training.use_gpu=true
```

### Evaluation

1. Basic evaluation with random checkpoint:
```bash
python src/ehr_classification/evaluate.py
```

2. Evaluate best checkpoint:
```bash
python src/ehr_classification/evaluate.py evaluation.mode=best
```

3. Evaluate specific checkpoint:
```bash
python src/ehr_classification/evaluate.py evaluation.checkpoint_path=/path/to/checkpoint.ckpt
```

4. Evaluate different split:
```bash
python src/ehr_classification/evaluate.py data.split_number=2
```

## Common Configuration Options

### Model Parameters
- `input_size`: Number of temporal features in the input data
- `hidden_size`: Size of hidden layers in the neural network
- `static_input_size`: Number of static features in the input data
- `num_classes`: Number of output classes (2 for binary classification)
- `dropout_rate`: Dropout probability for regularization
- `bidirectional`: Whether to use bidirectional LSTM

### Training Parameters
- `max_epochs`: Maximum number of training epochs
- `batch_size`: Number of samples per batch
- `learning_rate`: Learning rate for optimization
- `class_weights`: Weights for handling class imbalance
- `patience`: Number of epochs to wait before early stopping
- `use_gpu`: Whether to use GPU acceleration

### Evaluation Parameters
- `checkpoint_path`: Path to specific model checkpoint
- `mode`: Checkpoint selection mode ('random', 'best', 'last')
- `split_number`: Which data split to evaluate