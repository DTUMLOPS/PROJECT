"""
Training script for the DSSM model with cross-validation support.
"""

import logging
from pathlib import Path
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from ehr_classification.data import PhysionetDataModule
from ehr_classification.model import DSSMLightning

logger = logging.getLogger(__name__)


def train_single_split(cfg: DictConfig, split_number: int) -> dict:
    """Train model on a single data split."""
    logger.info(f"Training on split {split_number}")

    # Update split-specific paths
    split_checkpoint_dir = Path(cfg.training.checkpoint_dir) / f"split_{split_number}"
    split_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create data module for this split
    datamodule = PhysionetDataModule(
        data_dir=cfg.data.base_dir, split_number=split_number, batch_size=cfg.training.batch_size
    )

    # Initialize model
    model = DSSMLightning(
        **cfg.model, learning_rate=cfg.training.learning_rate, class_weights=cfg.training.class_weights
    )

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=split_checkpoint_dir,
            filename=f"split_{split_number}-{{epoch}}-{{val_loss:.2f}}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=cfg.training.patience, mode="min"),
    ]

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.training.use_gpu else "cpu",
        devices=1,
        callbacks=callbacks,
        deterministic=True,
    )

    # Train and test
    trainer.fit(model, datamodule)
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best")

    return test_results[0]


def print_results(results: dict, split_number: int = None):
    """Print formatted results with all metrics."""
    if split_number is not None:
        logger.info(f"\nResults for split {split_number}:")
        logger.info("=" * 50)
    else:
        logger.info("\nFinal Results:")
        logger.info("=" * 50)

    # Format each metric
    metrics = [("Loss", "test_loss"), ("Accuracy", "test_acc"), ("AUROC", "test_auroc"), ("AUPRC", "test_auprc")]

    # Calculate padding for alignment
    max_name_length = max(len(name) for name, _ in metrics)

    for name, key in metrics:
        if key in results:
            value = results[key]
            logger.info(f"{name:<{max_name_length}}: {value:.4f}")


def aggregate_metrics(all_results: list) -> dict:
    """Calculate mean and std of all metrics across splits."""
    metrics_dict = {}

    # Get all unique metric names
    metric_names = set()
    for result in all_results:
        metric_names.update(result.keys())

    # Calculate statistics for each metric
    for metric in metric_names:
        values = [result[metric] for result in all_results if metric in result]
        if values:
            metrics_dict[f"mean_{metric}"] = float(np.mean(values))
            metrics_dict[f"std_{metric}"] = float(np.std(values))

    return metrics_dict


@hydra.main(config_path="../../configs", config_name="train", version_base="1.1")
def train(cfg: DictConfig) -> None:
    """Train the model with cross-validation."""
    logger.info("Starting cross-validation training...")
    logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}")

    # Determine splits to use
    if cfg.data.splits:
        splits = cfg.data.splits
    else:
        splits = list(range(1, 6))  # Default: use all 5 splits

    logger.info(f"Training on splits: {splits}")

    # Train on each split and collect results
    all_results = []
    for split_number in splits:
        split_results = train_single_split(cfg, split_number)
        all_results.append(split_results)

    for split_number, split_results in enumerate(all_results, 1):
        print_results(split_results, split_number)

    aggregate_results = aggregate_metrics(all_results)

    logger.info("\nFinal Cross-Validation Results:")
    logger.info("Mean ± Std:")
    logger.info("-" * 50)

    # Print all metrics with their statistics
    metrics = ["test_loss", "test_acc", "test_auroc", "test_auprc"]
    for metric in metrics:
        if f"mean_{metric}" in aggregate_results:
            mean_value = aggregate_results[f"mean_{metric}"]
            std_value = aggregate_results[f"std_{metric}"]
            logger.info(f"{metric:15s}: {mean_value:.4f} ± {std_value:.4f}")

    logger.info("Training completed!")


if __name__ == "__main__":
    train()
