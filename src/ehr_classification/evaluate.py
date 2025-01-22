"""
Evaluation script for the DSSM model.
"""

import random
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from ehr_classification.data import PhysionetDataModule
from ehr_classification.model import DSSMLightning

logger = logging.getLogger(__name__)


def find_checkpoint(model_dir: Path, split_number: int, mode: str = "random") -> str:
    """Find a checkpoint for evaluation.

    Args:
        model_dir: Directory containing model checkpoints
        split_number: Split number to evaluate
        mode: How to select checkpoint - 'random', 'best', or 'last'

    Returns:
        str: Path to selected checkpoint
    """
    split_dir = model_dir / f"split_{split_number}"
    if not split_dir.exists():
        raise ValueError(f"Split directory {split_dir} does not exist")

    checkpoints = list(split_dir.glob(f"split_{split_number}-epoch=*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {split_dir}")

    if mode == "random":
        selected_checkpoint = random.choice(checkpoints)
        logger.info(f"Randomly selected checkpoint: {selected_checkpoint}")
    elif mode == "best":
        selected_checkpoint = min(checkpoints, key=lambda x: float(str(x).split("val_loss=")[-1].replace(".ckpt", "")))
        logger.info(f"Selected best checkpoint (lowest val_loss): {selected_checkpoint}")
    elif mode == "last":
        selected_checkpoint = max(checkpoints, key=lambda x: int(str(x).split("epoch=")[1].split("-")[0]))
        logger.info(f"Selected last checkpoint: {selected_checkpoint}")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'random', 'best', or 'last'")

    return str(selected_checkpoint)


@hydra.main(config_path="../../configs", config_name="evaluate", version_base="1.1")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model with given configuration."""
    logger.info("Starting evaluation...")

    logger.info("\nConfiguration:")
    logger.info("-" * 40)
    logger.info(OmegaConf.to_yaml(cfg))

    # Determine checkpoint path
    try:
        if cfg.evaluation.checkpoint_path and "${paths.output_dir}" not in cfg.evaluation.checkpoint_path:
            # Use explicitly provided checkpoint
            checkpoint_path = cfg.evaluation.checkpoint_path
            logger.info(f"\nUsing provided checkpoint: {checkpoint_path}")
        else:
            # Select checkpoint based on mode
            checkpoint_path = find_checkpoint(
                Path(cfg.paths.model_dir),
                cfg.data.split_number,
                mode=cfg.evaluation.get("mode", "random"),  # Default to random if not specified
            )
    except ValueError as e:
        logger.error(f"Error finding checkpoint: {e}")
        return

    # Create data module and load model
    try:
        datamodule = PhysionetDataModule(
            data_dir=cfg.data.base_dir, split_number=cfg.data.split_number, batch_size=cfg.training.batch_size
        )

        model = DSSMLightning.load_from_checkpoint(checkpoint_path)
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Error setting up evaluation: {e}")
        return

    trainer = pl.Trainer(accelerator="gpu" if cfg.training.use_gpu else "cpu", devices=1, deterministic=True)

    try:
        results = trainer.test(model, datamodule=datamodule)

        logger.info("\nEvaluation Results:")
        logger.info("-" * 40)
        for metric_name, value in results[0].items():
            logger.info(f"{metric_name:15s}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return

    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate()
