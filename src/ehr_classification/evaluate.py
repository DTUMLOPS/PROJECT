"""
Evaluation script for the DSSM model.
"""
import logging
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from src.ehr_classification.data import PhysionetDataModule
from src.ehr_classification.model import DSSMLightning

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="evaluate")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model with given configuration."""
    logger.info("Starting evaluation...")

    # Create data module
    datamodule = PhysionetDataModule(
        data_dir=cfg.data.base_dir,
        split_number=cfg.data.split_number,
        batch_size=cfg.training.batch_size
    )

    # Load trained model
    model = DSSMLightning.load_from_checkpoint(cfg.evaluation.checkpoint_path)

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu' if cfg.training.use_gpu else 'cpu',
        devices=1,
        deterministic=True
    )

    # Evaluate model
    results = trainer.test(model, datamodule=datamodule)

    logger.info("Evaluation completed!")
    logger.info(f"Test results: {results}")


if __name__ == "__main__":
    evaluate()
