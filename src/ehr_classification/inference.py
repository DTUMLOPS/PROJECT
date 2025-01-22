"""
interface.py

Core inference module for the DSSM model. 
This file includes utilities for loading models, validating input data, and performing predictions. 
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, List
from dataclasses import dataclass

import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from ehr_classification.model import DSSMLightning
from ehr_classification.evaluate import find_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class InferenceInput:
    """
    Data class for validating and storing inference input data.
    
    Attributes:
        temporal_data (np.ndarray): Time-series data for inference.
        static_data (np.ndarray): Static features for inference.
    """
    temporal_data: np.ndarray
    static_data: np.ndarray

    def validate(self) -> bool:
        """Validate input data dimensions and types."""
        try:
            # Check temporal data
            if not isinstance(self.temporal_data, np.ndarray):
                raise ValueError("temporal_data must be a numpy array")

            # Check static data
            if not isinstance(self.static_data, np.ndarray):
                raise ValueError("static_data must be a numpy array")

            # Validate dimensions for single sample
            if len(self.temporal_data.shape) == 2:
                if len(self.static_data.shape) != 1:
                    raise ValueError("For single sample, static_data must be 1D")

            # Validate dimensions for batch
            elif len(self.temporal_data.shape) == 3:
                if len(self.static_data.shape) != 2:
                    raise ValueError("For batch input, static_data must be 2D")
                if self.temporal_data.shape[0] != self.static_data.shape[0]:
                    raise ValueError("Batch sizes must match between temporal and static data")
            else:
                raise ValueError("temporal_data must be 2D (single sample) or 3D (batch)")

            return True
        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise


@dataclass
class InferenceOutput:
    """
    Data class for structured inference output.
    
    Attributes:
        probabilities (np.ndarray): Class probabilities for predictions.
        predicted_classes (np.ndarray): Predicted class labels.
    """
    probabilities: np.ndarray
    predicted_classes: np.ndarray

    def get_interpretation(self, confidence_threshold: float = 0.7) -> str:
        """
        Generate a human-readable interpretation of the prediction.

        Args:
            confidence_threshold (float): Threshold for low-confidence warnings.

        Returns:
            str: Interpretation of the prediction.
        """
        prob = self.probabilities[0]
        pred_class = self.predicted_classes[0]
        confidence = prob[pred_class]

        # Define class labels
        class_labels = {0: "negative", 1: "positive"}

        # Generate confidence level description
        if confidence >= 0.9:
            confidence_level = "very high"
        elif confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "moderate"
        else:
            confidence_level = "low"

        # Create interpretation message
        prediction = class_labels[pred_class]
        message = f"The model predicts a {prediction} outcome with {confidence_level} confidence ({confidence:.1%})"

        # Add uncertainty warning if confidence is low
        if confidence < confidence_threshold:
            message += ". Given the low confidence level, this prediction should be interpreted with caution"

        return message

    def to_dict(self) -> Dict:
        """
        Convert the output to a dictionary format.

        Returns:
            Dict: Output in dictionary format.
        """
        return {
            "probabilities": self.probabilities.tolist(),
            "predicted_classes": self.predicted_classes.tolist(),
            "interpretation": self.get_interpretation(),
        }


class ModelManager:
    """
    Handles model loading and management.
    """
    def __init__(self):
        self._models: Dict[str, DSSMLightning] = {}

    def load_model(self, model_path: str, model_id: str = "default") -> None:
        """
        Load a model from a checkpoint and cache it.

        Args:
            model_path (str): Path to the model checkpoint.
            model_id (str): Identifier for the model.
        """
        try:
            model = DSSMLightning.load_from_checkpoint(model_path)
            model.eval()
            self._models[model_id] = model
            logger.info(f"Model {model_id} loaded successfully from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def get_model(self, model_id: str = "default") -> Optional[DSSMLightning]:
        """
        Get a loaded model by ID.

        Args:
            model_id (str): Identifier for the model.

        Returns:
            Optional[DSSMLightning]: The loaded model or None if not found.
        """
        return self._models.get(model_id)

    def unload_model(self, model_id: str = "default") -> None:
        """
        Unload a model from memory.

        Args:
            model_id (str): Identifier for the model to unload.
        """
        if model_id in self._models:
            del self._models[model_id]
            logger.info(f"Model {model_id} unloaded")


class InferenceEngine:
    """
    Inference engine for processing input data and generating predictions.
    """
    def __init__(self, use_gpu: bool = False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_manager = ModelManager()

    def load_model(self, model_path: str, model_id: str = "default") -> None:
        """
        Load a model for inference.
        """
        self.model_manager.load_model(model_path, model_id)
        model = self.model_manager.get_model(model_id)
        if model:
            model.to(self.device)

    def _preprocess(self, inference_input: InferenceInput) -> tuple:
        """
        Preprocess input data.
        """
        inference_input.validate()

        temporal_data = inference_input.temporal_data
        static_data = inference_input.static_data

        # Add batch dimension if needed
        if len(temporal_data.shape) == 2:
            temporal_data = temporal_data[np.newaxis, ...]
            static_data = static_data[np.newaxis, ...]

        # Convert to tensors
        temporal_tensor = torch.tensor(temporal_data, dtype=torch.float32, device=self.device)
        static_tensor = torch.tensor(static_data, dtype=torch.float32, device=self.device)
        seq_lengths = torch.tensor([temporal_data.shape[1]] * temporal_data.shape[0], device=self.device)

        return temporal_tensor, static_tensor, seq_lengths

    @torch.no_grad()
    def predict(
        self, temporal_data: Union[np.ndarray, List], static_data: Union[np.ndarray, List], model_id: str = "default"
    ) -> InferenceOutput:
        """
        Perform predictions using the model.

        Args:
            temporal_data: Time series data [sequence_length, num_features] or [batch_size, sequence_length, num_features]
            static_data: Static features [num_static_features] or [batch_size, num_static_features]
            model_id: ID of the model to use for inference

        Returns:
            InferenceOutput: containing probabilities and predicted classes
        """
        # Convert lists to numpy arrays if necessary
        if isinstance(temporal_data, list):
            temporal_data = np.array(temporal_data)
        if isinstance(static_data, list):
            static_data = np.array(static_data)

        # Prepare input
        inference_input = InferenceInput(temporal_data, static_data)
        temporal_tensor, static_tensor, seq_lengths = self._preprocess(inference_input)

        # Get model
        model = self.model_manager.get_model(model_id)
        if model is None:
            raise RuntimeError(f"Model {model_id} not loaded")

        # Get predictions
        outputs = model(temporal_tensor, static_tensor, seq_lengths)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(outputs, dim=1)

        return InferenceOutput(
            probabilities=probabilities.cpu().numpy(), predicted_classes=predicted_classes.cpu().numpy()
        )


@hydra.main(config_path="../../configs", config_name="evaluate", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for testing the inference engine.
    """    
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Find checkpoint path
    if cfg.evaluation.checkpoint_path:
        checkpoint_path = cfg.evaluation.checkpoint_path
    else:
        checkpoint_path = find_checkpoint(
            Path(cfg.paths.model_dir),
            cfg.data.split_number,
            mode=cfg.evaluation.get("mode", "random"),
        )

    # Initialize inference engine
    engine = InferenceEngine(use_gpu=cfg.training.use_gpu)
    engine.load_model(checkpoint_path)

    # Example data
    temporal_data = np.random.rand(10, 37)  # [sequence_length, num_features]
    static_data = np.random.rand(8)  # [num_static_features]

    # Make predictions
    try:
        results = engine.predict(temporal_data, static_data)
        logger.info("\nPrediction Results:")
        logger.info(f"Predicted Classes: {results.predicted_classes}")
        logger.info(f"Class Probabilities:\n{results.probabilities}")
        logger.info("\nInterpretation:")
        logger.info(results.get_interpretation())
    except Exception as e:
        logger.error(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
