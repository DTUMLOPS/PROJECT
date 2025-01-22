"""
This file provides the API interface for deploying the machine learning model to the cloud.
It handles data processing, model inference, and communication with cloud services,
enabling remote operation.
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wandb

from ehr_classification.inference import InferenceEngine

logger = logging.getLogger(__name__)


class PredictionOutput(BaseModel):
    probabilities: List[List[float]]
    predicted_classes: List[int]
    interpretation: str


# Global inference engine
engine: Optional[InferenceEngine] = None


def get_wandb_token() -> str:
    """Get Wandb token from environment or secret manager."""
    try:
        from ehr_classification.utils.secret_manager import get_wandb_token as get_cloud_token

        return get_cloud_token()
    except Exception as e:
        logger.error(f"Could not get Wandb token from secret manager: {e}")
        raise ValueError("Could not get token from secret manager")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        # Initialize wandb
        wandb.login(key=get_wandb_token())

        # Initialize wandb run
        wandb.init(
            project="dtumlops",
            entity="alexcomas",
            job_type="api-inference",
        )

        # Use the artifact API to download the latest model
        artifact = wandb.use_artifact("ehr_classification:latest", type="model")
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / "model.ckpt"

        # Initialize inference engine
        engine = InferenceEngine(use_gpu=False)
        engine.load_model(str(model_path))

        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    yield
    # Cleanup
    if engine:
        engine.model_manager.unload_model()
    # Finish wandb run
    wandb.finish()


app = FastAPI(title="EHR Classification API", lifespan=lifespan)


@app.get("/predict")
async def predict():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate display data
        temporal_data = np.random.rand(10, 37)  # [sequence_length, num_features]
        static_data = np.random.rand(8)  # [num_static_features]

        # Make prediction
        results = engine.predict(temporal_data, static_data)

        # Convert numpy arrays to lists for JSON serialization
        return PredictionOutput(
            probabilities=results.probabilities.tolist(),
            predicted_classes=results.predicted_classes.tolist(),
            interpretation=results.get_interpretation(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API and model status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}
