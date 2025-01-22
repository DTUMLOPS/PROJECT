from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
from pathlib import Path
from typing import List

from ehr_classification.inference import InferenceEngine, InferenceInput
from ehr_classification.evaluate import find_checkpoint


class PredictionInput(BaseModel):
    temporal_data: List[List[float]]
    static_data: List[float]

class PredictionOutput(BaseModel):
    probabilities: List[List[float]]
    predicted_classes: List[int]
    interpretation: str

# Global inference engine
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        # Initialize inference engine
        engine = InferenceEngine(use_gpu=False)
        
        # Find best checkpoint
        checkpoint_path = find_checkpoint(
            Path("../../models"), 
            split_number=1,  # Use first split by default
            mode="best"
        )
        
        # Load model
        engine.load_model(checkpoint_path)
    except Exception as e:
        print(f"Error during startup: {e}")
        raise
    yield

app = FastAPI(title="EHR Classification API",lifespan=lifespan)

@app.get("/predict")
async def predict():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input data to numpy arrays
        temporal_data = np.random.rand(10, 37)  # [sequence_length, num_features]
        static_data = np.random.rand(8) 
        # Make prediction
        results = engine.predict(temporal_data, static_data)
        
        # Convert numpy arrays to lists for JSON serialization
        return PredictionOutput(
            probabilities=results.probabilities.tolist(),
            predicted_classes=results.predicted_classes.tolist(),
            interpretation=results.get_interpretation()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
