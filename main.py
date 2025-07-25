from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import xgboost as xgb
import joblib
import json
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Pydantic models
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    Male = "male"
    Female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Load model and artifacts
model_path = 'data/model.json'
preprocessor_path = 'data/preprocessor.pkl'
species_mapping_path = 'data/species_mapping.json'

if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(preprocessor_path):
    logger.error(f"Preprocessor file not found at {preprocessor_path}")
    raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
if not os.path.exists(species_mapping_path):
    logger.error(f"Species mapping file not found at {species_mapping_path}")
    raise FileNotFoundError(f"Species mapping file not found at {species_mapping_path}")

logger.info("Loading model, preprocessor, and species mapping...")
model = xgb.Booster()
model.load_model(model_path)
preprocessor = joblib.load(preprocessor_path)
with open(species_mapping_path, 'r') as f:
    species_mapping = json.load(f)

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(features: PenguinFeatures):
    """Predict penguin species based on input features."""
    try:
        logger.info("Received prediction request")
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Apply preprocessing
        input_encoded = preprocessor.transform(input_data)
        
        # Convert to DMatrix and predict
        dmatrix = xgb.DMatrix(input_encoded)
        prediction = model.predict(dmatrix)
        predicted_class = int(prediction[0])
        
        # Map to species name
        species = species_mapping[str(predicted_class)]
        logger.info(f"Predicted species: {species}")
        
        return {"predicted_species": species}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))