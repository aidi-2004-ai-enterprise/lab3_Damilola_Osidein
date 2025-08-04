from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import pandas as pd
import xgboost as xgb
import pickle
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app/penguins_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

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
    sex: Sex
    island: Island

# Load model and encoders
try:
    model = xgb.XGBClassifier()
    model.load_model('app/data/model.json')
    with open('app/data/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    logger.info("Model and encoders loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or encoders: {str(e)}")
    raise

@app.post("/predict")
async def predict(features: PenguinFeatures) -> Dict[str, Any]:
    """
    Predict penguin species based on input features.
    
    Args:
        features (PenguinFeatures): Input features for prediction.
    
    Returns:
        dict: Predicted species and confidence score.
    
    Raises:
        HTTPException: If input validation or prediction fails.
    """
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Apply one-hot encoding consistently with training
        input_encoded = pd.get_dummies(input_data, columns=['sex', 'island'], prefix=['sex', 'island'])
        
        # Ensure all expected columns are present
        for col in encoders['feature_columns']:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training
        input_encoded = input_encoded[encoders['feature_columns']]
        
        # Predict
        prediction = model.predict(input_encoded)[0]
        probabilities = model.predict_proba(input_encoded)[0]
        
        # Decode prediction
        predicted_species = encoders['label_encoder'].inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        logger.info(f"Prediction successful: species={predicted_species}, confidence={confidence:.4f}")
        return {"species": predicted_species, "confidence": confidence}
    
    except Exception as e:
        logger.debug(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")