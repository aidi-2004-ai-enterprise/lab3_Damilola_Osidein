import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import pickle
import os
from typing import Tuple, Dict

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Load the penguins dataset and preprocess it with one-hot encoding and label encoding.
    
    Returns:
        X (pd.DataFrame): Preprocessed features.
        y (pd.Series): Encoded target variable.
        encoders (dict): Dictionary containing label encoder and feature columns.
    """
    # Load dataset
    df = sns.load_dataset('penguins')
    df = df.dropna()  # Remove rows with missing values
    
    # Features and target
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island']]
    y = df['species']
    
    # Label encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X, columns=['sex', 'island'], prefix=['sex', 'island'])
    
    # Store encoders and feature columns for API consistency
    encoders = {
        'label_encoder': label_encoder,
        'feature_columns': X_encoded.columns.tolist()
    }
    
    return X_encoded, y_encoded, encoders

def train_and_evaluate_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """
    Train and evaluate an XGBoost classifier.
    
    Args:
        X (pd.DataFrame): Preprocessed features.
        y (pd.Series): Encoded target variable.
    
    Returns:
        model (xgb.XGBClassifier): Trained model.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    print(f"Training F1-score: {train_f1:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    
    return model

def save_model_and_encoders(model: xgb.XGBClassifier, encoders: Dict, model_path: str, encoder_path: str) -> None:
    """
    Save the trained model and encoders to disk.
    
    Args:
        model (xgb.XGBClassifier): Trained model.
        encoders (dict): Dictionary containing encoders and feature columns.
        model_path (str): Path to save the model.
        encoder_path (str): Path to save the encoders.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)

def main() -> None:
    """Main function to run the training pipeline."""
    X, y, encoders = load_and_preprocess_data()
    model = train_and_evaluate_model(X, y)
    save_model_and_encoders(model, encoders, 'app/data/model.json', 'app/data/encoders.pkl')

if __name__ == "__main__":
    main()