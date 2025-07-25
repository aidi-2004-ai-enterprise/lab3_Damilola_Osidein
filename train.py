import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
import joblib
import json
import os

def train_model():
    """Load data, preprocess, train XGBoost model, evaluate, and save artifacts."""
    # Load the penguins dataset
    penguins = sns.load_dataset('penguins')
    penguins = penguins.dropna()  # Drop rows with missing values

    # Separate features and target
    X = penguins.drop('species', axis=1)
    y = penguins['species']

    # Define categorical features for one-hot encoding
    categorical_features = ['sex', 'island']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop=None, sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Preprocess features
    X_encoded = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_encoded = pd.DataFrame(X_encoded, columns=feature_names)

    # Label encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Train XGBoost model
    model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    f1_train = f1_score(y_train, y_train_pred, average='macro')
    f1_test = f1_score(y_test, y_test_pred, average='macro')
    print(f"F1-score on training set: {f1_train:.4f}")
    print(f"F1-score on test set: {f1_test:.4f}")

    # Save artifacts
    os.makedirs('app/data', exist_ok=True)
    model.save_model('app/data/model.json')
    joblib.dump(preprocessor, 'app/data/preprocessor.pkl')
    species_mapping = {str(i): label for i, label in enumerate(le.classes_)}
    with open('app/data/species_mapping.json', 'w') as f:
        json.dump(species_mapping, f)

if __name__ == "__main__":
    train_model()