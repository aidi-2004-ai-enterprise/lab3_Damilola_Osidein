# lab3_Damilola_Osidein

# Lab 3: Penguins Classification with XGBoost and FastAPI

## Overview
This project implements a machine learning pipeline to classify penguin species using the Seaborn penguins dataset, an XGBoost model, and a FastAPI application for predictions. The pipeline includes data preprocessing, model training, and a prediction endpoint with input validation and logging.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aidi-2004-ai-enterprise/lab3_Damilola_Osidein.git
   cd lab3_Damilola_Osidein
   ```

2. **Install `uv`**:
   ```bash
   pip install uv
   ```

3. **Create Virtual Environment and Install Dependencies**:
   ```bash
   uv venv --python 3.10
   .\.venv\Scripts\activate  # On Windows
   uv pip install -r pyproject.toml --link-mode=copy
   ```

4. **Train the Model**:
   ```bash
   python train.py
   ```
   This generates the trained model (`app/data/model.json`) and encoders (`app/data/encoders.pkl`).

5. **Run the FastAPI Application**:
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Access the API**:
   - Open `http://127.0.0.1:8000/docs` in a browser to test the `/predict` endpoint.
   - Example request body:
     ```json
     {
       "bill_length_mm": 39.1,
       "bill_depth_mm": 18.7,
       "flipper_length_mm": 181.0,
       "body_mass_g": 3750.0,
       "sex": "male",
       "island": "Torgersen"
     }
     ```

## Demo Video
[Watch the demo video](./demo.mp4) for examples of successful and failed API requests. The video demonstrates:
- Successful requests with valid inputs.
- Failed requests with invalid `sex` or `island` values, showing graceful error handling.

https://drive.google.com/file/d/1AOQuLrsfpZ4ac_AhHvws-TOi_oXI_Lb-/view?usp=drive_link 

## Notes
- The model is trained on the Seaborn penguins dataset with one-hot encoding for `sex` and `island`, and label encoding for `species`.
- The FastAPI application uses Pydantic for input validation, ensuring only valid `sex` and `island` values are accepted.
- Logging is implemented to track model loading, predictions, and errors, with logs saved to `app/penguins_app.log` and displayed in the console.
- The code includes docstrings and type hints for readability and maintainability.

- The application handles invalid inputs gracefully, returning HTTP 400 errors with clear messages.
