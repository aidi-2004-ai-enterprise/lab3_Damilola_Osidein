Lab 3: Penguin Species Classification

This repository contains a FastAPI application that predicts penguin species using an XGBoost model trained on the Seaborn penguins dataset.

Setup Instructions





Clone the repository:

git clone https://github.com/aidi-2004-ai-enterprise/lab3_Damilola_Osidein.git
cd lab3_Damilola_Osidein



Install uv:

pip install uv



Create and activate virtual environment:

uv venv





Windows (PowerShell): . .\.venv\Scripts\activate.ps1



Windows (CMD): .venv\Scripts\activate.bat



Unix: source .venv/bin/activate



Install dependencies:

uv pip install --link-mode=copy -r requirements.txt



Train the model:

python train.py



Run the FastAPI app:

uvicorn app.main:app --reload



Test the API: Open http://127.0.0.1:8000/docs and use the /predict endpoint.