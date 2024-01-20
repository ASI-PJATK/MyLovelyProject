from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from joblib import load

preprocessing_pipeline = load('../../data/05_model_input/preprocessing_pipeline.joblib')

app = FastAPI()

# Load your trained model once when the application starts (adjust the file path as needed)
model_path = "../../data/06_models/best_model.pkl"
model = joblib.load(model_path)
current_model_name = ""

models_dir = "../../data/06_models/"


# Define the InputData model
class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float


@app.post("/predict")
def get_prediction(input_data: InputData):
    try:
        feature_names = ["longitude", "latitude", "housing_median_age", "total_rooms",
                         "total_bedrooms", "population", "households", "median_income"]

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data.dict()], columns=feature_names)

        # Process the data using the loaded preprocessing pipeline
        prepared_data = preprocessing_pipeline.transform(input_df)

        # Make a prediction
        prediction = model.predict(prepared_data)

        # Return the prediction in proper JSON format
        return {"Selected model": current_model_name, "Prediction": prediction.tolist()}
    except Exception as e:
        # Detailed logging for debugging
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


import os
from typing import List


@app.get("/models", response_model=List[str])
def list_models():
    models = [file for file in os.listdir(models_dir) if file.endswith('.pkl') and file != '.gitkeep']
    return models


@app.post("/select_model")
def select_model(model_name: str):
    global model, current_model_name
    try:
        if model_name in os.listdir(models_dir) and model_name.endswith('.pkl'):
            model_path = os.path.join(models_dir, model_name)
            model = joblib.load(model_path)
            current_model_name = model_name  # Update the current model name
            return {"message": f"Model changed to {model_name}"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    except Exception as e:
        print(f"Error during model selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))