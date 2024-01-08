from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI, HTTPException
import joblib
import os
import pandas as pd

app = FastAPI()

# Load your trained model (adjust the file path as needed)
model = joblib.load("../../data/06_models/tuned_model.pkl")
models_dir = "../../data/06_models/"
current_model = "tuned_model"

# Define a Pydantic model for input data validation
from pydantic import BaseModel


class InputData(BaseModel):
    # Define your input data structure here
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float


def endpoint_prepare_data(data):
    # Define numerical and categorical columns
    num_attribs = ["longitude", "latitude", "housing_median_age",
                   "total_rooms", "total_bedrooms", "population",
                   "households", "median_income"]

    #  cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        #      ("cat", OneHotEncoder, cat_attribs),
    ])

    return full_pipeline.fit_transform(data)


@app.post("/predict")
async def get_prediction(input_data: InputData):
    # Convert input data to a DataFrame (or the required format)
    housing = pd.DataFrame([input_data.dict()])

    print(housing.values)
    prepared_housing = endpoint_prepare_data(housing)
    # print(prepared_housing)
    string = current_model + ".pkl"
    model = joblib.load("../../data/06_models/" + current_model + ".pkl")

    prediction = model.predict(prepared_housing)

    # Return prediction
    return {"Selected model:" + current_model + "\nprediction": prediction.tolist()}


@app.get("/models")
async def list_models():
    # List all models in the models directory
    models = os.listdir(models_dir)
    return {"models": models}


class ModelName(BaseModel):
    model_name: str


@app.post("/models/select")
async def select_model(model_name: ModelName):
    global current_model
    model_path = os.path.join(models_dir, model_name.model_name)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    current_model = joblib.load(model_path)
    return {"message": f"Model {model_name.model_name} selected"}
