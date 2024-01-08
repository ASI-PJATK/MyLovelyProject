from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from fastapi import FastAPI
import joblib

app = FastAPI()

# Load your trained model (adjust the file path as needed)
model = joblib.load("/Users/ahmetduzduran/Projects/PJATK/7thSemester/ASI/project/MyLovelyProject/data/06_models/tuned_model.pkl")

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


import pandas as pd


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
    prediction = model.predict(prepared_housing)

    # Return prediction
    return {"prediction": prediction.tolist()}
