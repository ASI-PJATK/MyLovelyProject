import csv
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

@app.get("/models")
async def models():
    return {"models": ["data_raw_model","data_trained_model","data_raw_model_2","data_trained_model_2","data_raw_model_3","data_trained_model_3"]}

@app.post("/predict")
async def predict(model_name: str , longitude: float, latitude: float, housing_median_age: float, total_rooms: float, total_bedrooms: float, population: float, households: float, median_income: float, median_house_value: float, ocean_proximity: object):
    file_path='Task1/data/06_models/'+ str(model_name) +'.pkl'
    with open(file_path, 'rb') as file:
        model = pickle.load(file)

    features = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity]
    features = np.asarray(features)
    features = features.reshape(1, -1)
    prediction = model.predict(features)

    with open('outputs/'+ str(model_name) +'predictions.csv', 'wb') as output_file:
        output_file.write(features)
        output_file.write(prediction)
    return {"prediction": str(prediction)}

@app.post("/train")
async def train(model_name: str , longitude: float, latitude: float, housing_median_age: float, total_rooms: float, total_bedrooms: float, population: float, households: float, median_income: float, median_house_value: float, ocean_proximity: object):
    file_path = 'Task1/data/06_models/' + str(model_name) + '.pkl'
    with open(file_path, 'rb') as file:
        model = pickle.load(file)

    features = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value, ocean_proximity]])
    features = np.asarray(features)
    features = features.reshape(1, -1)
    model.fit(features, [SMPEP2])

    test_features = np.array([[-122.23, 37.88, 41.0, 880.0, 129.0, 322.00, 126.0, 8.3252, 452600.0, "NEAR BY"]])
    test_features = test_features.reshape(1, -1)
    
    test_SMPEP2 = np.asarray([53.47])
    test_SMPEP2 = test_SMPEP2.reshape(1, -1)
    
    prediction = model.predict(test_features)
    
    MeanAbsolError = metrics.mean_absolute_error(test_SMPEP2, prediction)
    MeanSquaredError = metrics.mean_squared_error(test_SMPEP2, prediction)
    MaxError = metrics.max_error(test_SMPEP2, prediction)

    with open('outputs/'+ str(model_name) +'trained_output.csv', 'wb') as output_file:
        output_file.write(MeanAbsolError)
        output_file.write(MeanSquaredError)
        output_file.write(MaxError)

    return {"Mean Absolute Error": MeanAbsolError,
            "Mean Squared Error": MeanSquaredError,
            "Max Error": MaxError}
