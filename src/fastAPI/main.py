import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

from fastapi import FastAPI
app = FastAPI()

@app.get("/models")
async def model():

    return {"Models used by us": ["trained_model"]}

@app.post("/predict")
async def prediction(model_name: str, longitude: float,latitude: float,housing_median_age: float,total_rooms: float,total_bedrooms: float,population: float,households: float,median_income: float,median_house_value: float,ocean_proximity: object,longitud: float,latitud: float,housing_median_ag: float,total_room: float,total_bedroom: float,populatio: float,household: float,median_incom: float,median_house_valu: float):
    
    with open('../../data/06_models/'+str(model_name) +'.pkl', 'rb') as file:
        model = pickle.load(file)
        
    features = [longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity,longitud,latitud,housing_median_ag,total_room,total_bedroom,populatio,household,median_incom,median_house_valu]
    
    features = np.asarray(features,dtype=np.float64)
    
    features = features.reshape(1, -1)
    
    prediction = model.predict(features)
    
    with open('outputs/'+ str(model_name) +'_result_of_prediction.csv', 'wb') as output_file:
        output_file.write(features)
        output_file.write(prediction)
    
    return {"This is the prediction:": str(prediction)}