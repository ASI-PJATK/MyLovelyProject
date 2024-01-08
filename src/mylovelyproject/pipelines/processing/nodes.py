"""
This is a boilerplate pipeline 'processing'
generated using Kedro 0.18.14
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
from pycaret.regression import setup, compare_models, predict_model, tune_model, save_model, load_model
import pickle


def load_housing_data(housing):
    return housing


def explore_housing_data(housing):
    print(housing.head())
    print(housing.info())
    print(housing.ocean_proximity.value_counts())
    return housing


# def add_features(housing):
#     housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
#     housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
#     housing["population_per_household"] = housing["population"] / housing["households"]
#
#     return housing


# def add_extra_features(X, add_bedrooms_per_room=True):
#     rooms_per_household = X["total_rooms"] / X["households"]
#     population_per_household = X["population"] / X["households"]
#     if add_bedrooms_per_room:
#         bedrooms_per_room = X["total_bedrooms"] / X["total_rooms"]
#         return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
#     else:
#         return np.c_[X, rooms_per_household, population_per_household]


def prepare_data(data,  test_size=0.2, random_state=42):
    # Separate the target variable
    features = data.drop("median_house_value", axis=1)
    features = data.drop("ocean_proximity", axis=1)

    labels = data["median_house_value"].copy()

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                        random_state=random_state)

    # features_with_extra_features = add_extra_features(features, add_bedrooms_per_room)

    # Define numerical and categorical columns
    num_attribs = ["longitude", "latitude", "housing_median_age",
                   "total_rooms", "total_bedrooms", "population",
                   "households", "median_income"]

    # if add_bedrooms_per_room:
    #     num_attribs.append("bedrooms_per_room")
#    cat_attribs = ["ocean_proximity"]

    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
#        ("cat", one_hot_encode_ocean_proximity(), cat_attribs),
    ])


    # Apply transformations to training and test sets
    X_train_prepared = full_pipeline.fit_transform(X_train)
    X_test_prepared = full_pipeline.transform(X_test)

    return X_train_prepared, y_train, X_test_prepared, y_test



def train_model_with_pycaret(features, labels, target_column_name='median_house_value'):
    print(features)
    # Combine features and labels into one DataFrame
    df = pd.DataFrame(features)
    df[target_column_name] = labels

    # Drop rows where the target variable is missing
    df.dropna(subset=[target_column_name], inplace=True)

    # Use PyCaret's setup function
    session_id = 123
    setup(data=df, target=target_column_name, session_id=session_id, verbose=False)
    best_model = compare_models()
    return best_model


def optimize_model_hyperparameters(model, n_trials=10):
    tuned_model = tune_model(model, n_iter=n_trials, custom_grid=None)
    return tuned_model


def predict_pycaret(model, features):
    # Convert features to DataFrame if it's a NumPy array
    if isinstance(features, np.ndarray):
        features = pd.DataFrame(features)

    predictions = predict_model(model, data=features)
    return predictions


def predict(model, features):
    print("FEATURES: ")
    print(features)
    predictions = model.predict(features)
    return predictions


def predictions_to_dataframe(predictions):
    df = pd.DataFrame(predictions, columns=['Predicted_Value'])
    return df


def features_to_dataframe(features):
    df = pd.DataFrame(features)
    return df


def train_model(features, labels):
    model = LinearRegression()
    model.fit(features, labels)
    return model
