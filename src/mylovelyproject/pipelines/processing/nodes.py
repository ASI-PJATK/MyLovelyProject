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

def load_housing_data(housing):
    return housing


def explore_housing_data(housing):
    print(housing.head())
    print(housing.info())
    print(housing.ocean_proximity.value_counts())
    return housing


def stratify_data(housing):
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    # Create 'income_cat' column for stratification
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    return strat_train_set, strat_test_set


def add_features(housing):
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    return housing


# Define the custom transformer for additional attributes
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# The main function for data preparation
def prepare_data(strat_train_set):
    # Separate features and labels
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    # Fill missing values in 'total_bedrooms'
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median, inplace=True)

    # Numerical attributes pipeline
    num_attribs = list(housing.drop("ocean_proximity", axis=1))
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # Full pipeline for both numerical and categorical attributes
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    # Apply the full pipeline
    housing_prepared = full_pipeline.fit_transform(housing)

    return housing_prepared, housing_labels


def train_model(features, labels):
    model = LinearRegression()
    model.fit(features, labels)
    return model


def train_model_with_pycaret(features, labels, target_column_name='median_house_value'):
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
    predictions = model.predict(features)
    return predictions

def predictions_to_dataframe(predictions):
    df = pd.DataFrame(predictions, columns=['Predicted_Value'])
    return df
