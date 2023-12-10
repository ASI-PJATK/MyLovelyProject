"""
This is a boilerplate pipeline 'processing'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_housing_data,
                inputs="housing",
                outputs="raw_housing_data",
                name="load_housing_data_node"
            ),
            node(
                func=explore_housing_data,
                inputs="raw_housing_data",
                outputs="explored_housing_data",
                name="explore_housing_data_node"
            ),
            node(
                func=stratify_data,
                inputs="explored_housing_data",
                outputs=["strat_train_set", "strat_test_set"],
                name="stratify_data_node"
            ),
            node(
                func=add_features,
                inputs="strat_train_set",
                outputs="feature_added_housing",
                name="add_features_node"
            ),
            node(
                func=prepare_data,
                inputs="feature_added_housing",
                outputs=["housing_prepared", "housing_labels"],
                name="prepare_data_node"
            ),

            node(
                func=train_model,
                inputs=["housing_prepared", "housing_labels"],
                outputs="trained_model",
                name="train_model_node"
            ),
            node(
                func=predict,
                inputs=["trained_model", "housing_prepared"],  # predicting on the training set
                outputs="predictions",
                name="predict_node"
            ),
            node(
                func=predictions_to_dataframe,
                inputs="predictions",
                outputs="predictions_df",
                name="convert_predictions_node"
        ),
        ])
