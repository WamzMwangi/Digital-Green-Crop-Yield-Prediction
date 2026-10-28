import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.data_cleaning import data_cleaning
from src.features.feature_eng import eng_features


def load_model(path: str) -> Pipeline:
    """
    Load a fitted pipeline saved by train_and_save.

    Args:
        path (str): The path to the saved pipeline.

    Returns:
        Pipeline: The fitted pipeline.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found {path}")

    return joblib.load(path)


def predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Predict yield, clipping negative predictions to zero.

    Args:
        pipeline (Pipeline): The fitted pipeline.
        X (pd.DataFrame): The cleaned and engineered features.

    Returns:
        np.ndarray: The predicted yields.
    """
    preds = pipeline.predict(X)

    # yield cannot be negative, but the model does not know that
    return np.clip(preds, 0, None)


def predict_raw(pipeline: Pipeline, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict yield for raw farm records, applying the same cleaning and features as training.

    Args:
        pipeline (Pipeline): The fitted pipeline.
        raw_df (pd.DataFrame): Raw records in the same format as Train.csv, without Yield.

    Returns:
        pd.DataFrame: One row per record with the ID and the predicted yield.
    """
    # keep the IDs before cleaning drops the ID column
    ids = raw_df['ID']

    # new data goes through exactly the same steps as the training data
    X = eng_features(data_cleaning(raw_df))

    return pd.DataFrame({'ID': ids, 'PredictedYield': predict(pipeline, X)})
