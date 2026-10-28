import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV

from src.features.feature_eng import build_preprocessor


def build_modelpipeline(model) -> Pipeline:
    """
    Build a pipeline that preprocesses the features and then fits the model.

    Args:
        model: Any scikit-learn compatible regressor (e.g. XGBRegressor, RandomForestRegressor).

    Returns:
        Pipeline: A scikit-learn Pipeline with a 'preprocessor' step and a 'model' step.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    """
    Evaluate a pipeline with K-fold cross-validation using RMSE.

    Args:
        pipeline (Pipeline): The pipeline to evaluate.
        X (pd.DataFrame): The features.
        y (pd.Series): The target.
        n_splits (int): The number of cross-validation folds.

    Returns:
        np.ndarray: The RMSE for each fold.
    """
    # shuffle with a fixed seed so every model is scored on the same folds
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # sklearn maximises scores, so it returns RMSE as a negative number; flip the sign back
    scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")

    print("RMSE for each fold:", scores.round(2))
    print(f"Mean RMSE: {scores.mean().round(2)} (+/- {scores.std().round(2)})")

    return scores


def tune(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, param_distributions: dict,
         n_iter: int = 30, n_splits: int = 5) -> Pipeline:
    """
    Tune the model's hyperparameters with randomized search and cross-validation.

    Args:
        pipeline (Pipeline): The pipeline to tune.
        X (pd.DataFrame): The features.
        y (pd.Series): The target.
        param_distributions (dict): The values to try, with keys prefixed by 'model__',
            e.g. {'model__max_depth': [3, 5, 7]}.
        n_iter (int): How many random combinations to try.
        n_splits (int): The number of cross-validation folds.

    Returns:
        Pipeline: A copy of the pipeline with the best hyperparameters set (not yet fitted).
    """
    # same folds as evaluate(), so tuned and untuned scores can be compared fairly
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # randomized search tries n_iter random combinations instead of every one, which is much faster
    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=n_iter, cv=cv,
                                scoring="neg_root_mean_squared_error", random_state=42,
                                n_jobs=-1, refit=False)
    search.fit(X, y)

    print("Best hyperparameters:", search.best_params_)
    print(f"Best CV RMSE: {-search.best_score_:.2f}")

    # return the pipeline with the best settings; fit it with train_and_save
    return pipeline.set_params(**search.best_params_)


def train_and_save(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, path: str) -> Pipeline:
    """
    Fit the pipeline on the full training data and save it to disk.

    Args:
        pipeline (Pipeline): The pipeline to fit.
        X (pd.DataFrame): The features.
        y (pd.Series): The target.
        path (str): Where to save the fitted pipeline, e.g. 'models/xgb_pipeline.joblib'.

    Returns:
        Pipeline: The fitted pipeline.
    """
    # fit on all rows: cross-validation was only for scoring, the final model should learn from everything
    pipeline.fit(X, y)

    # create the folder if it does not exist yet
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    # save the whole pipeline (preprocessor + model) so predictions use the exact same encoding
    joblib.dump(pipeline, path)
    print(f"Saved fitted pipeline to {path}")

    return pipeline
