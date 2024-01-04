import logging
import pandas as pd
from zenml import step
import numpy as np
from typing import Union, Tuple
from typing_extensions import Annotated

from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2, RMSE

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model:RegressorMixin, 
                   X_test: pd.DataFrame, 
                   y_test: pd.Series) -> Tuple[Annotated[float, 'mse'],
                                               Annotated[float, 'r2'],
                                               Annotated[float, 'rmse'],]:
    '''
        Evaluates the model on ingested data

        Args:
            df: The ingested data
    '''

    try:
        y_pred = model.predict(X_test)
        mse = MSE().calculate_scores(y_true=y_test, y_pred=y_pred)
        r2_score = R2().calculate_scores(y_true=y_test, y_pred=y_pred)
        rmse = RMSE().calculate_scores(y_true=y_test, y_pred=y_pred)

        mlflow.log_metric('mse', mse)
        mlflow.log_metric('r2_score', r2_score)
        mlflow.log_metric('rmse', rmse)

        return mse, r2_score, rmse
    
    except Exception as e:
        logging.error('Error evaluating model')
        raise e 


    