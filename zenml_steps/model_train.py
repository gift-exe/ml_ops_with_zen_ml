import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelConfig
import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train:pd.DataFrame, 
                X_test:pd.DataFrame, 
                y_train:pd.Series, 
                y_test:pd.Series,
                config: ModelConfig) -> RegressorMixin:
    try:
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train=X_train, y_train=y_train)
            
            logging.info('Model Training Complete')
            
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
        
    except Exception as e:
        logging.error('Error Training Model')
        raise e