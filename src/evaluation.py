import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    '''
        Abstract class defining strategy for evaluating models
    '''

    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray) -> Union[float, np.ndarray]:
        pass

class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        try:
            logging.info('Calculating MSE')
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f'MSE: {mse}')
            return mse
        
        except Exception as e:
            logging.error(f'Error Calculating MSE score : {e}')
            raise e

class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        try:
            logging.info('Calculating R2 score')
            r2 = r2_score(y_pred=y_pred, y_true=y_true)
            logging.info(f'R2 Score: {r2}')
            return r2
        
        except Exception as e:
            logging.error(f'Error Calculating R2 score : {e}')
            raise e


class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        try:
            logging.info('Calculating RMSE')
            rmse = mean_squared_error(y_pred=y_pred, y_true=y_true, squared=True)
            logging.info(f'RMSE: {rmse}')
            return rmse
        
        except Exception as e:
            logging.error(f'Error Calculating RMSE score : {e}')
            raise e

