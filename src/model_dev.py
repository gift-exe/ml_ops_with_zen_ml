import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin

class Model(ABC):
    '''
        Abstract class for all models
    '''

    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Linear Regression Model Training Complete')
            return reg
        except Exception as e:
            logging.error('Error training Linear Regression Model: {}'.format(e))
            raise e