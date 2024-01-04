import logging
from typing_extensions import Annotated
from typing import Tuple

import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessingStrategy, DataDivideStrategy


@step
def clean_data(df:pd.DataFrame) -> Tuple[
                                            Annotated[pd.DataFrame, 'X_train'],
                                            Annotated[pd.DataFrame, 'X_test'],
                                            Annotated[pd.Series, 'y_train'],
                                            Annotated[pd.Series, 'y_train'],
                                        ]:
    '''
        Cleans and divides data into train and test sets

        Args: 
            df: Loaded dataset
        
        Returns:
            X_train: Training data
            X_test: Testing data
            y_train: Training labels
            y_test: Testing labels
    '''
    try:
        #process data
        data_cleaning = DataCleaning(df, DataPreProcessingStrategy())
        processed_data = data_cleaning.handle_data()

        #split data
        data_cleaning = DataCleaning(processed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        logging.info('Data Cleaning Completed')
        
    except Exception as e:
        logging.error('Error cleaning data: {}'.format(e))
        raise e
