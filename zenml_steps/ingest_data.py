import logging
import pandas as pd
from zenml import step 

class IngestData:
    '''
        Ingesting data from datapath
    '''
    def __init__(self, datapath:str):
        '''
            Args:
                datapath: path to the data
        '''
        self.datapath = datapath

    def get_data(self):
        '''
            Returns:
                pd.Dataframe object of the data
        '''
        logging.info(f'Ingesting data from {self.datapath}')
        return pd.read_csv(self.datapath)

@step
def ingest_data(datapath:str) -> pd.DataFrame:
    '''
        Ingesting (Loading on) data from datapath

        Args:
            datapath: Path to data
        Returns:
            pd.Dataframe: The ingested data
    '''
    try:
        return IngestData(datapath=datapath).get_data()
    except Exception as e:
        logging.error(f'Error while ingesting the data')
        raise e


