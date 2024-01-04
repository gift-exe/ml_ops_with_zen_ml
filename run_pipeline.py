from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == '__main__':
    print(f'\nML Flow Dashboard link {Client().active_stack.experiment_tracker.get_tracking_uri()}\n')  
    train_pipeline(datapath='data/olist_customers_dataset.csv')