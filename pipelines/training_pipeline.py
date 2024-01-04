from zenml import pipeline
from zenml_steps.ingest_data import ingest_data
from zenml_steps.clean_data import clean_data
from zenml_steps.model_train import train_model
from zenml_steps.model_eval import evaluate_model


@pipeline
def train_pipeline(datapath: str):
    df = ingest_data(datapath)
    df = clean_data(df)
    train_model(df)
    evaluate_model(df)



