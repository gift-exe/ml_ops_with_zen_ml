from zenml import pipeline
from zenml_steps.ingest_data import ingest_data
from zenml_steps.clean_data import clean_data
from zenml_steps.model_train import train_model
from zenml_steps.model_eval import evaluate_model


@pipeline
def train_pipeline(datapath: str):
    df = ingest_data(datapath)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)
    



