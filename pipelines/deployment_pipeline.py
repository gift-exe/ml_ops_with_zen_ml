import numpy as np
import pandas as pd

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step

from zenml_steps.clean_data import clean_data
from zenml_steps.model_eval import evaluate_model
from zenml_steps.ingest_data import ingest_data
from zenml_steps.model_train import train_model


docker_setting = DockerSettings(required_integrations=[MLFLOW])

@pipeline(name='continuous_deployment_pipeline', enable_cache=False, settings={'docker':docker_setting})
def continuous_deployment_pipeline(min_accuracy:float = 0.92,
                                   workers: int = 1,
                                   timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    df = ingest_data(datapath='./data/olist_customers_dataset.csv')
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = evaluate_model(model, X_test, y_test)
    ml_service = mlflow_model_deployer_step(model=model,
                               deploy_decision = True,
                               workers=workers,
                               timeout=timeout,
                               )

@step(enable_cache=False)
def prediction_service_loader(pipeline_name:str,
                              pipeline_step_name:str,
                              running:bool = True,
                              model_name:str = 'model') -> MLFlowDeploymentService:
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(f'No MLflow deployment service found for pipeline "{pipeline_name}",'
                           f'step "{pipeline_step_name}", and model "{model_name}".'
                           f'Pipeline for the "{model_name}" model is currently running.')

    
    return  existing_services[0]

from .utils import get_data_for_test
import json

@step(enable_cache=False)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step
def predictor(service:MLFlowDeploymentService,
              data: str,) -> np.ndarray:
    
    service.start(timeout=10)
    data = json.loads(data)
    data.pop('columns')
    data.pop('index')
    columns_for_df = [
        'payment_sequential',
        'payment_installments',
        'payment_value',
        'price',
        'freight_value',
        'product_name_lenght',
        'product_description_length',
        'product_photos_qty',
        'products_weight_g',
        'products_length_cm',
        'product_height_cm',
        'product_width_cm',
    ]
    df = pd.DataFrame(data['data'], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(name='inference_pipeline', enable_cache=False, settings={'docker': docker_setting})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(pipeline_name=pipeline_name,
                                        pipeline_step_name=pipeline_step_name,
                                        running=False)
    prediction = predictor(service=service, data=data)
    return prediction