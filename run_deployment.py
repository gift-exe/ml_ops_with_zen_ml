from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline

from rich import print

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

import click
import typing

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'

@click.command
@click.option(
    '--config',
    '-c',
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help='You can also choose to run only the deployment '
    'pipeline to train and completly deploy model, or to '
    'only run a prediction against the deployed model '
    'By default, both would be run.',
)
@click.option(
    '--min-accuracy',
    default=0.92,
    help='Minimum accuracy requried to deploy the model',
)

def run_deployment(config:str, min_accuracy:float):
    mlflow_model_deployer_component_base = MLFlowModelDeployer.get_active_model_deployer()

    if isinstance(mlflow_model_deployer_component_base, MLFlowModelDeployer):
        #print('config: ', mlflow_model_deployer_component_base.config)
        mlflow_model_deployer_component = MLFlowModelDeployer(config=mlflow_model_deployer_component_base.config, **vars(mlflow_model_deployer_component_base))
        
    
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    if deploy:
        continuous_deployment_pipeline(min_accuracy=min_accuracy, workers=3, timeout=60)
    if predict:
        inference_pipeline(
            pipeline_name='continuous_deployment_pipeline',
            pipeline_step_name='mlflow_model_deployer_step',
        )
    
    print(
        'You can run: \n'
        f'[italic green]    mlflow ui --backend-store-uri {get_tracking_uri()} [/italic green]\n'
        '...to inspect your experiments runs within the MLflow UI.\n'
        'You can find your runs tracked with the `mlflow_example_pipeline` experiment.'
        'There you will also be able to compare two or more runs. \n\n'
    )
    
    existing_services = mlflow_model_deployer_component.find_model_server(pipeline_name='continous_deployment_pipeline',
                                                                          pipeline_step_name='mlflow_model_deployer_step',
                                                                          model_name='model')

    print('Existing services: ',existing_services)

    if existing_services:
        service = typing.cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(f'The MLflow prediction server is running locally as a daemon '
                  f'process service and accepts inference requests at: \n'
                  f'    {service.prediction_url}\n'
                  f'Top stop the service, run '
                  f'[italic green] `zenml model-deployer models delete {str(service.uuid)}`[/italic green].')
        elif service.is_failed:
            print(f'The MLflow prediction server is in a failed state: \n'
                  f'Last state: {service.status.last_state}\n'
                  f'Last error: {service.status.last_error}')
    else:
        print('No MLflow prediction server is currently running. '
              'The deployment pipeline must run first to train a model and deploy it. '
              'Execute the same command with the `--deploy` argument to deploy a model.')


if __name__ == '__main__':
    run_deployment()