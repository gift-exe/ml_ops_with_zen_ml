from zenml.steps import BaseParameters

class ModelConfig(BaseParameters):
    model_name: str = 'LinearRegression'
    
