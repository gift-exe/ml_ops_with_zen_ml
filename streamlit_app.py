import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment

def main():
    st.title("End to End customer statisfaction pipeline with ZenML")

    high_level_image = Image.open('_assets/high_level_overview.png')
    st.image(high_level_image, caption='High Level Overview of the Pipeline')

    whole_pipeline_image = Image.open('_assets/training_and_deployment_pipeline_updated.png')

    st.markdown(
        '''

            ###E Problem Statement
            The Objective here is to predict the customer satisfaction score for a given order based on features like:
            order status, price, payment, etc. 
            We will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the 
            customer satisfaction score for the next order or purchase.
        
        '''
    )

    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
            Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, 
            and evaluate the model, and if data source changes or any hyperparameter values changes, 
            deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy 
            requirement, the model will be deployed.
        """
    )

    st.markdown(
        """ 
            #### Description of Features 
            This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
            | Models        | Description   | 
            | ------------- | -     | 
            | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
            | Payment Installments   | Number of installments chosen by the customer. |  
            | Payment Value |       Total amount paid by the customer. | 
            | Price |       Price of the product. |
            | Freight Value |    Freight value of the product.  | 
            | Product Name length |    Length of the product name. |
            | Product Description length |    Length of the product description. |
            | Product photos Quantity |    Number of product published photos |
            | Product weight measured in grams |    Weight of the product measured in grams. | 
            | Product length (CMs) |    Length of the product measured in centimeters. |
            | Product height (CMs) |    Height of the product measured in centimeters. |
            | Product width (CMs) |    Width of the product measured in centimeters. |
        """
    )

    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("freight_value")
    product_name_length = st.number_input("Product name length")
    product_description_length = st.number_input("Product Description length")
    product_photos_qty = st.number_input("Product photos Quantity ")
    product_weight_g = st.number_input("Product weight measured in grams")
    product_length_cm = st.number_input("Product length (CMs)")
    product_height_cm = st.number_input("Product height (CMs)")
    product_width_cm = st.number_input("Product width (CMs)")

    if st.button("Predict"):
        service = prediction_service_loader(
                                            pipeline_name="continuous_deployment_pipeline",
                                            pipeline_step_name="mlflow_model_deployer_step",
                                            running=False,
                                            )
        if service is None:
            st.write(
                "No service could be found. The pipeline needs be run first to create a service. "
                "\n RUN: `python run_deployment.py --config deploy` first"
            )

        df = pd.DataFrame(
            {
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "price": [price],
                "freight_value": [freight_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(pred)
        )


if __name__ == "__main__":
    main()

