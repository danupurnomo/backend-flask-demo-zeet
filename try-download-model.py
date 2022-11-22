# This file is intended to test download model from HuggingFace Hub

# Import Libraries
from huggingface_hub import hf_hub_url, cached_download, hf_hub_download
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Create A New data
new_data = {
    'PassengerId': 1191,
    'Pclass': 1, 
    'Name': 'Sherlock Holmes', 
    'Sex': 'male', 
    'Age': 30, 
    'SibSp': 0,
    'Parch': 0, 
    'Ticket': 'C.A.29395', 
    'Fare': 12, 
    'Cabin': 'F44', 
    'Embarked': 'S'
}
new_data = pd.DataFrame([new_data])

schema = 2
if schema == 1:
    # Setting HuggingFace Repo and Model Names
    REPO_ID = 'danupurnomo/dummy-titanic'
    PIPELINE_FILENAME = 'final_pipeline.pkl'
    TF_FILENAME = 'titanic_model.h5'

    # Download the Models
    ## Download Model Pipeline
    model_pipeline = joblib.load(cached_download(
        hf_hub_url(REPO_ID, PIPELINE_FILENAME)
    ))

    ## Download TensorFlow Model
    model_seq = load_model(cached_download(
        hf_hub_url(REPO_ID, TF_FILENAME)
    ))

    # Transform Inference-Set
    new_data_transform = model_pipeline.transform(new_data)

    # Predict using Neural Networks
    y_pred_inf_single = model_seq.predict(new_data_transform)
    y_pred_inf_single = np.where(y_pred_inf_single >= 0.5, 1, 0)
    print('Result : ', y_pred_inf_single)

elif schema == 2: 
    # Download the Model
    model = joblib.load(
        hf_hub_download("danupurnomo/dummy-titanic", "sklearn_model.joblib")
        )

    # Transform Inference-Set
    new_data_transform = model.predict(new_data)
    print(new_data_transform)