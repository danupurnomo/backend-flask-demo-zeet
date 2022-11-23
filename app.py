from flask import Flask, jsonify, request
import pickle
import pandas as pd
import joblib
import numpy as np

# App Initialization
app = Flask(__name__)

# Load Pipeline Model
with open('final_pipeline.pkl', 'rb') as file_1:
  model_pipeline = joblib.load(file_1)

# Load Sequential Model
from tensorflow.keras.models import load_model
model_seq = load_model('titanic_model.h5')

@app.route("/")
def home():
    return "<h1>It Works!</h1>"

@app.route("/predict", methods=['POST'])
def titanic_predict():
    args = request.json

    new_data = {
      'PassengerId': args.get('PassengerId'),
      'Pclass': args.get('Pclass'), 
      'Name': args.get('Name'), 
      'Sex': args.get('Sex'), 
      'Age': args.get('Age'), 
      'SibSp': args.get('SibSp'),
      'Parch':  args.get('Parch'), 
      'Ticket': args.get('Ticket'), 
      'Fare': args.get('Fare'), 
      'Cabin': args.get('Cabin'), 
      'Embarked': args.get('Embarked')
    }

    new_data = pd.DataFrame([new_data])
    print('New Data : ', new_data)

    # Transform Inference-Set
    new_data_transform = model_pipeline.transform(new_data)
    new_data_transform

    # Predict using Neural Network
    y_pred_inf_single = model_seq.predict(new_data_transform)
    y_pred_inf_single = np.where(y_pred_inf_single >= 0.5, 1, 0)[0][0]
    if y_pred_inf_single == 0:
      label = 'Not Survived'
    else:
        label = 'Survived'
    print('result : ', y_pred_inf_single, label)

    # idx, label = titanic_inference(new_data, model_titanic)
    # response = jsonify(result=str(idx), label_names=label)
    response = jsonify(
      result = str(y_pred_inf_single), 
      label_names = label)
    
    print('response : ', response)

    return response

# jika deploy ke heroku, komen baris dibawah
# app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)