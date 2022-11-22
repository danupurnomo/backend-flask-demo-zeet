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

import requests

URL = "http://127.0.0.1:5001/predict"
# URL = "http://44.208.85.154:8501/predict"
# URL = 'https://huggingface.co/spaces/danupurnomo/test-flask-backend2/predict'
r = requests.post(URL, json=new_data)
print(r, dir(r), dir(r.json))
print(r.status_code, r.text, r.raw)

# res = r.json()
# print(res)