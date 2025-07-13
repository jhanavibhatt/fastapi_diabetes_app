from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import json

# FastAPI app initialization
app = FastAPI()

# Allow all origins for CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for input data
class model_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load trained model
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))

# API endpoint
@app.post('/diabetes_prediction')
def diabetes_predd(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    input_list = [
        input_dictionary['Pregnancies'],
        input_dictionary['Glucose'],
        input_dictionary['BloodPressure'],
        input_dictionary['SkinThickness'],
        input_dictionary['Insulin'],
        input_dictionary['BMI'],
        input_dictionary['DiabetesPedigreeFunction'],
        input_dictionary['Age']
    ]

    prediction = diabetes_model.predict([input_list])

    if prediction[0] == 0:
        return {'result': 'The person is not diabetic'}
    else:
        return {'result': 'The person is diabetic'}
