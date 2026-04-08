from fastapi import FastAPI
import numpy as np
import pickle

# Initialize app
app = FastAPI()

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))

# Home route
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: list):
    try:
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]

        return {
            "prediction": int(prediction[0]),
            "probability": float(probability)
        }
    except Exception as e:
        return {"error": str(e)}
