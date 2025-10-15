from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import logging
import os
from tensorflow.keras.models import load_model
import numpy as np
from fastapi import Body

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


model = load_model('D:\\Fashion_mnist\\model\\my_fashion_mnist_model.keras')


class PredictionRequest(BaseModel):
    instances: list

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Backend Server!"}


@app.post("/prediction")
def predict(data: PredictionRequest = Body(...)):
    try:
        instances = np.array(data.instances)
        predictions = model.predict(instances)[0]
        predicted_classes = np.argmax(predictions).tolist()
        print(f"Predicted classes: {predicted_classes}")
        return {"predictions": predicted_classes}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000)
