from fastapi import FastAPI
# from main import read_root
from pydantic import BaseModel
from ultralytics import YOLO
import numpy as np
import os
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()

def get_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

class Data(BaseModel):
    imageArr: list[str] = []

@app.get("/")
def read_main():
    return {"message": "Welcome to the API"}

@app.post("/predict")
async def predict(data: Data):
    imageArr = data.imageArr
    namrArr = []

  
    modelPath = os.path.join("best.pt")
    model = YOLO(modelPath)

    for img_url in imageArr:
        img = get_image(img_url)

 
        results = model.predict(img, stream=False)

      
        names_dict = results[0].names if hasattr(results[0], 'names') else {}
        probs = results[0].probs.data.tolist() if hasattr(results[0], 'probs') else []

        if probs:  
            namrArr.append(names_dict[np.argmax(probs)])

  
    resultsName = max(set(namrArr), key=namrArr.count) if namrArr else "No prediction found"
    
    return {"prediction": resultsName}
