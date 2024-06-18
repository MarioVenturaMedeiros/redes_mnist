from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from tempfile import NamedTemporaryFile
import time

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('C:/Users/mario/OneDrive/Documentos/GitHub/redes_conv/src/backend/pesos.h5')

class LinearModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out

input_size = 784  # 28x28
num_classes = 10
model2 = LinearModel(input_size, num_classes)
model2.load_state_dict(torch.load('C:/Users/mario/OneDrive/Documentos/GitHub/redes_conv/src/backend/model.ckpt'))
model2.eval()

templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("ai.jsx", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Process the image and make prediction
        img = cv2.imread(temp_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / img.max()
        img = img.reshape(1, 28, 28, 1)

        start_time = time.time()
        prediction = model.predict(img)
        end_time = time.time()
        predicted_class = np.argmax(prediction)
        prediction_time = end_time - start_time

        # Clean up
        os.remove(temp_file_path)

        return JSONResponse(content={"prediction": int(predicted_class), "prediction_time": prediction_time})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_linear")
async def predict_linear(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Process the image and make prediction
        img = cv2.imread(temp_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img / img.max()
        img = img.reshape(1, 28*28)  # Flatten the image

        # Convert to PyTorch tensor
        img_tensor = torch.tensor(img, dtype=torch.float32)

        # Make prediction using PyTorch model
        start_time = time.time()
        with torch.no_grad():
            output = model2(img_tensor)
            _, predicted_class = torch.max(output.data, 1)
        end_time = time.time()
        prediction_time = end_time - start_time

        # Clean up
        os.remove(temp_file_path)

        print(f"prediction_time: {prediction_time}")

        return JSONResponse(content={"prediction": int(predicted_class.item()), "prediction_time": prediction_time})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, debug=True)