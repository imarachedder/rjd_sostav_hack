from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from typing import Dict

from utils.ml_functions import process_audio, predict_label
from models.model import Model

app = FastAPI(
    title="Audio Processing API",
    description="API for processing audio files and predicting labels.",
    version="1.0.0"
)

# Enable CORS with wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize your ML model (replace with your actual model)
model = Model()


@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        start_time = time.time()
        audio_bytes = await file.read()
        features = process_audio(audio_bytes)
        # Predict using the ML model
        label = predict_label(model, features)

        # Calculate runtime
        end_time = time.time()
        runtime = end_time - start_time

        # Prepare response
        response: Dict = {
            "label": label,
            "misc": "Additional information can go here.",
            "runtime_seconds": runtime
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

