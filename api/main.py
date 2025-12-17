# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.routes import prediction

app = FastAPI(
    title="Carseats Sales Prediction API",
    description="Predict Carseats sales using Regression Tree, Pruned Tree, or Bagging",
    version="1.0.0",
)

# Enable CORS (for local frontend or other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction API routes
app.include_router(prediction.router)

# Mount static files folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve single-page frontend at /predict
@app.get("/predict", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/")
def root():
    return {"message": "Carseats Sales Prediction API is running."}
