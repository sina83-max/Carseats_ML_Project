# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api.routes import prediction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="Carseats Sales Prediction API",
    description="Predict Carseats sales using Regression Tree, Pruned Tree, or Bagging",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(prediction.router, prefix="/api")  # all API routes under /api

# Serve SPA static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# SPA entry point
@app.get("/", include_in_schema=False)
@app.get("/predict", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
