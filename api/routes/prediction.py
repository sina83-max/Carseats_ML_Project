from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas.request import PredictionRequest
from src.predict import predict

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/tree")
def predict_tree(request: PredictionRequest):
    try:
        X = pd.DataFrame([request.dict()])
        y_pred = predict("tree", X)
        return {"prediction": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/pruned_tree")
def predict_pruned_tree(request: PredictionRequest):
    try:
        X = pd.DataFrame([request.dict()])
        y_pred = predict("pruned_tree", X)
        return {"prediction": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/bagging")
def predict_bagging(request: PredictionRequest):
    try:
        X = pd.DataFrame([request.model_dump()])
        y_pred = predict("bagging", X)
        return {"prediction": y_pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
