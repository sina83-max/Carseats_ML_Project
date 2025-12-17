from pydantic import BaseModel

class PredictionRequest(BaseModel):
    CompPrice: float
    Income: float
    Advertising: float
    Population: float
    Price: float
    ShelveLoc: int   # Assuming categorical encoded as int
    Age: float
    Education: float
    Urban: int       # 0 or 1
    US: int          # 0 or 1
