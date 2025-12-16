import pandas as pd

def load_carseats(path: str = "data/carseats.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df