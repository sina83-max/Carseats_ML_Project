import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Encode categorical columns using LabelEncoder
# (converts each category to an integer)

# LabelEncoder transforms non-numeric categorical values into integer codes,
# so machine learning models can use them.
# It assigns a unique integer to each category,
# but it may introduce an artificial order between categories,
# so it's mainly suitable for tree-based models or target encoding,
# not linear/distance-based models.
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode ordinal categorical
    df["ShelveLoc"] = df["ShelveLoc"].map({"Bad": 0, "Medium": 1, "Good": 2})

    # Encode binary columns
    df["Urban"] = df["Urban"].map({"Yes": 1, "No": 0})
    df["US"] = df["US"].map({"Yes": 1, "No": 0})

    return df
