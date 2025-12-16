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

    cat_cols = ["ShelveLoc", "Urban", "US"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df