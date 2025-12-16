import os

import joblib
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import load_carseats
from src.logger import get_logger
from src.preprocess import preprocess

logger = get_logger()

def train_bagging(
        data_path: str = "../data/Carseats.csv",
        model_path: str = "../models/bagging_model.joblib",
):
    # 1. Load data
    df = load_carseats(data_path)
    logger.info(f"Loaded dataset with shape {df.shape}")

    # 2. Preprocess
    df = preprocess(df)
    logger.info(f"Preprocessing complete")

    # 3. Split features and target
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 5. Define base Estimator
    base_tree = DecisionTreeRegressor(random_state=42)

    # 6. Bagging model
    bagging = BaggingRegressor(
        estimator=base_tree,
        n_estimators=100,
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )

    # 7. Train
    bagging.fit(X_train, y_train)
    logger.info("Bagging model trained")

    # 8. Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bagging, model_path)
    logger.info(f"Bagging model saved to {model_path}")

    # 9. Evaluate
    y_pred = bagging.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Bagging Test MSE: {test_mse:.4f}")

    return test_mse


if __name__ == "__main__":
    mse = train_bagging()
    print(f"Bagging Test MSE: {mse:.4f}")