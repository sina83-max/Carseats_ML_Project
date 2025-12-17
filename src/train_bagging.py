import os
import joblib

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

from src.data_loader import load_carseats
from src.plot_utils import plot_pred_vs_actual, plot_feature_importance
from src.preprocess import preprocess
from src.logger import get_logger

logger = get_logger()


def train_bagging(
    data_path: str = "../data/Carseats.csv",
    model_path: str = "../models/bagging_model.joblib",
):
    mlflow.set_experiment("carseats-sales-regression")

    with mlflow.start_run(run_name="bagging_tree"):

        # 1. Load data
        df = load_carseats(data_path)
        logger.info(f"Loaded dataset with shape {df.shape}")

        # 2. Preprocess
        df = preprocess(df)
        logger.info("Preprocessing complete")

        # 3. Split features and target
        X = df.drop("Sales", axis=1)
        y = df["Sales"]

        # 4. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 5. Base estimator
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

        plot_pred_vs_actual(
            y_test,
            y_pred,
            save_path="../reports/figures/bagging_pred_vs_actual.png"
        )
        plot_feature_importance(
            bagging,
            feature_names=X.columns,
            save_path="../reports/figures/bagging_feature_importance.png"
        )

        # 10. MLflow logging
        mlflow.log_param("model_type", "bagging_regressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("bootstrap", True)
        mlflow.log_metric("test_mse", test_mse)

        mlflow.sklearn.log_model(bagging, artifact_path="model")

        return test_mse


if __name__ == "__main__":
    mse = train_bagging()
    print(f"Bagging Test MSE: {mse:.4f}")
