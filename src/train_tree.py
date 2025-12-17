import os

import joblib
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.data_loader import load_carseats
from src.logger import get_logger
from src.plot_utils import plot_tree_model, plot_pred_vs_actual
from src.preprocess import preprocess
from src.tracking import ExperimentTracker

logger = get_logger()


def train_regression_tree(
        data_path: str = "../data/Carseats.csv",
        model_path: str = "../models/tree_model.joblib"
):
    # Initialize Experiment Tracker
    tracker = ExperimentTracker("carseats-sales-regression")

    # Load Data
    df = load_carseats(data_path)
    logger.info(f"Loaded dataset with the shape {df.shape}")

    # Preprocess
    df = preprocess(df)
    logger.info(f"Preprocessing complete")

    # Split X and Ys
    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    # Train and test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    logger.info(f"Training set: {x_train.shape}, test set: {x_test.shape}")

    with tracker.start_run(run_name="regression_tree"):
        # Initialize and train the tree
        tree = DecisionTreeRegressor(random_state=42)
        tree.fit(x_train, y_train)
        logger.info("Regression tree trained")

        # Evaluate
        y_pred = tree.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Test MSE: {mse:.4f}")

        # Log params
        tracker.log_params(
            {
                "model": "decision_tree",
                "random_state": 42,
            }
        )

        # Log metrics
        tracker.log_metrics(
            {
                "test_mse": mse
            }
        )

        # Save model locally
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(tree, model_path)
        logger.info(f"Model saved to {model_path}")

        # Log model to ML flow
        tracker.log_model(tree, artifact_path="model")

        # Plot tree
        plot_tree_model(
            tree,
            feature_names=X.columns,
            save_path="../reports/figures/tree_plot.png"
        )

        # Predicted vs Actual
        plot_pred_vs_actual(
            y_test,
            y_pred,
            save_path="../reports/figures/tree_pred_vs_actual.png"
        )

    return mse


if __name__ == "__main__":
    mse = train_regression_tree()
    print(f"Regression Tree Test MSE: {mse:.4f}")