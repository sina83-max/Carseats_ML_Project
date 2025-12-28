import os
import joblib
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import randint

from src.data_loader import load_carseats
from src.logger import get_logger
from src.plot_utils import plot_tree_model, plot_pred_vs_actual
from src.preprocess import preprocess
from src.tracking import ExperimentTracker

logger = get_logger()


def train_regression_tree(
        data_path: str = "../data/Carseats.csv",
        model_path: str = "../models/tree_model.joblib",
        n_iter_search: int = 50
):
    tracker = ExperimentTracker("carseats-sales-regression")

    # Load and preprocess data
    df = load_carseats(data_path)
    df = preprocess(df)
    logger.info(f"Loaded and preprocessed dataset: {df.shape}")

    X = df.drop("Sales", axis=1)
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    logger.info(f"Train/Test split: {X_train.shape}, {X_test.shape}")

    with tracker.start_run(run_name="regression_tree"):
        base_tree = DecisionTreeRegressor(random_state=42)

        # Randomized hyperparameter search (continuous ranges)
        param_dist = {
            'max_depth': randint(3, 20),           # integer between 3 and 20
            'min_samples_split': randint(2, 50),   # integer between 2 and 50
            'min_samples_leaf': randint(1, 32),    # integer between 1 and 32
        }

        random_search = RandomizedSearchCV(
            estimator=base_tree,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring='neg_mean_squared_error',
            cv=5,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        random_search.fit(X_train, y_train)

        tree = random_search.best_estimator_
        logger.info(f"Best hyperparameters: {random_search.best_params_}")

        y_pred = tree.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = tree.score(X_test, y_test)
        logger.info(f"Test MSE: {mse:.4f}, Test R²: {r2*100:.2f}%")

        tracker.log_params(random_search.best_params_)
        tracker.log_metrics({"test_mse": mse, "test_r2": r2})

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(tree, model_path)
        logger.info(f"Model saved to {model_path}")

        tracker.log_model(tree, artifact_path="model")

        plot_tree_model(
            tree,
            feature_names=X.columns,
            save_path="../reports/figures/tree_plot.png"
        )

        plot_pred_vs_actual(
            y_test,
            y_pred,
            save_path="../reports/figures/tree_pred_vs_actual.png"
        )

    return mse, r2, random_search.best_params_


if __name__ == "__main__":
    mse, r2, best_params = train_regression_tree()
    print(f"Regression Tree Test MSE: {mse:.4f}")
    print(f"Regression Tree Test R²: {r2*100:.2f}%")
    print(f"Best hyperparameters: {best_params}")
