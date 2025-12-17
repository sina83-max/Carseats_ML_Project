import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

from src.data_loader import load_carseats
from src.plot_utils import plot_tree_model, plot_pred_vs_actual
from src.preprocess import preprocess
from src.logger import get_logger

logger = get_logger()


def train_pruned_tree(
    data_path="../data/Carseats.csv",
    model_path="../models/pruned_tree_model.joblib",
):
    mlflow.set_experiment("carseats-sales-regression")

    with mlflow.start_run(run_name="pruned_tree"):

        # 1. Load data
        df = load_carseats(data_path)
        logger.info(f"Loaded dataset with shape {df.shape}")

        # 2. Preprocess
        df = preprocess(df)
        logger.info("Preprocessing complete")

        # 3. Split features and target
        X = df.drop("Sales", axis=1)
        y = df["Sales"]

        # 4. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 5. Cost-complexity pruning path
        base_tree = DecisionTreeRegressor(random_state=42)
        path = base_tree.cost_complexity_pruning_path(X_train, y_train)

        ccp_alphas = path.ccp_alphas[:-1]  # last alpha prunes everything
        logger.info(f"Evaluating {len(ccp_alphas)} alpha values")

        # 6. Cross-validation to select alpha
        cv_mse = []

        for alpha in ccp_alphas:
            tree = DecisionTreeRegressor(
                random_state=42,
                ccp_alpha=alpha,
            )

            scores = cross_val_score(
                tree,
                X_train,
                y_train,
                cv=5,
                scoring="neg_mean_squared_error",
            )

            cv_mse.append(-scores.mean())

        optimal_alpha = ccp_alphas[np.argmin(cv_mse)]
        logger.info(f"Optimal alpha: {optimal_alpha:.6f}")

        # 7. Train final pruned tree
        final_tree = DecisionTreeRegressor(
            random_state=42,
            ccp_alpha=optimal_alpha,
        )
        final_tree.fit(X_train, y_train)
        logger.info("Pruned tree trained")

        # 8. Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(final_tree, model_path)

        # 9. Evaluation
        y_pred = final_tree.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)

        logger.info(f"Pruned Tree Test MSE: {test_mse:.4f}")

        plot_tree_model(
            final_tree,
            feature_names=X.columns,
            save_path="../reports/figures/pruned_tree_plot.png"
        )
        plot_pred_vs_actual(
            y_test,
            y_pred,
            save_path="../reports/figures/pruned_tree_pred_vs_actual.png"
        )

        # 10. MLflow logging
        mlflow.log_param("model_type", "pruned_decision_tree")
        mlflow.log_param("ccp_alpha", optimal_alpha)
        mlflow.log_metric("test_mse", test_mse)

        mlflow.sklearn.log_model(final_tree, artifact_path="model")

        return {
            "optimal_alpha": optimal_alpha,
            "test_mse": test_mse,
        }


if __name__ == "__main__":
    results = train_pruned_tree()
    print("Pruned Tree Results")
    print(f"Optimal alpha: {results['optimal_alpha']:.6f}")
    print(f"Test MSE: {results['test_mse']:.4f}")
