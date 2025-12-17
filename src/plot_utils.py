import os
import matplotlib.pyplot as plt
import numpy as np
import mlflow
from sklearn.tree import plot_tree

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_tree_model(model, feature_names, save_path="../reports/figures/tree_plot.png", log_mlflow=True):
    """
    Plot a DecisionTreeRegressor model and save figure
    """
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(20, 15))
    plot_tree(model, feature_names=feature_names, filled=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if log_mlflow:
        mlflow.log_artifact(save_path, artifact_path="figures")


def plot_pred_vs_actual(y_true, y_pred, save_path="../reports/figures/pred_vs_actual.png", log_mlflow=True):
    """
    Scatter plot: predicted vs actual
    """
    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if log_mlflow:
        mlflow.log_artifact(save_path, artifact_path="figures")


def plot_feature_importance(model, feature_names, save_path="../reports/figures/feature_importance.png", log_mlflow=True):
    """
    Plot feature importance for tree-based models (Bagging / Tree / Pruned Tree)
    """
    importances = None

    # Bagging models store feature_importances_ in base estimator average
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Model does not have feature_importances_ attribute.")

    ensure_dir(os.path.dirname(save_path))
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    if log_mlflow:
        mlflow.log_artifact(save_path, artifact_path="figures")
