import os

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

from src.data_loader import load_carseats
from src.logger import get_logger
from src.preprocess import preprocess

logger = get_logger()

def train_regression_tree(
        data_path: str = "../data/Carseats.csv",
        model_path: str = "../models/tree_model.joblib"
):
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

    # Initialize and train the tree
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(x_train, y_train)
    logger.info("Regression tree trained")

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(tree, model_path)
    logger.info(f"Model saved to {model_path}")

    # Evaluate
    y_pred = tree.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Test MSE: {mse:.4f}")

    # Plot tree
    plt.figure(figsize=(20, 15))
    plot_tree(tree, feature_names=X.columns, filled=True, fontsize=10)
    plt.tight_layout()
    os.makedirs("../reports/figures", exist_ok=True)
    tree_plot_path = "../reports/figures/full_tree.png"
    plt.savefig(tree_plot_path)
    logger.info(f"Tree Plot saved to {tree_plot_path}")
    plt.close()

    return mse


if __name__ == "__main__":
    mse = train_regression_tree()
    print(f"Regression Tree Test MSE: {mse:.4f}")