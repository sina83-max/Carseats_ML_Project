import mlflow
import mlflow.sklearn

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        for k, v in params.items():
            mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

    def log_model(self, model, artifact_path: str):
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)