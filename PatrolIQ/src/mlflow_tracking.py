# src/mlflow_tracking.py

import mlflow


def start_experiment(experiment_name="PatrolIQ_Clustering"):
    mlflow.set_experiment(experiment_name)


def log_clustering_run(algorithm_name, params, metrics):
    with mlflow.start_run():
        mlflow.log_param("algorithm", algorithm_name)

        for key, value in params.items():
            mlflow.log_param(key, value)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)
