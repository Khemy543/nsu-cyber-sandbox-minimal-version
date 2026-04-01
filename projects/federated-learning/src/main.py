import boto3
import pandas as pd
from pathlib import Path

from data_preprocessing import load_and_preprocess_data
from experiments import run_federated_experiments, run_scalability_experiment, run_accuracy_experiments

s3 = boto3.client(
    "s3",
    endpoint_url="http://storage:9000",
    aws_access_key_id="admin",
    aws_secret_access_key="password123",
)

FILE_PATH = Path("/app/data/federated-learning/liver_cirrhosis.csv")
FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

s3.download_file(
    "datasets",
    "liver_cirrhosis.csv",
    str(FILE_PATH)
)


def main():
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(str(FILE_PATH))
    run_federated_experiments(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_runs=3, num_clients=20)
    run_scalability_experiment(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, client_counts=range(2, 21, 2))
    run_accuracy_experiments(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, str(FILE_PATH))

if __name__ == "__main__":
    main()
