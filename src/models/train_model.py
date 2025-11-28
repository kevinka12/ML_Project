import datetime

def get_training_config():
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    data_gold_path = "./artifacts/train_data_gold.csv"
    data_version = "00000"
    experiment_name = current_date

    return {
        "current_date": current_date,
        "data_gold_path": data_gold_path,
        "data_version": data_version,
        "experiment_name": experiment_name
    }



import os
import shutil
import mlflow

def setup_training_environment(experiment_name):
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)

    mlflow.set_experiment(experiment_name)



import pandas as pd

def load_training_data(data_gold_path):
    data = pd.read_csv(data_gold_path)
    print(f"Training data length: {len(data)}")
    print(data.head(5))
    return data
