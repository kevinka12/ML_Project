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



from sklearn.model_selection import train_test_split

def split_train_test(data):
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )

    return X_train, X_test, y_train, y_test



from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def train_xgboost_model(X_train, y_train):
    model = XGBRFClassifier(random_state=42)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"]
    }

    model_grid = RandomizedSearchCV(
        model, 
        param_distributions=params, 
        n_jobs=-1, 
        verbose=3, 
        n_iter=10, 
        cv=10
    )

    model_grid.fit(X_train, y_train)

    return model_grid
