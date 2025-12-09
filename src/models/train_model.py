import datetime
import os
import shutil
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pprint import pprint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import json
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import time
import mlflow.pyfunc

# ---------------------
#  GLOBAL ARTIFACT PATH
# ---------------------
ARTIFACT_DIR = "notebooks/artifacts"


# ---------------------
#  CONFIG FUNCTIONS
# ---------------------
def get_training_config():
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    data_gold_path = f"{ARTIFACT_DIR}/train_data_gold.csv"
    data_version = "00000"
    experiment_name = current_date

    return {
        "current_date": current_date,
        "data_gold_path": data_gold_path,
        "data_version": data_version,
        "experiment_name": experiment_name
    }


def setup_training_environment(experiment_name):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)

    mlflow.set_experiment(experiment_name)


# ---------------------
#  DATA LOADING
# ---------------------
def load_training_data(data_gold_path):
    data = pd.read_csv(data_gold_path)
    print(f"Training data length: {len(data)}")
    print(data.head(5))
    return data


# ---------------------
#  TRAIN / TEST SPLIT
# ---------------------
def split_train_test(data):
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)

    return train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )


# ---------------------
#  XGBOOST MODEL TRAINING
# ---------------------
def train_xgboost_model(X_train, y_train):
    model = XGBRFClassifier(random_state=42)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["binary:logistic"],
        "eval_metric": ["error"]
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


def evaluate_model(model_grid, X_train, y_train, X_test, y_test):
    best_params = model_grid.best_params_
    print("Best XGBoost params:")
    pprint(best_params)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    print("Train accuracy:", accuracy_score(y_pred_train, y_train))
    print("Test accuracy:", accuracy_score(y_pred_test, y_test))

    return best_params, y_pred_train, y_pred_test


# ---------------------
#  PERFORMANCE
# ---------------------
def performance_overview(y_train, y_pred_train, y_test, y_pred_test):
    print("TEST CONFUSION MATRIX\n")
    print(pd.crosstab(y_test, y_pred_test))

    print("\nTEST REPORT\n")
    print(classification_report(y_test, y_pred_test))

    print("\nTRAIN CONFUSION MATRIX\n")
    print(pd.crosstab(y_train, y_pred_train))

    print("\nTRAIN REPORT\n")
    print(classification_report(y_train, y_pred_train))

    return


# ---------------------
#  SAVE XGBOOST MODEL
# ---------------------
def save_best_xgboost_model(model_grid, y_train, y_pred_train):
    xgboost_model = model_grid.best_estimator_
    xgboost_path = f"{ARTIFACT_DIR}/lead_model_xgboost.json"

    xgboost_model.save_model(xgboost_path)

    model_results = {
        xgboost_path: classification_report(
            y_train, y_pred_train, output_dict=True
        )
    }

    return xgboost_path, model_results


# ---------------------
#  LOGISTIC REGRESSION
# ---------------------
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


def train_logistic_regression(
    experiment_name,
    X_train,
    y_train,
    X_test,
    y_test,
    model_results
):
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    with mlflow.start_run(experiment_id=experiment_id):
        model = LogisticRegression()
        lr_path = f"{ARTIFACT_DIR}/lead_model_lr.pkl"

        params = {
            "solver": ["lbfgs", "liblinear"],
            "C": [1.0, 0.1, 0.01]
        }

        model_grid = RandomizedSearchCV(
            model, param_distributions=params, verbose=3, n_iter=5, cv=3
        )
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        # Log metrics
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))

        # Save model
        joblib.dump(best_model, lr_path)

    model_results[lr_path] = classification_report(
        y_test, y_pred_test, output_dict=True
    )

    return (
        lr_path,
        model_grid.best_params_,
        model_results[lr_path],
        y_pred_train,
        y_pred_test,
        model_results
    )


# ---------------------
#  SAVE GENERAL ARTIFACTS
# ---------------------
def save_artifacts(X_train, model_results):
    columns_path = f"{ARTIFACT_DIR}/columns_list.json"
    results_path = f"{ARTIFACT_DIR}/model_results.json"

    with open(columns_path, "w+") as f:
        json.dump({"column_names": list(X_train.columns)}, f)

    with open(results_path, "w+") as f:
        json.dump(model_results, f)

    return columns_path, results_path


# ---------------------
#  MODEL SELECTION
# ---------------------
def select_best_model(experiment_name):
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    experiment_best = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]

    with open(f"{ARTIFACT_DIR}/model_results.json", "r") as f:
        model_results = json.load(f)

    results_df = pd.DataFrame({
        model_path: result["weighted avg"]
        for model_path, result in model_results.items()
    }).T

    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name

    print(f"Best model: {best_model}")
    return experiment_best, best_model, results_df

