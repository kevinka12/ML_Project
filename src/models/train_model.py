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



from sklearn.metrics import accuracy_score
from pprint import pprint

def evaluate_model(model_grid, X_train, y_train, X_test, y_test):
    best_model_xgboost_params = model_grid.best_params_
    print("Best xgboost params")
    pprint(best_model_xgboost_params)

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    print("Accuracy train", accuracy_score(y_pred_train, y_train))
    print("Accuracy test", accuracy_score(y_pred_test, y_test))

    return best_model_xgboost_params, y_pred_train, y_pred_test



from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def performance_overview(y_train, y_pred_train, y_test, y_pred_test):
    # Test confusion matrix
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test), '\n')

    # Train confusion matrix
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train), '\n')

    return conf_matrix_train, conf_matrix_test



from sklearn.metrics import classification_report

def save_best_xgboost_model(model_grid, y_train, y_pred_train):
    xgboost_model = model_grid.best_estimator_
    xgboost_model_path = "./artifacts/lead_model_xgboost.json"
    
    # Save model
    xgboost_model.save_model(xgboost_model_path)

    # Store classification report
    model_results = {
        xgboost_model_path: classification_report(
            y_train, y_pred_train, output_dict=True
        )
    }

    return xgboost_model_path, model_results

# src/models/train_model.py

import mlflow.pyfunc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix
)
import joblib
import pandas as pd
from pprint import pprint

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

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = LogisticRegression()
        lr_model_path = "./artifacts/lead_model_lr.pkl"

        params = {
            'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            'penalty': ["none", "l1", "l2", "elasticnet"],
            'C': [100, 10, 1.0, 0.1, 0.01]
        }

        model_grid = RandomizedSearchCV(
            model,
            param_distributions=params,
            verbose=3,
            n_iter=10,
            cv=3
        )
        model_grid.fit(X_train, y_train)

        best_model = model_grid.best_estimator_

        y_pred_train = model_grid.predict(X_train)
        y_pred_test = model_grid.predict(X_test)

        # MLflow logging
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")

        # Save LR model
        joblib.dump(value=best_model, filename=lr_model_path)

        # Log custom Python model
        mlflow.pyfunc.log_model(
            'model',
            python_model=lr_wrapper(best_model)
        )

    model_classification_report = classification_report(
        y_test,
        y_pred_test,
        output_dict=True
    )

    best_model_lr_params = model_grid.best_params_

    print("Best lr params")
    pprint(best_model_lr_params)

    print("Accuracy train:", accuracy_score(y_pred_train, y_train))
    print("Accuracy test:", accuracy_score(y_pred_test, y_test))

    # Test metrics
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    print("Test actual/predicted\n")
    print(pd.crosstab(y_test, y_pred_test,
          rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
    print("Classification report\n")
    print(classification_report(y_test, y_pred_test), '\n')

    # Train metrics
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    print("Train actual/predicted\n")
    print(pd.crosstab(y_train, y_pred_train,
          rownames=['Actual'], colnames=['Predicted'], margins=True), '\n')
    print("Classification report\n")
    print(classification_report(y_train, y_pred_train), '\n')

    model_results[lr_model_path] = model_classification_report
    print(model_classification_report["weighted avg"]["f1-score"])

    return (
        lr_model_path,
        best_model_lr_params,
        model_classification_report,
        y_pred_train,
        y_pred_test,
        model_results
    )


import json
from pprint import pprint

def save_artifacts(X_train, model_results):
    # Save column list
    column_list_path = './artifacts/columns_list.json'
    with open(column_list_path, 'w+') as columns_file:
        columns = {'column_names': list(X_train.columns)}
        pprint(columns)
        json.dump(columns, columns_file)

    print('Saved column list to ', column_list_path)

    # Save model results
    model_results_path = "./artifacts/model_results.json"
    with open(model_results_path, 'w+') as results_file:
        json.dump(model_results, results_file)

    return column_list_path, model_results_path


import datetime

def get_model_selection_config():
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    artifact_path = "model"
    model_name = "lead_model"
    experiment_name = current_date

    return {
        "current_date": current_date,
        "artifact_path": artifact_path,
        "model_name": model_name,
        "experiment_name": experiment_name
    }


import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

def wait_until_ready(model_name, model_version):
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)



import json
import mlflow
import pandas as pd

def select_best_model(experiment_name):
    # Get experiment id
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    
    # Search for best experiment run
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    
    # Load local model results
    with open("./artifacts/model_results.json", "r") as f:
        model_results = json.load(f)

    results_df = pd.DataFrame({
        model: val["weighted avg"] 
        for model, val in model_results.items()
    }).T

    best_model = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    print(f"Best model: {best_model}")

    return experiment_best, best_model, results_df



from mlflow.tracking import MlflowClient

def get_production_model(model_name):
    client = MlflowClient()
    
    # Find model versions in Production stage
    prod_model = [
        model
        for model in client.search_model_versions(f"name='{model_name}'")
        if dict(model)['current_stage'] == 'Production'
    ]
    
    prod_model_exists = len(prod_model) > 0
    
    if prod_model_exists:
        prod_model_version = dict(prod_model[0])['version']
        prod_model_run_id = dict(prod_model[0])['run_id']
        
        print('Production model name: ', model_name)
        print('Production model version:', prod_model_version)
        print('Production model run id:', prod_model_run_id)
        
        return {
            "model_name": model_name,
            "model_version": prod_model_version,
            "run_id": prod_model_run_id
        }
    
    else:
        print('No model in production')
        return None



import mlflow

def compare_models_and_select(
    experiment_best,
    prod_model_exists,
    prod_model_run_id
):
    train_model_score = experiment_best["metrics.f1_score"]
    model_status = {}
    run_id = None

    if prod_model_exists:
        data, details = mlflow.get_run(prod_model_run_id)
        prod_model_score = data[1]["metrics.f1_score"]

        model_status["current"] = train_model_score
        model_status["prod"] = prod_model_score

        if train_model_score > prod_model_score:
            print("Registering new model")
            run_id = experiment_best["run_id"]
    else:
        print("No model in production")
        run_id = experiment_best["run_id"]

    print(f"Registered model: {run_id}")

    return run_id, model_status


import mlflow

def register_best_model(run_id, artifact_path, model_name):
    if run_id is not None:
        print(f'Best model found: {run_id}')

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id,
            artifact_path=artifact_path
        )

        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        wait_until_ready(model_details.name, model_details.version)

        model_details = dict(model_details)
        print(model_details)

        return model_details
    
    return None



def get_deployment_config():
    model_version = 1
    return {
        "model_version": model_version
    }



import time
from mlflow.tracking import MlflowClient

client = MlflowClient()

def wait_for_deployment(model_name, model_version, stage='Staging'):
    status = False
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name, version=model_version)
        )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status


def transition_to_staging(model_name, model_version):
    model_version_details = dict(
        client.get_model_version(name=model_name, version=model_version)
    )
    model_status = True

    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Staging",
            archive_existing_versions=True
        )
        model_status = wait_for_deployment(model_name, model_version, 'Staging')
    else:
        print('Model already in staging')

    return model_status
