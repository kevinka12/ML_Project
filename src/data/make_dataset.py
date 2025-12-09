# -*- coding: utf-8 -*-
import os
import pandas as pd
import datetime
import json

from src.features.build_features import describe_numeric_col, impute_missing_values

# Same notebook variables
max_date = "2024-01-31"
min_date = "2024-01-01"

ARTIFACT_DIR = "notebooks/artifacts"


def create_artifact_directory():
    """
    Creates notebooks/artifacts/ directory if it does not exist.
    """
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print("Created artifacts directory")


def load_data():
    print("Loading training data")

    data = pd.read_csv(f"{ARTIFACT_DIR}/raw_data.csv")

    print("Total rows:", data.count())
    print(data.head(5))

    return data


def clean_data(data):
    global max_date, min_date

    if not max_date:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()

    # Time limit conversion
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    # Save new min/max
    min_date = data["date_part"].min()
    max_date = data["date_part"].max()

    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}

    with open(f"{ARTIFACT_DIR}/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    return data


def drop_columns(data):
    """
    Drops irrelevant columns exactly as in the notebook.
    """
    data = data.drop(
        [
            "is_active",
            "marketing_consent",
            "first_booking",
            "existing_customer",
            "last_seen",
        ],
        axis=1
    )

    data = data.drop(
        [
            "domain",
            "country",
            "visited_learn_more_before_booking",
            "visited_faq",
        ],
        axis=1
    )

    return data


def clean_missing_and_invalid(data):
    import numpy as np

    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    # Keep only signup rows
    data = data[data.source == "signup"]

    result = data.lead_indicator.value_counts(normalize=True)

    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)

    return data


def convert_categorical_columns(data):
    vars = [
        "lead_id",
        "lead_indicator",
        "customer_group",
        "onboarding",
        "source",
        "customer_code",
    ]

    for col in vars:
        data[col] = data[col].astype("object")
        print(f"Changed {col} to object type")

    return data


def separate_categorical_and_continuous(data):
    from pprint import pprint

    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]
    cat_vars = data.loc[:, (data.dtypes == "object")]

    print("\nContinuous columns: \n")
    pprint(list(cont_vars.columns), indent=4)

    print("\nCategorical columns: \n")
    pprint(list(cat_vars.columns), indent=4)

    return cat_vars, cont_vars


def handle_outliers(cont_vars):
    cont_vars = cont_vars.apply(
        lambda x: x.clip(
            lower=(x.mean() - 2 * x.std()),
            upper=(x.mean() + 2 * x.std())
        )
    )

    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(f"{ARTIFACT_DIR}/outlier_summary.csv")

    print("Outlier summary saved to notebooks/artifacts/outlier_summary.csv")

    return cont_vars


def impute_missing_values_dataset(cat_vars, cont_vars):
    # 1. Mode for categorical variables
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(f"{ARTIFACT_DIR}/cat_missing_impute.csv")

    # 2. Continuous variables
    cont_vars = cont_vars.apply(impute_missing_values)
    cont_vars.apply(describe_numeric_col).T

    # 3. Customer_code special rule
    cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

    # 4. Impute categorical
    cat_vars = cat_vars.apply(impute_missing_values)

    return cat_vars, cont_vars


def scale_continuous_variables(cont_vars):
    from sklearn.preprocessing import MinMaxScaler
    import joblib

    scaler_path = f"{ARTIFACT_DIR}/scaler.pkl"

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(value=scaler, filename=scaler_path)
    print("Saved scaler in notebooks/artifacts")

    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    return cont_vars


def combine_processed_data(cat_vars, cont_vars):
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)

    data = pd.concat([cat_vars, cont_vars], axis=1)

    print(f"Data cleansed and combined.\nRows: {len(data)}")

    return data


def save_drift_artifacts(data):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    data_columns = list(data.columns)

    with open(f"{ARTIFACT_DIR}/columns_drift.json", "w+") as f:
        json.dump(data_columns, f)

    data.to_csv(f"{ARTIFACT_DIR}/training_data.csv", index=False)

    print("Saved drift artifacts and training data.")


def bin_source_column(data):
    data["bin_source"] = data["source"]

    values_list = ["li", "organic", "signup", "fb"]

    data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"

    mapping = {
        "li": "socials",
        "fb": "socials",
        "organic": "group1",
        "signup": "group1",
    }

    data["bin_source"] = data["source"].map(mapping)

    return data


def save_gold_dataset(data):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    output_path = f"{ARTIFACT_DIR}/train_data_gold.csv"
    data.to_csv(output_path, index=False)

    print(f"Saved gold medallion dataset at: {output_path}")
