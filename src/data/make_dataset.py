# -*- coding: utf-8 -*-
import os
import pandas as pd
import datetime
import json
# Same notebook variables
max_date = "2024-01-31"
min_date = "2024-01-01"


def create_artifact_directory():
    """
    Creates artifacts/ directory if it does not exist.
    """
    os.makedirs("artifacts", exist_ok=True)
    print("Created artifacts directory")


def load_data():
    print("Loading training data")

    data = pd.read_csv("./artifacts/raw_data.csv")

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

    # Time limit data
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

    min_date = data["date_part"].min()
    max_date = data["date_part"].max()

    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("./artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    return data

def drop_columns(data):
    """
    Drops irrelevant columns exactly as in the notebook.
    """

    # Drop columns not relevant for modelling
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

    # Drop columns that will be added back after the EDA
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

    # Replace empty strings with NaN
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)

    # Drop rows with missing target variable and lead_id
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])

    # Keep only signup rows
    data = data[data.source == "signup"]

    # Print target distribution
    result = data.lead_indicator.value_counts(normalize=True)

    print("Target value counter")
    for val, n in zip(result.index, result):
        print(val, ": ", n)

    return data

def convert_categorical_columns(data):

    """
    Converts specific columns to object type exactly as in the notebook.
    """

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
    """
    Separates continuous and categorical variables exactly as in the notebook.
    """

    from pprint import pprint

    # Continuous variables: float64 or int64
    cont_vars = data.loc[:, ((data.dtypes == "float64") | (data.dtypes == "int64"))]

    # Categorical variables: object type
    cat_vars = data.loc[:, (data.dtypes == "object")]

    print("\nContinuous columns: \n")
    pprint(list(cont_vars.columns), indent=4)

    print("\n Categorical columns: \n")
    pprint(list(cat_vars.columns), indent=4)

    return cat_vars, cont_vars

def handle_outliers(cont_vars):
    """
    Clips continuous variables to remove outliers using Z-score logic
    exactly as in the notebook.
    """

    # Apply clipping to each continuous column
    cont_vars = cont_vars.apply(
        lambda x: x.clip(
            lower=(x.mean() - 2 * x.std()),
            upper=(x.mean() + 2 * x.std())
        )
    )

    # Generate outlier summary
    outlier_summary = cont_vars.apply(describe_numeric_col).T

    # Save summary to artifacts
    outlier_summary.to_csv('./artifacts/outlier_summary.csv')

    print("Outlier summary saved to artifacts/outlier_summary.csv")

    return cont_vars