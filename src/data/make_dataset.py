# -*- coding: utf-8 -*-
import os
# Same notebook variables
max_date = "2024-01-31"
min_date = "2024-01-01"


def create_artifact_directory():
    """
    Creates artifacts/ directory if it does not exist.
    """
    os.makedirs("artifacts", exist_ok=True)
    print("Created artifacts directory")

import pandas as pd

def load_data():
    print("Loading training data")

    data = pd.read_csv("./artifacts/raw_data.csv")

    print("Total rows:", data.count())
    print(data.head(5))

    return data

def clean_data(data):
    import pandas as pd
    import datetime
    import json
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


