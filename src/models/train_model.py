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

