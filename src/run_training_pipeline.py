# src/run_training_pipeline.py

import shutil
import os

"""
Full training pipeline that:
1. Loads raw data
2. Cleans and preprocesses data
3. Builds features
4. Trains models
5. Selects best model
6. Saves artifacts
"""

from src.data.make_dataset import (
    create_artifact_directory,
    load_data,
    clean_data,
    drop_columns,
    clean_missing_and_invalid,
    convert_categorical_columns,
    separate_categorical_and_continuous,
    handle_outliers,
    impute_missing_values_dataset,
    scale_continuous_variables,
    combine_processed_data,
    save_drift_artifacts,
    bin_source_column,
    save_gold_dataset,
)

from src.features.build_features import (
    split_data_types,
    encode_and_combine_features,
)

from src.models.train_model import (
    get_training_config,
    setup_training_environment,
    split_train_test,
    train_xgboost_model,
    evaluate_model,
    performance_overview,
    save_best_xgboost_model,
    train_logistic_regression,
    save_artifacts,
    get_model_selection_config,
    select_best_model,
    get_production_model,
    compare_models_and_select,
)


def run_pipeline():
    # Phase 1: Dataset preparation
    create_artifact_directory()

    data = load_data()
    data = clean_data(data)
    data = drop_columns(data)
    data = clean_missing_and_invalid(data)
    data = convert_categorical_columns(data)
    cat_vars, cont_vars = separate_categorical_and_continuous(data)

    cont_vars = handle_outliers(cont_vars)
    cat_vars, cont_vars = impute_missing_values_dataset(cat_vars, cont_vars)
    cont_vars = scale_continuous_variables(cont_vars)

    data = combine_processed_data(cat_vars, cont_vars)
    data = bin_source_column(data)

    save_drift_artifacts(data)
    save_gold_dataset(data)

    # Phase 2: Feature engineering
    data, cat_vars, other_vars = split_data_types(data)
    data_encoded = encode_and_combine_features(cat_vars, other_vars)

    # Phase 3: Model training
    config = get_training_config()
    setup_training_environment(config["experiment_name"])

    X_train, X_test, y_train, y_test = split_train_test(data_encoded)

    # XGBoost
    model_grid_xgb = train_xgboost_model(X_train, y_train)
    params_xgb, y_pred_train_xgb, y_pred_test_xgb = evaluate_model(
        model_grid_xgb, X_train, y_train, X_test, y_test
    )
    performance_overview(y_train, y_pred_train_xgb, y_test, y_pred_test_xgb)

    xgb_model_path, model_results = save_best_xgboost_model(
        model_grid_xgb, y_train, y_pred_train_xgb
    )

    # Logistic Regression
    (
        lr_model_path,
        best_lr_params,
        lr_report,
        y_pred_train_lr,
        y_pred_test_lr,
        model_results,
    ) = train_logistic_regression(
        config["experiment_name"],
        X_train,
        y_train,
        X_test,
        y_test,
        model_results,
    )

    # Save metadata artifacts
    save_artifacts(X_train, model_results)

    # Phase 6: Model selection
    model_selection_cfg = get_model_selection_config()

    experiment_best, best_model, results_df = select_best_model(
        config["experiment_name"]
    )

    prod_info = get_production_model(model_selection_cfg["model_name"])

    if prod_info:
        run_id, status = compare_models_and_select(
            experiment_best,
            True,
            prod_info["run_id"],
        )
    else:
        run_id, status = compare_models_and_select(
            experiment_best,
            False,
            None,
        )

    # --- Save final model as artifacts/model ---

    os.makedirs("artifacts", exist_ok=True)

    # Delete existing file if present
    final_model_path = "artifacts/model"
    if os.path.exists(final_model_path):
        os.remove(final_model_path)

    # Determine which file to copy
    if best_model.endswith(".json"):
        source_model_path = "artifacts/lead_model_xgboost.json"
    else:
        source_model_path = "artifacts/lead_model_lr.pkl"

    # Copy model -> artifacts/model
    shutil.copy(source_model_path, final_model_path)

    print(f"Saved best model to artifacts/model")

    print("Training pipeline finished.")
    print("Best model:", best_model)
    print("Model comparison:", status)


if __name__ == "__main__":
    run_pipeline()
