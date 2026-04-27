from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from forecast_plotting import save_prediction_bundle, save_test_comparison_plot, to_prediction_frame
from tree_based_models.data_processing import prepare_tree_based_dataset
from tree_based_models.hyperparameter_tuning import build_tree_model
from tree_based_models.model_evaluation import rolling_retrain_evaluation


DATASET_PATH = Path(__file__).resolve().parent / "Data" / "data_csv" / "dataset_CET_20260226T2242.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "forecast_plots"


def fit_predict_frames(model, X_train, X_val, X_test, y_train, y_val, y_test) -> dict[str, pd.DataFrame]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    return {
        "train": to_prediction_frame(y_train, fitted.predict(X_train), "train"),
        "validation": to_prediction_frame(y_val, fitted.predict(X_val), "validation"),
        "test": to_prediction_frame(y_test, fitted.predict(X_test), "test"),
    }


def fit_predict_rolling_test_frames(model, X_train_val, X_test, y_train, y_val, y_test) -> dict[str, pd.DataFrame]:
    train_val_y = pd.concat([y_train, y_val])
    fitted = clone(model)
    fitted.fit(X_train_val, train_val_y)
    rolling_pred, _ = rolling_retrain_evaluation(
        model=model,
        X_initial_train=X_train_val,
        y_initial_train=train_val_y,
        X_test=X_test,
        y_test=y_test,
        retrain_every=24 * 7,
    )
    return {
        "train": to_prediction_frame(y_train, fitted.predict(X_train_val.iloc[: len(y_train)]), "train"),
        "validation": to_prediction_frame(y_val, fitted.predict(X_train_val.iloc[len(y_train) :]), "validation"),
        "test": to_prediction_frame(y_test.loc[rolling_pred.index], rolling_pred, "test"),
    }


def main():
    prepared = prepare_tree_based_dataset(
        dataset_path=DATASET_PATH,
        tz="Europe/Berlin",
        target_col="price",
        n_val=24 * 30,
        n_test=24 * 90,
    )
    feature_set = prepared["feature_sets"][-1]
    X_train_all, X_val_all, X_test_all, y_train, y_val, y_test = prepared["splits"]
    X_train = X_train_all[feature_set]
    X_val = X_val_all[feature_set]
    X_test = X_test_all[feature_set]
    X_train_val = pd.concat([X_train, X_val])

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(random_state=0, n_estimators=300, n_jobs=-1),
        "XGBoost": build_tree_model("xgboost", random_state=0),
    }

    saved: dict[str, dict[str, str]] = {}
    for model_name, model in models.items():
        frames = fit_predict_frames(model, X_train, X_val, X_test, y_train, y_val, y_test)
        saved[model_name] = save_prediction_bundle(frames, model_name, OUTPUT_DIR)

    xgboost_rolling_frames = fit_predict_rolling_test_frames(
        models["XGBoost"], X_train_val, X_test, y_train, y_val, y_test
    )
    saved["XGBoost Rolling"] = save_prediction_bundle(xgboost_rolling_frames, "XGBoost Rolling", OUTPUT_DIR)
    comparison_plot_path = OUTPUT_DIR / "test_set_model_comparison.png"
    save_test_comparison_plot(
        {
            "Linear Regression": saved_test_frame(models["Linear Regression"], X_train, X_val, X_test, y_train, y_val, y_test),
            "Random Forest": saved_test_frame(models["Random Forest"], X_train, X_val, X_test, y_train, y_val, y_test),
            "XGBoost": saved_test_frame(models["XGBoost"], X_train, X_val, X_test, y_train, y_val, y_test),
            "XGBoost Rolling": xgboost_rolling_frames["test"],
        },
        comparison_plot_path,
    )

    print("Saved forecast plots:")
    for model_name, paths in saved.items():
        print(model_name)
        for split_name, path in paths.items():
            print(f"  {split_name}: {path}")
    print(f"Test comparison: {comparison_plot_path}")


def saved_test_frame(model, X_train, X_val, X_test, y_train, y_val, y_test) -> pd.DataFrame:
    fitted = clone(model)
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    fitted.fit(X_train_val, y_train_val)
    return to_prediction_frame(y_test, fitted.predict(X_test), "test")


if __name__ == "__main__":
    main()
