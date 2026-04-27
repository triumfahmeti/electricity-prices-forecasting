import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo
from hyperparameter_tuning import tune_regularized_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1) Load + timestamp handling

REQUIRED_DATA_COLUMNS = {
    "timestamp",
    "price",
    "load",
    "solar",
    "wind onshore",
    "wind offshore",
}


def validate_dataset_schema(df: pd.DataFrame, target_col: str) -> None:
    required_columns = REQUIRED_DATA_COLUMNS | {target_col}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")


def load_dataset(dataset_path: str | Path, tz: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    validate_dataset_schema(df, target_col)
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        .dt.tz_convert(ZoneInfo(tz))
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    return df

# 2) Feature engineering

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["month"] = df.index.month

    # cyclical encodings (better than raw hour/month for linear models)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["residual_load"] = df["load"] - df["solar"] - df["wind onshore"] - df["wind offshore"]
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "price") -> pd.DataFrame:
    df = df.copy()
    lag_feature_names = {
        24: f"{target_col}_lag_24",
        168: f"{target_col}_lag_168",
    }
    for lag_hours, feature_name in lag_feature_names.items():
        df[feature_name] = df[target_col].shift(lag_hours)
    return df


# 3) Splitting + evaluation

def align_dataset_for_comparison(
    df: pd.DataFrame, feature_sets: list[list[str]], target_col: str
) -> pd.DataFrame:
    required_cols = sorted({target_col, *(col for feats in feature_sets for col in feats)})
    usable = df.dropna(subset=required_cols)
    if usable.empty:
        raise ValueError("No rows remain after aligning all feature sets on a common timestamp range.")
    return usable


def time_train_val_test_split(df: pd.DataFrame, target_col: str, n_val: int, n_test: int):
    if len(df) <= n_val + n_test:
        raise ValueError(
            f"Not enough rows for train/validation/test split. Have {len(df)}, need > {n_val + n_test}."
        )

    train_end = len(df) - (n_val + n_test)
    val_end = len(df) - n_test

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return (
        train_df.drop(columns=[target_col]),
        val_df.drop(columns=[target_col]),
        test_df.drop(columns=[target_col]),
        train_df[target_col],
        val_df[target_col],
        test_df[target_col],
    )


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

# 4) Model factory (scaled)

def get_base_models() -> dict:
    return {
        "LinReg_scaled": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    }

# 5) Main: compare models/features

def train_and_compare_models(
    dataset_path: str | Path = Path(__file__).resolve().parent.parent / "Data" / "data_csv" / "dataset_CET_20260226T2242.csv",
    tz: str = "Europe/Berlin",
    n_val: int = 24 * 30,    # prior 30 days (hourly)
    n_test: int = 24 * 90,   # last 90 days (hourly)
    tuning_search_type: str = "grid",
    tuning_cv_splits: int = 5,
):
    target_col = "price"
    lag_24_col = f"{target_col}_lag_24"
    lag_168_col = f"{target_col}_lag_168"

    df = load_dataset(dataset_path, tz, target_col)
    df = add_time_features(df)
    df = add_power_features(df)
    df = add_lag_features(df, target_col=target_col)

    # Feature sets to test (now using cyclical time features)
    feature_sets = [
        ["residual_load"],
        ["residual_load", "hour_sin", "hour_cos"],
        ["residual_load", "hour_sin", "hour_cos", "month_sin", "month_cos"],
        ["residual_load", lag_24_col],
        ["residual_load", "hour_sin", "hour_cos", lag_24_col],
        ["residual_load", "hour_sin", "hour_cos", lag_24_col, lag_168_col],
        ["solar", "wind onshore", "wind offshore", "load", "hour_sin", "hour_cos", "month_sin", "month_cos"],
        ["solar", "wind onshore", "wind offshore", "load", "hour_sin", "hour_cos", lag_24_col],
        ["solar", "wind onshore", "wind offshore", "load", "hour_sin", "hour_cos", lag_24_col, lag_168_col],
    ]
    df = align_dataset_for_comparison(df, feature_sets, target_col)

    models = get_base_models()
    results = []
    X_train_all, X_val_all, X_test_all, y_train, y_val, y_test = time_train_val_test_split(
        df, target_col, n_val, n_test
    )

    for feats in feature_sets:
        X_train = X_train_all[feats]
        X_val = X_val_all[feats]

        for model_name, model in models.items():
            metrics = evaluate(model, X_train, y_train, X_val, y_val)
            results.append({
                "model": model_name,
                "feature_tuple": tuple(feats),
                "features": ", ".join(feats),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "best_alpha": np.nan,
                "train_cv_rmse": np.nan,
                **{f"val_{metric}": value for metric, value in metrics.items()},
            })

        for regularized_model_name in ["ridge", "lasso"]:
            tuning_result = tune_regularized_model(
                model_name=regularized_model_name,
                X_train=X_train,
                y_train=y_train,
                search_type=tuning_search_type,
                n_splits=tuning_cv_splits,
            )
            metrics = evaluate(tuning_result["best_estimator"], X_train, y_train, X_val, y_val)
            results.append({
                "model": f"{regularized_model_name.capitalize()}_tuned",
                "feature_tuple": tuple(feats),
                "features": ", ".join(feats),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "best_alpha": tuning_result["best_alpha"],
                "train_cv_rmse": tuning_result["best_cv_rmse"],
                **{f"val_{metric}": value for metric, value in metrics.items()},
            })

    results_df = pd.DataFrame(results).sort_values(["val_RMSE", "val_MAE"], ascending=True)

    print("\n=== VALIDATION MODEL COMPARISON (sorted by validation RMSE, then MAE) ===")
    print(results_df.drop(columns=["feature_tuple"]).to_string(index=False))

    best_row = results_df.iloc[0]
    best_features = list(best_row["feature_tuple"])
    X_train_best = X_train_all[best_features]
    X_val_best = X_val_all[best_features]
    X_train_val = pd.concat([X_train_best, X_val_best])
    y_train_val = pd.concat([y_train, y_val])

    if best_row["model"] == "LinReg_scaled":
        best_model = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    elif best_row["model"] == "Ridge_tuned":
        best_model = tune_regularized_model(
            model_name="ridge",
            X_train=X_train_val,
            y_train=y_train_val,
            search_type=tuning_search_type,
            n_splits=tuning_cv_splits,
        )["best_estimator"]
    elif best_row["model"] == "Lasso_tuned":
        best_model = tune_regularized_model(
            model_name="lasso",
            X_train=X_train_val,
            y_train=y_train_val,
            search_type=tuning_search_type,
            n_splits=tuning_cv_splits,
        )["best_estimator"]
    else:
        raise ValueError(f"Unsupported best model selection: {best_row['model']}")

    test_metrics = evaluate(best_model, X_train_val, y_train_val, X_test_all[best_features], y_test)

    print("\n=== BEST MODEL (chosen on validation set) ===")
    print(best_row.drop(labels=["feature_tuple"]))

    print("\n=== FINAL TEST METRICS (untouched during model selection) ===")
    print(pd.Series({
        "model": best_row["model"],
        "features": best_row["features"],
        "n_train_val": len(X_train_val),
        "n_test": len(X_test_all),
        **test_metrics,
    }))

    return best_model, results_df

if __name__ == "__main__":
    train_and_compare_models()
