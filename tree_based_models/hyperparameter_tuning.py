from __future__ import annotations

import pandas as pd
from importlib.util import find_spec
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import mean_squared_error


def make_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def is_optional_model_available(model_name: str) -> bool:
    if model_name == "random_forest":
        return True
    if model_name == "optuna":
        return find_spec("optuna") is not None
    if model_name == "xgboost":
        return find_spec("xgboost") is not None
    if model_name == "lightgbm":
        return find_spec("lightgbm") is not None
    raise ValueError(f"Unsupported model_name: {model_name}")


def require_optional_dependency(model_name: str) -> None:
    if is_optional_model_available(model_name):
        return

    package_name = {"optuna": "optuna", "xgboost": "xgboost", "lightgbm": "lightgbm"}[model_name]
    raise ImportError(
        f"{model_name} tuning requested, but the '{package_name}' package is not installed in the active environment."
    )


def build_tree_model(model_name: str, random_state: int = 0):
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1)

    if model_name == "xgboost":
        require_optional_dependency(model_name)
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=random_state,
            objective="reg:squarederror",
            n_estimators=300,
            tree_method="hist",
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        require_optional_dependency(model_name)
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            random_state=random_state,
            n_estimators=300,
            verbosity=-1,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def default_param_grid(model_name: str) -> dict:
    if model_name == "random_forest":
        return {
            "n_estimators": [200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, "sqrt"],
        }
    if model_name == "xgboost":
        return {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": [200, 400],
            "num_leaves": [31, 63, 127],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def default_tpe_search_space(model_name: str) -> dict:
    if model_name == "random_forest":
        return {
            "n_estimators": {"type": "int", "low": 200, "high": 500, "step": 100},
            "max_depth": {"type": "categorical", "choices": [None, 8, 12, 16, 20]},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
            "min_samples_leaf": {"type": "int", "low": 1, "high": 5},
            "max_features": {"type": "categorical", "choices": [1.0, "sqrt", "log2"]},
        }
    if model_name == "xgboost":
        return {
            "n_estimators": {"type": "int", "low": 200, "high": 500, "step": 100},
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.7, "high": 1.0},
            "min_child_weight": {"type": "int", "low": 1, "high": 10},
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": {"type": "int", "low": 200, "high": 500, "step": 100},
            "num_leaves": {"type": "int", "low": 31, "high": 127, "step": 8},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.2, "log": True},
            "subsample": {"type": "float", "low": 0.7, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.7, "high": 1.0},
            "min_child_samples": {"type": "int", "low": 5, "high": 40},
        }
    raise ValueError(f"Unsupported model_name: {model_name}")


def suggest_optuna_value(trial, name: str, spec: dict):
    suggestion_type = spec["type"]
    if suggestion_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if suggestion_type == "int":
        step = spec.get("step", 1)
        log = spec.get("log", False)
        return trial.suggest_int(name, spec["low"], spec["high"], step=step, log=log)
    if suggestion_type == "float":
        step = spec.get("step")
        log = spec.get("log", False)
        if step is not None:
            return trial.suggest_float(name, spec["low"], spec["high"], step=step, log=log)
        return trial.suggest_float(name, spec["low"], spec["high"], log=log)
    raise ValueError(f"Unsupported Optuna suggestion type: {suggestion_type}")


def run_tpe_search(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: dict | None,
    n_splits: int,
    n_iter: int,
    random_state: int,
):
    require_optional_dependency("optuna")
    import optuna
    from optuna.samplers import TPESampler

    estimator = build_tree_model(model_name, random_state=random_state)
    search_space = default_tpe_search_space(model_name) if search_space is None else search_space
    cv = make_time_series_cv(n_splits=n_splits)

    def objective(trial) -> float:
        params = {name: suggest_optuna_value(trial, name, spec) for name, spec in search_space.items()}
        model = clone(estimator)
        model.set_params(**params)

        fold_rmses = []
        for train_idx, val_idx in cv.split(X_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            fold_rmses.append(mean_squared_error(y_fold_val, preds) ** 0.5)

        mean_rmse = float(sum(fold_rmses) / len(fold_rmses))
        trial.set_user_attr("fold_rmses", fold_rmses)
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_iter)

    best_estimator = clone(estimator)
    best_estimator.set_params(**study.best_params)
    best_estimator.fit(X_train, y_train)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state")).copy()
    trials_df = trials_df.rename(
        columns={
            "number": "trial_number",
            "value": "mean_cv_rmse",
            "state": "trial_state",
        }
    )
    trials_df["rank_test_score"] = trials_df["mean_cv_rmse"].rank(method="dense").astype(int)
    trials_df = trials_df.sort_values(["rank_test_score", "trial_number"]).reset_index(drop=True)

    return {
        "best_estimator": best_estimator,
        "best_params": study.best_params,
        "best_cv_rmse": study.best_value,
        "cv_results": trials_df,
    }


def tune_tree_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    search_type: str = "random",
    n_splits: int = 5,
    n_iter: int = 12,
    random_state: int = 0,
):
    if search_type == "tpe":
        return run_tpe_search(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            search_space=param_grid,
            n_splits=n_splits,
            n_iter=n_iter,
            random_state=random_state,
        )

    estimator = build_tree_model(model_name, random_state=random_state)
    param_grid = default_param_grid(model_name) if param_grid is None else param_grid
    cv = make_time_series_cv(n_splits=n_splits)

    if search_type == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
        )
    else:
        raise ValueError("search_type must be 'grid', 'random', or 'tpe'")

    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    cv_results["mean_cv_rmse"] = -cv_results["mean_test_score"]
    cv_results["std_cv_rmse"] = cv_results["std_test_score"]
    cv_results = cv_results.sort_values(["rank_test_score"]).reset_index(drop=True)

    return {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_rmse": -search.best_score_,
        "cv_results": cv_results,
    }
