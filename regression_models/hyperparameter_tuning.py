import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def default_alpha_grid() -> np.ndarray:
    return np.logspace(-4, 2, num=25)


def make_time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)


def build_regularized_pipeline(model_name: str):
    if model_name == "ridge":
        model = Ridge(random_state=0)
    elif model_name == "lasso":
        model = Lasso(random_state=0, max_iter=20000)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def tune_regularized_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha_grid: np.ndarray | None = None,
    search_type: str = "grid",
    n_splits: int = 5,
    n_iter: int = 12,
    random_state: int = 0,
):
    alpha_grid = default_alpha_grid() if alpha_grid is None else np.asarray(alpha_grid, dtype=float)
    pipeline = build_regularized_pipeline(model_name)
    cv = make_time_series_cv(n_splits=n_splits)
    param_grid = {"model__alpha": alpha_grid}

    if search_type == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
    elif search_type == "random":
        if n_iter > len(alpha_grid):
            n_iter = len(alpha_grid)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=random_state,
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].copy()
    cv_results["mean_cv_rmse"] = -cv_results["mean_test_score"]
    cv_results["std_cv_rmse"] = cv_results["std_test_score"]
    cv_results["alpha"] = cv_results["params"].apply(lambda p: p["model__alpha"])
    cv_results = cv_results[["alpha", "mean_cv_rmse", "std_cv_rmse", "rank_test_score"]]
    cv_results = cv_results.sort_values(["rank_test_score", "alpha"]).reset_index(drop=True)

    return {
        "best_estimator": search.best_estimator_,
        "best_alpha": search.best_params_["model__alpha"],
        "best_cv_rmse": -search.best_score_,
        "cv_results": cv_results,
    }
