# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from typing import Any

DEFAULT_RF_REGRESSION_SCORING = "neg_mean_absolute_error"

DEFAULT_RF_REGRESSION_CRITERION = "squared_error"

DEFAULT_RF_REGRESSION_SCORERS = {
    "Mean Absolute Error (MAE)": "neg_mean_absolute_error",
    "Root Mean Squared Error (RMSE)": "neg_root_mean_squared_error",
    "R-squared (R2)": "r2",
    "Median Absolute Error (MedAE)": "neg_median_absolute_error",
}

DEFAULT_RF_REGRESSION_SEARCH_ITERATIONS = 60

DEFAULT_RF_REGRESSION_PERM_IMPORTANCE_REPEATS = 30


def hyperparam_search_randomized(
    X,
    y,
    groups,
    estimator,
    param_distributions: dict[str, Any],
    n_iter: int = 10,
    scoring=None,
    n_jobs=None,
    cv=None,
    verbose=0,
    random_state=None,
    error_score=np.nan,
    output_csv=None,
):

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        scoring=scoring,
        n_jobs=n_jobs,
        n_iter=n_iter,
        cv=cv,
        verbose=verbose,
        random_state=random_state,
        error_score=error_score,
        return_train_score=True,
    )

    search.fit(X, y, groups=groups)

    results = pd.DataFrame(search.cv_results_)

    if output_csv is not None:
        results.to_csv(f"{output_csv}", index=False)

    best_params_idx = results.sort_values(by="rank_test_score")[
        "rank_test_score"
    ].idxmin()
    best_params = results.loc[best_params_idx, "params"]

    return search, results, best_params


def report_cross_validation_scores(model_name: str, scores: dict[str, np.ndarray]):
    from dse_research_utils.console.console import print_table
    from dse_research_utils.console.sections import section_header
    from dse_research_utils.console.tables import metrics_table

    section_header(f"{model_name}: Cross-validation scores")
    rows = []
    for key, score in scores.items():
        if not key.startswith("test_"):
            continue
        label = key[5:]
        rows.append(
            {
                "metric": label,
                "mean": float(np.abs(np.mean(score))),
                "sd": float(np.std(score)),
            }
        )
    print_table(metrics_table(rows, columns=["metric", "mean", "sd"]))


# from https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/16-Gaussian_Processes.html
def quadratic_distance_kernel(X0, X1, eta=1, sigma=0.5):
    # Use linear algebra identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
    X0_norm = np.sum(X0**2, axis=-1)
    X1_norm = np.sum(X1**2, axis=-1)
    squared_distances = X0_norm[:, None] + X1_norm[None, :] - 2 * X0 @ X1.T
    rho = 1 / sigma**2
    return eta**2 * np.exp(-rho * squared_distances)


# from https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/16-Gaussian_Processes.html
def ornstein_uhlenbeck_kernel(X0, X1, eta_squared=1, rho=4):
    # ``plot_kernel_function`` and similar helpers pass 2D column vectors
    # (shape ``(n, 1)``); flatten so the pairwise-distance broadcast
    # produces the expected ``(n, n)`` matrix rather than ``(n, n, 1)``.
    X0 = np.asarray(X0).reshape(-1)
    X1 = np.asarray(X1).reshape(-1)
    distances = np.abs(X1[None, :] - X0[:, None])
    return eta_squared * np.exp(-rho * distances)


# from https://www.pymc.io/projects/examples/en/latest/statistical_rethinking_lectures/16-Gaussian_Processes.html
def periodic_kernel(X0, X1, eta=1, sigma=1, periodicity=0.5):
    X0 = np.asarray(X0).reshape(-1)
    X1 = np.asarray(X1).reshape(-1)
    distances = np.sin((X1[None, :] - X0[:, None]) * periodicity) ** 2
    rho = 2 / sigma**2
    return eta**2 * np.exp(-rho * distances)
