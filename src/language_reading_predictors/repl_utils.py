# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""REPL / notebook environment banner.

Delegates to the shared ``dse_research_utils`` helpers so the environment and
package-version report stays consistent across DSE research repos. torch is no
longer a dependency (it was only used for a CUDA banner); GPU acceleration is an
opt-in jax overlay and is surfaced via the reported ``jax``/``jaxlib`` versions
rather than a bespoke CUDA probe.
"""

from dse_research_utils.environment.info import report_environment_info
from dse_research_utils.metadata.packages import report_package_versions

RANDOM_SEED = 47

# Versions worth pinning down when debugging model runs.
_REPORTED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pymc",
    "pytensor",
    "nutpie",
    "numpyro",
    "jax",
    "jaxlib",
    "arviz",
    "lightgbm",
    "xgboost",
    "shap",
    "dse-research-utils",
]


def print_environment_info() -> None:
    report_environment_info()
    report_package_versions(_REPORTED_PACKAGES)
