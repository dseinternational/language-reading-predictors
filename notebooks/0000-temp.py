# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dse-language-reading-predictors
#     language: python
#     name: python3
# ---

# %%
# %config InlineBackend.figure_format = 'retina'

# %%
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata

import language_reading_predictors.data_utils as data_utils

from language_reading_predictors.data_variables import Variables as vars

# constrained layout causes issues with some of the SHAP plots
mpl.rcParams["figure.constrained_layout.use"] = False

setup.init_workbook()

# %%
WORKBOOK = "0000-temp"
OUTPUT_DIR = f"../output/notebooks/{WORKBOOK}"
REPORT_FIGS_DIR = f"../docs/report/figures/{WORKBOOK}"
SAVE_PLOTS = True

RANDOM_SEED = 47
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_FIGS_DIR, exist_ok=True)

print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print()

package_list = [
    "duckdb",
    "matplotlib",
    "numpy",
    "pandas",
]

print()

package_metadata.report_package_versions(package_list)

# %%
df = data_utils.load_data()

# %%
# no progress: start?

# %%
df[vars.EWRSWR]
