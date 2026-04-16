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

# %% [markdown]
# # 0002 - Associations

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
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.metadata.packages as package_metadata

import language_reading_predictors.data_utils as data_utils

from language_reading_predictors.data_variables import Variables as vars

# constrained layout causes issues with some of the SHAP plots
mpl.rcParams["figure.constrained_layout.use"] = False

setup.init_workbook()

# %%
WORKBOOK = "0002-associations"
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
    "matplotlib",
    "numpy",
    "pandas",
]

print()

package_metadata.report_package_versions(package_list)

# %%
df = data_utils.load_data()


# %%
def scatter_plot(x, y):
    plt.figure(figsize=plot_styles.FIGSIZE_LG)
    plt.scatter(df[x], df[y], alpha=0.5)
    plt.xlabel(vars.get_variable_name(x))
    plt.ylabel(vars.get_variable_name(y))


# %% [markdown]
# ## Influence of letter sounds score

# %%
scatter_plot(vars.YARCLET, vars.EWRSWR)

# %%
scatter_plot(vars.YARCLET, vars.SPPHON)

# %%
scatter_plot(vars.YARCLET, vars.BLENDING)

# %%
scatter_plot(vars.YARCLET, vars.NONWORD)

# %% [markdown]
# ## Factors influencing early word reading composite

# %%
scatter_plot(vars.BLENDING, vars.EWRSWR)

# %%
scatter_plot(vars.SPPHON, vars.EWRSWR)

# %%
scatter_plot(vars.NONWORD, vars.EWRSWR)

# %% [markdown]
# ## Factors influencing non-word reading score

# %%
scatter_plot(vars.ROWPVT, vars.NONWORD)

# %%
scatter_plot(vars.YARCLET, vars.NONWORD)

# %%
scatter_plot(vars.SPPHON, vars.NONWORD)

# %%
scatter_plot(vars.EWRSWR, vars.NONWORD)
