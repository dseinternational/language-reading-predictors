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

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata

import language_reading_predictors.data_utils as data_utils

from language_reading_predictors.data_variables import Variables as vars
from language_reading_predictors.plot_utils import scatter_plot, violin_plot

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
    "matplotlib",
    "numpy",
    "pandas",
]

print()

package_metadata.report_package_versions(package_list)

# %%
all_df = data_utils.load_data()
t1_df = all_df[all_df[vars.TIME] == 1]
intervention_df = t1_df[t1_df[vars.GROUP] == 1]
control_df = t1_df[t1_df[vars.GROUP] == 2]

# %%

# %%

# %%
scatter_plot(t1_df, vars.EWRSWR, vars.EWRSWR_GAIN, color=vars.GROUP, categorical=True)

# %%
scatter_plot(t1_df, vars.YARCLET, vars.EWRSWR_GAIN, color=vars.GROUP, categorical=True)

# %%
violin_plot(t1_df, vars.GROUP, vars.EWRSWR_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.ROWPVT_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.B1RETO_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.B1EXTO_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.EOWPVT_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.EOWPVT)

# %%
violin_plot(t1_df, vars.GROUP, vars.CELF_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.TROG_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.SPPHON_GAIN)

# %%
violin_plot(t1_df, vars.GROUP, vars.BLENDING_GAIN)
