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
# # Notebook 1 - **Variables**

# %% [markdown]
#

# %% [markdown]
# ## Preparation

# %%
import datetime, joblib, os, shap, matplotlib, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.stats import randint, loguniform
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.inspection import permutation_importance

import language_reading_predictors.data_utils as data_utils
import language_reading_predictors.repl_utils as repl_utils
import language_reading_predictors.stats_utils as stats_utils

from language_reading_predictors.data_variables import Variables as vars
from language_reading_predictors.data_variables import Predictors as pred
from language_reading_predictors.data_variables import Categories as cats



RANDOM_SEED = repl_utils.RANDOM_SEED
np.random.seed(RANDOM_SEED)
RNG = np.random.default_rng(RANDOM_SEED)

N_CORES = joblib.cpu_count(only_physical_cores=True)
START_TIME = datetime.datetime.now()
OUTPUT_DIR = f"output/0001-data-variables/{START_TIME:%Y%m%d-%H%M%S}"

SAVE_PLOTS = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

repl_utils.print_environment_info()

print(f"\n--------------------\nOutput directory: {OUTPUT_DIR}\n--------------------\n")

# %%
df = data_utils.load_data()
print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# %%
for col in df.columns:
    print(f"{col}: {vars.get_variable_name(col)}")

# %%
predictors = vars.NUMERIC + vars.CATEGORICAL
predictors_df = df[predictors]
predictors_np = predictors_df.to_numpy(dtype=np.float64)

# %%
distance, corr = stats_utils.distance_corr_dissimilarity(predictors_np)
condensed = squareform(distance, checks=True)
dist_linkage_0 = hierarchy.ward(condensed)

# %%
fig, ax = plt.subplots(figsize=(7, 12))
dendro_0 = hierarchy.dendrogram(dist_linkage_0, labels=predictors_df.columns.to_list(), orientation="right", ax=ax)
plt.xlabel("Ward linkage distance (increase in within-cluster variance)")
plt.ylabel("Predictors")
plt.title(f"Hierarchical clustering of predictors")
ax.tick_params(axis="y", labelsize=10)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/hierarchical-clustering.svg", format="svg", bbox_inches="tight")
plt.show()

# %%
dendro_0_idx = np.arange(0, len(dendro_0["ivl"]))

with plt.rc_context({'ytick.labelsize': 12, 'xtick.labelsize': 12, 'axes.titlesize': 12}):
    plt.figure(figsize=(14, 14))
    plt.set_cmap("viridis")
    ax = plt.axes()
    im = ax.imshow(corr[dendro_0["leaves"], :][:, dendro_0["leaves"]])
    ax.set_title(f"Correlation heatmap of predictors")
    ax.set_xticks(dendro_0_idx)
    ax.set_yticks(dendro_0_idx)
    ax.set_xticklabels(dendro_0["ivl"], rotation="vertical")
    ax.set_yticklabels(dendro_0["ivl"])
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.025)
    if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/correlation-heatmap.svg", format="svg", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Study variables

# %% [markdown]
# ## Child and family variables

# %%
child_family_vars = ([v for v in cats.CHILD_FAMILY if v in vars.NUMERIC])
stats_utils.describe_all(df[child_family_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[child_family_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/child-family-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Health variables

# %%
fig, axes = plt.subplots(figsize=(6,4))
bottom = np.zeros(len(cats.HEALTH))
colors = matplotlib.color_sequences["tab10"]
i=0

for v in cats.HEALTH:
    t = df[df[v] == 1][v].count()
    f = df[df[v] == 0][v].count()
    tot = t + f
    p = axes.bar(v, t/tot, bottom=0, color=colors[i])
    p = axes.bar(v, f/tot, bottom=p[0].get_height(), color=colors[i], alpha=0.5)
    i=i+1

plt.title("Proportions of children with health issues")
plt.ylabel("Proportion")
plt.xlabel("Health issue (darker = present, lighter = absent)")
plt.xticks(rotation=45)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/health-issues-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Cognition variables

# %%
cognition_vars = ([v for v in cats.COGNITION if v in vars.NUMERIC])
stats_utils.describe_all(df[cognition_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[cognition_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/cognition-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Language variables

# %%
language_vars = ([v for v in cats.LANGUAGE if v in vars.NUMERIC])
stats_utils.describe_all(df[language_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[language_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/language-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Speech variables

# %%
speech_vars = ([v for v in cats.SPEECH if v in vars.NUMERIC])
stats_utils.describe_all(df[speech_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[speech_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/speech-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Reading variables

# %%
reading_vars = ([v for v in cats.READING if v in vars.NUMERIC])
stats_utils.describe_all(df[reading_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[reading_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/reading-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Social variables

# %%
social_vars = ([v for v in cats.SOCIAL if v in vars.NUMERIC])
stats_utils.describe_all(df[social_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[social_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/social-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Teaching variables

# %%
teaching_vars = ([v for v in cats.TEACHING if v in vars.NUMERIC])
stats_utils.describe_all(df[teaching_vars], 0.05).T

# %%
fig, axes = plot_utils.plot_histograms(df[reading_vars], name_lookup=vars.NAMES)
if SAVE_PLOTS: plt.savefig(f"{OUTPUT_DIR}/teaching-variables-distributions.svg", format="svg", bbox_inches="tight")
plt.show()
