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
#     display_name: dse-research-reading-language
#     language: python
#     name: python3
# ---

# %%
# %config InlineBackend.figure_format = 'retina'

# %%
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

# constrained layout causes issues with some of the SHAP plots
mpl.rcParams["figure.constrained_layout.use"] = False

setup.init_workbook()

# %%
WORKBOOK = "0010-predictors-word-reading"
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
analysis_df = df[df["ewrswr"].notna()].copy()
analysis_df["ewrswr"].describe()

# %%
plt.figure(figsize=(8, 2))
sns.boxplot(analysis_df["ewrswr"], orient="h")

# %%
predictor_vars = [
    "time",
    "group",
    "age",
    "area",
    "attend",
    "hearing",
    "vision",
    "behav",
#    "ewrswr",
    "rowpvt",
    "eowpvt",
    "yarclet",
    "spphon",
    "blending",
    "nonword",
    "trog",
    "aptgram",
    "aptinfo",
    #    "b1extau",
    #    "b1exnt",
    "b1exto",
    #    "b1retau",
    #    "b1rent",
    "b1reto",
    #    "b2extau",
    #    "b2exnt",
    "b2exto",
    #    "b2retau",
    #    "b2rent",
    "b2reto",
    "celf",
    "deappin",
    "deappvo",
    "deappfi",
    "deappav"
]

X = analysis_df[predictor_vars]
y = analysis_df["ewrswr"]

groups = analysis_df["subject_id"]

rf = RandomForestRegressor(
    n_estimators=1200,
    max_depth=8,
    min_samples_leaf=16,
    min_samples_split=4,
    max_features=0.5,
    bootstrap=False,
    criterion="squared_error",
    random_state=RANDOM_SEED,
    n_jobs=16,
)

model = Pipeline([
    ("rf", rf)
])

cv = GroupKFold(n_splits=51)

scores = cross_val_score(
    model, X, y, cv=cv, groups=groups,
    scoring="neg_root_mean_squared_error"
)

print("Group-aware CV RMSE:", -scores)
print("Mean RMSE:", -scores.mean())

# %%
model.fit(X, y)

# %%
result = permutation_importance(
    model, X, y, n_repeats=50, random_state=RANDOM_SEED
)

perm_importance = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std,
}).sort_values("importance_mean", ascending=False)

perm_importance

# %%
X_shap = X.copy()
X_shap = X_shap.replace({pd.NA: np.nan})
X_shap = X_shap.astype("float64")
X_shap = X_shap.fillna(X_shap.mean())

rf_fitted = model.named_steps["rf"]
explainer = shap.TreeExplainer(rf_fitted)
shap_values = explainer.shap_values(X_shap)

# %%
shap.plots.bar(explainer(X_shap), max_display=20)

# %%

shap.summary_plot(shap_values, X_shap, plot_size=0.25)
