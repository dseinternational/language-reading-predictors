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
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import partial_dependence, permutation_importance

import dse_research_utils.environment.setup as setup
import dse_research_utils.metadata.packages as package_metadata

import language_reading_predictors.data_utils as data_utils

# constrained layout causes issues with some of the SHAP plots
mpl.rcParams["figure.constrained_layout.use"] = False

setup.init_workbook()

# %%
WORKBOOK = "0003-predictors-word-reading-gain-random-forest-exc-outliers"
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
ewrswr_gain_df = df[df["ewrswr_gain"].notna()].copy()
ewrswr_gain_df["ewrswr_gain"].describe()

# %%
ewrswr_gain_df = ewrswr_gain_df[ewrswr_gain_df["ewrswr_gain"] < 15.0]
ewrswr_gain_df["ewrswr_gain"].describe()

# %%
predictor_vars = [
    "time",
    #   "group",
    "age",
    #    "area",
    "attend",
    "hearing",
    "vision",
    "behav",
    "ewrswr",
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
    "deappav",
]

X = ewrswr_gain_df[predictor_vars]
y = ewrswr_gain_df["ewrswr_gain"]

groups = ewrswr_gain_df["subject_id"]

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

model = Pipeline([("rf", rf)])

cv = GroupKFold(n_splits=53)

scores = cross_val_score(
    model, X, y, cv=cv, groups=groups, scoring="neg_root_mean_squared_error"
)

print("Group-aware CV RMSE:", -scores)
print("Mean RMSE:", -scores.mean())

# %%
model.fit(X, y)

# %%
y_pred = model.predict(X)

df_eval = pd.DataFrame(
    {
        "group": groups,
        "y_true": y,
        "y_pred": y_pred,
    }
)

df_eval["sq_error"] = (df_eval["y_true"] - df_eval["y_pred"]) ** 2
df_eval

# %%
result = permutation_importance(model, X, y, n_repeats=50, random_state=RANDOM_SEED)

perm_importance = pd.DataFrame(
    {
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
).sort_values("importance_mean", ascending=False)

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
shap.plots.bar(explainer(X_shap), max_display=12)

# %%
clustering = shap.utils.hclust(X_shap, y)

# %%
shap.plots.bar(
    explainer(X_shap), clustering=clustering, clustering_cutoff=0.4, max_display=15
)

# %%
df_eval["residual"] = df_eval["y_true"] - df_eval["y_pred"]
df_eval["abs_residual"] = df_eval["residual"].abs()
typical_by_fit = df_eval.sort_values("abs_residual").index.tolist()

# Compute median feature vector
median_vec = X_shap.median(axis=0)

# Compute L1 distance from median for each child
dist_from_median = (X_shap - median_vec).abs().sum(axis=1)

df_eval["dist_from_median"] = dist_from_median

df_eval["representative_score"] = (
    df_eval["abs_residual"] / df_eval["abs_residual"].max()
    + df_eval["dist_from_median"] / df_eval["dist_from_median"].max()
)

# %%
best_idx = df_eval["representative_score"].idxmin()
pos_idx = df_eval.index.get_loc(best_idx)

explainer_2 = shap.Explainer(rf_fitted, X_shap)
explanation_2 = explainer(X_shap)

shap.plots.waterfall(explanation_2[pos_idx], max_display=12)

# %%
explanation_2

# %%
shap.summary_plot(shap_values, X_shap, plot_size=0.25)

# %%
shap.plots.scatter(explainer(X_shap)[:, "attend"])

# %%
shap.plots.scatter(explainer(X_shap)[:, "yarclet"])

# %%
shap.plots.scatter(explainer(X_shap)[:, "celf"])

# %%
X_fp = X.astype("float32").copy()

plot_features = [
    "age",
    "attend",
    "yarclet",
    "celf",
    "trog",
    "eowpvt",
    "b1exto",
    "blending",
    "time",
    "nonword",
    "b1reto",
    "b2reto",
    "aptinfo",
    "deappin",
    "aptgram",
    "rowpvt",
    "behav",
    "b2exto",
    "vision",
]

mean_imp = result.importances_mean  # shape: (n_features,)
std_imp = result.importances_std  # optional, if you want to print

feature_names = X.columns
imp_mean_dict = dict(zip(feature_names, mean_imp))

features_ordered = sorted(
    plot_features, key=lambda f: imp_mean_dict.get(f, 0.0), reverse=True
)

n_features = len(features_ordered)
n_cols = 3
n_rows = math.ceil(n_features / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

axes = axes.reshape(n_rows, n_cols)

# Loop and plot
for idx, feature in enumerate(features_ordered):
    row = idx // n_cols
    col = idx % n_cols
    ax = axes[row, col]

    pdp = partial_dependence(model, X_fp, [feature])
    grid = pdp["grid_values"][0]
    avg = pdp["average"][0]

    ax.plot(grid, avg)
    ax.set_title(f"Partial dependence of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Predicted reading gain")
    ax.set_ylim(1.0, 5.0)

# Hide empty subplots (if any)
for j in range(idx + 1, n_rows * n_cols):
    row = j // n_cols
    col = j % n_cols
    axes[row, col].axis("off")

# %%
shap_inter = explainer.shap_interaction_values(X_shap)

feature_names = X_shap.columns.tolist()

idx_yarclet = feature_names.index("yarclet")

yarclet_interactions = shap_inter[:, idx_yarclet, :]
interaction_strength = np.abs(yarclet_interactions).mean(axis=0)

yarclet_interactions_df = pd.DataFrame(
    {"feature": feature_names, "mean_abs_interaction": interaction_strength}
).sort_values("mean_abs_interaction", ascending=False)
yarclet_interactions_df

# %%
top_feature = yarclet_interactions_df.iloc[1]["feature"]  # skip 'yarclet' itself
shap.dependence_plot(("yarclet", top_feature), shap_inter, X_shap)

# %%
plt.scatter(ewrswr_gain_df["yarclet"], ewrswr_gain_df["ewrswr_gain"], alpha=0.3)
plt.xlabel("Letter sounds at t")
plt.ylabel("Gain in word reading t to t + 1")
plt.title("Gain in word reading t to t + 1, compared to letter–sound knowledge at t")

# %%
shap.dependence_plot(
    ("yarclet", "celf"), shap_inter, X_shap  # primary feature, interacting feature
)

# %%
shap.dependence_plot(
    ("yarclet", "age"), shap_inter, X_shap  # primary feature, interacting feature
)

# %%
feature_names = X_shap.columns.tolist()
idx_yarclet = feature_names.index("yarclet")
idx_celf = feature_names.index("celf")

# Extract the interaction term for YARCLET×CELF for each sample
interaction_yarclet_celf = shap_inter[:, idx_yarclet, idx_celf]

df_int = pd.DataFrame(
    {
        "yarclet": X_shap["yarclet"].values,
        "celf": X_shap["celf"].values,
        "interaction": interaction_yarclet_celf,
    }
)

# %%
# Choose grid resolution
n_bins_y = 25
n_bins_c = 25

y_bins = np.linspace(df_int["yarclet"].min(), df_int["yarclet"].max(), n_bins_y + 1)
c_bins = np.linspace(df_int["celf"].min(), df_int["celf"].max(), n_bins_c + 1)

# Digitise values to bins
y_idx = np.digitize(df_int["yarclet"], y_bins) - 1
c_idx = np.digitize(df_int["celf"], c_bins) - 1

# Initialise grid with NaNs
Z = np.full((n_bins_c, n_bins_y), np.nan)

# Aggregate mean interaction per bin
for i in range(n_bins_y):
    for j in range(n_bins_c):
        mask = (y_idx == i) & (c_idx == j)
        if mask.any():
            Z[j, i] = df_int.loc[mask, "interaction"].mean()

# Build grid coordinates at bin midpoints
Y_grid = 0.5 * (y_bins[:-1] + y_bins[1:])
C_grid = 0.5 * (c_bins[:-1] + c_bins[1:])
Y_mesh, C_mesh = np.meshgrid(Y_grid, C_grid)

# %%

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Mask NaNs so they don't create spikes/holes
Z_plot = np.ma.masked_invalid(Z)

ax.plot_surface(Y_mesh, C_mesh, Z_plot, linewidth=0, antialiased=True)

ax.set_xlabel("YARCLET (letter–sound)")
ax.set_ylabel("CELF (language)")
ax.set_zlabel("SHAP interaction (YARCLET×CELF)")
ax.set_title("3D surface of YARCLET × CELF interaction")

# %%
# Assuming:
# shap_inter = explainer.shap_interaction_values(X_shap)
# X_shap is your float64-filled DataFrame

feature_names = X_shap.columns.tolist()
idx_attend = feature_names.index("attend")
idx_celf = feature_names.index("celf")

interaction_attend_celf = shap_inter[:, idx_attend, idx_celf]

df_int = pd.DataFrame(
    {
        "attend": X_shap["attend"],
        "celf": X_shap["celf"],
        "interaction": interaction_attend_celf,
    }
)
df_int

# %%
import numpy as np
from scipy.interpolate import griddata

# Points and values
points = df_int[["attend", "celf"]].values
values = df_int["interaction"].values

# Regular grid over the observed range
att_min, att_max = df_int["attend"].min(), df_int["attend"].max()
celf_min, celf_max = df_int["celf"].min(), df_int["celf"].max()

att_grid = np.linspace(att_min, att_max, 60)
celf_grid = np.linspace(celf_min, celf_max, 60)
A_mesh, C_mesh = np.meshgrid(att_grid, celf_grid)

# Interpolate interaction values onto the grid
Z = griddata(points, values, (A_mesh, C_mesh), method="linear")

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 7))

Z_plot = np.ma.masked_invalid(Z)

contours = plt.contourf(A_mesh, C_mesh, Z_plot, levels=15, cmap="coolwarm")

plt.colorbar(contours, label="SHAP interaction (attend × celf)")
plt.xlabel("Attendance")
plt.ylabel("CELF")
plt.title("Contour plot of SHAP interaction between attend and celf")

# %%
idx_blending = feature_names.index("blending")

blending_interactions = shap_inter[:, idx_blending, :]
interaction_strength = np.abs(blending_interactions).mean(axis=0)

blending_interactions_df = pd.DataFrame(
    {"feature": feature_names, "mean_abs_interaction": interaction_strength}
).sort_values("mean_abs_interaction", ascending=False)
blending_interactions_df

# %%
shap.dependence_plot(
    ("blending", "celf"), shap_inter, X_shap  # primary feature, interacting feature
)
