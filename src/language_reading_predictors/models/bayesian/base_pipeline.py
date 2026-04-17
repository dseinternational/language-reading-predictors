# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Base class for Bayesian modelling pipelines.

Mirrors the ``EstimatorPipeline`` pattern used by the tree-model family:
all generic steps live on the base class; concrete subclasses override
:meth:`build_model` to declare the PyMC model for a particular
DAG / likelihood combination.

Pipeline steps operate on ``self.context`` (a :class:`BayesianFitContext`
defined below) so any single step can be called from a notebook for
debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from rich import print

import language_reading_predictors.data_utils as data_utils
from language_reading_predictors.models.bayesian.definition import (
    BayesianDAGSpec,
    NodeSpec,
    ParentSpec,
)
from language_reading_predictors.models.bayesian.hsgp import standardise
from language_reading_predictors.models.common import ModelConfig, RunConfig

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
_DOCS_DIR = _ROOT_DIR / "docs"
_OUTPUT_DIR = _ROOT_DIR / "output" / "models"


def _section(title: str) -> None:
    print(
        "\n[green]------------------------------------------------------------[/green]"
    )
    print(f"[bold green]{title}[/bold green]")
    print(
        "[green]------------------------------------------------------------[/green]"
    )


def _code_binary(raw: np.ndarray, label: str) -> np.ndarray:
    """Recode a binary column to 0/1.

    Accepts already-0/1-coded data unchanged. For 2-level columns with
    any other coding (e.g. the ``group`` column's 1/2 convention), the
    lowest observed value maps to 0 and the highest to 1.
    """
    finite = raw[~np.isnan(raw)]
    uniq = sorted({float(v) for v in finite})
    if uniq == [0.0, 1.0]:
        return raw.astype(float)
    if len(uniq) == 2:
        lo, hi = uniq
        return np.where(raw == hi, 1.0, np.where(raw == lo, 0.0, np.nan))
    msg = (
        f"Binary parent {label!r} has {len(uniq)} unique values {uniq}; "
        "expected exactly 2."
    )
    raise ValueError(msg)


def _presets(run_config: RunConfig) -> dict[str, Any]:
    """Return (draws, tune, chains, target_accept) for this run config.

    Defaults kick in when the :class:`RunConfig` did not override them —
    e.g. a reporting run used for a tree model and then re-used here.
    """
    defaults = {
        "dev": dict(draws=500, tune=500, chains=2, target_accept=0.90),
        "test": dict(draws=1000, tune=1000, chains=4, target_accept=0.95),
        "reporting": dict(draws=2000, tune=2000, chains=4, target_accept=0.99),
    }
    base = defaults.get(run_config.name, defaults["dev"])
    return {
        "draws": run_config.draws if run_config.draws is not None else base["draws"],
        "tune": run_config.tune if run_config.tune is not None else base["tune"],
        "chains": (
            run_config.chains if run_config.chains is not None else base["chains"]
        ),
        "target_accept": (
            run_config.target_accept
            if run_config.target_accept is not None
            else base["target_accept"]
        ),
    }


@dataclass
class ParentArrays:
    """Per-parent standardised input arrays and their standardisation params."""

    spec: ParentSpec
    raw: np.ndarray
    std: np.ndarray
    mean: float
    sd: float


@dataclass
class NodeArrays:
    """Per-node outcome and parent arrays after preparation."""

    spec: NodeSpec
    outcome: np.ndarray
    parents: dict[str, ParentArrays] = field(default_factory=dict)


@dataclass
class BayesianFitContext:
    """Shared state passed through every pipeline step."""

    config: ModelConfig
    run_config: RunConfig
    output_dir: Path
    dag_spec: BayesianDAGSpec

    df: pd.DataFrame | None = None
    """Filtered input data (one row per subject for the t1 → t2 scope)."""

    nodes: dict[str, NodeArrays] = field(default_factory=dict)
    """Per-node prepared arrays, keyed by node name."""

    pm_model: pm.Model | None = None
    trace: az.InferenceData | None = None
    posterior_predictive: az.InferenceData | None = None
    summary: pd.DataFrame | None = None


class BayesianPipeline:
    """Base class for Bayesian modelling pipelines.

    Subclasses override :meth:`build_model` to declare the PyMC model for
    a particular DAG / likelihood combination.
    """

    def __init__(self, config: ModelConfig, run_config: RunConfig) -> None:
        if config.dag_spec is None:
            msg = (
                f"Model {config.model_id!r} has no dag_spec set; "
                "BayesianPipeline cannot run without one."
            )
            raise ValueError(msg)
        self.context = BayesianFitContext(
            config=config,
            run_config=run_config,
            output_dir=_OUTPUT_DIR / config.model_id,
            dag_spec=config.dag_spec,
        )

    # ── pipeline steps ───────────────────────────────────────────────────

    def prepare_data(self) -> None:
        """Load data, filter to the DAG scope, and build per-node arrays."""
        _section("Prepare data")

        ctx = self.context
        spec = ctx.dag_spec

        df = data_utils.load_data()

        if spec.time_filter is not None and spec.time_column in df.columns:
            df = df[df[spec.time_column] == spec.time_filter].copy()

        required = [c for c in spec.required_columns() if c in df.columns]
        missing_cols = [c for c in spec.required_columns() if c not in df.columns]
        if missing_cols:
            msg = (
                f"Columns required by the DAG spec are missing from the "
                f"input data: {missing_cols}"
            )
            raise ValueError(msg)

        before = len(df)
        df = df.dropna(subset=required).copy()
        dropped = before - len(df)
        if dropped:
            print(
                f"  Dropped {dropped} rows with missing values "
                f"(complete-case analysis on {len(required)} columns)."
            )
        print(f"  Observations after filtering: {len(df)}")

        df = df.reset_index(drop=True)
        ctx.df = df

        for node in spec.nodes:
            y = df[node.outcome_column].to_numpy(dtype=float)
            if node.likelihood == "beta_binomial":
                y = np.clip(np.round(y), 0, node.n_trials).astype(int)
            parents: dict[str, ParentArrays] = {}
            for parent in node.parents:
                raw = df[parent.column].to_numpy(dtype=float)
                if parent.kind == "binary":
                    raw = _code_binary(raw, parent.column)
                    z, mu, sd = raw.copy(), 0.0, 1.0
                else:
                    z, mu, sd = standardise(raw)
                parents[parent.name] = ParentArrays(
                    spec=parent, raw=raw, std=z, mean=mu, sd=sd
                )
            ctx.nodes[node.name] = NodeArrays(
                spec=node, outcome=y, parents=parents
            )
            print(
                f"  Node {node.name}: outcome={node.outcome_column} "
                f"(likelihood={node.likelihood}, "
                f"{'N=' + str(node.n_trials) if node.n_trials else 'continuous'}) "
                f"parents={[p.name + ':' + p.kind for p in node.parents]}"
            )

    def build_model(self) -> None:
        """Subclass hook — must populate ``self.context.pm_model``."""
        msg = "Subclasses must implement build_model()."
        raise NotImplementedError(msg)

    def sample(self) -> None:
        _section("Sample posterior")
        ctx = self.context
        settings = _presets(ctx.run_config)
        print(
            f"  draws={settings['draws']} tune={settings['tune']} "
            f"chains={settings['chains']} target_accept={settings['target_accept']}"
        )
        with ctx.pm_model:
            ctx.trace = pm.sample(
                draws=settings["draws"],
                tune=settings["tune"],
                chains=settings["chains"],
                target_accept=settings["target_accept"],
                random_seed=ctx.config.random_seed,
                progressbar=True,
                return_inferencedata=True,
            )
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        ctx.trace.to_netcdf(str(ctx.output_dir / "trace.nc"))
        print(f"  Saved trace: {ctx.output_dir / 'trace.nc'}")

    def run_diagnostics(self) -> None:
        _section("Diagnostics")
        ctx = self.context
        summary = az.summary(ctx.trace, hdi_prob=0.9)
        ctx.summary = summary
        csv_path = ctx.output_dir / "posterior_summary.csv"
        summary.to_csv(csv_path)
        print(f"  Wrote {csv_path}")

        # Divergences / energy / rhat extremes
        n_divergent = int(
            ctx.trace.sample_stats.get("diverging", None).sum().item()
            if "diverging" in ctx.trace.sample_stats
            else 0
        )
        rhat_max = float(summary["r_hat"].max())
        ess_min = float(summary[["ess_bulk", "ess_tail"]].min().min())
        diag = {
            "divergences": n_divergent,
            "r_hat_max": rhat_max,
            "ess_min": ess_min,
            "n_parameters": int(len(summary)),
        }
        with open(ctx.output_dir / "diagnostics.json", "w") as fp:
            json.dump(diag, fp, indent=2)
        print(
            f"  divergences={n_divergent}  r_hat_max={rhat_max:.3f}  "
            f"ess_min={ess_min:.0f}"
        )

    def posterior_predictive(self) -> None:
        _section("Posterior predictive")
        ctx = self.context
        with ctx.pm_model:
            pp = pm.sample_posterior_predictive(
                ctx.trace,
                random_seed=ctx.config.random_seed,
                progressbar=False,
            )
        ctx.posterior_predictive = pp
        pp.to_netcdf(str(ctx.output_dir / "posterior_predictive.nc"))

    def plot_traces(self) -> None:
        _section("Plot traces")
        import matplotlib.pyplot as plt

        ctx = self.context
        # Focus trace plot on intercepts and linear coefficients (the
        # interpretable fixed effects). GP unit vectors are skipped to
        # keep the plot legible.
        interesting = [
            v
            for v in ctx.summary.index
            if v.startswith(("b0__", "beta__", "sigma__", "eta__", "ell__"))
        ]
        if not interesting:
            interesting = list(ctx.summary.index[:20])
        az.plot_trace(ctx.trace, var_names=interesting, compact=True)
        fig = plt.gcf()
        fig.tight_layout()
        out = ctx.output_dir / "trace.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  Wrote {out}")

        # Energy plot for NUTS health
        az.plot_energy(ctx.trace)
        fig = plt.gcf()
        fig.tight_layout()
        out = ctx.output_dir / "energy.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  Wrote {out}")

    def plot_posterior_predictive(self) -> None:
        _section("Plot posterior predictive")
        import matplotlib.pyplot as plt

        ctx = self.context
        if ctx.posterior_predictive is None:
            return

        for node_name, node in ctx.nodes.items():
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                az.plot_ppc(
                    ctx.posterior_predictive,
                    var_names=[f"y__{node_name}"],
                    num_pp_samples=50,
                    observed=True,
                    ax=ax,
                )
                ax.set_title(
                    f"Node {node_name} — posterior predictive vs observed"
                )
                fig.tight_layout()
                out = ctx.output_dir / f"ppc_{node_name}.png"
                fig.savefig(out, dpi=110)
                plt.close(fig)
                print(f"  Wrote {out}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [yellow]Skipping PPC for node {node_name}: "
                    f"{exc}[/yellow]"
                )

    def save_config(self) -> None:
        _section("Save config")
        ctx = self.context
        cfg = ctx.config
        settings = _presets(ctx.run_config)

        # Parent standardisation params for reproducibility
        standardisations = {
            node.spec.name: {
                p.spec.name: {"mean": p.mean, "sd": p.sd}
                for p in node.parents.values()
            }
            for node in ctx.nodes.values()
        }

        out = {
            "model_id": cfg.model_id,
            "description": cfg.description,
            "pipeline_cls": type(self).__name__,
            "variant_of": cfg.variant_of,
            "notes": cfg.notes,
            "random_seed": cfg.random_seed,
            "run_config": {
                "name": ctx.run_config.name,
                **settings,
            },
            "n_observations": int(len(ctx.df)) if ctx.df is not None else 0,
            "dag": ctx.dag_spec.to_json_dict(),
            "standardisations": standardisations,
        }
        with open(ctx.output_dir / "config.json", "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"  Wrote {ctx.output_dir / 'config.json'}")

    def report(self) -> None:
        """Copy the per-model Quarto template into ``output_dir``."""
        _section("Copy Quarto template")
        ctx = self.context
        import shutil

        cfg = ctx.config
        candidates = [_DOCS_DIR / "models" / cfg.model_id / "index.qmd"]
        if cfg.variant_of:
            candidates.append(_DOCS_DIR / "models" / cfg.variant_of / "index.qmd")

        for src in candidates:
            if src.exists():
                dst = ctx.output_dir / "index.qmd"
                shutil.copyfile(src, dst)
                print(f"  Copied {src} -> {dst}")
                return
        print(
            f"  [yellow]No Quarto template found for {cfg.model_id}; "
            "skipping.[/yellow]"
        )

    # ── orchestrator ─────────────────────────────────────────────────────

    def fit(self) -> BayesianFitContext:
        ctx = self.context
        print(
            f"\n[green]{'=' * 60}[/green]\n"
            f"[bold green]{ctx.config.model_id.upper()}: "
            f"{ctx.config.description}[/bold green]\n"
            f"[green]{'=' * 60}[/green]"
        )
        self.prepare_data()
        self.build_model()
        self.sample()
        self.run_diagnostics()
        self.posterior_predictive()
        self.plot_traces()
        self.plot_posterior_predictive()
        self.save_config()
        self.report()
        print(
            f"\n[green]{'=' * 60}[/green]\n"
            f"[bold green]{ctx.output_dir}[/bold green]\n"
            f"[green]{'=' * 60}[/green]"
        )
        return ctx
