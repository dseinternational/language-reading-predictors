# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI entry point for fitting the Bayesian statistical models (LRP52-LRP60).

Usage::

    python scripts/fit_statistical_model.py lrp52 --config dev
    python scripts/fit_statistical_model.py all --config dev
    python scripts/fit_statistical_model.py lrp52 --config reporting --render
"""

from __future__ import annotations

import argparse
import os
import subprocess
import uuid
from multiprocessing import freeze_support

from rich import print as rprint
from rich.panel import Panel

from language_reading_predictors.models._reporting import (
    metrics_table,
    print_panel,
    print_table,
)
from language_reading_predictors.storage import upload_to_blob_storage
from language_reading_predictors.statistical_models import (
    lrp52,
    lrp53,
    lrp54,
    lrp55,
    lrp56,
    lrp57,
    lrp58,
    lrp59,
    lrp60,
    lrp60a,
    lrp62,
    lrp67,
    lrp68,
    lrp71,
    lrp72,
    lrp72base,
    lrp73,
    lrp73base,
)


MODELS = {
    "lrp52": lrp52,
    "lrp53": lrp53,
    "lrp54": lrp54,
    "lrp55": lrp55,
    "lrp56": lrp56,
    "lrp57": lrp57,
    "lrp58": lrp58,
    # LRP59: ITT-phase mediation (does G raise W via L?). New 'mediation' family.
    "lrp59": lrp59,
    "lrp60": lrp60,
    # LRP60a: matched complete-case comparator to LRP60 (unadjusted, SES subset).
    "lrp60a": lrp60a,
    # LRP62: reading-route decomposition (phonics-route composite mediation).
    "lrp62": lrp62,
    # LRP67: latent change-score model — within-child predictors of reading
    # change (the longitudinal extension of LRP65's between-child story).
    "lrp67": lrp67,
    # LRP68: constrained RI-CLPM — within-child cross-lagged comparison
    # (AR-only vs L→R vs reverse vs reciprocal) by LOO.
    "lrp68": lrp68,
    # LRP70 (celf moderator) is reserved but deferred pending a DAG review of
    # conditioning on a descendant of L. LRP71 (eowpvt) is the first built
    # interaction model.
    "lrp71": lrp71,
    # LRP72: phonics route (letter-sound × blending -> decoding). lrp72base is
    # its no-interaction companion for the PSIS-LOO comparison.
    "lrp72": lrp72,
    "lrp72base": lrp72base,
    # LRP73: age-moderated letter-sound -> word reading. lrp73base is its
    # no-interaction companion.
    "lrp73": lrp73,
    "lrp73base": lrp73base,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="Model id (lrp52..lrp60, lrp60a, lrp62, lrp71, lrp72, lrp73) or 'all'",
    )
    parser.add_argument(
        "--config",
        default="dev",
        help="Sampling configuration: dev, test, reporting (see dse_research_utils.sampling)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Quarto report after fitting",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=None,
        help="Override NUTS target_accept (default: preset from --config)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload model output to Azure Blob Storage after fitting.",
    )
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include trace files (.nc) in the upload (excluded by default).",
    )
    args = parser.parse_args()

    if args.target_accept is not None:
        import dse_research_utils.statistics.models.sampling as _S

        _orig = _S.get_sampling_configuration

        def _override(cfg: str = "dev", random_seed: int = 47):
            s = _orig(cfg, random_seed=random_seed)
            s.target_accept = args.target_accept
            return s

        _S.get_sampling_configuration = _override
        rprint(
            f"[yellow]Overriding target_accept -> {args.target_accept}[/yellow]"
        )

    if args.model == "all":
        to_fit = list(MODELS.items())
    elif args.model in MODELS:
        to_fit = [(args.model, MODELS[args.model])]
    else:
        rprint(f"[red]Unknown model: {args.model}[/red]")
        rprint(f"[yellow]Available: {', '.join(MODELS)}[/yellow]")
        raise SystemExit(1)

    rprint()
    print_panel(
        Panel(
            "[bold]Statistical model run[/bold]\n\n"
            f"[dim]Models:[/dim]     {', '.join(mid.upper() for mid, _ in to_fit)}\n"
            f"[dim]Run config:[/dim] {args.config}"
            + (
                f"\n[dim]Override:[/dim]   target_accept={args.target_accept}"
                if args.target_accept is not None
                else ""
            ),
            border_style="green",
            padding=(1, 2),
        )
    )

    contexts: list = []
    failed: list[tuple[str, Exception]] = []
    for model_id, module in to_fit:
        try:
            contexts.append(module.fit(args.config))
        except Exception as exc:
            failed.append((model_id, exc))
            rprint(f"[bold red]Error fitting {model_id}: {exc}[/bold red]")

    if len(to_fit) > 1 or failed:
        rows = []
        for ctx in contexts:
            spec = ctx.spec
            prepared = ctx.prepared
            rows.append(
                {
                    "model": spec.model_id.upper(),
                    "kind": spec.kind,
                    "outcome": spec.outcome_symbol or "-",
                    "mechanism": spec.mechanism_symbol or "-",
                    "n_obs": int(prepared.n_obs) if prepared is not None else None,
                    "loo_elpd": float(ctx.loo.elpd) if ctx.loo is not None else None,
                    "status": "ok",
                }
            )
        for model_id, exc in failed:
            rows.append(
                {
                    "model": model_id.upper(),
                    "kind": "-",
                    "outcome": "-",
                    "mechanism": "-",
                    "n_obs": None,
                    "loo_elpd": None,
                    "status": f"FAILED: {type(exc).__name__}",
                }
            )
        rprint()
        print_table(
            metrics_table(
                rows,
                title=f"Summary ({len(contexts)} fitted, {len(failed)} failed)",
                columns=["model", "kind", "outcome", "mechanism", "n_obs", "loo_elpd", "status"],
            )
        )

    if args.render:
        for ctx in contexts:
            qmd = os.path.join(ctx.output_dir, "index.qmd")
            if os.path.exists(qmd):
                rprint(f"[bold green]quarto render {qmd}[/bold green]")
                subprocess.run(["quarto", "render", qmd], check=False)

    if args.upload:
        run_id = str(uuid.uuid7())
        for ctx in contexts:
            upload_to_blob_storage(
                str(ctx.output_dir),
                os.path.basename(str(ctx.output_dir)),
                include_traces=args.include_traces,
                run_id=run_id,
            )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    freeze_support()
    main()
