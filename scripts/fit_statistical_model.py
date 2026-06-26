# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""CLI entry point for fitting the Bayesian statistical models.

Usage::

    python scripts/fit_statistical_model.py lrpitt07 --config dev
    python scripts/fit_statistical_model.py all --config dev
    python scripts/fit_statistical_model.py lrpitt10 --config reporting --render
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
    lrp52d,
    lrp56,
    lrp57,
    lrp58,
    lrp59,
    lrp62,
    lrp71,
    lrp72,
    lrp72base,
    lrp73,
    lrp73base,
    lrp77,
    lrp77a,
    lrp77base,
    lrpitt01,
    lrpitt02,
    lrpitt03,
    lrpitt04,
    lrpitt05,
    lrpitt06,
    lrpitt07,
    lrpitt08,
    lrpitt09,
    lrpitt10,
    lrpitt11,
    lrpitt12,
    lrpitt13,
    lrpitt13b,
    lrpitt14,
    lrpitt14b,
    lrpitt15,
    lrpitt15b,
    lrpitt17,
    lrpitt18,
    lrpitt19,
    lrpitt20,
    lrpitt21,
    lrpitt22,
    lrpitt23,
    lrpitt24,
    lrpdid01,
    lrpdid02,
    lrpdid03,
    lrpdid04,
    lrpdid05,
    lrpdid06,
)


MODELS = {
    # LRPITT01-11: the uniform DAG-faithful ITT suite (issue #119). One outcome
    # each, RCT phase; cross-baselines dropped (the ITT effect is identified by
    # the empty adjustment set), own baseline + linear age as precision terms.
    # P (09) and N (11) take the floor-rule branch (binary off-floor PRIMARY +
    # graded SECONDARY). Supersede LRP52 (W), LRP53 (R), LRP54 (E), LRP74 (TE),
    # LRP75 (TR), which have been deleted.
    "lrpitt01": lrpitt01,
    "lrpitt02": lrpitt02,
    "lrpitt03": lrpitt03,
    "lrpitt04": lrpitt04,
    "lrpitt05": lrpitt05,
    "lrpitt06": lrpitt06,
    "lrpitt07": lrpitt07,
    "lrpitt08": lrpitt08,
    "lrpitt09": lrpitt09,
    "lrpitt10": lrpitt10,
    "lrpitt11": lrpitt11,
    # LRPITT12-15: companions (issue #119). 12 = joint over the ten baseline-
    # bearing suite outcomes (migrates LRP55; N read from LRPITT11). 13/13b =
    # SES-adjusted W/L with 14/14b matched complete-case comparators (migrate
    # LRP60/60a). 15/15b = taught-vs-not-taught generalisation contrast,
    # expressive/receptive (migrates LRP76).
    "lrpitt12": lrpitt12,
    "lrpitt13": lrpitt13,
    "lrpitt13b": lrpitt13b,
    "lrpitt14": lrpitt14,
    "lrpitt14b": lrpitt14b,
    "lrpitt15": lrpitt15,
    "lrpitt15b": lrpitt15b,
    # LRPITT17-24: general-ability robustness companions. Each adds block design
    # (the baseline nonverbal-ability measure, t1-only) as a linear precision
    # covariate on top of the own baseline + linear age, asking whether the ITT
    # effect survives the immediate arm's mild (~0.27 SD) baseline-ability
    # head-start. Block design is complete, so no rows drop and no matched
    # comparator is needed: each is a same-sample adjusted-vs-unadjusted contrast
    # with its base model. Ordered by the suite reference order; covers the
    # vocabulary family (TR/TE/UR/UE/R/E) and the two reading anchors (L, W).
    # (16 reserved for the deferred descriptive trajectory.)
    "lrpitt17": lrpitt17,  # TR taught-receptive  (base lrpitt01)
    "lrpitt18": lrpitt18,  # TE taught-expressive (base lrpitt02)
    "lrpitt19": lrpitt19,  # UR not-taught-recept (base lrpitt03)
    "lrpitt20": lrpitt20,  # UE not-taught-expr   (base lrpitt04)
    "lrpitt21": lrpitt21,  # R  receptive-vocab   (base lrpitt05)
    "lrpitt22": lrpitt22,  # E  expressive-vocab  (base lrpitt06)
    "lrpitt23": lrpitt23,  # L  letter-sounds     (base lrpitt07)
    "lrpitt24": lrpitt24,  # W  word-reading      (base lrpitt10)
    # LRPDID01-06: waitlist-crossover / difference-in-differences family (new
    # 'did' kind). Within-person replication of the randomised ITT effect using
    # the waitlist arm's P1 (untreated) vs P2 (crossover) periods, each child its
    # own control, the immediate arm anchoring the time trend. Beta-Binomial logit
    # so the ceiling is respected. 01 W, 02 L, 03 B, 04 TE, 05 R; 06 = W
    # dose-response (sessions) sensitivity variant of 01.
    "lrpdid01": lrpdid01,  # W  word-reading      (vs lrpitt10)
    "lrpdid02": lrpdid02,  # L  letter-sounds     (vs lrpitt07)
    "lrpdid03": lrpdid03,  # B  blending          (vs lrpitt08)
    "lrpdid04": lrpdid04,  # TE taught-expressive (vs lrpitt02)
    "lrpdid05": lrpdid05,  # R  receptive-vocab   (vs lrpitt05)
    "lrpdid06": lrpdid06,  # W  dose-response     (variant of lrpdid01)
    "lrp56": lrp56,
    "lrp57": lrp57,
    "lrp58": lrp58,
    # LRP59: ITT-phase mediation (does G raise W via L?). New 'mediation' family.
    "lrp59": lrp59,
    # LRP62: reading-route decomposition (phonics-route composite mediation).
    "lrp62": lrp62,
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
    # LRP77 (#104 Phase 2): period-resolved dose-response on word reading.
    # lrp77base is the pooled-dose comparator (LOO); lrp77a the ability-adjusted
    # sensitivity fit (no-g->dose assumption).
    "lrp77": lrp77,
    "lrp77base": lrp77base,
    "lrp77a": lrp77a,
    # LRP52d: period-1 ITT (W) entering group + dose - dose-absorbs-group restatement.
    "lrp52d": lrp52d,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help=(
            "Model id (lrpitt01..lrpitt15b, lrp56..lrp59, lrp62, lrp71, lrp72, "
            "lrp73) or 'all'"
        ),
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
