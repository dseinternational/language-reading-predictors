# Survival family: tau estimand relabel and child-level LOO (#631 findings 11 and 14)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).

Date: 2026-08-26. Scope: `lrp-rli-surv-009` / `lrp-rli-surv-011` (the `treatment_window="randomised"` default). The pooled comparator (`treatment_window="pooled"`) is unchanged and remains a prognostic, prior-mediated association.

## Decision 1 — what `tau` is called (#631 finding 11)

The randomised-window `tau` was described everywhere as "a prognostic association, not a causal effect of record". That undersells what the model actually estimates and is inconsistent with the suite's own terminology: the same first-window quantity in the sibling ITT floor rules (`lrp-rli-itt-009`/`011`) is labelled an available-case modified ITT floor-rule estimate.

`tau` is now described as a **model-based, available-case modified-ITT randomised-window assignment contrast in the baseline at-floor subgroup**: the covariate-adjusted immediate-versus-waitlist off-floor hazard contrast in the randomised first interval (t1 to t2) among children at the outcome floor at t1. Randomisation anchors the first-interval arm comparison; the estimate is qualified by (a) the pre-randomisation at-floor subgroup restriction, (b) available-case selection — a child without an observed wave-2 outcome contributes no first-interval row, and nothing in the design repairs that selection, (c) mean-imputed baseline covariates, and (d) the discrete-time hazard-model form. None of these qualifications is lifted by the relabel, and the family's release policy is unchanged: `Status.ASSOCIATION`, no causal headline.

Surfaces updated: `survival.py` (module docstring, resolver `estimand`/`causal_status`, recipe, factory comment), `lrp_rli_surv_009.py`/`011.py`, `pipelines/survival.py` (docstrings and the `tau_reading` table caption), `reporting.py` (`_kf_build_survival` randomised branch), `prior_artifacts.py` (tau role rationale), `docs/models/lrp-rli-surv-009|011/index.qmd`, `docs/models/_partials/_results_survival.qmd`, `docs/models/README.md`, and the `lrp-fit-statistical` skill table (which had additionally gone stale by describing the retired pooled default). Earlier dated notes are deliberately not retro-edited; this note supersedes their "prognostic association" wording for the randomised window.

## Decision 2 — PSIS-LOO leaves out children, not person-period rows (#631 finding 14)

The person-period expansion emits row *k+1* only when the interval-*k* event was zero, so holding out one row while retaining the child's later rows conditions on information about the held-out event — the same leakage argument that moved `dose_response` to the child unit (#587 finding 4). The survival factory now persists `loo_child_idx` and the shared diagnostics aggregate the pointwise log likelihood within child before PSIS-LOO (`_joint_log_likelihood_by_child`, generalised to accept the `y_event` node). Because the model deliberately carries no child frailty, the child-summed LOO is an exact marginal leave-one-child-out. `resolved_run_plan.loo_unit` is now `"child"`.

Stored SURV-009/011 reporting fits predate both changes: their LOO tables use the leaking row unit and their recipes carry the old labels, so both fits must be refitted (or at minimum re-diagnosed from the stored traces) before republication.
