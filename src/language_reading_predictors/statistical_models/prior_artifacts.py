# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior-side artefacts: the prior panel, ``priors_table.csv`` and pushforwards.

Two responsibilities, both prior-side and shared by every family: the pruned
prior panel plus the per-parameter ``priors_table.csv`` (with the constructor /
role / rationale overrides that keep the table honest about what a model
actually registered), and the estimand-scale prior pushforward rows that answer
"what does this prior imply on the scale the reader cares about". Split out of
``pipeline.py`` for #394.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from rich import print as rprint

from language_reading_predictors.models._reporting import (
    print_table,
    ranked_dataframe_table,
)
from language_reading_predictors.statistical_models import (
    priors as _priors,
    reporting as _report,
)
from language_reading_predictors.statistical_models.artifacts import save_table
from language_reading_predictors.statistical_models.context import (
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.measures import is_distal


def emit_priors(context: StatisticalFitContext) -> None:
    """Write the pruned prior panel + ``priors_table.csv`` (issue #125 Area 1).

    Only the priors the model actually registered are panelled (no more 4–6 dead
    panels per model), and ``priors_table.csv`` documents every parameter's
    distribution, role (causal / precision / association / nuisance / GP) and
    rationale, driven by the built model so it cannot drift from the source.
    """
    model = context.model
    # Clear stale prior-PDF panels from a previous run so only the used set
    # remains (one file per named prior; not the prior-predictive / overlay PNGs).
    for key in _priors.ALL_PRIORS:
        for ext in ("png", "svg"):
            stale = os.path.join(context.output_dir, f"prior_{key}.{ext}")
            try:
                os.remove(stale)
            except OSError:
                pass
    ctor_overrides, role_overrides, rationale_overrides = _prior_table_overrides(context)
    _priors.save_shared_prior_panel(
        context.output_dir,
        used=_priors.used_prior_keys(model, ctor_overrides=ctor_overrides),
    )
    table = _priors.priors_table(
        model,
        ctor_overrides=ctor_overrides,
        role_overrides=role_overrides,
        rationale_overrides=rationale_overrides,
    )
    save_table(context, "priors_table", table)


def _prior_table_overrides(
    context: StatisticalFitContext,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Context-specific prior-table corrections for reused RV names.

    Some factories reuse a PyMC variable name with a different prior constructor
    or a different reporting role. Keep the model code stable and teach the
    artifact writer about those contextual meanings here.
    """
    ctor: dict[str, str] = {}
    role: dict[str, str] = {}
    rationale: dict[str, str] = {}
    spec = context.spec

    if spec.kind == "dose_response":
        ctor.update(
            {
                "beta_dose": "beta_mech",
                "beta_dose_phase": "beta_mech",
                "mu_dose": "beta_mech",
                "sigma_dose": "sigma_dose",
            }
        )
        role.update(
            {
                "beta_dose": "association",
                "beta_dose_phase": "association",
                "mu_dose": "association",
                "sigma_dose": "nuisance",
                "beta_G": "association",
            }
        )
        # role is demoted above, but without a rationale override each RV would
        # inherit its reused constructor's docstring ("Treatment effect tau…" for
        # beta_G, "Linear-mechanism slope beta_mech…" for the dose slopes).
        rationale.update(
            {
                "beta_G": (
                    "Intervention-arm (G) backdoor adjustment: the confounder of the "
                    "dose->outcome edge; an adjusted association, not the randomised "
                    "treatment effect."
                ),
                "mu_dose": (
                    "Average (pooled) per-period dose-response slope; outcome-logit "
                    "change per 1 SD of per-period dose — the model's focal "
                    "adjusted-association estimand."
                ),
                "beta_dose_phase": (
                    "Partial-pooled per-period dose-response slopes; each period's "
                    "outcome-logit change per 1 SD of dose, an adjusted association."
                ),
                "beta_dose": (
                    "Single pooled dose-response slope (no period variation); the "
                    "comparator's focal adjusted-association estimand, not a mechanism "
                    "slope."
                ),
            }
        )
    elif spec.kind == "gain_factors" and spec.extra.get("moderation_variant", False):
        # Moderation variants (#391 finding 3): beta_trt keeps the tau-tier prior
        # but is never presented as causal — its interaction-aware marginal is
        # model-dependent (the trt interactions are estimated on all stacked
        # periods, partly post-crossover). The causal headline lives in the
        # interaction-free primary; every artefact of a variant fit must agree.
        role["beta_trt"] = "association"
        rationale["beta_trt"] = (
            "On-intervention log-odds contrast inside an explicitly associational "
            "moderation variant: netted with the fitted treatment interactions it "
            "is a model-dependent association, partly informed by post-crossover "
            "data — read the randomised causal headline from the interaction-free "
            "primary model."
        )
    elif spec.kind == "did":
        # Time offsets and every post-crossover term are associations.  Only the
        # saturated arm-by-wave model's t2 arm gap is licensed by randomisation.
        role["beta_period"] = "association"
        role["arm_gap_t1"] = "association"
        role["arm_gap_t3"] = "association"
        role["delta_crossover"] = "association"
        rationale["beta_period"] = (
            "Wave/period offset; an age, maturation and treatment-history association, "
            "not a randomised treatment effect."
        )
        rationale["arm_gap_t1"] = (
            "Pre-randomisation immediate-minus-waitlist balance quantity; regularised "
            "as an association, not interpreted as an effect."
        )
        rationale["tau_t2"] = (
            "Immediate-minus-waitlist t2 contrast identified by the original "
            "randomisation; the only causal coefficient in the binary crossover model."
        )
        rationale["arm_gap_t3"] = (
            "Post-crossover immediate-minus-waitlist t3 association comparing different "
            "treatment histories (approximately 40 versus 20 weeks)."
        )
        rationale["sigma_delta"] = (
            "Exploratory between-waitlist-child SD of unexplained t3 catch-up; may mix "
            "response, maturation, history, period shocks and measurement variation."
        )
        if not spec.extra.get("dose", False):
            role["tau_t2"] = "causal"
            if spec.extra.get("use_intercept_anchor", True):
                role["alpha_offset"] = "nuisance"
                # The empirical-Bayes sentence comes from ``priors`` rather than
                # being written again here, so the family prose and the suite-wide
                # label cannot drift (#390 P1, Frank's 2026-07-24 ruling, condition
                # 2). Scoped to the *anchored* arm-by-wave models, which is correct
                # rather than incidental: the dose variants and the LRPDID101
                # independent-prior companion build an ordinary free
                # ``alpha ~ Normal(0, 1.5)`` and have no anchor to label.
                rationale["alpha_offset"] = (
                    "Zero-centred offset around the pooled observed t1 logit anchor; "
                    "the deterministic alpha is the anchored t1 level. "
                    f"{_priors.EMPIRICAL_BAYES_SENTENCE}"
                )
        if spec.extra.get("dose", False):
            role["beta_group"] = "association"
            role["theta_treated"] = "association"
            role["gamma_t1"] = "precision"
            rationale["beta_group"] = (
                "Randomised-arm and prior-treatment-history adjustment in the transition "
                "dose model; not itself the t2 randomised arm contrast."
            )
            rationale["theta_treated"] = (
                "Modelled current-treatment presence at the mean treated dose; a "
                "crossover association, not a second randomised ITT effect."
            )
            rationale["gamma_t1"] = (
                "Shared pre-randomisation t1 outcome precision term broadcast to both "
                "period rows; never the treatment-affected t2 period-start score."
            )
            rationale["beta_dose"] = (
                "Observational intensive-margin association per treated-row SD of raw "
                "sessions, with untreated rows coded at zero intensity."
            )
            rationale["mu_dose"] = (
                "Average observational intensive-margin session association across P1/P2."
            )
            rationale["beta_dose_phase"] = (
                "Partial-pooled observational intensive-margin session associations by period."
            )
            # Dose slopes now share build_dose_response_model's ``beta_mech`` prior
            # (Normal(0, 1)) so the shared summary compares like with like.
            if spec.extra.get("period_varying_dose", False):
                ctor.update(
                    {
                        "mu_dose": "beta_mech",
                        "beta_dose_phase": "beta_mech",
                        "sigma_dose": "sigma_dose",
                    }
                )
                role.update(
                    {
                        "mu_dose": "association",
                        "beta_dose_phase": "association",
                        "sigma_dose": "nuisance",
                    }
                )
            else:
                ctor["beta_dose"] = "beta_mech"
                role["beta_dose"] = "association"
    elif spec.kind in ("mediation", "mediation_multi"):
        # The mediation coefficients ``a_G`` (group→mediator) and ``b_G``
        # (group→outcome direct path) reuse the ``tau`` constructor's scale but
        # are structural building blocks of the g-formula, not the reported
        # estimand: the NDE/NIE come from the counterfactual simulation
        # (``mediation_summary.csv``), never a raw coefficient. Label them adjusted
        # associations so the prior table does not imply a bare coefficient is the
        # reported quantity. The simulated NDE/NIE are **not** causal either: they
        # are not identified natural effects (latent GA confounds the
        # mediator->outcome path, and dose ``IS`` is a treatment-induced
        # mediator-outcome confounder). See the :mod:`mediation` module docstring.
        role["a_G"] = "association"
        role["b_G"] = "association"
        rationale["a_G"] = (
            "Group->mediator (a-path) coefficient (tau-scaled Normal(0, 0.5)); a "
            "structural g-formula building block, an adjusted association, not the "
            "reported estimand."
        )
        rationale["b_G"] = (
            "Group->outcome direct-path (c') coefficient (tau-scaled Normal(0, 0.5)); "
            "a structural g-formula building block, an adjusted association, not an "
            "identified natural effect and not the reported estimand."
        )
        # B3 (review 2026-07-13; generalised #384). A confounder coefficient in the
        # a-/b-legs is built from gamma_cross_prior (Normal(0, 0.3)); a genuine
        # mediator b-path is b_path (Normal(0, 1)) and an own-baseline autoregression
        # is gamma_own (Normal(1, 0.25)). Reused names are ctor-mapped by NAME to the
        # wrong panel — b_E/b_B are globally mapped to b_path (mediators) yet are
        # confounders in LRP66/75; a_L is mapped to gamma_own (own-baseline) yet is a
        # cross-baseline confounder in LRP68/80 where the own-baseline is a_TE/a_TR —
        # so the rationale + panel misreport them (the distribution column, read off
        # the RV, stays correct). Detect confounders by their fitted scale and route
        # to gamma_cross for BOTH kinds. a_G/b_G (tau, 0.5) and the reported b_M
        # (b_path, 1.0) never match Normal(0, 0.3), so their explicit labels stand.
        if context.model is not None:
            for rv in context.model.free_RVs:
                # Per-mediator group->mediator a-paths in the two-mediator model are
                # named a{sym}_G (aL_G / aE_G / aB_G) rather than a_G; they are the
                # tau-scaled a-paths and otherwise carry an empty rationale.
                if (
                    rv.name != "a_G"
                    and rv.name.startswith("a")
                    and rv.name.endswith("_G")
                ):
                    rationale.setdefault(
                        rv.name,
                        "Group->mediator (a-path) coefficient for one mediator "
                        "(tau-scaled Normal(0, 0.5)); a structural g-formula building "
                        "block, an adjusted association, not the reported estimand.",
                    )
                    continue
                if rv.name in ("a_G", "b_G"):
                    continue
                if not (rv.name.startswith("a_") or rv.name.startswith("b_")):
                    continue
                dist = (_priors._dist_from_rv(rv) or "").replace(" ", "")
                # Scale-string-fragile (#384 review, Frank, non-blocking): this keys
                # the confounder reroute off the exact ``Normal(0, 0.3)`` scale. The
                # explicit a_G/b_G (tau 0.5) and reported b_M (b_path 1.0) carve-outs
                # above never match it, so it is correct today — but a future
                # confounder built at a different scale, or a genuine reported path
                # that happens to be Normal(0, 0.3), would be silently mislabelled.
                # Labelling-only risk; no estimand is affected.
                if dist == "Normal(0,0.3)":
                    ctor[rv.name] = "gamma_cross"
                    role[rv.name] = "association"
                    rationale[rv.name] = (
                        "Cross-baseline confounder coupling in the mediation legs "
                        "(Normal(0, 0.3)); an adjusted association, not a mediator "
                        "a-/b-path and not the reported estimand."
                    )
        # Period-stacked two-mediator model (med-092). b_trt (direct path, tau 0.5)
        # and b_phase (per-phase offset, Normal(0, 0.5)) are not rerouted above (not
        # 0.3) but would inherit empty/misleading rationales; b_trtM (exposure x
        # mediator, gamma_cross 0.3) IS rerouted above but wants a specific
        # description. These names are unique to med-092, so the overrides are inert
        # on other models (no matching row). Set after the loop so b_trtM wins.
        rationale["b_trt"] = (
            "Per-period on-intervention direct-path coefficient (tau-scaled "
            "Normal(0, 0.5)); a structural g-formula building block leaning on "
            "gain-factor ignorability, an adjusted association, not a cross-baseline "
            "coupling."
        )
        rationale["b_phase"] = (
            "Per-phase intercept/period offset (Normal(0, 0.5)); an "
            "age/maturation/period association, not a cross-baseline skill coupling."
        )
        rationale["b_trtM"] = (
            "Exposure x mediator interaction (on-intervention x standardised "
            "mediator; Normal(0, 0.3)); admits exposure-mediator interaction in the "
            "g-formula, not a cross-baseline coupling."
        )
    elif spec.kind == "mechanism":
        # ``beta_G`` reuses the tau constructor (its Normal(0, 0.5) scale) but here
        # it is the group main effect entered as a DAG backdoor adjustment, not the
        # randomised ITT effect — an adjusted association, not a causal term. The
        # role is demoted but the rationale still inherits the tau docstring, so set
        # it explicitly. ``f_mech__ell`` is built with ell_prior_mech() = IG(5, 5)
        # (#265) but the ``__ell`` suffix routes it to the default ell constructor
        # whose docstring says IG(3, 1); the distribution column (read off the RV)
        # correctly shows IG(5, 5), so the rationale contradicts its own row.
        role["beta_G"] = "association"
        rationale["beta_G"] = (
            "Group main effect entered as a DAG backdoor adjustment (reuses the tau "
            "Normal(0, 0.5) scale); an adjusted association, not the randomised "
            "treatment effect."
        )
        rationale["f_mech__ell"] = (
            "Mechanism-curve GP lengthscale ell ~ InverseGamma(5, 5) on standardised "
            "inputs (issue #265)."
        )
    elif spec.kind == "aligned":
        ctor["beta_cohort"] = "tau"
        role["beta_cohort"] = "association"
        rationale["beta_cohort"] = (
            "Per-protocol cohort contrast (immediate vs wait-list) at onset-aligned "
            "endpoints; an adjusted association confounded by age-at-onset and "
            "cohort/timing, never the randomised treatment effect."
        )
        rationale["gamma_ability"] = (
            "Cognitive-ability (block design) covariate coupling ~ Normal(0, 0.3); an "
            "adjusted association, not a cross-baseline coupling."
        )
        rationale["gamma_dose"] = (
            "Within-arm cumulative-session dose coupling ~ Normal(0, 0.3); a "
            "collider-adjusted sensitivity association, never a causal dose effect."
        )
    elif spec.kind == "adjusted" and context.model is not None:
        for rv in context.model.free_RVs:
            # Cohort group-nuisance dummies are classified as inline nuisances in
            # priors.prior_info_for_rv (prefix match) — do not sweep them into the
            # predictor-slope/association bucket here.
            if rv.name.startswith("beta_group_nuisance"):
                continue
            # Missing-data indicators (beta_{cov}_missing) are handled by the
            # universal missing-indicator sweep below (role nuisance, #384 review) —
            # skip them here so they are not tagged as predictor-slope associations.
            if rv.name.endswith("_missing"):
                continue
            if rv.name.startswith("beta_"):
                ctor[rv.name] = "predictor_slope"
                role[rv.name] = "association"
    elif spec.kind == "growth":
        # Baseline non-verbal ability -> trajectory shape (gamma on the growth rate,
        # delta on the baseline level): adjusted, latent-GA-confounded associations,
        # never causal — routed to the predictor-slope panel / association role.
        # gamma_age (baseline-age main effect) and gamma_int (the #228 item-10
        # baseline age x ability interaction) are also association slopes, but their
        # names fall through the ``gamma`` prefix to the gamma_cross panel + its
        # "cross-baseline coupling gamma_k" docstring — the wrong quantity.
        for _rv in ("gamma", "delta", "gamma_age", "gamma_int"):
            ctor[_rv] = "predictor_slope"
            role[_rv] = "association"
        rationale["gamma_age"] = (
            "Baseline (t1) age main effect on the growth rate (gamma_age * age0); an "
            "adjusted, GA-confounded association, not a cross-baseline coupling."
        )
        rationale["gamma_int"] = (
            "Baseline age x ability interaction on the growth rate (the #228 item-10 "
            "headline: older-and-more-able children grow faster than age and ability "
            "predict separately); an adjusted, GA-confounded association, never "
            "causal."
        )
        # ``loading`` (rank-1 growth-tempo factor loading) otherwise inherits the
        # CFA test->domain measurement-loading fallback text, which is the wrong
        # model — override the rationale (role/association already correct).
        rationale["loading"] = (
            "Positive loading (HalfNormal(0.5)) of the shared child-level "
            "growth-tempo factor G onto measure k's growth rate; a rank-1 stand-in "
            "for cross-measure slope covariation, not a CFA test->domain measurement "
            "loading."
        )
    elif spec.kind == "level_factors" and spec.extra.get("group_by_time", True):
        # The prior table is one row per RV, while ``b_grp_time`` is a vector whose
        # elements have different interpretation: only b_grp_time[1] is the clean
        # randomised t2 contrast. Keep the vector row conservative and let
        # factor_summary.csv carry the element-level causal label.
        role["b_grp_time"] = "association"
        rationale["b_grp_time"] = (
            "Level-model group-by-time vector; only b_grp_time[1] is the "
            "randomised t2 contrast, while the vector row is documented "
            "conservatively because other elements are pre-randomisation or "
            "post-crossover associations."
        )
    elif spec.kind == "itt":
        # adjust_for covariates are built as gamma_{covariate} from gamma_cross_prior,
        # so they inherit the gamma_cross panel's "cross-baseline coupling gamma_k"
        # rationale + association role. They are pre-randomisation adjustment/precision
        # covariates, not cross-baseline skill couplings: under randomisation a
        # baseline covariate is balanced across arms in expectation, so it cannot
        # confound tau and only sharpens it — the definition of a precision covariate.
        # ``blocks``/``area`` and the SES adjusters (parental education, age first
        # exposed to books) are all documented "precision covariate" in their modules,
        # so the role is quoted, not inferred (#384 review, Frank: promote SES to
        # precision — identical causal status to blocks/area).
        _quoted_precision = {"blocks", "area", "mumedupost16", "dadedupost16", "agebooks"}
        for c in spec.extra.get("adjust_for", ()):
            name = f"gamma_{c}"
            if c in _quoted_precision:
                role[name] = "precision"
                rationale[name] = (
                    f"Baseline adjustment/precision covariate ({c}) ~ Normal(0, 0.3); "
                    "a pre-randomisation term that sharpens tau and cannot confound "
                    "the randomised effect, not a cross-baseline coupling."
                )
            else:
                rationale[name] = (
                    f"Pre-randomisation adjustment covariate ({c}) ~ Normal(0, 0.3); "
                    "a robustness adjustment that cannot confound the randomised "
                    "effect (balanced across arms in expectation), not a "
                    "cross-baseline coupling."
                )
    elif spec.kind == "corr_factor" and context.model is not None:
        _rv_names = {rv.name for rv in context.model.free_RVs}
        if "beta_G" in _rv_names:
            # The randomised arm G enters mm-002 as a mech-058 backdoor covariate on
            # the predictor_slope prior (Normal(0, 0.3)); it reuses the ``beta_G``
            # name, so _RV_TO_CTOR maps it to ``tau`` (role causal + "Treatment
            # effect tau" rationale) — the most severe mislabel, a causal claim the
            # model explicitly disowns. Route to predictor_slope + association.
            ctor["beta_G"] = "predictor_slope"
            role["beta_G"] = "association"
            rationale["beta_G"] = (
                "Randomised arm G entered as an adjusted-association (mech-058) "
                "backdoor covariate on the standardised predictor_slope prior, not "
                "the randomised ITT effect (the causal claim lives in the ITT suite)."
            )
        if "factor_corr_chol" in _rv_names:
            # ``factor_corr_chol``'s off-diagonals are the reported factor-correlation
            # matrix (exposed as the ``factor_corr_pairs`` deterministic the strict
            # gate evaluates), so it is an ``association`` — the same carve-out this
            # branch already applies to ``measure_corr_chol`` / ``trait_corr_chol`` /
            # ``state_corr_chol_w`` (#384 review, Frank: promote nuisance ->
            # association).
            #
            # Formerly ``factor_cov``, an ``LKJCholeskyCov`` whose ``sd_dist`` scales
            # were discarded. That observation — scale is carried by the loadings, so
            # the sds do nothing — is exactly why they were unidentified, and why the
            # all-free-RV gate failed on them while every reported quantity converged.
            # The bare ``LKJCorr`` has no such component to leave dangling.
            role["factor_corr_chol"] = "association"
            rationale["factor_corr_chol"] = (
                "LKJ(eta=2) prior on the domain-factor correlation matrix, sampled as "
                "its Cholesky factor (R = L L'); scale is carried by the loadings. Its "
                "off-diagonals are the reported factor-correlation matrix — the "
                "study's headline descriptive association."
            )
    elif spec.kind == "concurrent" and context.model is not None:
        # The focal concurrent skill coefficients are ``beta``/``beta_age``; every
        # ``gamma_{c}`` is a trait-covariate adjustment (non-verbal ability, hearing,
        # speech, phonological memory) built from predictor_slope_prior (Normal(0,
        # 0.3)). The ``gamma`` prefix routes them to the gamma_cross panel + its
        # "cross-baseline coupling" docstring — the wrong quantity.
        for rv in context.model.free_RVs:
            if rv.name.startswith("gamma_"):
                ctor[rv.name] = "predictor_slope"
                role[rv.name] = "association"
                rationale[rv.name] = (
                    "Trait-covariate adjustment slope (non-verbal ability / hearing / "
                    "speech / phonological-memory t1 baseline; Normal(0, 0.3)); a "
                    "regularised adjusted association, not a between-skill "
                    "cross-baseline coupling."
                )
    elif spec.kind == "block_exposure":
        # ``delta`` reuses the tau constructor (role causal), but it is the
        # block-active exposure shift in the block-2 taught-vocabulary logit — a
        # parallel-trends association, not a randomised treatment effect. Plain
        # assignment (not setdefault) so the distal `is_distal` block below keeps its
        # tau_distal *panel* for bx-003/004 while the role stays association.
        role["delta"] = "association"
        rationale["delta"] = (
            "Block-active exposure shift in the block-2 taught-vocabulary logit; a "
            "parallel-trends association ('block-2-active vs block-1-active'), not a "
            "randomised treatment effect."
        )
    elif spec.kind == "survival":
        # The cloglog survival models set causal_status='none' (by t4 both arms are
        # treated), so ``tau`` is a prognostic association anchored on the immediate
        # arm's randomised first interval, not a randomised treatment effect.
        role["tau"] = "association"
        rationale["tau"] = (
            "Intervention-aligned treatment hazard shift; a prognostic association "
            "anchored on the immediate arm's randomised first interval, not a "
            "randomised treatment effect of record (both arms are treated by t4)."
        )

    # Distal outcomes take the tighter tau prior (issue #141): the factory built
    # the single-outcome causal treatment term at Normal(0, 0.3), so route it to
    # the ``tau_distal`` panel + distribution here so the report panel matches the
    # fitted scale. Only the randomised treatment terms are listed (never the
    # adjusted-association ``beta_G`` / ``beta_cohort``).
    if is_distal(getattr(spec, "outcome_symbol", None)):
        for _name in (
            "tau",
            "beta_trt",
            "b_grp_time",
            "beta_grp",
            "delta",
            "tau_t2",
            "arm_gap_t3",
            "theta_treated",
        ):
            ctor.setdefault(_name, "tau_distal")
            role.setdefault(_name, "causal")
        # The ANCOVA intercept is likewise tiered for distal outcomes (Normal(0,
        # 1.0); prior-critical-review 2026-07-07, Finding 1). Route it to the
        # ``alpha_distal`` panel so the report rationale matches the fitted scale
        # (the distribution column already reads the true 1.0 off the built RV).
        ctor.setdefault("alpha", "alpha_distal")
        ctor.setdefault("alpha_offset", "alpha_distal")

    # Missing-data-indicator coefficients (beta_{cov}_missing) are subgroup
    # mean-offsets under the missing-indicator method — confounded with the constant
    # fill value and well known to be uninterpretable as an effect (Greenland &
    # Finkle 1995, Am J Epidemiol 142(12):1255-64; Groenwold et al. 2012, CMAJ
    # 184(11):1265-9) — so they are nuisance, not predictor-slope associations, in
    # every family that carries them (currently the adjusted LRP65 and the
    # correlated-factor mm-002). Swept once here rather than per kind (#384 review,
    # Frank). The distribution column, read off the RV, still shows the true
    # predictor_slope Normal(0, 0.3). See also the predictor_associations.csv filter
    # in the adjusted/RLM writers, which keeps the reported-associations table from
    # contradicting this nuisance label.
    if context.model is not None:
        for rv in context.model.free_RVs:
            if rv.name.startswith("beta_") and rv.name.endswith("_missing"):
                ctor.setdefault(rv.name, "predictor_slope")
                role[rv.name] = "nuisance"
                rationale[rv.name] = (
                    f"Missing-data indicator ({rv.name[len('beta_') :]} = 1 when the "
                    "value is unknown/imputed); a subgroup mean-offset under the "
                    "missing-indicator method, confounded with the fill value and not "
                    "interpretable as a substantive standardised-trait association."
                )

    return ctor, role, rationale


def growth_contrast_pushforward_rows(
    ctx: StatisticalFitContext,
    panel,
    measure: str,
    *,
    fitted_var: str = "fitted_mean_items_obs",
    prefix: str = "",
) -> list[dict[str, object]]:
    """Prior pushforward for a historical-growth family's group contrasts (#381).

    The reported estimands are the pairwise ``total_growth_X_minus_Y`` rows over
    the common window — how much more a comparison group grows than the
    Down-syndrome group across the whole observed span. Running
    :func:`historical.growth_summary` on the ``prior`` group answers how much of
    that difference the priors alone permit; the cohorts are not randomised, so
    the rows are descriptive contrasts rather than effects.
    """
    from language_reading_predictors.statistical_models import historical as _hist

    source = getattr(ctx, "prior_samples", None) or ctx.trace
    try:
        prior_growth = _hist.growth_summary(
            source, panel, measure, fitted_var=fitted_var, group="prior"
        )
    except Exception as exc:  # noqa: BLE001 - absence must stay legible
        return [
            _report.unavailable_pushforward(
                estimand=f"{prefix}total_growth",
                estimand_label="the between-group total-growth contrasts",
                role="descriptive",
                reason=str(exc),
            )
        ]
    contrasts = prior_growth[
        prior_growth["quantity"].astype(str).str.startswith("total_growth")
    ]
    rows: list[dict[str, object]] = []
    for _, r in contrasts.iterrows():
        rows.append(
            _report.labelled_pushforward(
                {
                    # The growth summary is already in items; there is no separate
                    # linear-predictor contrast to report, since the quantity is a
                    # difference of fitted means rather than a coefficient.
                    "prior_logit_median": float("nan"),
                    "prior_logit_lo": float("nan"),
                    "prior_logit_hi": float("nan"),
                    "prior_items_median": float(r["q50"]),
                    "prior_items_lo50": float(r["q25"]),
                    "prior_items_hi50": float(r["q75"]),
                    "prior_items_lo": float(r["q_lo"]),
                    "prior_items_hi": float(r["q_hi"]),
                    "n_trials": 0,
                },
                estimand=f"{prefix}{r['quantity']}",
                estimand_label=str(r["label"]),
                role="descriptive",
            )
        )
    return rows


def write_indicator_prior_check(
    ctx: StatisticalFitContext, nodes: Sequence[str]
) -> None:
    """Write ``indicator_prior_check.csv`` for a measurement family (#381).

    The CFA families have no outcome-scale estimand to push a prior through —
    they report loadings, communalities and factor correlations — so #381 asks
    them for this instead, on the scale they do observe: the standardised
    indicator matrix. Without it these families were exempt from the coverage
    guarantee by construction rather than by argument.
    """
    try:
        df = _report.indicator_prior_check(
            ctx.trace, nodes=list(nodes), ci_prob=ctx.reporting.ci_prob
        )
    except Exception as exc:  # noqa: BLE001 - a report extra must not fail a fit
        rprint(f"[yellow]indicator prior check skipped: {exc}[/yellow]")
        return
    if df.empty:
        rprint("[yellow]indicator prior check: no indicator nodes found[/yellow]")
        return
    save_table(ctx, "indicator_prior_check", df)
    print_table(
        ranked_dataframe_table(
            df,
            title="Indicator-scale prior check (SD ratio 1 = prior matches the data)",
            columns=["indicator", "observed_sd", "prior_sd", "sd_ratio", "coverage_90", "verdict"],
            rank_column=False,
            precision=3,
        )
    )


def write_prior_pushforward(
    ctx: StatisticalFitContext, rows: Sequence[Mapping[str, object]]
) -> None:
    """Write ``prior_pushforward.csv`` — including when the check is unavailable (#381).

    The meta-finding behind #381 is that a *missing* artefact reads as a clean
    one: a family that never emitted the estimand-scale prior check looked, in the
    rendered report, exactly like one whose prior was checked and found harmless.
    So every family that reaches this point writes the file, and a row whose
    ``status`` is ``unavailable`` carries the reason instead of being dropped.
    """
    df = pd.DataFrame(list(rows))
    save_table(ctx, "prior_pushforward", df)


def horseshoe_pushforward_rows(
    ctx: StatisticalFitContext, predictors: Sequence[str], outcome: str
) -> list[dict[str, object]]:
    """Per-predictor prior pushforward for a horseshoe ranking fit (#381).

    The horseshoe deliverable is a ranking by ``P(|beta| > delta)``, which the
    prior-analysis review flagged as a direct function of ``tau0`` / ``slab_scale``
    — so "no signal" and "shrunk to nothing by the prior" are the two readings that
    have to be told apart. The ``prior_logit_*`` columns are the shrinkage prior's
    own implied spread for a single coefficient, against which the ranking's
    ``delta`` can be judged; the items columns put the same ``+1 SD`` shift on the
    outcome scale. Every predictor shares the global-local prior, so the rows
    differ only through each coefficient's own local scale draws.
    """
    n_trials = pushforward_n_trials(ctx, outcome)
    label = pushforward_outcome_label(ctx, outcome)
    return marginal_pushforward_rows(
        ctx,
        [
            (
                "beta",
                f"the shrunk association of +1 SD {p} with {label}",
                {"predictor": p},
            )
            for p in predictors
        ],
        n_trials=n_trials,
        convention="forward",
    )


def pushforward_outcome_label(ctx: StatisticalFitContext, outcome: str) -> str:
    """Reader-facing name for the pushforward's outcome, falling back to the symbol.

    The rows are read by a science reader, not by whoever picked the symbols, so
    ``W`` and ``basread`` should render as their measure labels. The study's own
    measure table is the source: RLI symbols resolve through ``measures.MEASURES``
    and the Byrne-cohort ones through their dataset's table, so neither study's
    labels are hard-coded here.
    """
    from language_reading_predictors.statistical_models import datasets as _datasets
    from language_reading_predictors.statistical_models.measures import MEASURES

    if outcome in MEASURES:
        return str(MEASURES[outcome].label)
    try:
        _, measures = _datasets.resolve_dataset(ctx.spec.extra.get("study_id", "rlm"))
        return str(measures[outcome].label)
    except Exception:  # noqa: BLE001 - a label is cosmetic; the symbol still names it
        return outcome


def pushforward_n_trials(ctx: StatisticalFitContext, outcome: str) -> int:
    """The pushforward's item denominator, or 1 when the fit carries none (#381).

    Most families know their outcome's item ceiling and the check is most
    readable in items. Where the fit does not carry one, return 1 rather than
    inventing a denominator: the marginal is then a probability difference, and
    :func:`reporting.pushforward_scale_for` labels it in percentage points off
    that same 1 — one rule, so a denominator and its scale cannot disagree.
    """
    trials = getattr(ctx.prepared, "n_trials", None) or {}
    try:
        return int(trials[outcome])
    except (KeyError, TypeError, ValueError):
        return 1


def marginal_pushforward_rows(
    ctx: StatisticalFitContext,
    terms: Sequence[tuple],
    *,
    n_trials: int,
    role: str = "association",
    convention: str = "forward",
    eta_name: str = "eta",
    row_mask: np.ndarray | None = None,
    scale: str | None = None,
) -> list[dict[str, object]]:
    """Build one labelled pushforward row per term (#381).

    Each entry is ``(term, label)``, or ``(term, label, index)`` to select one
    element of a vector-valued coefficient — ``("beta", "...", {"predictor":
    "age"})`` for the horseshoe families, whose ``beta`` carries a labelled
    predictor dimension.

    ``convention`` is passed straight through to
    :func:`reporting.marginal_prior_pushforward` and must match the convention the
    family's own posterior marginal uses. A term the prior group does not carry
    yields an ``unavailable`` row naming it, rather than a silently shorter table.
    """
    # ``prior_samples`` carries the prior group from ``run_prior_predictive``; the
    # trace only carries it after ``save_trace`` grafts it on, so prefer the former
    # and the call site stays free to sit either side of that step.
    source = getattr(ctx, "prior_samples", None) or ctx.trace
    rows: list[dict[str, object]] = []
    for entry in terms:
        term, label = entry[0], entry[1]
        index = entry[2] if len(entry) > 2 else None
        named = term if not index else f"{term}[{'/'.join(map(str, index.values()))}]"
        try:
            values = _report.marginal_prior_pushforward(
                source,
                term=term,
                n_trials=n_trials,
                eta_name=eta_name,
                ci_prob=ctx.reporting.ci_prob,
                convention=convention,
                row_mask=row_mask,
                term_index=index,
            )
        except Exception as exc:  # noqa: BLE001 - an absent term must stay legible
            rows.append(
                _report.unavailable_pushforward(
                    estimand=named,
                    estimand_label=label,
                    role=role,
                    reason=str(exc),
                    scale=scale,
                )
            )
        else:
            rows.append(
                _report.labelled_pushforward(
                    values,
                    estimand=named,
                    estimand_label=label,
                    role=role,
                    scale=scale,
                )
            )
    return rows
