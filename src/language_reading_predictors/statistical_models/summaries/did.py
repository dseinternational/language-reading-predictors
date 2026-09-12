# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Did calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

from collections.abc import Mapping
import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.statistics.evidence import (
    evidence_label,
)
from scipy.special import expit
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    apply_score_mean_link,
)
from language_reading_predictors.statistical_models.posteriors import (
    band50,
)


def did_summary(
    trace: xr.DataTree,
    *,
    ci_prob: float,
    n_trials: int,
    dose: bool = False,
    off_floor: bool = False,
    child_idx: np.ndarray | None = None,
    standardization_cells: Mapping[str, np.ndarray] | None = None,
    wave: np.ndarray | None = None,
    score_mean_link: ScoreMeanLink = "logit",
    subject_ids: np.ndarray | None = None,
) -> dict[str, float | bool | str]:
    """Summarise a waitlist-crossover arm-by-wave model (kind="did").

    The current binary model exposes three immediate-minus-waitlist logit contrasts:
    ``arm_gap_t1`` (pre-randomisation balance), ``tau_t2`` (the randomised
    immediate-treatment-versus-no-treatment assignment contrast at t2) and
    ``arm_gap_t3`` (a **different randomised contrast** — assignment to the
    early-start rather than the delayed-start treatment schedule, both arms being
    treated by t3). Its derived ``delta_crossover = tau_t2 - arm_gap_t3`` is the
    change between those two randomised regime contrasts: positive means the gap
    between the arms is smaller at t3 than at t2. It is **not** an identified
    catch-up mechanism — duration, carryover, maturation, ceiling effects and
    different taught blocks are inseparable in it (#576 finding 3) — and it is not a
    second treated-versus-untreated effect.

    ``score_mean_link`` is the inverse link of the fitted score model. The
    phoneme-blending guessing-floor companion maps the mean onto ``[1/3, 1]``, so
    every outcome-scale quantity here must go through the same link the likelihood
    used; reading ``expit(eta)`` directly would understate the fitted score by up to
    a third of the test (#576 finding 2).

    ``wave`` must contain the fitted row's zero-based t1/t2/t3 code (0/1/2). For
    each wave, the function standardises both arms over that wave's fitted rows
    using ``eta_base``, which excludes the arm term. It reports the two standardised
    arm means and their immediate-minus-waitlist difference on the outcome scale.
    ``delta_crossover_items_*`` is the t2 standardised arm gap minus the t3
    standardised arm gap, not ``expit(delta_crossover)``. Because the logit link is
    nonlinear, this outcome-scale change-in-gap depends on the wave-specific
    operating points. These are fitted-sample standardisations and the t2 quantity
    is not numerically interchangeable with an ITT summary standardised over a
    different fitted sample or covariate distribution.

    For the exploratory varying-crossover model, ``delta_crossover_i`` is averaged
    over the fitted waitlist children per posterior draw and reported separately as
    ``delta_crossover_sample_average_*``. The outcome-scale change-in-gap is omitted
    for that model because a scalar arm-gap toggle would fail to integrate the
    fitted child-specific catch-up terms. For the same reason its t3 standardised
    quantities (``t3_waitlist_items_*``, ``t3_immediate_items_*``,
    ``arm_gap_t3_items_*``) are omitted: ``eta_base`` excludes the fitted
    ``v_delta`` deviations, so a population-mean t3 toggle would misstate the
    fitted waitlist t3 level. ``arm_gap_t3_items_available`` records the omission;
    the t1/t2 quantities are unaffected (``v_delta`` enters only waitlist t3 rows),
    and the logit-scale ``arm_gap_t3`` posterior is always reported.

    The legacy ``beta_period``/``delta`` branch remains readable so existing traces
    fail gracefully during the refit transition. Its ``delta_items_*`` quantity is
    a fitted-row model-implied treated-versus-untreated toggle, not a four-cell DiD
    cross-difference and not automatically comparable with the corresponding
    available-case modified ITT estimate.

    When the posterior contains child-specific ``delta_i`` draws, ``child_idx`` is
    required and must map each fitted row to the corresponding ``child`` position.
    The marginal effect then uses the fitted child's posterior slope rather than the
    population-mean ``delta``. This is conditional standardisation over the fitted
    children; it does not integrate a new child's random slope from the population
    distribution. For a constant-effect fit, ``child_idx`` is ignored.

    ``standardization_cells`` optionally maps short, identifier-like names (for
    example ``{"p1": phase == 0, "waitlist_p1": ...}``) to boolean masks aligned
    with the fitted rows. Each cell receives a companion
    ``delta_items_{name}_*`` summary. These remain model-implied treatment toggles
    at that cell's covariate distribution, rather than observed arm contrasts.

    With ``off_floor=True`` (the off-floor prevalence DiD for heavily-floored P / N,
    fitted as a Bernoulli on the off-floor indicator) the caller passes
    ``n_trials=1``. Every ``*_items_*`` field is then on the probability scale:
    arm-gap fields are off-floor risk differences and cell fields are probabilities
    of *being* off the floor at that wave, not item counts or transitions from the
    floor. The returned ``off_floor`` flag lets the report label the scale.
    """
    posterior = trace.posterior
    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q

    def _summ_draws(name: str, draws: np.ndarray) -> dict[str, float | str]:
        d = np.asarray(draws).ravel()
        prob_pos = float(np.mean(d > 0))
        lo50, hi50 = band50(d)
        return {
            # Median-first to match the ITT tau_summary_itt convention (#144 / #271);
            # the mean is kept as a secondary column.
            f"{name}_median": float(np.median(d)),
            f"{name}_mean": float(np.mean(d)),
            f"{name}_lo": float(np.quantile(d, lo_q)),
            f"{name}_hi": float(np.quantile(d, hi_q)),
            f"{name}_lo50": lo50,
            f"{name}_hi50": hi50,
            f"prob_{name}_pos": prob_pos,
            f"{name}_direction_label": evidence_label(prob_pos),
            f"{name}_favoured_direction": "positive" if prob_pos >= 0.5 else "negative",
            f"{name}_favoured_label": evidence_label(max(prob_pos, 1.0 - prob_pos)),
        }

    def _summ(name: str) -> dict[str, float | str]:
        return _summ_draws(
            name, posterior[name].stack(sample=("chain", "draw")).values
        )

    out: dict[str, float | bool | str] = {}

    def _effect_summary(draws: np.ndarray, *, prefix: str) -> None:
        scaled: np.ndarray = draws * n_trials
        out[f"{prefix}_median"] = float(np.median(scaled))
        out[f"{prefix}_mean"] = float(np.mean(scaled))
        out[f"{prefix}_lo"] = float(np.quantile(scaled, lo_q))
        out[f"{prefix}_hi"] = float(np.quantile(scaled, hi_q))
        out[f"{prefix}_lo50"], out[f"{prefix}_hi50"] = band50(scaled)

    if "tau_t2" in posterior:
        required = {"arm_gap_t1", "arm_gap_t3", "delta_crossover", "eta_base"}
        missing = sorted(required.difference(posterior.data_vars))
        if missing:
            raise KeyError(
                "arm-by-wave DiD trace is missing required posterior nodes: "
                + ", ".join(missing)
            )
        for name in ("arm_gap_t1", "tau_t2", "arm_gap_t3", "delta_crossover"):
            out.update(_summ(name))
        if "delta_crossover_i" in posterior:
            child_draws = (
                posterior["delta_crossover_i"]
                .stack(sample=("chain", "draw"))
                .transpose("waitlist_child", "sample")
                .values
            )
            out.update(
                _summ_draws(
                    "delta_crossover_sample_average", child_draws.mean(axis=0)
                )
            )
            out["delta_crossover_sample_n_children"] = int(child_draws.shape[0])
        if dose and "beta_dose" in posterior:
            out.update(_summ("beta_dose"))

        if wave is None:
            raise ValueError(
                "wave is required for arm-by-wave outcome-scale standardisation; "
                "pass the fitted prepared.phase array."
            )
        wave_arr = np.asarray(wave)
        if wave_arr.ndim != 1:
            raise ValueError(f"wave must be 1-D, got a {wave_arr.ndim}-D array.")
        if not np.issubdtype(wave_arr.dtype, np.integer):
            raise ValueError(f"wave must contain integer phase codes, got {wave_arr.dtype}.")
        eta_base = (
            posterior["eta_base"]
            .stack(sample=("chain", "draw"))
            .transpose("obs_id", "sample")
            .values
        )  # (n_obs, S)
        if wave_arr.shape[0] != eta_base.shape[0]:
            raise ValueError(
                f"wave has {wave_arr.shape[0]} rows but eta_base has "
                f"{eta_base.shape[0]} observations; pass the fitted-subset phases."
            )

        varying_catch_up = "delta_crossover_i" in posterior
        wave_effects: dict[str, np.ndarray] = {}
        wave_terms = (
            (0, "t1", "arm_gap_t1"),
            (1, "t2", "tau_t2"),
            (2, "t3", "arm_gap_t3"),
        )
        for wave_code, wave_name, term_name in wave_terms:
            rows = wave_arr == wave_code
            if not np.any(rows):
                raise ValueError(
                    f"wave contains no {wave_name} rows (expected phase code {wave_code})."
                )
            if varying_catch_up and wave_code == 2:
                # The fitted waitlist-child catch-up deviations (v_delta) enter
                # the waitlist t3 rows but are absent from eta_base, so a scalar
                # arm-gap toggle would misstate the fitted t3 levels — the same
                # partial-integration reason delta_crossover_items is withheld
                # below. Omit rather than publish a partially-integrated summary.
                continue
            gap = (
                posterior[term_name]
                .stack(sample=("chain", "draw"))
                .values.ravel()
            )
            waitlist = apply_score_mean_link(
                expit(eta_base[rows]), score_mean_link
            ).mean(axis=0)
            immediate = apply_score_mean_link(
                expit(eta_base[rows] + gap[None, :]), score_mean_link
            ).mean(axis=0)
            arm_gap = immediate - waitlist
            _effect_summary(waitlist, prefix=f"{wave_name}_waitlist_items")
            _effect_summary(immediate, prefix=f"{wave_name}_immediate_items")
            _effect_summary(arm_gap, prefix=f"{term_name}_items")
            out[f"{term_name}_items_n_rows"] = int(rows.sum())
            wave_effects[term_name] = arm_gap

        out["arm_gap_t3_items_available"] = not varying_catch_up
        if varying_catch_up:
            out["arm_gap_t3_items_omission_reason"] = (
                "the fitted waitlist-child catch-up deviations are not "
                "integrated by a scalar arm-gap toggle"
            )

        if not varying_catch_up:
            _effect_summary(
                wave_effects["tau_t2"] - wave_effects["arm_gap_t3"],
                prefix="delta_crossover_items",
            )
            out["delta_crossover_items_available"] = True
            out["delta_crossover_items_population"] = "wave_specific_fitted_rows"
            # Common-child gap change (#576 material qualification 6). The two legs
            # above are each standardised over their own wave's fitted rows. When a
            # child is observed at t2 but not t3 those row sets differ, and the
            # difference then mixes the change over time with a change in *who* is
            # being averaged over. Recomputing both legs on the children present at
            # both waves separates the two; the wave-specific quantity is retained
            # beside it, and the recorded flag says whether they can differ at all.
            if subject_ids is not None:
                ids = np.asarray(subject_ids).astype(str)
                if ids.shape[0] != eta_base.shape[0]:
                    raise ValueError(
                        f"subject_ids has {ids.shape[0]} rows but eta_base has "
                        f"{eta_base.shape[0]} observations; pass the fitted-subset ids."
                    )
                common = np.intersect1d(ids[wave_arr == 1], ids[wave_arr == 2])
                out["delta_crossover_items_common_n_children"] = int(common.size)
                out["delta_crossover_items_common_population_identical"] = bool(
                    common.size == np.unique(ids[wave_arr == 1]).size
                    and common.size == np.unique(ids[wave_arr == 2]).size
                )
                if common.size:
                    in_common = np.isin(ids, common)
                    common_effects: dict[str, np.ndarray] = {}
                    for wave_code, term_name in ((1, "tau_t2"), (2, "arm_gap_t3")):
                        rows = (wave_arr == wave_code) & in_common
                        gap = (
                            posterior[term_name]
                            .stack(sample=("chain", "draw"))
                            .values.ravel()
                        )
                        base = eta_base[rows]
                        common_effects[term_name] = (
                            apply_score_mean_link(
                                expit(base + gap[None, :]), score_mean_link
                            ).mean(axis=0)
                            - apply_score_mean_link(
                                expit(base), score_mean_link
                            ).mean(axis=0)
                        )
                    for term_name, prefix in (
                        ("tau_t2", "tau_t2_items_common"),
                        ("arm_gap_t3", "arm_gap_t3_items_common"),
                    ):
                        _effect_summary(common_effects[term_name], prefix=prefix)
                    _effect_summary(
                        common_effects["tau_t2"] - common_effects["arm_gap_t3"],
                        prefix="delta_crossover_items_common",
                    )
                    out["delta_crossover_items_common_available"] = True
                else:
                    out["delta_crossover_items_common_available"] = False
            else:
                out["delta_crossover_items_common_available"] = False
        else:
            out["delta_crossover_items_available"] = False
            out["delta_crossover_items_omission_reason"] = (
                "child-specific catch-up requires an explicitly integrated "
                "waitlist-child counterfactual"
            )
        out["arm_wave_marginal_estimand"] = (
            "wave-specific fitted-row standardized immediate-minus-waitlist arm gap"
        )
        out["arm_wave_marginal_effect_source"] = (
            "population-mean arm gaps; child-specific catch-up is not integrated"
            if "delta_crossover_i" in posterior
            else "fixed arm gaps"
        )
        out["score_mean_link"] = str(score_mean_link)
        out["tau_t2_interpretation"] = (
            "randomised assignment contrast: immediate treatment versus no treatment "
            "yet, read at t2"
        )
        out["arm_gap_t3_interpretation"] = (
            "randomised assignment contrast between treatment schedules: early-start "
            "versus delayed-start treatment history at t3; not a treated-versus-"
            "untreated effect"
        )
        out["delta_crossover_interpretation"] = (
            "change between two randomised regime contrasts (t2 gap minus t3 gap); "
            "not an identified catch-up mechanism"
        )
        out["off_floor"] = bool(off_floor)
        return out

    out.update(_summ("beta_period"))
    if dose:
        # The redesigned dose model separates the saturated arm-by-period cell
        # structure from intensive session variation. Report the arm and cell
        # coefficients whenever the trace carries them so the observational
        # beta_dose is not presented as though it were the randomised on/off
        # contrast. Under that saturated coding (treated = immediate arm OR
        # period 2) theta_treated at the mean treated dose is the crossover
        # *cell* contrast, not an isolated treatment-presence effect (#631
        # finding 12; the resolver in did.py says the same).
        for name in ("beta_group", "theta_treated", "gamma_t1", "beta_dose"):
            if name in posterior:
                out.update(_summ(name))
        out["dose_interpretation"] = (
            "beta_dose is an observational intensive-margin association; "
            "theta_treated is the crossover cell contrast at the mean treated "
            "dose, not an isolated treatment-presence effect"
        )
        return out

    out.update(_summ("delta"))
    # Model-implied treated-vs-untreated contrast, standardised over the fitted
    # rows. For the varying-slope fit, map each child's posterior delta_i to every
    # row belonging to that child; using the scalar population mean here would not
    # report the model that was actually fitted.
    delta = posterior["delta"].stack(sample=("chain", "draw")).values.ravel()  # (S,)
    eta_base = (
        posterior["eta_base"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)
    if "delta_i" in posterior:
        if child_idx is None:
            raise ValueError(
                "child_idx is required when the DiD posterior contains child-specific "
                "delta_i draws."
            )
        idx = np.asarray(child_idx)
        if idx.ndim != 1:
            raise ValueError(f"child_idx must be 1-D, got a {idx.ndim}-D array.")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"child_idx must contain integer positions, got {idx.dtype}.")
        if idx.shape[0] != eta_base.shape[0]:
            raise ValueError(
                f"child_idx has {idx.shape[0]} rows but eta_base has "
                f"{eta_base.shape[0]} observations; pass the fitted-subset mapping."
            )
        child_delta = (
            posterior["delta_i"]
            .stack(sample=("chain", "draw"))
            .transpose("child", "sample")
            .values
        )  # (n_child, S)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= child_delta.shape[0]):
            raise ValueError(
                f"child_idx contains positions outside [0, {child_delta.shape[0]})."
            )
        row_delta = child_delta[idx]  # (n_obs, S)
        effect_source = "child_specific_delta_i"
    else:
        row_delta = delta[None, :]  # (1, S), broadcast over observations
        effect_source = "population_mean_delta"

    row_effect = expit(eta_base + row_delta) - expit(eta_base)  # (n_obs, S)

    _effect_summary(row_effect.mean(axis=0), prefix="delta_items")
    out["delta_standardization_n_rows"] = int(eta_base.shape[0])
    cell_names: list[str] = []
    for name, raw_mask in (standardization_cells or {}).items():
        if not name.isascii() or not name.isidentifier():
            raise ValueError(
                "standardization cell names must be non-empty ASCII identifiers; "
                f"got {name!r}."
            )
        mask = np.asarray(raw_mask)
        if mask.ndim != 1:
            raise ValueError(
                f"standardization cell {name!r} must be 1-D, got {mask.ndim}-D."
            )
        if mask.dtype != bool:
            raise ValueError(
                f"standardization cell {name!r} must be a boolean mask, got "
                f"{mask.dtype}."
            )
        if mask.shape[0] != eta_base.shape[0]:
            raise ValueError(
                f"standardization cell {name!r} has {mask.shape[0]} rows but "
                f"eta_base has {eta_base.shape[0]} observations."
            )
        if not np.any(mask):
            raise ValueError(f"standardization cell {name!r} selects no observations.")
        _effect_summary(row_effect[mask].mean(axis=0), prefix=f"delta_items_{name}")
        out[f"delta_items_{name}_n_rows"] = int(mask.sum())
        cell_names.append(name)

    out["delta_marginal_estimand"] = (
        "fitted-row sample-average model-implied treated-versus-untreated contrast"
    )
    out["delta_marginal_effect_source"] = effect_source
    out["delta_standardization_cells"] = ",".join(cell_names)
    out["off_floor"] = bool(off_floor)
    return out


def did_cell_ppc(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    G: np.ndarray,
    dose: bool = False,
    node: str = "y_post",
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Posterior-predictive checks for every fitted DiD arm-by-time cell.

    A pooled posterior-predictive plot can hide a cell-specific failure by letting
    well-fitted cells compensate for a badly fitted one. This helper therefore
    compares the observed cell mean and zero rate with their replicated posterior-
    predictive distributions for every wave/arm (binary model) or period/arm (dose
    model). The mean uses the upper-tail probability ``P(rep >= obs)``; the discrete
    zero rate uses a **mid-p** upper tail (``P(rep > obs) + 0.5 P(rep == obs)``) so a
    boundary cell — observed zero-rate exactly 0 or 1, where a plain ``>=`` tail is
    degenerate — is not falsely flagged. These are diagnostics, not hypothesis-test
    p-values. Values near zero or one flag an observed statistic in a predictive tail
    and should be investigated before interpreting contrasts. The ``*_tail_flag``
    columns use fixed 2.5% / 97.5% cutoffs (a 95% two-sided convention) regardless
    of ``ci_prob``, which shapes only the reported interval columns.
    """
    phase_arr = np.asarray(phase)
    group_arr = np.asarray(G)
    if phase_arr.ndim != 1 or group_arr.ndim != 1:
        raise ValueError("phase and G must both be one-dimensional")
    if phase_arr.shape != group_arr.shape:
        raise ValueError(
            f"phase and G must align; got {phase_arr.shape} and {group_arr.shape}"
        )
    if not np.issubdtype(phase_arr.dtype, np.integer):
        raise ValueError(f"phase must contain integer codes, got {phase_arr.dtype}")
    if not set(np.unique(group_arr)).issubset({0, 1}):
        raise ValueError("G must use 0=waitlist and 1=immediate coding")
    if not 0 < ci_prob < 1:
        raise ValueError(f"ci_prob must lie in (0, 1), got {ci_prob}")

    try:
        pp_da = trace.posterior_predictive[node]
        observed = np.asarray(trace.observed_data[node].values).reshape(-1)
    except (AttributeError, KeyError) as exc:
        raise KeyError(
            f"trace must contain posterior_predictive and observed_data for {node!r}"
        ) from exc

    sample_dims = {"chain", "draw"}
    obs_dims = [d for d in pp_da.dims if d not in sample_dims]
    if len(obs_dims) != 1:
        raise ValueError(
            f"{node!r} must have one observation dimension, got {pp_da.dims}"
        )
    replicated = (
        pp_da.stack(sample=("chain", "draw"))
        .transpose(obs_dims[0], "sample")
        .values
    )
    n_obs = phase_arr.shape[0]
    if replicated.shape[0] != n_obs or observed.shape[0] != n_obs:
        raise ValueError(
            f"fitted arrays are misaligned: phase={n_obs}, replicated="
            f"{replicated.shape[0]}, observed={observed.shape[0]}"
        )

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | int | str | bool]] = []
    prefix = "P" if dose else "t"
    for phase_code in sorted(np.unique(phase_arr)):
        for arm_code, arm_name in ((0, "waitlist"), (1, "immediate")):
            mask = (phase_arr == phase_code) & (group_arr == arm_code)
            if not np.any(mask):
                raise ValueError(
                    f"no rows for {prefix}{int(phase_code) + 1}/{arm_name}"
                )
            observed_cell = observed[mask]
            replicated_cell = replicated[mask]
            observed_mean = float(np.mean(observed_cell))
            observed_zero = float(np.mean(observed_cell == 0))
            replicated_mean = replicated_cell.mean(axis=0)
            replicated_zero = (replicated_cell == 0).mean(axis=0)
            p_mean = float(np.mean(replicated_mean >= observed_mean))
            # Mid-p upper tail for the discrete zero-rate statistic: split ties so a
            # boundary cell is not falsely flagged. A plain P(rep >= obs) is
            # necessarily 1.0 when the observed zero-rate is exactly 0 (and small when
            # it is exactly 1), which spuriously flagged well-fitting cells (#390 P2).
            zero_mid_p = float(
                np.mean(replicated_zero > observed_zero)
                + 0.5 * np.mean(replicated_zero == observed_zero)
            )
            rows.append(
                {
                    "cell": f"{prefix}{int(phase_code) + 1}_{arm_name}",
                    "time": f"{prefix}{int(phase_code) + 1}",
                    "phase_code": int(phase_code),
                    "arm": arm_name,
                    "n": int(mask.sum()),
                    "observed_mean": observed_mean,
                    "replicated_mean_median": float(np.median(replicated_mean)),
                    "replicated_mean_lo": float(np.quantile(replicated_mean, lo_q)),
                    "replicated_mean_hi": float(np.quantile(replicated_mean, hi_q)),
                    "p_rep_mean_ge_observed": p_mean,
                    "mean_tail_flag": bool(p_mean <= 0.025 or p_mean >= 0.975),
                    "observed_zero_rate": observed_zero,
                    "replicated_zero_rate_median": float(
                        np.median(replicated_zero)
                    ),
                    "replicated_zero_rate_lo": float(
                        np.quantile(replicated_zero, lo_q)
                    ),
                    "replicated_zero_rate_hi": float(
                        np.quantile(replicated_zero, hi_q)
                    ),
                    "zero_rate_ppc_mid_p": zero_mid_p,
                    "zero_tail_flag": bool(
                        zero_mid_p <= 0.025 or zero_mid_p >= 0.975
                    ),
                }
            )
    return pd.DataFrame(rows)


def did_within_child_ppc(
    trace: xr.DataTree,
    *,
    phase: np.ndarray,
    subject_ids: np.ndarray,
    G: np.ndarray,
    node: str = "y_post",
    ci_prob: float = REPORTING_CI_PROB,
) -> pd.DataFrame:
    """Posterior-predictive checks of the model's **within-child** structure (#576 MQ3).

    A single stable child random intercept plus conditionally independent
    Beta-Binomial rows imposes a restrictive repeated-measures covariance: it says
    every pair of a child's waves is equicorrelated, with the correlation set by one
    variance ratio, and it fixes how much a child can move between consecutive waves.
    The family's existing checks cannot see a failure of that assumption. The
    arm-by-time cell PPC compares *marginal* cell means and zero rates, which a model
    with badly wrong within-child dependence can still reproduce; the pooled score
    density likewise.

    This cross-checks the structure directly. For every child observed at both waves
    of a pair, it compares the observed **within-child change** (its mean and SD, per
    arm where the pair spans the randomised window) and the observed **across-child
    correlation** of the paired scores with the same statistics recomputed on each
    posterior-predictive replicate. A replicate distribution that systematically
    understates the spread of within-child changes, or overstates the wave-to-wave
    correlation, is the signature of an over-restrictive covariance — invisible in
    the marginal checks.

    Tail probabilities are the usual ``P(replicated >= observed)`` upper tails and the
    flag uses the family's fixed 2.5 % / 97.5 % convention (matching
    :func:`did_cell_ppc`), while the interval columns render at the house ``ci_prob``.
    These are predictive diagnostics, not hypothesis tests.
    """
    posterior_predictive = getattr(trace, "posterior_predictive", None)
    if posterior_predictive is None or node not in posterior_predictive:
        raise KeyError(f"posterior predictive group has no node {node!r}")
    replicated = (
        posterior_predictive[node]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values.astype(float)
    )
    observed = np.asarray(trace.observed_data[node].values, dtype=float)
    phase_arr = np.asarray(phase, dtype=int)
    ids = np.asarray(subject_ids).astype(str)
    arm_arr = np.asarray(G, dtype=int)
    n_obs = phase_arr.shape[0]
    for name, array in (
        ("subject_ids", ids), ("G", arm_arr), ("observed", observed),
        ("replicated", replicated),
    ):
        if array.shape[0] != n_obs:
            raise ValueError(
                f"fitted arrays are misaligned: phase={n_obs}, {name}={array.shape[0]}"
            )

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | int | str | bool]] = []

    def _record(
        statistic: str, pair: str, arm: str, obs_value: float, rep_values: np.ndarray
    ) -> None:
        finite = np.isfinite(rep_values)
        if not np.isfinite(obs_value) or not finite.any():
            return
        rep = rep_values[finite]
        tail = float(np.mean(rep >= obs_value))
        rows.append(
            {
                "statistic": statistic,
                "wave_pair": pair,
                "arm": arm,
                # Filled in by the wave-pair loop below, which knows the pairing.
                "n_children": 0,
                "observed": float(obs_value),
                "replicated_median": float(np.median(rep)),
                "replicated_lo": float(np.quantile(rep, lo_q)),
                "replicated_hi": float(np.quantile(rep, hi_q)),
                "p_rep_ge_observed": tail,
                "tail_flag": bool(tail <= 0.025 or tail >= 0.975),
            }
        )

    def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Across-child Pearson correlation of two row blocks, per draw/column."""
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        a_c = a - a.mean(axis=0, keepdims=True)
        b_c = b - b.mean(axis=0, keepdims=True)
        denominator = np.sqrt((a_c**2).sum(axis=0) * (b_c**2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denominator > 0, (a_c * b_c).sum(axis=0) / denominator, np.nan)

    wave_pairs = ((0, 1, "t1_t2"), (1, 2, "t2_t3"), (0, 2, "t1_t3"))
    for first, second, pair in wave_pairs:
        first_rows = {i: r for r, i in enumerate(ids) if phase_arr[r] == first}
        second_rows = {i: r for r, i in enumerate(ids) if phase_arr[r] == second}
        common = sorted(set(first_rows) & set(second_rows))
        if len(common) < 3:
            continue
        idx_a = np.asarray([first_rows[i] for i in common])
        idx_b = np.asarray([second_rows[i] for i in common])
        change_obs = observed[idx_b] - observed[idx_a]
        change_rep = replicated[idx_b] - replicated[idx_a]
        arms_here = np.asarray([arm_arr[r] for r in idx_a])
        started = len(rows)
        _record(
            "within_child_change_sd", pair, "both",
            float(np.std(change_obs, ddof=1)), change_rep.std(axis=0, ddof=1),
        )
        _record(
            "across_child_correlation", pair, "both",
            float(_corr(observed[idx_a][:, None], observed[idx_b][:, None])[0]),
            _corr(replicated[idx_a], replicated[idx_b]),
        )
        for arm_code, arm_name in ((0, "waitlist"), (1, "immediate")):
            arm_mask = arms_here == arm_code
            if arm_mask.sum() < 2:
                continue
            _record(
                "within_child_change_mean", pair, arm_name,
                float(np.mean(change_obs[arm_mask])),
                change_rep[arm_mask].mean(axis=0),
            )
        for row in rows[started:]:
            row["n_children"] = (
                int(len(common))
                if row["arm"] == "both"
                else int((arms_here == (1 if row["arm"] == "immediate" else 0)).sum())
            )
    return pd.DataFrame(rows)
