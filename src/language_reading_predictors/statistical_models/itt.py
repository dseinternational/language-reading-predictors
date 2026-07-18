# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Family-owned specification, preparation, construction, and audit for ITT models.

The family module is the shortest executable route through one statistical model:
it validates a declared specification, resolves the data rows and factory
arguments, names the parameters that diagnostics must inspect, and writes the
family-specific analysis/PPC audits.  The general pipeline remains responsible
for shared sampling and report stages.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from language_reading_predictors.statistical_models.definitions import FLOORED
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES

if TYPE_CHECKING:
    from language_reading_predictors.statistical_models.context import (
        ModelSpec,
        StatisticalFitContext,
    )
    from language_reading_predictors.statistical_models.factories import BuiltModel
    from language_reading_predictors.statistical_models.preprocessing import (
        PreparedData,
    )


_LEGACY_KEYS = frozenset(
    {
        "adjust_for",
        "alpha_sigma",
        "cross_symbols",
        "drop_missing_pre",
        "floor_estimand_role",
        "floor_rule",
        "floor_rule_provenance",
        "gamma_own_sigma",
        "kappa_sigma",
        "outcomes",
        "pre_required",
        "restrict_complete",
        "tau_moderator_interaction",
        "tau_moderator_is_covariate",
        "tau_moderator_symbol",
        "tau_sigma",
        "use_age_gp",
        "use_age_linear",
        "use_own_baseline",
        "use_own_baseline_gp",
        "use_varying_tau",
    }
)


def _tuple_of_strings(
    value: Any,
    *,
    name: str,
    optional: bool = False,
) -> tuple[str, ...] | None:
    if value is None and optional:
        return None
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item for item in value
    ):
        suffix = " or None" if optional else ""
        raise TypeError(f"{name} must be a sequence of non-empty strings{suffix}")
    return tuple(value)


def _legacy_bool(extra: Mapping[str, Any], name: str, default: bool) -> bool:
    value = extra.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"ITT setting {name!r} must be bool, got {type(value).__name__}")
    return value


def _legacy_optional_float(extra: Mapping[str, Any], name: str) -> float | None:
    value = extra.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"ITT setting {name!r} must be a positive number or None")
    return float(value)


@dataclass(frozen=True, slots=True)
class IttModelSettings:
    """Immutable settings declared by a single-outcome ITT model module.

    The defaults encode the project's DAG-faithful teaching model: one outcome,
    no cross-baselines, a linear age precision term, and the outcome's own baseline.
    The outcome itself lives on :class:`ModelSpec` and is resolved into the run plan.
    """

    outcomes: tuple[str, ...] | None = None
    cross_symbols: tuple[str, ...] | None = ()
    adjust_for: tuple[str, ...] = ()
    restrict_complete: tuple[str, ...] = ()
    pre_required: tuple[str, ...] | None = None
    drop_missing_pre: bool = True
    use_age_gp: bool = False
    use_own_baseline_gp: bool = False
    use_varying_tau: bool = False
    use_age_linear: bool = True
    use_own_baseline: bool = True
    tau_moderator_symbol: str | None = None
    tau_moderator_is_covariate: bool = False
    tau_moderator_interaction: bool = True
    tau_sigma: float | None = None
    alpha_sigma: float | None = None
    gamma_own_sigma: float | None = None
    kappa_sigma: float | None = None
    floor_rule: bool = False
    floor_rule_provenance: str | None = None
    floor_estimand_role: str | None = None

    def __post_init__(self) -> None:
        for name in ("outcomes", "cross_symbols", "pre_required"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _tuple_of_strings(value, name=name))
        for name in ("adjust_for", "restrict_complete"):
            object.__setattr__(
                self,
                name,
                _tuple_of_strings(getattr(self, name), name=name),
            )
        for name in (
            "drop_missing_pre",
            "use_age_gp",
            "use_own_baseline_gp",
            "use_varying_tau",
            "use_age_linear",
            "use_own_baseline",
            "tau_moderator_is_covariate",
            "tau_moderator_interaction",
            "floor_rule",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        for name in ("tau_sigma", "alpha_sigma", "gamma_own_sigma", "kappa_sigma"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive number or None")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.tau_moderator_symbol is not None and (
            not isinstance(self.tau_moderator_symbol, str)
            or not self.tau_moderator_symbol
        ):
            raise TypeError("tau_moderator_symbol must be a non-empty string or None")
        for name in ("floor_rule_provenance", "floor_estimand_role"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"{name} must be a non-empty string or None")

    @classmethod
    def for_floor_outcome(cls) -> IttModelSettings:
        """Return the shared P/N exploratory floor-rule specification."""
        return cls(
            pre_required=(),
            use_own_baseline=False,
            floor_rule=True,
            floor_rule_provenance="post_hoc_data_adaptive_t2_zero_rate",
            floor_estimand_role="exploratory_headline",
        )

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> IttModelSettings:
        """Strictly translate the former dictionary boundary during migration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown ITT setting(s): {', '.join(unknown)}. "
                "Use IttModelSettings so misspellings fail before data loading."
            )
        outcomes = (
            _tuple_of_strings(extra["outcomes"], name="outcomes", optional=True)
            if "outcomes" in extra
            else None
        )
        cross_symbols = (
            _tuple_of_strings(
                extra["cross_symbols"], name="cross_symbols", optional=True
            )
            if "cross_symbols" in extra
            else None
        )
        pre_required = (
            _tuple_of_strings(
                extra["pre_required"], name="pre_required", optional=True
            )
            if "pre_required" in extra
            else None
        )
        tau_moderator_symbol = extra.get("tau_moderator_symbol")
        if tau_moderator_symbol is not None and not isinstance(
            tau_moderator_symbol, str
        ):
            raise TypeError("ITT setting 'tau_moderator_symbol' must be str or None")
        return cls(
            outcomes=outcomes,
            cross_symbols=cross_symbols,
            adjust_for=_tuple_of_strings(
                extra.get("adjust_for", ()), name="adjust_for"
            ),
            restrict_complete=_tuple_of_strings(
                extra.get("restrict_complete", ()), name="restrict_complete"
            ),
            pre_required=pre_required,
            drop_missing_pre=_legacy_bool(extra, "drop_missing_pre", True),
            use_age_gp=_legacy_bool(extra, "use_age_gp", False),
            use_own_baseline_gp=_legacy_bool(
                extra, "use_own_baseline_gp", False
            ),
            use_varying_tau=_legacy_bool(extra, "use_varying_tau", False),
            use_age_linear=_legacy_bool(extra, "use_age_linear", False),
            use_own_baseline=_legacy_bool(extra, "use_own_baseline", True),
            tau_moderator_symbol=tau_moderator_symbol,
            tau_moderator_is_covariate=_legacy_bool(
                extra, "tau_moderator_is_covariate", False
            ),
            tau_moderator_interaction=_legacy_bool(
                extra, "tau_moderator_interaction", True
            ),
            tau_sigma=_legacy_optional_float(extra, "tau_sigma"),
            alpha_sigma=_legacy_optional_float(extra, "alpha_sigma"),
            gamma_own_sigma=_legacy_optional_float(extra, "gamma_own_sigma"),
            kappa_sigma=_legacy_optional_float(extra, "kappa_sigma"),
            floor_rule=_legacy_bool(extra, "floor_rule", False),
            floor_rule_provenance=extra.get("floor_rule_provenance"),
            floor_estimand_role=extra.get("floor_estimand_role"),
        )


@dataclass(frozen=True, slots=True)
class IttRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    outcome_symbol: str
    settings_source: str
    outcomes: tuple[str, ...]
    cross_symbols: tuple[str, ...]
    adjust_for: tuple[str, ...]
    covariates_to_load: tuple[str, ...]
    restrict_complete: tuple[str, ...]
    pre_required: tuple[str, ...] | None
    drop_missing_pre: bool
    use_age_gp: bool
    use_own_baseline_gp: bool
    use_varying_tau: bool
    use_age_linear: bool
    use_own_baseline: bool
    tau_moderator_symbol: str | None
    tau_moderator_is_covariate: bool
    tau_moderator_interaction: bool
    tau_sigma: float | None
    alpha_sigma: float | None
    gamma_own_sigma: float | None
    kappa_sigma: float | None
    floor_rule: bool
    floor_rule_provenance: str | None
    floor_estimand_role: str | None
    headline_likelihood: str

    @property
    def age_effect(self) -> str:
        return "gp" if self.use_age_gp else "linear" if self.use_age_linear else "none"

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        return {
            "phase_mode": "itt",
            "outcomes": self.outcomes,
            "covariates": self.covariates_to_load,
            "restrict_complete": self.restrict_complete,
            "drop_missing_pre": self.drop_missing_pre,
            "pre_required": self.pre_required,
        }

    def factory_kwargs(
        self,
        *,
        effective_adjustment: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Arguments shared by every factory call for this ITT plan."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "use_age_gp": self.use_age_gp,
            "use_own_baseline_gp": self.use_own_baseline_gp,
            "use_varying_tau": self.use_varying_tau,
            "adjust_for": self.adjust_for
            if effective_adjustment is None
            else effective_adjustment,
            "cross_symbols": self.cross_symbols,
            "use_age_linear": self.use_age_linear,
            "use_own_baseline": self.use_own_baseline,
            "tau_moderator_symbol": self.tau_moderator_symbol,
            "tau_moderator_is_covariate": self.tau_moderator_is_covariate,
            "tau_moderator_interaction": self.tau_moderator_interaction,
            "tau_sigma": self.tau_sigma,
            "alpha_sigma": self.alpha_sigma,
            "gamma_own_sigma": self.gamma_own_sigma,
            "kappa_sigma": self.kappa_sigma,
        }

    def recipe_markdown(self, *, title: str) -> str:
        """Explain the executable plan in undergraduate-friendly Markdown."""
        if self.floor_rule:
            question = (
                "What is the effect of random assignment on the probability of "
                "moving above the test floor at t2 among children observed at the "
                "floor at t1?"
            )
            likelihood = (
                "The headline uses a Bernoulli model for whether the child moves "
                "above zero. Graded Beta-Binomial fits are secondary checks."
            )
        else:
            question = (
                f"What is the effect of random assignment on the t2 "
                f"{self.outcome_symbol} score in the fitted available-case sample?"
            )
            likelihood = (
                "The observed score is treated as a bounded count and modelled with "
                "a Beta-Binomial likelihood, which allows more between-child "
                "variation than a plain Binomial model."
            )
        baseline_terms = []
        if self.use_own_baseline:
            baseline_terms.append("the outcome's t1 score as a linear precision term")
        if self.use_own_baseline_gp:
            baseline_terms.append("a flexible smooth function of the t1 score")
        if self.cross_symbols:
            baseline_terms.append(
                "cross-baselines: " + ", ".join(self.cross_symbols)
            )
        baseline_text = "; ".join(baseline_terms) if baseline_terms else "none"
        adjustment_text = ", ".join(self.adjust_for) if self.adjust_for else "none"
        if self.tau_moderator_symbol is None:
            moderation_text = "none"
        else:
            moderation_text = (
                f"{self.tau_moderator_symbol}; the model includes its main effect"
                + (
                    " and allows the treatment effect to vary with it"
                    if self.tau_moderator_interaction
                    else " but keeps the treatment effect constant"
                )
            )
        restriction_text = (
            ", ".join(self.restrict_complete)
            if self.restrict_complete
            else "none"
        )
        return (
            "Note: Generated from the validated ITT run plan; template drafted by "
            "a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Question\n\n{question}\n\n"
            "## Analysis rows\n\n"
            "The randomised comparison uses the t1 to t2 trial window, with one "
            "available row per child. Requested outcomes: "
            f"{', '.join(self.outcomes)}. Complete-case-only restrictions: "
            f"{restriction_text}.\n\n"
            f"## Model\n\n{likelihood}\n\n"
            "The treatment term compares the assigned intervention and wait-list "
            "groups. Positive values mean the intervention helps. Random assignment "
            "supports the causal interpretation of this term within the observed "
            "analysis set; extending it to every randomised child also requires a "
            "missing-outcome assumption. Random assignment does not make the other "
            "coefficients causal.\n\n"
            f"Age effect: {self.age_effect}. Baseline terms: {baseline_text}. "
            f"Requested additional adjustment terms: {adjustment_text}. Treatment-"
            f"effect moderator: {moderation_text}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution: a range of effect values and "
            "how much support the data and model give each one. Interpret it only "
            "after the convergence gate and posterior-predictive checks pass. The "
            "report translates the treatment effect from log-odds to items or "
            "percentage points, as appropriate, for interpretation.\n\n"
            "The saved `config.json` contains the same resolved run plan in a "
            "machine-readable form.\n"
        )


def declared_itt_settings(spec: ModelSpec) -> tuple[IttModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: ITT settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, IttModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='itt' requires IttModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return IttModelSettings.from_legacy_extra(
        spec.extra, model_id=spec.model_id
    ), "legacy_extra"


def _tag_settings_source(
    spec: ModelSpec,
    serialized: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    """Add declaration provenance without allowing a settings field to replace it."""
    if "source" in serialized:
        raise ValueError(
            f"{spec.model_id}: model_settings field 'source' is reserved for "
            "declaration provenance"
        )
    return {"source": source, **serialized}


def declared_settings_dict(spec: ModelSpec) -> dict[str, Any]:
    """Serialize the declared family settings without resolving data-dependent terms."""
    if spec.kind == "itt":
        settings, source = declared_itt_settings(spec)
        return _tag_settings_source(spec, asdict(settings), source=source)
    if spec.model_settings is not None:
        settings = spec.model_settings
        if not is_dataclass(settings) or isinstance(settings, type):
            raise TypeError(
                f"{spec.model_id}: typed model_settings must be a dataclass instance, "
                f"got {type(settings).__name__}"
            )
        return _tag_settings_source(spec, asdict(settings), source="typed")
    return dict(spec.extra)


def _reject_duplicates(model_id: str, name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{model_id}: {name} contains duplicate symbols: {values!r}")


def resolve_itt_run_plan(spec: ModelSpec) -> IttRunPlan:
    """Resolve and validate an ITT specification before any data are loaded."""
    if spec.kind != "itt":
        raise ValueError(f"{spec.model_id}: expected kind 'itt', got {spec.kind!r}")
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: outcome_symbol is required for an ITT model")

    settings, source = declared_itt_settings(spec)
    own = spec.outcome_symbol
    outcomes = settings.outcomes
    if outcomes is None:
        outcomes = (own,) if source == "typed" else ITT_OUTCOMES
    cross_symbols = settings.cross_symbols
    if cross_symbols is None:
        # The legacy factory inferred cross-baselines from the outcomes actually
        # loaded. Preserve that behaviour for saved or ad-hoc legacy declarations
        # that specify ``outcomes`` but omit ``cross_symbols``.
        cross_symbols = tuple(symbol for symbol in outcomes if symbol != own)

    for name, values in (
        ("outcomes", outcomes),
        ("cross_symbols", cross_symbols),
        ("adjust_for", settings.adjust_for),
        ("restrict_complete", settings.restrict_complete),
    ):
        _reject_duplicates(spec.model_id, name, values)
    if own not in outcomes:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol {own!r} must appear in outcomes={outcomes!r}"
        )
    if own in cross_symbols:
        raise ValueError(f"{spec.model_id}: the outcome cannot be its own cross-baseline")
    missing_cross = sorted(set(cross_symbols) - set(outcomes))
    if missing_cross:
        raise ValueError(
            f"{spec.model_id}: cross-baselines are not loaded as outcomes: "
            f"{', '.join(missing_cross)}"
        )
    overlap = sorted(set(settings.adjust_for) & set(settings.restrict_complete))
    if overlap:
        raise ValueError(
            f"{spec.model_id}: settings cannot both adjust for and only restrict on "
            f"the same covariate(s): {', '.join(overlap)}"
        )
    if settings.pre_required is not None:
        missing_pre = sorted(set(settings.pre_required) - set(outcomes))
        if missing_pre:
            raise ValueError(
                f"{spec.model_id}: pre_required contains unloaded outcome(s): "
                f"{', '.join(missing_pre)}"
            )
    if settings.use_age_gp and settings.use_age_linear:
        raise ValueError(
            f"{spec.model_id}: use_age_gp and use_age_linear are mutually exclusive"
        )
    if settings.floor_rule and settings.use_varying_tau:
        raise ValueError(
            f"{spec.model_id}: floor_rule cannot use a varying treatment effect"
        )
    if settings.floor_rule and settings.tau_moderator_symbol is not None:
        raise ValueError(
            f"{spec.model_id}: floor_rule cannot use treatment-effect moderation"
        )
    if (
        settings.tau_moderator_symbol is None
        and not settings.tau_moderator_interaction
    ):
        raise ValueError(
            f"{spec.model_id}: tau_moderator_interaction has no effect without a "
            "tau_moderator_symbol"
        )
    if (
        settings.tau_moderator_symbol is None
        and settings.tau_moderator_is_covariate
    ):
        raise ValueError(
            f"{spec.model_id}: tau_moderator_is_covariate requires a "
            "tau_moderator_symbol"
        )
    if settings.tau_moderator_symbol is not None:
        moderator = settings.tau_moderator_symbol
        if settings.tau_moderator_is_covariate:
            if moderator in settings.adjust_for:
                raise ValueError(
                    f"{spec.model_id}: covariate moderator {moderator!r} already gets "
                    "a main effect and cannot also appear in adjust_for"
                )
            if moderator == "A" and settings.use_age_linear:
                raise ValueError(
                    f"{spec.model_id}: age moderation already supplies a linear age "
                    "main effect; disable use_age_linear"
                )
        elif moderator not in outcomes:
            raise ValueError(
                f"{spec.model_id}: baseline moderator {moderator!r} must be loaded "
                "in outcomes"
            )
        elif (moderator == own and settings.use_own_baseline) or (
            moderator in cross_symbols
        ):
            raise ValueError(
                f"{spec.model_id}: baseline moderator {moderator!r} already gets a "
                "linear main effect; remove the duplicate baseline term"
            )
    if settings.floor_rule:
        if own not in FLOORED:
            raise ValueError(
                f"{spec.model_id}: floor_rule is only registered for "
                f"{sorted(FLOORED)}, got {own!r}"
            )
        if settings.use_own_baseline or settings.use_own_baseline_gp:
            raise ValueError(
                f"{spec.model_id}: floor_rule uses baseline only for eligibility; "
                "disable own-baseline model terms"
            )
        if cross_symbols:
            raise ValueError(f"{spec.model_id}: floor_rule cannot use cross-baselines")
        if settings.pre_required != ():
            raise ValueError(
                f"{spec.model_id}: floor_rule must set pre_required=() so missing "
                "eligibility remains visible"
            )
        if not settings.floor_rule_provenance or not settings.floor_estimand_role:
            raise ValueError(
                f"{spec.model_id}: floor_rule requires provenance and estimand role"
            )
    elif settings.floor_rule_provenance or settings.floor_estimand_role:
        raise ValueError(
            f"{spec.model_id}: floor metadata is only valid when floor_rule=True"
        )

    covariates_to_load = list(settings.adjust_for)
    if (
        settings.tau_moderator_is_covariate
        and settings.tau_moderator_symbol not in {None, "A"}
    ):
        covariates_to_load.append(settings.tau_moderator_symbol)

    return IttRunPlan(
        model_id=spec.model_id,
        outcome_symbol=own,
        settings_source=source,
        outcomes=outcomes,
        cross_symbols=cross_symbols,
        adjust_for=settings.adjust_for,
        covariates_to_load=tuple(covariates_to_load),
        restrict_complete=settings.restrict_complete,
        pre_required=settings.pre_required,
        drop_missing_pre=settings.drop_missing_pre,
        use_age_gp=settings.use_age_gp,
        use_own_baseline_gp=settings.use_own_baseline_gp,
        use_varying_tau=settings.use_varying_tau,
        use_age_linear=settings.use_age_linear,
        use_own_baseline=settings.use_own_baseline,
        tau_moderator_symbol=settings.tau_moderator_symbol,
        tau_moderator_is_covariate=settings.tau_moderator_is_covariate,
        tau_moderator_interaction=settings.tau_moderator_interaction,
        tau_sigma=settings.tau_sigma,
        alpha_sigma=settings.alpha_sigma,
        gamma_own_sigma=settings.gamma_own_sigma,
        kappa_sigma=settings.kappa_sigma,
        floor_rule=settings.floor_rule,
        floor_rule_provenance=settings.floor_rule_provenance,
        floor_estimand_role=settings.floor_estimand_role,
        headline_likelihood=(
            "bernoulli_offfloor" if settings.floor_rule else "beta_binomial"
        ),
    )


def prepare_itt_data(
    plan: IttRunPlan,
    *,
    loader: Callable[..., PreparedData] | None = None,
) -> tuple[PreparedData, tuple[str, ...]]:
    """Load the rows named by ``plan`` and return adjusters actually available.

    The loader may drop a covariate or missingness indicator that is constant on
    the fitted rows.  Returning the effective set here keeps preprocessing,
    construction, diagnostics, and metadata on the same family-owned contract.
    """

    if loader is None:
        from language_reading_predictors.statistical_models.preprocessing import (
            load_and_prepare,
        )

        loader = load_and_prepare
    prepared = loader(**plan.prepare_kwargs())
    adjustment = tuple(
        name for name in plan.adjust_for if name in prepared.covariates
    )
    return prepared, adjustment


def build_itt_from_plan(
    plan: IttRunPlan,
    prepared: PreparedData,
    *,
    effective_adjustment: tuple[str, ...],
    likelihood: str | None = None,
    builder: Callable[..., BuiltModel] | None = None,
) -> BuiltModel:
    """Build exactly the model described by a validated ITT run plan."""

    if builder is None:
        from language_reading_predictors.statistical_models.factories import (
            build_itt_model,
        )

        builder = build_itt_model
    return builder(
        prepared,
        likelihood=likelihood or plan.headline_likelihood,
        **plan.factory_kwargs(effective_adjustment=effective_adjustment),
    )


def itt_diagnostic_variables(
    plan: IttRunPlan,
    effective_adjustment: tuple[str, ...],
    *,
    likelihood: str | None = None,
) -> list[str]:
    """Name the scalar parameters present in this particular ITT model."""

    effective_likelihood = likelihood or plan.headline_likelihood
    variables = ["alpha", "tau"]
    if plan.use_own_baseline:
        variables.append("gamma_own")
    if plan.use_age_linear:
        variables.append("gamma_A")
    if effective_likelihood == "beta_binomial":
        variables.append("kappa")
    variables.extend(f"gamma_{name}" for name in effective_adjustment)
    if plan.tau_moderator_symbol is not None:
        variables.append("gamma_tau_mod")
        if plan.tau_moderator_interaction:
            variables.append("gamma_tau_int")
    return variables


def write_itt_analysis_audit(
    context: StatisticalFitContext,
    prepared: PreparedData,
    outcomes: Sequence[str],
    *,
    loader: Callable[..., PreparedData] | None = None,
) -> None:
    """Persist fitted-arm counts and full-randomised extreme-case bounds."""

    from language_reading_predictors.statistical_models.itt_audit import (
        analysis_set_table,
        randomised_postscore_bounds,
    )

    if loader is None:
        from language_reading_predictors.statistical_models.preprocessing import (
            load_and_prepare,
        )

        loader = load_and_prepare

    analysis_frames = []
    bound_frames = []
    multiple = len(outcomes) > 1
    for symbol in outcomes:
        analysis = analysis_set_table(prepared, outcome_symbol=symbol)
        if multiple:
            analysis.insert(0, "outcome", symbol)
        analysis_frames.append(analysis)

        attrition_source = loader(
            path=getattr(prepared, "data_path", None) or None,
            phase_mode="itt",
            outcomes=(symbol,),
            pre_required=(),
        )
        bound_frames.append(randomised_postscore_bounds(attrition_source, symbol))

    analysis_df = pd.concat(analysis_frames, ignore_index=True)
    bounds_df = pd.concat(bound_frames, ignore_index=True)
    analysis_df.to_csv(
        os.path.join(context.output_dir, "analysis_set.csv"), index=False
    )
    bounds_df.to_csv(
        os.path.join(context.output_dir, "attrition_bounds.csv"), index=False
    )
    context.tables["analysis_set"] = analysis_df
    context.tables["attrition_bounds"] = bounds_df


def write_itt_ppc_calibration(
    context: StatisticalFitContext,
    prepared: PreparedData,
    outcomes: Sequence[str],
    *,
    node: str = "y_post",
    filename: str = "posterior_predictive_calibration.csv",
) -> pd.DataFrame:
    """Save bounded-score posterior-predictive metrics for ITT rows."""

    from language_reading_predictors.statistical_models.ppc_audit import (
        score_ppc_by_arm_and_baseline,
        score_ppc_distribution_shape,
    )

    predictive = np.asarray(
        context.trace.posterior_predictive[node].values, dtype=float
    )
    frames = []
    shape_frames = []
    if len(outcomes) == 1 and predictive.shape[-1] == prepared.n_obs:
        symbol = outcomes[0]
        observed = None
        denominator = None
        if node == "y_offfloor":
            observed = (prepared.post_counts[symbol] > 0).astype(float)
            denominator = 1
        frames.append(
            score_ppc_by_arm_and_baseline(
                prepared,
                symbol,
                predictive,
                observed_counts=observed,
                n_trials=denominator,
                ci_prob=context.reporting.ci_prob,
            )
        )
    else:
        constant = context.trace.constant_data
        cell_rows = np.asarray(
            constant["y_post_cell_row"].values, dtype=int
        ).ravel()
        cell_outcomes = np.asarray(
            constant["y_post_cell_outcome"].values, dtype=int
        ).ravel()
        if predictive.shape[-1] != cell_rows.size:
            raise ValueError("joint posterior-predictive cells do not match cell map")
        for outcome_index, symbol in enumerate(outcomes):
            selected = cell_outcomes == outcome_index
            outcome_rows = cell_rows[selected]
            frames.append(
                score_ppc_by_arm_and_baseline(
                    prepared,
                    symbol,
                    predictive[..., selected],
                    row_indices=outcome_rows,
                    ci_prob=context.reporting.ci_prob,
                )
            )
            shape_frames.append(
                score_ppc_distribution_shape(
                    symbol,
                    predictive[..., selected],
                    np.asarray(prepared.post_counts[symbol], dtype=float)[outcome_rows],
                    n_trials=int(prepared.n_trials[symbol]),
                    ci_prob=context.reporting.ci_prob,
                )
            )

    calibration = pd.concat(frames, ignore_index=True)
    calibration.to_csv(os.path.join(context.output_dir, filename), index=False)
    context.tables[os.path.splitext(filename)[0]] = calibration
    if shape_frames:
        shape_calibration = pd.concat(shape_frames, ignore_index=True)
        shape_filename = "posterior_predictive_shape_calibration.csv"
        shape_calibration.to_csv(
            os.path.join(context.output_dir, shape_filename), index=False
        )
        context.tables[os.path.splitext(shape_filename)[0]] = shape_calibration
    return calibration
