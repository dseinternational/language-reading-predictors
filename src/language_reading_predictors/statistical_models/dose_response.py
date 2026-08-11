# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for dose-response models.

The family estimates observational associations between intervention-session dose
and bounded skill outcomes.  Resolution happens before an output transaction is
opened or RLI data are loaded, while preserving the existing fitted equations and
artefacts (#394 pillar 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec

__all__ = [
    "DoseResponseModelSettings",
    "DoseResponseRunPlan",
    "declared_dose_response_settings",
    "resolve_dose_response_run_plan",
]


_FAMILY_KEYS = frozenset(
    {
        "adjust_baseline_symbol",
        "dose_covariate",
        "dose_stage_covariate",
        "period_varying_dose",
        "use_subject_random_intercept",
        "ability_adjust_symbols",
        "outcomes",
        "adjust_group",
        "adjust_age",
    }
)
_GLOBAL_KEYS = frozenset({"target_accept"})
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        _string(item, name=name)
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class DoseResponseModelSettings:
    """Immutable declaration for one dose-response model."""

    adjust_baseline_symbol: str = "W"
    dose_covariate: str = "attend"
    dose_stage_covariate: str | None = None
    period_varying_dose: bool = True
    use_subject_random_intercept: bool = True
    ability_adjust_symbols: tuple[str, ...] = ()
    outcomes: tuple[str, ...] = ()
    adjust_group: bool = True
    adjust_age: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adjust_baseline_symbol",
            _string(self.adjust_baseline_symbol, name="adjust_baseline_symbol"),
        )
        object.__setattr__(
            self,
            "dose_covariate",
            _string(self.dose_covariate, name="dose_covariate"),
        )
        object.__setattr__(
            self,
            "dose_stage_covariate",
            _optional_string(
                self.dose_stage_covariate,
                name="dose_stage_covariate",
            ),
        )
        object.__setattr__(
            self,
            "ability_adjust_symbols",
            _tuple_of_strings(
                self.ability_adjust_symbols,
                name="ability_adjust_symbols",
            ),
        )
        object.__setattr__(
            self,
            "outcomes",
            _tuple_of_strings(self.outcomes, name="outcomes"),
        )
        for name in (
            "period_varying_dose",
            "use_subject_random_intercept",
            "adjust_group",
            "adjust_age",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name=name))

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> DoseResponseModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown dose-response setting(s): "
                f"{', '.join(unknown)}. Declare DoseResponseModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            adjust_baseline_symbol=extra.get("adjust_baseline_symbol", "W"),
            dose_covariate=extra.get("dose_covariate", "attend"),
            dose_stage_covariate=extra.get("dose_stage_covariate"),
            period_varying_dose=extra.get("period_varying_dose", True),
            use_subject_random_intercept=extra.get(
                "use_subject_random_intercept",
                True,
            ),
            ability_adjust_symbols=extra.get("ability_adjust_symbols", ()),
            outcomes=extra.get("outcomes", ()),
            adjust_group=extra.get("adjust_group", True),
            adjust_age=extra.get("adjust_age", True),
        )


@dataclass(frozen=True, slots=True)
class DoseResponseRunPlan:
    """Concrete, validated instructions for a complete dose-response fit."""

    model_id: str
    settings_source: str
    study_id: str
    outcome_symbol: str
    adjust_baseline_symbol: str
    dose_covariate: str
    dose_stage_covariate: str | None
    period_varying_dose: bool
    use_subject_random_intercept: bool
    ability_adjust_symbols: tuple[str, ...]
    outcomes: tuple[str, ...]
    adjust_group: bool
    adjust_age: bool
    phase_mode: str
    loader_covariates: tuple[str, ...]
    observation_node: str
    compute_loo: bool
    loo_unit: str
    focal_term: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan."""
        return {
            "phase_mode": self.phase_mode,
            "outcomes": self.outcomes,
            "covariates": self.loader_covariates,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_dose_response_model``."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "adjust_baseline_symbol": self.adjust_baseline_symbol,
            "dose_covariate": self.dose_covariate,
            "dose_stage_covariate": self.dose_stage_covariate,
            "period_varying_dose": self.period_varying_dose,
            "use_subject_random_intercept": self.use_subject_random_intercept,
            "adjust_group": self.adjust_group,
            "adjust_age": self.adjust_age,
            "ability_adjust_symbols": self.ability_adjust_symbols,
        }

    def diagnostic_vars(self) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        names = ["alpha", "gamma_own", "kappa"]
        if self.use_subject_random_intercept:
            names.append("sigma_child")
        if self.adjust_group:
            names.append("beta_G")
        if self.adjust_age:
            names.append("gamma_A")
        if self.period_varying_dose:
            names.extend(["mu_dose", "sigma_dose", "beta_dose_phase"])
        else:
            names.append("beta_dose")
        if self.dose_stage_covariate is not None:
            names.append("gamma_dose_stage")
        names.extend(f"gamma_{symbol}_pre" for symbol in self.ability_adjust_symbols)
        return names

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        outcomes = ", ".join(self.outcomes)
        ability = (
            ", ".join(self.ability_adjust_symbols)
            if self.ability_adjust_symbols
            else "none"
        )
        return (
            "Note: Generated from the validated dose-response run plan; template "
            "drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. Loaded outcomes: {outcomes}. "
            f"Own baseline: `{self.adjust_baseline_symbol}`. Dose: "
            f"`{self.dose_covariate}` (period-varying: "
            f"{self.period_varying_dose}). Stage-dose covariate: "
            f"{self.dose_stage_covariate or 'none'}. Ability adjustments: "
            f"{ability}. Group adjustment: {self.adjust_group}. Age adjustment: "
            f"{self.adjust_age}. Child random intercept: "
            f"{self.use_subject_random_intercept}.\n\n"
            "## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. Interpret the posterior only after the "
            "zero-divergence convergence gate, posterior-predictive checks and "
            "power-scaling sensitivity diagnostics pass. The saved `config.json` "
            "contains the same resolved run plan in machine-readable form.\n"
        )


def declared_dose_response_settings(
    spec: ModelSpec,
) -> tuple[DoseResponseModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: dose-response settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, DoseResponseModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='dose_response' requires "
                f"DoseResponseModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        DoseResponseModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_dose_response_run_plan(spec: ModelSpec) -> DoseResponseRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "dose_response":
        raise ValueError(
            f"{spec.model_id}: expected kind 'dose_response', got {spec.kind!r}"
        )
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: dose_response requires study_id='rli', got "
            f"{spec.study_id!r}"
        )
    if not spec.outcome_symbol:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol is required for dose_response"
        )

    settings, source = declared_dose_response_settings(spec)
    outcome = spec.outcome_symbol
    outcomes = settings.outcomes or (outcome,)
    required = {
        outcome,
        settings.adjust_baseline_symbol,
        *settings.ability_adjust_symbols,
    }
    missing = sorted(required - set(outcomes))
    if missing:
        raise ValueError(
            f"{spec.model_id}: outcomes must load every fitted outcome/baseline/"
            f"ability symbol; missing {missing!r} from {outcomes!r}"
        )
    if settings.dose_stage_covariate == settings.dose_covariate:
        raise ValueError(
            f"{spec.model_id}: dose_stage_covariate must differ from dose_covariate"
        )
    loader_covariates = tuple(
        value
        for value in (settings.dose_covariate, settings.dose_stage_covariate)
        if value is not None
    )
    focal = "mu_dose" if settings.period_varying_dose else "beta_dose"

    return DoseResponseRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        outcome_symbol=outcome,
        adjust_baseline_symbol=settings.adjust_baseline_symbol,
        dose_covariate=settings.dose_covariate,
        dose_stage_covariate=settings.dose_stage_covariate,
        period_varying_dose=settings.period_varying_dose,
        use_subject_random_intercept=settings.use_subject_random_intercept,
        ability_adjust_symbols=settings.ability_adjust_symbols,
        outcomes=outcomes,
        adjust_group=settings.adjust_group,
        adjust_age=settings.adjust_age,
        phase_mode="all",
        loader_covariates=loader_covariates,
        observation_node="y_post",
        compute_loo=True,
        loo_unit="observation_row",
        focal_term=focal,
        design=(
            "Period-resolved conditional-change model over all RLI transitions, "
            "with intervention-session dose standardised over the fitted rows and "
            "entered as partially pooled period slopes or one pooled slope."
        ),
        estimand=(
            "The adjusted association between a one-standard-deviation increase in "
            "session dose and the post-score, conditional on the selected own "
            "baseline, group, age and declared ability terms."
        ),
        causal_status=(
            "Observational association, not a randomised treatment effect. Session "
            "dose is post-randomisation and may be confounded by attendance and "
            "engagement processes."
        ),
        analysis_population=(
            f"Available RLI transition rows with observed {outcome}, "
            f"{settings.adjust_baseline_symbol} baseline and dose covariates."
        ),
        missing_data_assumption=(
            "Available-case analysis under ignorable missingness conditional on the "
            "modelled variables; rows missing a required score, dose or group value "
            "are excluded before fitting."
        ),
    )
