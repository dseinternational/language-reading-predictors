# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and resolved plans for RLI and RLM factor models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from language_reading_predictors.statistical_models.context import ModelSpec

__all__ = [
    "CorrFactorModelSettings",
    "CorrFactorRunPlan",
    "declared_corr_factor_settings",
    "resolve_corr_factor_run_plan",
]


DomainItems = tuple[tuple[str, tuple[str, ...]], ...]

_RLI_DEFAULT_DOMAINS: DomainItems = (
    ("vocabulary", ("R", "E")),
    ("code", ("L", "B")),
    ("grammar", ("F", "T")),
)
_FAMILY_KEYS = frozenset(
    {
        "domains",
        "structural_covariates",
        "structural_factors",
        "use_group",
        "use_age",
        "post_time",
        "loading_prior",
        "comm_alpha",
        "comm_beta",
        "loading_mu",
        "loading_sigma",
        "residual_sigma",
        "predictor_slope_sigma",
        "focal_slope_sigma",
        "lkj_eta",
        "wave",
        "single_indicator_reliability",
        "study_id",
    }
)
_GLOBAL_KEYS = frozenset({"target_accept"})
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _optional_tuple_of_strings(
    value: Any,
    *,
    name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _tuple_of_strings(value, name=name)


def _domains(value: Any) -> DomainItems:
    if isinstance(value, Mapping):
        raw = tuple(value.items())
    elif isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("domains must be a mapping or sequence of (name, indicators)")
    else:
        raw = tuple(value)
    out: list[tuple[str, tuple[str, ...]]] = []
    for entry in raw:
        if not isinstance(entry, Sequence) or isinstance(entry, str) or len(entry) != 2:
            raise TypeError("each domain must be a (name, indicators) pair")
        name, indicators = entry
        if not isinstance(name, str) or not name:
            raise TypeError(f"domain names must be non-empty strings, got {name!r}")
        symbols = _tuple_of_strings(indicators, name=f"domain {name!r} indicators")
        if not symbols:
            raise ValueError(f"domain {name!r} must contain at least one indicator")
        out.append((name, symbols))
    if not out:
        raise ValueError("domains must contain at least one latent domain")
    names = [name for name, _ in out]
    if len(names) != len(set(names)):
        raise ValueError("domains contains duplicate domain names")
    all_symbols = [symbol for _, symbols in out for symbol in symbols]
    if len(all_symbols) != len(set(all_symbols)):
        raise ValueError("an indicator may belong to only one domain")
    return tuple(out)


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean or None, got {value!r}")
    return value


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, name=name)


def _optional_finite_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True, slots=True)
class CorrFactorModelSettings:
    """Immutable declaration shared by the RLI and historical RLM ports."""

    domains: DomainItems = _RLI_DEFAULT_DOMAINS
    structural_covariates: tuple[str, ...] | None = None
    structural_factors: tuple[str, ...] | None = None
    use_group: bool | None = None
    use_age: bool | None = None
    post_time: int | None = None
    loading_prior: Literal["communality", "free"] | None = None
    comm_alpha: float | None = None
    comm_beta: float | None = None
    loading_mu: float | None = None
    loading_sigma: float | None = None
    residual_sigma: float | None = None
    predictor_slope_sigma: float | None = None
    focal_slope_sigma: float | None = None
    lkj_eta: float = 2.0
    wave: int | None = None
    single_indicator_reliability: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", _domains(self.domains))
        for name in ("structural_covariates", "structural_factors"):
            object.__setattr__(
                self,
                name,
                _optional_tuple_of_strings(getattr(self, name), name=name),
            )
        for name in ("use_group", "use_age"):
            object.__setattr__(
                self,
                name,
                _optional_bool(getattr(self, name), name=name),
            )
        for name in ("post_time", "wave"):
            object.__setattr__(
                self,
                name,
                _optional_positive_int(getattr(self, name), name=name),
            )
        if self.loading_prior not in {None, "communality", "free"}:
            raise ValueError(
                "loading_prior must be 'communality', 'free' or None, got "
                f"{self.loading_prior!r}"
            )
        for name in (
            "comm_alpha",
            "comm_beta",
            "loading_sigma",
            "residual_sigma",
            "predictor_slope_sigma",
            "focal_slope_sigma",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_float(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "loading_mu",
            _optional_finite_float(self.loading_mu, name="loading_mu"),
        )
        object.__setattr__(self, "lkj_eta", _positive_float(self.lkj_eta, name="lkj_eta"))
        reliability = self.single_indicator_reliability
        if reliability is not None:
            reliability = _positive_float(
                reliability,
                name="single_indicator_reliability",
            )
            if reliability >= 1:
                raise ValueError("single_indicator_reliability must be in (0, 1)")
            object.__setattr__(self, "single_indicator_reliability", reliability)

    @classmethod
    def from_legacy_extra(
        cls,
        extra: Mapping[str, Any],
        *,
        model_id: str,
    ) -> CorrFactorModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown correlated-factor setting(s): "
                f"{', '.join(unknown)}. Declare CorrFactorModelSettings so "
                "misspellings fail fast."
            )
        return cls(
            domains=_domains(extra.get("domains") or _RLI_DEFAULT_DOMAINS),
            structural_covariates=extra.get("structural_covariates"),
            structural_factors=extra.get("structural_factors"),
            use_group=extra.get("use_group"),
            use_age=extra.get("use_age"),
            post_time=extra.get("post_time"),
            loading_prior=extra.get("loading_prior"),
            comm_alpha=extra.get("comm_alpha"),
            comm_beta=extra.get("comm_beta"),
            loading_mu=extra.get("loading_mu"),
            loading_sigma=extra.get("loading_sigma"),
            residual_sigma=extra.get("residual_sigma"),
            predictor_slope_sigma=extra.get("predictor_slope_sigma"),
            focal_slope_sigma=extra.get("focal_slope_sigma"),
            lkj_eta=extra.get("lkj_eta", 2.0),
            wave=extra.get("wave"),
            single_indicator_reliability=extra.get(
                "single_indicator_reliability"
            ),
        )


@dataclass(frozen=True, slots=True)
class CorrFactorRunPlan:
    """Concrete, validated instructions for one RLI or RLM factor fit."""

    model_id: str
    settings_source: str
    study_id: Literal["rli", "rlm"]
    port: Literal["rli", "rlm"]
    outcome_symbol: str | None
    domains: DomainItems
    indicators: tuple[str, ...]
    structural_covariates: tuple[str, ...]
    active_structural_covariates: tuple[str, ...]
    structural_factors: tuple[str, ...] | None
    use_group: bool
    use_age: bool
    post_time: int | None
    loading_prior: Literal["communality", "free"] | None
    comm_alpha: float
    comm_beta: float
    loading_mu: float
    loading_sigma: float
    residual_sigma: float
    predictor_slope_sigma: float
    focal_slope_sigma: float | None
    lkj_eta: float
    wave: int | None
    single_indicator_reliability: float | None
    observation_nodes: tuple[str, ...]
    compute_loo: bool
    loo_unit: str | None
    focal_term: str | None
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def domain_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return the factory's ordered domain mapping."""
        return dict(self.domains)

    def with_active_structural_covariates(
        self,
        covariates: tuple[str, ...],
    ) -> CorrFactorRunPlan:
        """Record the covariates retained after constant-column filtering."""
        if self.port != "rli":
            raise ValueError("active structural covariates apply only to the RLI port")
        unknown = sorted(set(covariates) - set(self.structural_covariates))
        if unknown:
            raise ValueError(f"active structural covariates were not declared: {unknown}")
        return replace(self, active_structural_covariates=tuple(covariates))

    def rli_prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for RLI ``load_and_prepare``."""
        if self.port != "rli" or self.outcome_symbol is None:
            raise ValueError("rli_prepare_kwargs requires an RLI factor plan")
        outcomes = tuple(dict.fromkeys((self.outcome_symbol, *self.indicators)))
        return {
            "phase_mode": "span",
            "post_time": self.post_time,
            "outcomes": outcomes,
            "covariates": self.structural_covariates,
        }

    def rli_factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_correlated_factor_model``."""
        if self.port != "rli" or self.outcome_symbol is None:
            raise ValueError("rli_factory_kwargs requires an RLI factor plan")
        return {
            "outcome_symbol": self.outcome_symbol,
            "domains": self.domain_mapping(),
            "structural_covariates": self.active_structural_covariates,
            "structural_factors": self.structural_factors,
            "use_group": self.use_group,
            "use_age": self.use_age,
            "loading_prior": self.loading_prior,
            "comm_alpha": self.comm_alpha,
            "comm_beta": self.comm_beta,
            "loading_mu": self.loading_mu,
            "loading_sigma": self.loading_sigma,
            "residual_sigma": self.residual_sigma,
            "predictor_slope_sigma": self.predictor_slope_sigma,
            "focal_slope_sigma": self.focal_slope_sigma,
            "lkj_eta": self.lkj_eta,
        }

    def rlm_prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_rlm_wave_battery``."""
        if self.port != "rlm":
            raise ValueError("rlm_prepare_kwargs requires an RLM factor plan")
        return {"wave": self.wave, "measure_symbols": self.indicators}

    def rlm_factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_rlm_corr_factor_model``."""
        if self.port != "rlm":
            raise ValueError("rlm_factory_kwargs requires an RLM factor plan")
        return {
            "domains": self.domain_mapping(),
            "single_indicator_reliability": self.single_indicator_reliability,
            "comm_alpha": self.comm_alpha,
            "comm_beta": self.comm_beta,
            "lkj_eta": self.lkj_eta,
        }

    def diagnostic_vars(self) -> list[str]:
        """Released quantities scanned by summaries and convergence checks."""
        if self.port == "rlm":
            return ["lambda_free", "sigma_free", "factor_corr_pairs"]
        variables = [
            "alpha",
            "gamma_own",
            "kappa",
            "beta_factor",
            "lambda_load",
            "sigma_indicator",
            "communality",
            "factor_z",
        ]
        if len(self.domains) > 1:
            variables.append("factor_corr_pairs")
        if self.use_age:
            variables.append("beta_age")
        variables.extend(f"beta_{name}" for name in self.active_structural_covariates)
        if self.use_group:
            variables.append("beta_G")
        return variables

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language recipe generated from the validated plan."""
        domains = "; ".join(
            f"{name}: {', '.join(symbols)}" for name, symbols in self.domains
        )
        if self.port == "rli":
            structural = (
                "All domains"
                if self.structural_factors is None
                else ", ".join(self.structural_factors)
            )
            terms = (
                f"Outcome: `{self.outcome_symbol}`. Structural factors: {structural}. "
                "Active structural covariates: "
                f"{', '.join(self.active_structural_covariates) or 'none'}. "
                f"Age term: {self.use_age}. Group term: {self.use_group}."
            )
        else:
            terms = (
                f"Wave: {self.wave}. Single-indicator reliability: "
                f"{self.single_indicator_reliability:g}. No structural outcome leg."
            )
        return (
            "Note: Generated from the validated correlated-factor run plan; "
            "template drafted by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            f"## Terms\n\nStudy port: `{self.port}`. Domains: {domains}. {terms} "
            f"LKJ eta: {self.lkj_eta:g}.\n\n"
            "## Uncertainty and checks\n\n"
            "This measurement family does not compute PSIS-LOO because its "
            "multiple likelihood nodes do not define one unambiguous predictive "
            "unit. Release requires the zero-divergence convergence gate, "
            "posterior-predictive checks, indicator-scale prior checks and "
            "power-scaling sensitivity diagnostics. The saved `config.json` "
            "contains the same resolved run plan.\n"
        )


def declared_corr_factor_settings(
    spec: ModelSpec,
) -> tuple[CorrFactorModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: correlated-factor settings cannot be split "
                f"between model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, CorrFactorModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='corr_factor' requires "
                f"CorrFactorModelSettings, got {type(settings).__name__}"
            )
        return settings, "typed"
    return (
        CorrFactorModelSettings.from_legacy_extra(
            spec.extra,
            model_id=spec.model_id,
        ),
        "legacy_extra",
    )


def resolve_corr_factor_run_plan(spec: ModelSpec) -> CorrFactorRunPlan:
    """Resolve and validate either study port before context or data I/O."""
    if spec.kind != "corr_factor":
        raise ValueError(
            f"{spec.model_id}: expected kind 'corr_factor', got {spec.kind!r}"
        )
    if spec.study_id not in {"rli", "rlm"}:
        raise ValueError(
            f"{spec.model_id}: corr_factor study_id must be 'rli' or 'rlm', got "
            f"{spec.study_id!r}"
        )
    settings, source = declared_corr_factor_settings(spec)
    legacy_study = spec.extra.get("study_id")
    if legacy_study is not None and legacy_study != spec.study_id:
        raise ValueError(
            f"{spec.model_id}: legacy study_id {legacy_study!r} conflicts with "
            f"ModelSpec.study_id {spec.study_id!r}"
        )
    indicators = tuple(symbol for _, symbols in settings.domains for symbol in symbols)
    observation_nodes: tuple[str, ...]

    if spec.study_id == "rli":
        if not spec.outcome_symbol:
            raise ValueError(f"{spec.model_id}: RLI corr_factor requires an outcome")
        rlm_fields = {
            "wave": settings.wave,
            "single_indicator_reliability": settings.single_indicator_reliability,
        }
        supplied = [name for name, value in rlm_fields.items() if value is not None]
        if supplied:
            raise ValueError(
                f"{spec.model_id}: RLM-only settings are invalid for the RLI port: "
                f"{', '.join(supplied)}"
            )
        short = [name for name, symbols in settings.domains if len(symbols) < 2]
        if short:
            raise ValueError(
                f"{spec.model_id}: RLI correlated-factor domains require at least "
                f"two indicators: {', '.join(short)}"
            )
        structural_covariates = settings.structural_covariates or ("blocks",)
        structural_factors = settings.structural_factors
        if structural_factors == ():
            raise ValueError(f"{spec.model_id}: structural_factors cannot be empty")
        domain_names = {name for name, _ in settings.domains}
        bad_factors = sorted(set(structural_factors or ()) - domain_names)
        if bad_factors:
            raise ValueError(
                f"{spec.model_id}: structural_factors are not fitted domains: "
                f"{', '.join(bad_factors)}"
            )
        loading_prior = settings.loading_prior or "communality"
        free_knobs = sorted(
            name
            for name in ("loading_mu", "loading_sigma", "residual_sigma")
            if getattr(settings, name) is not None
        )
        comm_knobs = sorted(
            name
            for name in ("comm_alpha", "comm_beta")
            if getattr(settings, name) is not None
        )
        if loading_prior == "communality" and free_knobs:
            raise ValueError(
                f"{spec.model_id}: {free_knobs} only apply to loading_prior='free'"
            )
        if loading_prior == "free" and comm_knobs:
            raise ValueError(
                f"{spec.model_id}: {comm_knobs} only apply to "
                "loading_prior='communality'"
            )
        port: Literal["rli", "rlm"] = "rli"
        outcome = spec.outcome_symbol
        use_group = False if settings.use_group is None else settings.use_group
        use_age = True if settings.use_age is None else settings.use_age
        post_time = 4 if settings.post_time is None else settings.post_time
        if post_time < 2:
            raise ValueError(f"{spec.model_id}: post_time must be at least 2")
        wave = None
        reliability = None
        observation_nodes = ("Z_obs", "y_post")
        design = (
            "RLI between-child correlated-domain measurement model at baseline "
            "with a Beta-Binomial post-score structural leg."
        )
        estimand = (
            "Latent domain correlations, indicator communalities and adjusted "
            "factor-to-outcome associations."
        )
        population = (
            f"Available RLI children with observed {outcome} post-score and all "
            "required baseline indicators and active structural covariates."
        )
    else:
        if spec.outcome_symbol is not None:
            raise ValueError(
                f"{spec.model_id}: RLM corr_factor is measurement-only and "
                "requires outcome_symbol=None"
            )
        rli_fields = {
            "structural_covariates": settings.structural_covariates,
            "structural_factors": settings.structural_factors,
            "use_group": settings.use_group,
            "use_age": settings.use_age,
            "post_time": settings.post_time,
            "loading_prior": settings.loading_prior,
            "loading_mu": settings.loading_mu,
            "loading_sigma": settings.loading_sigma,
            "residual_sigma": settings.residual_sigma,
            "predictor_slope_sigma": settings.predictor_slope_sigma,
            "focal_slope_sigma": settings.focal_slope_sigma,
        }
        supplied = [name for name, value in rli_fields.items() if value is not None]
        if supplied:
            raise ValueError(
                f"{spec.model_id}: RLI-only settings are invalid for the RLM port: "
                f"{', '.join(supplied)}"
            )
        port = "rlm"
        outcome = None
        structural_covariates = ()
        structural_factors = None
        use_group = False
        use_age = False
        post_time = None
        loading_prior = None
        wave = 3 if settings.wave is None else settings.wave
        reliability = (
            0.8
            if settings.single_indicator_reliability is None
            else settings.single_indicator_reliability
        )
        observation_nodes = ("Z_obs",)
        design = (
            "Historical Byrne-cohort one-wave correlated-domain measurement model "
            "with no structural outcome leg."
        )
        estimand = (
            "Latent domain correlations and indicator communalities under the "
            "declared single-indicator reliability assumption."
        )
        population = (
            f"Complete-case RLM children at wave {wave} across every declared "
            "measurement indicator."
        )

    return CorrFactorRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        port=port,
        outcome_symbol=outcome,
        domains=settings.domains,
        indicators=indicators,
        structural_covariates=structural_covariates,
        active_structural_covariates=structural_covariates,
        structural_factors=structural_factors,
        use_group=use_group,
        use_age=use_age,
        post_time=post_time,
        loading_prior=loading_prior,
        comm_alpha=2.0 if settings.comm_alpha is None else settings.comm_alpha,
        comm_beta=2.0 if settings.comm_beta is None else settings.comm_beta,
        loading_mu=0.0 if settings.loading_mu is None else settings.loading_mu,
        loading_sigma=(
            1.0 if settings.loading_sigma is None else settings.loading_sigma
        ),
        residual_sigma=(
            1.0 if settings.residual_sigma is None else settings.residual_sigma
        ),
        predictor_slope_sigma=(
            0.3
            if settings.predictor_slope_sigma is None
            else settings.predictor_slope_sigma
        ),
        focal_slope_sigma=settings.focal_slope_sigma,
        lkj_eta=settings.lkj_eta,
        wave=wave,
        single_indicator_reliability=reliability,
        observation_nodes=observation_nodes,
        compute_loo=False,
        loo_unit=None,
        focal_term=None,
        design=design,
        estimand=estimand,
        causal_status=(
            "Descriptive measurement associations only; neither factor "
            "correlations nor structural slopes are causal effects."
        ),
        analysis_population=population,
        missing_data_assumption=(
            "Complete-case analysis over the required measurement and structural "
            "variables, under ignorable missingness conditional on observed model "
            "variables."
        ),
    )
