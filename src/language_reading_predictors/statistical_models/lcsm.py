# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for latent change-score models.

The LCSM family covers the original reading-change coupling model and the
crossover-aware reverse, reciprocal-dominance and lagged change-on-change
variants.  This module replaces their free-form ``ModelSpec.extra`` boundary
with immutable settings and validates the complete graph shape before a fit
context is created or the RLI panel is loaded (#394 pillar 4).

The migration is structural: registered models retain the same measures,
couplings, priors, rows, diagnostic variables and artefacts.  Cross-process
couplings remain adjusted or exploratory associations; only a window-1 arm
contrast has randomised causal warrant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec

CouplingItems = tuple[tuple[str, tuple[str, ...]], ...]

__all__ = [
    "CouplingItems",
    "LcsmModelSettings",
    "LcsmRunPlan",
    "declared_lcsm_settings",
    "resolve_lcsm_run_plan",
]


_DEFAULT_OUTCOMES = ("W", "L", "E")
_FAMILY_KEYS = frozenset(
    {
        "outcomes",
        "couplings",
        "lagged_change_couplings",
        "arm_window_intercepts",
        "covariate_block",
        "covariate_targets",
        "dominance_pair",
        "coupling_prior_sigma",
        "use_process_noise",
        "shared_process_noise",
    }
)
_GLOBAL_KEYS = frozenset({"target_accept"})
_LEGACY_KEYS = _FAMILY_KEYS | _GLOBAL_KEYS


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        _string(item, name=name)
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _coupling_items(value: Any, *, name: str) -> CouplingItems | None:
    if value is None:
        return None
    raw_items = value.items() if isinstance(value, Mapping) else value
    if isinstance(raw_items, (str, bytes)) or not hasattr(raw_items, "__iter__"):
        raise TypeError(f"{name} must be a target-to-sources mapping, got {value!r}")
    out: list[tuple[str, tuple[str, ...]]] = []
    for item in raw_items:
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, Sequence)
            or len(item) != 2
        ):
            raise TypeError(f"{name} entries must be (target, sources) pairs")
        target, sources = item
        target = _string(target, name=f"{name} target")
        out.append(
            (target, _tuple_of_strings(sources, name=f"{name}[{target!r}]") )
        )
    targets = [target for target, _ in out]
    if len(targets) != len(set(targets)):
        raise ValueError(f"{name} contains duplicate targets: {targets!r}")
    return tuple(out)


def _dominance_pair(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    pair = _tuple_of_strings(value, name="dominance_pair")
    if len(pair) != 2:
        raise ValueError("dominance_pair must contain exactly two distinct outcomes")
    return pair[0], pair[1]


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return out


@dataclass(frozen=True, slots=True)
class LcsmModelSettings:
    """Immutable declaration for one latent change-score model."""

    outcomes: tuple[str, ...] = _DEFAULT_OUTCOMES
    couplings: CouplingItems | None = None
    lagged_change_couplings: CouplingItems = ()
    arm_window_intercepts: bool = False
    covariate_block: tuple[str, ...] = ()
    covariate_targets: tuple[str, ...] = ()
    dominance_pair: tuple[str, str] | None = None
    coupling_prior_sigma: float = 0.3
    use_process_noise: bool = True
    shared_process_noise: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcomes",
            _tuple_of_strings(self.outcomes, name="outcomes"),
        )
        object.__setattr__(
            self,
            "couplings",
            _coupling_items(self.couplings, name="couplings"),
        )
        object.__setattr__(
            self,
            "lagged_change_couplings",
            _coupling_items(
                self.lagged_change_couplings,
                name="lagged_change_couplings",
            )
            or (),
        )
        object.__setattr__(
            self,
            "covariate_block",
            _tuple_of_strings(self.covariate_block, name="covariate_block"),
        )
        object.__setattr__(
            self,
            "covariate_targets",
            _tuple_of_strings(self.covariate_targets, name="covariate_targets"),
        )
        object.__setattr__(self, "dominance_pair", _dominance_pair(self.dominance_pair))
        object.__setattr__(
            self,
            "coupling_prior_sigma",
            _positive_float(self.coupling_prior_sigma, name="coupling_prior_sigma"),
        )
        for name in (
            "arm_window_intercepts",
            "use_process_noise",
            "shared_process_noise",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> LcsmModelSettings:
        """Strictly translate the former ``spec.extra`` declaration."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown LCSM setting(s): {', '.join(unknown)}. "
                "Declare LcsmModelSettings so misspellings fail fast."
            )
        return cls(
            outcomes=extra.get("outcomes", _DEFAULT_OUTCOMES),
            couplings=_coupling_items(extra.get("couplings"), name="couplings"),
            lagged_change_couplings=_coupling_items(
                extra.get("lagged_change_couplings", ()),
                name="lagged_change_couplings",
            )
            or (),
            arm_window_intercepts=extra.get("arm_window_intercepts", False),
            covariate_block=extra.get("covariate_block", ()),
            covariate_targets=extra.get("covariate_targets", ()),
            dominance_pair=extra.get("dominance_pair"),
            coupling_prior_sigma=extra.get("coupling_prior_sigma", 0.3),
            use_process_noise=extra.get("use_process_noise", True),
            shared_process_noise=extra.get("shared_process_noise", False),
        )


@dataclass(frozen=True, slots=True)
class LcsmRunPlan:
    """Concrete, validated instructions for a complete LCSM fit."""

    model_id: str
    settings_source: str
    study_id: str
    reading_symbol: str
    outcomes: tuple[str, ...]
    couplings: CouplingItems
    lagged_change_couplings: CouplingItems
    arm_window_intercepts: bool
    covariate_block: tuple[str, ...]
    covariate_targets: tuple[str, ...]
    dominance_pair: tuple[str, str] | None
    coupling_prior_sigma: float
    use_process_noise: bool
    shared_process_noise: bool
    include_hearing: bool
    wave_covariates: tuple[str, ...]
    observation_node: str
    compute_loo: bool
    loo_unit: str
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        result = asdict(self)
        result["couplings"] = self.coupling_mapping()
        result["lagged_change_couplings"] = self.lagged_coupling_mapping()
        return result

    def coupling_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return the level-to-change graph in factory form."""
        return dict(self.couplings)

    def lagged_coupling_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return the change-to-later-change graph in factory form."""
        return dict(self.lagged_change_couplings)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_wave_panel`` from the resolved plan."""
        return {
            "outcomes": self.outcomes,
            "wave_covariates": self.wave_covariates,
            "include_hearing": self.include_hearing,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_lcsm_model`` from the same plan."""
        return {
            "reading_symbol": self.reading_symbol,
            "couplings": self.coupling_mapping(),
            "lagged_change_couplings": self.lagged_coupling_mapping() or None,
            "arm_window_intercepts": self.arm_window_intercepts,
            "covariate_block": self.covariate_block,
            "covariate_targets": self.covariate_targets,
            "coupling_prior_sigma": self.coupling_prior_sigma,
            "use_process_noise": self.use_process_noise,
            "shared_process_noise": self.shared_process_noise,
        }

    def coupling_names(self) -> dict[tuple[str, str], str]:
        """Map source-target pairs to their fitted level-coupling names."""
        single_target = len(self.couplings) == 1
        return {
            (source, target): (
                f"g_{source}" if single_target else f"g_{source}_{target}"
            )
            for target, sources in self.couplings
            for source in sources
        }

    def lagged_names(self) -> dict[tuple[str, str], str]:
        """Map source-target pairs to their fitted change-coupling names."""
        single_target = len(self.lagged_change_couplings) == 1
        return {
            (source, target): (
                f"h_{source}" if single_target else f"h_{source}_{target}"
            )
            for target, sources in self.lagged_change_couplings
            for source in sources
        }

    def diagnostic_vars(self) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        names = list(self.coupling_names().values())
        names.extend(self.lagged_names().values())
        names.extend(f"b_{name}" for name in self.covariate_block)
        names.extend(["a_change", "b_self", "d_age", "sigma1", "kappa"])
        if self.use_process_noise:
            names.append("sigma_proc")
        return names

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated run plan."""
        couplings = ", ".join(
            f"{source} to {target} change"
            for target, sources in self.couplings
            for source in sources
        )
        lagged = ", ".join(
            f"prior {source} change to {target} change"
            for target, sources in self.lagged_change_couplings
            for source in sources
        ) or "none"
        adjusters = ", ".join(self.covariate_block) or "none"
        return (
            "Note: Generated from the validated LCSM run plan; template drafted "
            "by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Measures: {', '.join(self.outcomes)}. Prior-level couplings: "
            f"{couplings}. Lagged change couplings: {lagged}. Shared adjustment "
            f"terms: {adjusters}. Arm-by-window change intercepts: "
            f"{self.arm_window_intercepts}. Process noise: "
            f"{self.use_process_noise}; shared process scale: "
            f"{self.shared_process_noise}. Coupling prior sigma: "
            f"{self.coupling_prior_sigma:g}.\n\n"
            "## Uncertainty and checks\n\n"
            f"The observation node is `{self.observation_node}` and PSIS-LOO uses "
            f"the `{self.loo_unit}` unit. Interpret latent couplings only after "
            "the zero-divergence convergence gate, posterior-predictive checks and "
            "power-scaling sensitivity diagnostics pass. The saved `config.json` "
            "contains the same resolved run plan in machine-readable form.\n"
        )


def declared_lcsm_settings(spec: ModelSpec) -> tuple[LcsmModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        family_extra = sorted(set(spec.extra) - _GLOBAL_KEYS)
        if family_extra:
            raise ValueError(
                f"{spec.model_id}: LCSM settings cannot be split between "
                f"model_settings and extra: {', '.join(family_extra)}"
            )
        if not isinstance(settings, LcsmModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='lcsm' requires LcsmModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        LcsmModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def _validate_couplings(
    items: CouplingItems,
    *,
    outcomes: tuple[str, ...],
    name: str,
) -> None:
    known = set(outcomes)
    for target, sources in items:
        if target not in known:
            raise ValueError(f"{name} target {target!r} is not in outcomes {outcomes!r}")
        unknown = sorted(set(sources) - known)
        if unknown:
            raise ValueError(
                f"{name}[{target!r}] contains symbols outside outcomes: {unknown!r}"
            )
        if target in sources:
            raise ValueError(
                f"{name} target {target!r} cannot couple to itself; b_self owns "
                "self-feedback"
            )


def resolve_lcsm_run_plan(spec: ModelSpec) -> LcsmRunPlan:
    """Resolve and validate the family contract before context or data I/O."""
    if spec.kind != "lcsm":
        raise ValueError(f"{spec.model_id}: expected kind 'lcsm', got {spec.kind!r}")
    if spec.study_id != "rli":
        raise ValueError(
            f"{spec.model_id}: LCSM requires study_id='rli', got {spec.study_id!r}"
        )
    reading_symbol = spec.outcome_symbol
    if reading_symbol is None:
        raise ValueError(f"{spec.model_id}: LCSM requires outcome_symbol")

    settings, source = declared_lcsm_settings(spec)
    outcomes = settings.outcomes
    if len(outcomes) < 2:
        raise ValueError("LCSM outcomes must contain at least two measures")
    if reading_symbol not in outcomes:
        raise ValueError(
            f"{spec.model_id}: outcome_symbol {reading_symbol!r} is not in "
            f"outcomes {outcomes!r}"
        )

    couplings = settings.couplings or (
        (reading_symbol, tuple(symbol for symbol in outcomes if symbol != reading_symbol)),
    )
    lagged = settings.lagged_change_couplings
    _validate_couplings(couplings, outcomes=outcomes, name="couplings")
    _validate_couplings(
        lagged,
        outcomes=outcomes,
        name="lagged_change_couplings",
    )
    if lagged and not settings.arm_window_intercepts:
        raise ValueError(
            "lagged_change_couplings requires arm_window_intercepts=True because "
            "the fitted transitions are post-crossover"
        )
    if bool(settings.covariate_block) != bool(settings.covariate_targets):
        raise ValueError(
            "covariate_block and covariate_targets must be declared together"
        )
    unknown_targets = sorted(set(settings.covariate_targets) - set(outcomes))
    if unknown_targets:
        raise ValueError(
            f"covariate_targets contains symbols outside outcomes: {unknown_targets!r}"
        )
    if settings.shared_process_noise and not settings.use_process_noise:
        raise ValueError(
            "shared_process_noise=True requires use_process_noise=True"
        )
    if settings.dominance_pair is not None:
        left, right = settings.dominance_pair
        if left not in outcomes or right not in outcomes:
            raise ValueError("dominance_pair symbols must both appear in outcomes")
        coupling_set = {
            (source_symbol, target)
            for target, sources in couplings
            for source_symbol in sources
        }
        if {(left, right), (right, left)} - coupling_set:
            raise ValueError(
                "dominance_pair requires reciprocal level couplings in both directions"
            )

    include_hearing = any(
        name in {"hs", "hs_missing"} for name in settings.covariate_block
    )
    wave_covariates = tuple(
        dict.fromkeys(
            name
            for name in settings.covariate_block
            if name not in {"hs", "hs_missing"} and not name.endswith("_missing")
        )
    )
    arm_design = (
        "Arm-by-window change intercepts separate the randomised first window from "
        "post-crossover windows."
        if settings.arm_window_intercepts
        else "A pooled change intercept is used across arms and transitions."
    )
    return LcsmRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        study_id=spec.study_id,
        reading_symbol=reading_symbol,
        outcomes=outcomes,
        couplings=couplings,
        lagged_change_couplings=lagged,
        arm_window_intercepts=settings.arm_window_intercepts,
        covariate_block=settings.covariate_block,
        covariate_targets=settings.covariate_targets,
        dominance_pair=settings.dominance_pair,
        coupling_prior_sigma=settings.coupling_prior_sigma,
        use_process_noise=settings.use_process_noise,
        shared_process_noise=settings.shared_process_noise,
        include_hearing=include_hearing,
        wave_covariates=wave_covariates,
        observation_node="y_obs",
        compute_loo=True,
        loo_unit="observed_measure_wave_cell",
        design=(
            "A coupled McArdle latent change-score model over four RLI waves, with "
            "bounded scores observed through Beta-Binomial measurement models and "
            "time-invariant cross-process coupling coefficients. "
            + arm_design
        ),
        estimand=(
            "Each g coefficient is the association between a source measure's "
            "prior latent level and a target measure's subsequent latent change. "
            "Each h coefficient analogously relates a prior latent change to the "
            "next target change. The optional dominance contrast compares reciprocal "
            "couplings after standardising by their latent source and change scales."
        ),
        causal_status=(
            "Cross-process couplings are adjusted or exploratory associations, not "
            "causal effects. When arm-by-window intercepts are fitted, only their "
            "window-1 immediate-minus-wait-list contrast inherits randomisation; "
            "later windows occur after crossover."
        ),
        analysis_population=(
            "The 54 archived RLI children across four waves. Each observed outcome "
            "cell contributes even when another measure or wave is missing; requested "
            "covariates follow the panel loader's complete-data policy."
        ),
        missing_data_assumption=(
            "The masked likelihood treats unobserved score cells as missing and uses "
            "the remaining cells under an ignorable-missingness assumption conditional "
            "on the fitted latent trajectories and covariates. Missing-indicator terms "
            "for declared covariates retain the established preprocessing policy."
        ),
    )
