# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the joint ITT family.

The registered ``kind="joint"`` models — the ten-outcome suite fit, the three
two-outcome contrast parents and their #551 LKJ residual-correlation companions
(``docs/models/README.md`` is the authoritative catalogue) — share one
multivariate Beta-Binomial construction. This module makes that contract explicit
and validates it before an output transaction is opened or intervention data are
loaded (#394 pillar 4). One resolved plan then drives preparation, factory
arguments, diagnostics, contrast metadata and the ``config.json`` /
``model_recipe.md`` audit trail.

This is a behaviour-preserving boundary. It does not change prepared rows,
likelihoods, priors, fitted equations, sampling settings or published table schemas.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES, MEASURES

__all__ = [
    "JointContrastSettings",
    "JointModelSettings",
    "JointRunPlan",
    "declared_joint_settings",
    "resolve_joint_run_plan",
]


_LEGACY_KEYS = frozenset(
    {
        "outcomes",
        "use_age_gp",
        "partial_pool_age_gp",
        "use_residual_correlation",
        "use_cross_baselines",
        "use_age_linear",
        "joint_structure",
        "loo_unit",
        "difference",
        "difference_metadata",
        # Global sampler setting resolved by ``make_context``, not this family.
        "target_accept",
    }
)

_CONTRAST_METADATA_KEYS = frozenset(
    {
        "contrast_kind",
        "contrast_label",
        "positive_interpretation",
        "negative_interpretation",
        "transfer_outcome",
        "transfer_interpretation",
        "dependence_note",
    }
)


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    if len(out) != len(set(out)):
        raise ValueError(f"{name} contains duplicate symbols: {out!r}")
    return out


def _optional_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string or None")
    return value


@dataclass(frozen=True, slots=True)
class JointContrastSettings:
    """Typed declaration for one reported between-outcome treatment contrast.

    ``dependence_companion`` is the machine-readable half of the dependence
    pairing (#551; 2026-08-21 joint review, finding 3): a **factorised** parent
    names its registered LKJ residual-correlation companion here, so the release
    decision can verify the companion is release-ready beside the parent instead
    of relying on the prose ``dependence_note`` alone. It is deliberately *not*
    part of the contrast metadata written to ``tau_difference.csv`` — it drives
    the release decision through the resolved plan in ``config.json``. A
    residual-correlated fit is itself the dependence model and must not name one
    (enforced in :func:`resolve_joint_run_plan`).
    """

    left: str
    right: str
    contrast_kind: str | None = None
    contrast_label: str | None = None
    positive_interpretation: str | None = None
    negative_interpretation: str | None = None
    transfer_outcome: str | None = None
    transfer_interpretation: str | None = None
    dependence_note: str | None = None
    dependence_companion: str | None = None

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        if self.left == self.right:
            raise ValueError("joint contrast outcomes must be different")
        for name in _CONTRAST_METADATA_KEYS:
            object.__setattr__(
                self,
                name,
                _optional_text(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "dependence_companion",
            _optional_text(self.dependence_companion, name="dependence_companion"),
        )
        if (self.transfer_outcome is None) != (self.transfer_interpretation is None):
            raise ValueError("transfer_outcome and transfer_interpretation must be declared together")

    @property
    def pair(self) -> tuple[str, str]:
        """Ordered outcome pair used by ``tau_difference_summary``."""
        return self.left, self.right

    def metadata(self) -> dict[str, str] | None:
        """Legacy-compatible metadata mapping, excluding absent optional fields."""
        values = {name: getattr(self, name) for name in _CONTRAST_METADATA_KEYS if getattr(self, name) is not None}
        return values or None

    @classmethod
    def from_legacy(
        cls,
        difference: Any,
        metadata: Any,
        *,
        model_id: str,
    ) -> JointContrastSettings | None:
        """Translate the former ``difference`` plus metadata dictionary."""
        if difference is None:
            if metadata is not None:
                raise ValueError(f"{model_id}: difference_metadata requires a difference pair")
            return None
        pair = _tuple_of_strings(difference, name="difference")
        if len(pair) != 2:
            raise ValueError("difference must contain exactly two outcome symbols")
        if metadata is None:
            values: dict[str, Any] = {}
        elif isinstance(metadata, Mapping):
            values = dict(metadata)
        else:
            raise TypeError("difference_metadata must be a mapping or None")
        unknown = sorted(set(values) - _CONTRAST_METADATA_KEYS)
        if unknown:
            raise ValueError(f"{model_id}: unknown joint contrast metadata: {', '.join(unknown)}")
        return cls(left=pair[0], right=pair[1], **values)


@dataclass(frozen=True, slots=True)
class JointModelSettings:
    """Immutable settings declared by a joint-model module."""

    outcomes: tuple[str, ...] | None = None
    use_age_gp: bool = False
    partial_pool_age_gp: bool = True
    use_residual_correlation: bool = False
    use_cross_baselines: bool = True
    use_age_linear: bool = False
    joint_structure: str | None = None
    loo_unit: str = "child"
    contrast: JointContrastSettings | None = None

    def __post_init__(self) -> None:
        if self.outcomes is not None:
            object.__setattr__(self, "outcomes", _tuple_of_strings(self.outcomes, name="outcomes"))
            if not self.outcomes:
                raise ValueError("outcomes must list at least one measure")
        for flag in (
            "use_age_gp",
            "partial_pool_age_gp",
            "use_residual_correlation",
            "use_cross_baselines",
            "use_age_linear",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        if self.use_age_gp and self.use_age_linear:
            raise ValueError("use_age_gp and use_age_linear are mutually exclusive")
        if self.joint_structure is not None and (not isinstance(self.joint_structure, str) or not self.joint_structure):
            raise TypeError("joint_structure must be a non-empty string or None")
        if self.loo_unit != "child":
            raise ValueError("joint loo_unit must be 'child'")
        if self.contrast is not None and not isinstance(self.contrast, JointContrastSettings):
            raise TypeError("contrast must be JointContrastSettings or None")

    @classmethod
    def from_legacy_extra(cls, extra: Mapping[str, Any], *, model_id: str) -> JointModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown joint setting(s): {', '.join(unknown)}. "
                "Declare JointModelSettings so misspellings fail fast."
            )
        return cls(
            outcomes=extra.get("outcomes"),
            use_age_gp=extra.get("use_age_gp", False),
            partial_pool_age_gp=extra.get("partial_pool_age_gp", True),
            use_residual_correlation=extra.get("use_residual_correlation", False),
            use_cross_baselines=extra.get("use_cross_baselines", True),
            use_age_linear=extra.get("use_age_linear", False),
            joint_structure=extra.get("joint_structure"),
            loo_unit=extra.get("loo_unit", "child"),
            contrast=JointContrastSettings.from_legacy(
                extra.get("difference"),
                extra.get("difference_metadata"),
                model_id=model_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class JointRunPlan:
    """Concrete, validated instructions consumed by the whole joint fit."""

    model_id: str
    settings_source: str
    outcomes: tuple[str, ...]
    outcomes_explicit: bool
    likelihood: str
    observation_node: str
    use_age_gp: bool
    partial_pool_age_gp: bool
    use_residual_correlation: bool
    use_cross_baselines: bool
    use_age_linear: bool
    joint_structure: str
    loo_unit: str
    contrast: JointContrastSettings | None
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    @property
    def difference(self) -> tuple[str, str] | None:
        """Ordered outcome pair for the optional reported contrast."""
        return self.contrast.pair if self.contrast is not None else None

    def difference_metadata(self) -> dict[str, str] | None:
        """Metadata supplied to the existing contrast summary function."""
        return self.contrast.metadata() if self.contrast is not None else None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        return asdict(self)

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from this plan."""
        kwargs: dict[str, Any] = {"phase_mode": "itt"}
        if self.outcomes_explicit:
            kwargs["outcomes"] = self.outcomes
        return kwargs

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_joint_model`` from this plan."""
        return {
            "outcomes": self.outcomes,
            "use_age_gp": self.use_age_gp,
            "partial_pool_age_gp": self.partial_pool_age_gp,
            "use_residual_correlation": self.use_residual_correlation,
            "use_cross_baselines": self.use_cross_baselines,
            "use_age_linear": self.use_age_linear,
        }

    def diagnostic_vars(self) -> list[str]:
        """Curated summary variables for this fitted equation."""
        variables = ["alpha", "tau", "gamma_own", "kappa"]
        if self.use_age_linear:
            variables.append("gamma_A")
        if self.use_residual_correlation:
            # The dependence block's reported quantities (#551): the per-outcome
            # residual SDs and the free within-child residual correlations (one
            # scalar per outcome pair — the full ``u_corr`` matrix carries a
            # constant unit diagonal that breaks the density plots), so the
            # summary, the prior-vs-posterior overlay and the psense selection show
            # how far the block is informed by the data rather than its prior.
            variables.extend(["sigma_outcome", "u_corr_pair"])
        return variables

    @property
    def psense_vars(self) -> list[str]:
        """Variables power-scaling covers: the causal ``tau`` vector, plus the
        dependence block's ``sigma_outcome`` / ``u_corr_pair`` when it is on
        (#551)."""
        names = ["tau"]
        if self.use_residual_correlation:
            names.extend(["sigma_outcome", "u_corr_pair"])
        return names

    def recipe_markdown(self, *, title: str) -> str:
        """Plain-language account generated from the validated plan."""
        outcomes = ", ".join(self.outcomes)
        contrast = (
            f" The reported treatment-effect contrast is `{self.difference[0]} - {self.difference[1]}`."
            if self.difference is not None
            else ""
        )
        age = "Gaussian process" if self.use_age_gp else "linear" if self.use_age_linear else "none"
        return (
            "Note: Generated from the validated joint run plan; template drafted "
            "by a LLM-based AI tool (Codex/GPT-5).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}{contrast}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcomes: {outcomes}. Likelihood: {self.likelihood} "
            f"(`{self.observation_node}`). Own baselines: true. Cross-baselines: "
            f"{self.use_cross_baselines}. Age term: {age}. Residual dependence: "
            f"{self.joint_structure}. LOO unit: {self.loo_unit}.\n\n"
            "## Uncertainty and checks\n\n"
            "The fit reports posterior distributions. Interpret them only after the "
            "convergence gate, posterior-predictive checks and child-level PSIS-LOO "
            "diagnostics pass. "
            + (
                "This fit carries a per-child LKJ residual-correlation block, so a "
                "declared contrast is a posterior difference that includes the "
                "estimated within-child cross-outcome covariance; read `u_corr` and "
                "`sigma_outcome` against their priors to see how far the block is "
                "informed by the data.\n"
                if self.use_residual_correlation
                else "Factorised models do not estimate within-child "
                "cross-outcome residual covariance, so a paired contrast also requires "
                "the dependence sensitivity stated in its metadata.\n"
            )
        )


def declared_joint_settings(spec: ModelSpec) -> tuple[JointModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(f"{spec.model_id}: joint settings cannot be split between model_settings and extra")
        if not isinstance(settings, JointModelSettings):
            raise TypeError(f"{spec.model_id}: kind='joint' requires JointModelSettings, got {type(settings).__name__}")
        return settings, "typed"
    return (
        JointModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_joint_run_plan(spec: ModelSpec) -> JointRunPlan:
    """Resolve and validate a joint specification before context or data I/O."""
    if spec.kind != "joint":
        raise ValueError(f"{spec.model_id}: expected kind 'joint', got {spec.kind!r}")

    settings, source = declared_joint_settings(spec)
    outcomes = settings.outcomes or ITT_OUTCOMES
    unknown = sorted(set(outcomes) - set(MEASURES))
    if unknown:
        raise ValueError(f"{spec.model_id}: unrecognised bounded outcome symbol(s): {', '.join(unknown)}")
    expected_structure = "residual_correlated" if settings.use_residual_correlation else "factorised_outcome_marginals"
    if settings.joint_structure is not None and settings.joint_structure != expected_structure:
        raise ValueError(
            f"{spec.model_id}: joint_structure={settings.joint_structure!r} "
            f"contradicts use_residual_correlation={settings.use_residual_correlation}"
        )
    if settings.contrast is not None:
        missing = sorted(set(settings.contrast.pair) - set(outcomes))
        if missing:
            raise ValueError(f"{spec.model_id}: contrast outcome(s) not in outcomes: {', '.join(missing)}")
        transfer = settings.contrast.transfer_outcome
        if transfer is not None and transfer not in outcomes:
            raise ValueError(f"{spec.model_id}: transfer_outcome {transfer!r} is not in outcomes")
        if settings.use_residual_correlation and settings.contrast.dependence_companion is not None:
            raise ValueError(
                f"{spec.model_id}: a residual-correlated fit is itself the dependence "
                "model and must not declare a dependence_companion"
            )

    design = (
        "Joint multi-outcome Beta-Binomial model over the randomised t1-to-t2 "
        "window. Each outcome has its own intercept, assigned-arm term and own "
        "baseline precision term; optional age and cross-baseline terms follow the "
        "declared settings."
        + (
            " A per-child multivariate-normal residual offset with an LKJ "
            "correlation prior (eta = 4) and HalfNormal(0.5) outcome scales links "
            "the outcomes, so within-child cross-outcome covariance is estimated "
            "rather than assumed away (#551)."
            if settings.use_residual_correlation
            else ""
        )
    )
    estimand = (
        "Each tau is the available-case modified intention-to-treat assigned-arm "
        "contrast for one outcome, reported as an average marginal effect on the "
        "proportion-correct scale. Any declared difference compares two of those "
        "outcome-specific average marginal effects."
    )
    causal_status = (
        "The assigned-arm effects use randomisation and are causal for the observed "
        "analysis cases under the stated missing-data assumptions. Precision terms "
        "are adjusted associations. "
        + (
            "The residual-correlation block is a dependence model for the paired "
            "contrast's uncertainty, not an effect: u_corr and sigma_outcome are "
            "descriptive of within-child covariance."
            if settings.use_residual_correlation
            else "A factorised fit does not estimate paired cross-outcome residual "
            "covariance."
        )
    )
    # The loader's default ``pre_required`` covers every declared outcome, so a
    # multi-outcome fit is a cross-outcome baseline complete-case intersection —
    # a child missing any one declared baseline is excluded from every outcome,
    # not only its own. State that mechanism rather than the previous
    # "outcome-specific available cases" wording, which described only the
    # post-score side and misread the ten-outcome fit's 53-child set as
    # per-outcome availability (2026-08-21 joint review, finding 1).
    analysis_population = (
        "The archived RLI cohort in the randomised t1-to-t2 window, restricted to "
        "children with an observed baseline for **every** declared outcome (the "
        "loader's joint baseline complete-case rule"
        + (
            " — with more than one outcome this is the cross-outcome "
            "intersection, so one missing baseline excludes a child from every "
            "outcome, and the fitted set can be smaller than the matching "
            "single-outcome fits'"
            if len(outcomes) > 1
            else ""
        )
        + "). Post-scores remain available-case per outcome, so observed child "
        "sets can differ between outcomes only through post-score missingness."
    )
    missing_data_assumption = (
        "Available-case modified ITT: missing post-scores are assumed ignorable for "
        "each outcome conditional on assigned arm and the declared precision terms. "
        "Randomisation does not by itself repair selection into observed cases."
    )

    return JointRunPlan(
        model_id=spec.model_id,
        settings_source=source,
        outcomes=outcomes,
        outcomes_explicit=settings.outcomes is not None,
        likelihood="beta_binomial",
        observation_node="y_post",
        use_age_gp=settings.use_age_gp,
        partial_pool_age_gp=settings.partial_pool_age_gp,
        use_residual_correlation=settings.use_residual_correlation,
        use_cross_baselines=settings.use_cross_baselines,
        use_age_linear=settings.use_age_linear,
        joint_structure=expected_structure,
        loo_unit=settings.loo_unit,
        contrast=settings.contrast,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
