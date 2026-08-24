# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Typed settings and a resolved run plan for the difference-in-differences family (#394 pillar 4).

Mirrors the ITT / gain-factor / level-factor run-plan pattern (:mod:`itt`,
:mod:`gain_factors`, :mod:`level_factors`) for the waitlist-crossover
difference-in-differences (``kind="did"``) models. A model module declares its
settings; the plan is resolved and **validated before any data are loaded or an
output directory is reset**, then a single object drives data preparation, factory
construction and the ``config.json`` / ``model_recipe.md`` audit trail. This removes
the untyped ``spec.extra`` boundary (where a misspelled key silently defaulted) and
records the resolved design, estimand, causal status, analysis population and
missing-data assumption alongside every fit.

Two designs share the family. **Binary** models fit the t1-t3 *levels* frame and
estimate the arm gap at each wave separately: ``tau_t2`` is the randomised
immediate-treatment-versus-no-treatment assignment contrast — the covariate-adjusted
t2 arm-gap *level*, not the differenced quantity ``tau_t2 - arm_gap_t1``; the shared
child random intercept and the tight ``arm_gap_t1`` prior supply a partial,
prior-weighted baseline adjustment rather than exact differencing (the level-factor
family's t1-referenced ``d_grp_time[t2]``, #552, is the gap-*change* estimand) —
``arm_gap_t3`` the randomised **early-start-versus-delayed-start treatment-schedule**
contrast, and ``delta_crossover = tau_t2 - arm_gap_t3`` the change between those two
randomised regime contrasts. **Dose** variants keep the P1/P2 *transition* frame
because sessions are interval exposures, carry an explicit crossover cell term with
the ``attend`` session covariate, and their session coefficient is observational,
never randomised.

**What t3 is, and is not** (2026-08-24 review, #576 finding 3). Original assignment
is still randomised at t3, so ``arm_gap_t3`` identifies the effect of *assignment to
the early-start treatment history versus the delayed-start one*, under the same
available-case selection and model assumptions as ``tau_t2``. It is **not** a
treated-versus-untreated effect, because both arms are treated by t3; and latent
ability does not become a confounder of randomised assignment merely because the
waitlist arm has crossed over. What is genuinely unavailable is the *mechanism*:
duration, carryover, maturation, ceiling effects and different taught blocks are
inseparable in that one number, so ``delta_crossover`` is a change between two
randomised regime contrasts and never evidence of an identified catch-up process.
That mechanistic limitation is a different thing from observational confounding, and
the family's prose, metadata and reports must not conflate them.
"""

from __future__ import annotations

import hashlib
import json
import math

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import KAPPA_PRIOR_FAMILIES
from language_reading_predictors.statistical_models.likelihood import SCORE_MEAN_LINKS

#: The registered phoneme-blending response-link pair for this family (#576
#: finding 2): the ordinary-logit primary and its one-in-three guessing-floor
#: companion. Blending items are three-alternative forced choice, so the ordinary
#: inverse-logit mean permits fitted means below chance while the floor link
#: cannot. Neither fit is sufficient evidence on its own, so the ids live here —
#: the one place that already knows which outcome a plan fits — rather than being
#: restated by the release gate, the pair evaluator and the report.
DID_BLENDING_PRIMARY_MODEL_ID = "lrp-rli-did-003"
DID_BLENDING_COMPANION_MODEL_ID = "lrp-rli-did-103"

# The complete, closed set of legacy ``spec.extra`` keys the DiD family understands.
# Anything else is a typo and must fail before a fit starts.
_LEGACY_KEYS = frozenset(
    {
        "dose",
        "period_varying_dose",
        "likelihood",
        "outcomes",
        "waves",
        "periods",
        "use_child_re",
        "use_age",
        "use_varying_delta",
        "use_intercept_anchor",
        "tau_t2_prior_sigma",
        "score_mean_link",
        "arm_gap_t1_prior_sigma",
        "sigma_child_prior_sigma",
        "kappa_prior_family",
        "kappa_prior_sigma",
        # Sampler knob, not a model setting: ``target_accept`` is resolved centrally by
        # ``context.make_context`` (CLI override > spec default > preset) and is never
        # read by this family's settings. Listed so a legitimate per-model declaration
        # is not rejected as a misspelling by the strict unknown-key check.
        "target_accept",
    }
)

_LIKELIHOODS = frozenset({"beta_binomial", "bernoulli_offfloor"})


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of strings, got {value!r}")
    out = tuple(value)
    for item in out:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{name} must contain non-empty strings, got {item!r}")
    return out


def _tuple_of_ints(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        raise TypeError(f"{name} must be a sequence of integers, got {value!r}")
    out = tuple(value)
    for item in out:
        # bool is an int subclass but is never a valid wave/period index.
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{name} must contain integers, got {item!r}")
    return out


def _optional_positive_float(value: Any, *, name: str) -> float | None:
    """Validate an optional prior width, rejecting ``bool`` explicitly (#576).

    ``bool`` is a subclass of ``int``, so a bare ``isinstance(value, (int, float))``
    check accepted ``tau_t2_prior_sigma=True`` and silently fitted the causal term
    at ``Normal(0, 1.0)`` — a real prior change disguised as a typo'd flag. Every
    optional prior width in this family goes through here so the same slip cannot
    reappear on a new setting.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be a number when set, got {value!r} "
            f"({type(value).__name__}); bool is not a prior width"
        )
    width = float(value)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError(f"{name} must be finite and positive when set; got {value!r}")
    return width


#: The run-plan fields that define the fitted equation, with the default a plan
#: written before the field existed would have taken (#576 finding 6). Ordered and
#: closed on purpose: adding a field here changes every digest, so a new modelling
#: setting must be added deliberately, and the prose fields are excluded so a
#: wording revision cannot invalidate a sweep for an unchanged model.
_RUN_PLAN_DIGEST_FIELDS: tuple[tuple[str, Any], ...] = (
    ("outcome_symbol", ""),
    ("dose", False),
    ("period_varying", False),
    ("likelihood", "beta_binomial"),
    ("off_floor", False),
    ("outcomes", ()),
    ("waves", (0, 1, 2)),
    ("periods", (0, 1)),
    ("use_child_re", True),
    ("use_age", True),
    ("use_varying_delta", False),
    ("use_intercept_anchor", True),
    ("tau_t2_prior_sigma", None),
    ("score_mean_link", "logit"),
    ("arm_gap_t1_prior_sigma", None),
    ("sigma_child_prior_sigma", None),
    ("kappa_prior_family", "halfnormal_concentration"),
    ("kappa_prior_sigma", None),
)

#: Digest schema version. Bumped only when :data:`_RUN_PLAN_DIGEST_FIELDS` changes
#: in a way that must invalidate previously attached evidence; it is part of the
#: hashed payload so two schemas can never collide.
RUN_PLAN_DIGEST_VERSION = 1


def did_run_plan_digest(resolved_run_plan: Mapping[str, Any]) -> str:
    """Canonical SHA-256 of a resolved (or persisted) DiD run plan's modelling fields.

    Accepts either :meth:`DiDRunPlan.as_dict` output or the ``resolved_run_plan``
    block read back from a stored ``config.json``, so the sweep runner, the
    sensitivity reference and the release gate all compute the same value from
    whichever form they hold.
    """
    payload: dict[str, Any] = {"schema_version": RUN_PLAN_DIGEST_VERSION}
    for key, default in _RUN_PLAN_DIGEST_FIELDS:
        value = resolved_run_plan.get(key, default)
        if isinstance(value, (list, tuple)):
            value = list(value)
        payload[key] = value
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DiDModelSettings:
    """Immutable settings declared by a single difference-in-differences model module.

    Defaults encode the primary binary DiD: the t1-t3 levels frame with a child
    random intercept and age, no dose term, and the Beta-Binomial working likelihood.
    """

    dose: bool = False
    period_varying_dose: bool = False
    likelihood: str = "beta_binomial"
    outcomes: tuple[str, ...] = ()
    waves: tuple[int, ...] = (0, 1, 2)
    periods: tuple[int, ...] = (0, 1)
    use_child_re: bool = True
    use_age: bool = True
    use_varying_delta: bool = False
    # #390 P1 (Frank's 2026-07-24 option-B ruling, condition 1): False replaces
    # the empirical-Bayes pooled-t1 intercept anchor with a genuinely
    # independent zero-centred Normal(0, alpha-tier) prior — not a wider sigma
    # around the same anchor, whose mean would stay data-dependent. LRPDID101
    # is the registered sensitivity companion. Arm-by-wave models only: the
    # dose variants already build a free intercept, so False there would claim
    # a change that is not one.
    use_intercept_anchor: bool = True
    # #382 recommendation 3: a one-off wider prior on the single causal term.
    # None keeps the outcome-tier default (proximal 0.5 / distal 0.3); LRPDID102
    # sets 1.0 to test whether the right-tail letter-sound tau_t2 is
    # prior-attenuated. arm_gap_t3 keeps the tier scale either way — the
    # sensitivity question is about tau_t2 alone.
    tau_t2_prior_sigma: float | None = None
    # #576 finding 2: the phoneme-blending response link. ``"logit"`` is the
    # ordinary Beta-Binomial inverse-logit score mean (every other outcome, and
    # the ``B`` primary LRPDID03); ``"three_choice_guessing_floor"`` maps it onto
    # [1/3, 1] because each blending item has three response alternatives, so an
    # expected score cannot fall below chance. Arm-by-wave graded models only.
    score_mean_link: str = "logit"
    # #576 finding 4: the soft, prior-weighted baseline adjustment. ``tau_t2`` is
    # the t2 arm-gap *level*, and any realised t1 imbalance is allocated between
    # the tightly regularised ``arm_gap_t1`` and the shared child intercepts.
    # These two widths are that allocation's only free knobs, so an
    # estimand-matched prior sensitivity varies them rather than ``tau_t2``
    # (whose sweep is the treatment-prior grid). None keeps the shared defaults —
    # ``gamma_cross`` Normal(0, 0.3) for the gap, HalfNormal(0.5) for the
    # intercept SD.
    arm_gap_t1_prior_sigma: float | None = None
    sigma_child_prior_sigma: float | None = None
    # #576 material qualification 2: the dispersion prior. The family default,
    # ``kappa ~ HalfNormal(50)``, cannot reach the near-Binomial limit at a high
    # denominator — at n = 170 its prior median implies about 5.9x Binomial
    # variance — so ``"halfnormal_inverse_sqrt"`` (1/sqrt(kappa) ~ HalfNormal)
    # is the registered sensitivity. The default is deliberately *not* changed:
    # every stored DiD fit was sampled under the concentration prior.
    kappa_prior_family: str = "halfnormal_concentration"
    kappa_prior_sigma: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "outcomes", _tuple_of_strings(self.outcomes, name="outcomes")
        )
        object.__setattr__(self, "waves", _tuple_of_ints(self.waves, name="waves"))
        object.__setattr__(self, "periods", _tuple_of_ints(self.periods, name="periods"))
        for flag in (
            "dose",
            "period_varying_dose",
            "use_child_re",
            "use_age",
            "use_varying_delta",
            "use_intercept_anchor",
        ):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be bool")
        if not self.use_intercept_anchor and self.dose:
            raise ValueError(
                "use_intercept_anchor=False is the arm-by-wave independent-prior "
                "sensitivity; the dose models already build a free intercept, so "
                "the setting would claim a change that is not one"
            )
        if self.period_varying_dose and not self.dose:
            raise ValueError("period_varying_dose requires dose=True")
        if self.likelihood not in _LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(_LIKELIHOODS)}, got {self.likelihood!r}"
            )
        # The remaining cross-field constraints build_did_model enforces (#455). They
        # depend on nothing but these settings, so the factory would only reject them
        # after make_context had reset an output directory and the loader had read the
        # panel. Checked here, incoherent settings fail at settings construction time
        # (often at model-module import time, otherwise at resolve time). The factory keeps its own copies as belt-and-braces for direct callers.
        if self.dose and self.likelihood == "bernoulli_offfloor":
            raise ValueError(
                "bernoulli_offfloor is the binary prevalence estimand; use dose=False"
            )
        if self.use_varying_delta and self.dose:
            raise ValueError("use_varying_delta is unavailable for dose models")
        if self.use_varying_delta and not self.use_child_re:
            raise ValueError("use_varying_delta=True requires use_child_re=True")
        if self.dose and self.periods != (0, 1):
            raise ValueError(
                f"DiD dose variants require periods=(0, 1); got {self.periods}."
            )
        for name in (
            "tau_t2_prior_sigma",
            "arm_gap_t1_prior_sigma",
            "sigma_child_prior_sigma",
            "kappa_prior_sigma",
        ):
            object.__setattr__(
                self, name, _optional_positive_float(getattr(self, name), name=name)
            )
        if self.tau_t2_prior_sigma is not None and self.dose:
            raise ValueError(
                "tau_t2_prior_sigma applies to the arm-by-wave tau_t2 contrast; "
                "a dose model has no tau_t2 (its estimand is the dose slope), "
                "so the setting would be silently ignored (#382 rec 3 scope)."
            )
        if self.arm_gap_t1_prior_sigma is not None and self.dose:
            raise ValueError(
                "arm_gap_t1_prior_sigma applies to the arm-by-wave baseline-gap "
                "term; a dose model has no arm_gap_t1"
            )
        if self.sigma_child_prior_sigma is not None and not self.use_child_re:
            raise ValueError(
                "sigma_child_prior_sigma requires use_child_re=True; without the "
                "child random intercept there is no scale to widen"
            )
        if self.score_mean_link not in SCORE_MEAN_LINKS:
            raise ValueError(
                f"score_mean_link must be one of {list(SCORE_MEAN_LINKS)}, "
                f"got {self.score_mean_link!r}"
            )
        if self.score_mean_link != "logit":
            if self.likelihood != "beta_binomial":
                raise ValueError(
                    "score_mean_link applies to the graded Beta-Binomial score "
                    f"mean; the {self.likelihood!r} branch has no score mean to map"
                )
            if self.dose:
                raise ValueError(
                    "score_mean_link is the arm-by-wave response-link sensitivity; "
                    "the dose companions report an observational session slope and "
                    "have no published response-link pair"
                )
        if self.kappa_prior_family not in KAPPA_PRIOR_FAMILIES:
            raise ValueError(
                f"kappa_prior_family must be one of {sorted(KAPPA_PRIOR_FAMILIES)}, "
                f"got {self.kappa_prior_family!r}"
            )
        if self.likelihood == "bernoulli_offfloor" and (
            self.kappa_prior_family != "halfnormal_concentration"
            or self.kappa_prior_sigma is not None
        ):
            raise ValueError(
                "the off-floor Bernoulli branch has no dispersion parameter, so a "
                "kappa prior declaration would be silently ignored"
            )
        # Design windows the factory hard-requires (#576 lower-severity 4). Both
        # used to survive resolution and fail inside ``build_did_model`` — after
        # ``make_context`` had reset an output directory and the loader had read
        # the panel. They depend on nothing but these settings, so they belong here.
        if self.dose:
            if self.waves != (0, 1, 2):
                raise ValueError(
                    "a DiD dose variant fits the P1/P2 transition frame and never "
                    f"reads ``waves``; declaring {self.waves} would be silently "
                    "ignored — set ``periods`` instead"
                )
        elif self.waves != (0, 1, 2):
            raise ValueError(
                "the binary DiD triangulation requires waves=(0, 1, 2) — the t1/t2/t3 "
                f"levels frame the three arm gaps are defined on; got {self.waves}"
            )

    @classmethod
    def from_legacy_extra(
        cls, extra: Mapping[str, Any], *, model_id: str
    ) -> DiDModelSettings:
        """Strictly translate the former ``spec.extra`` dictionary boundary.

        Rejects unknown keys so a misspelling fails before data loading rather than
        silently taking a default."""
        unknown = sorted(set(extra) - _LEGACY_KEYS)
        if unknown:
            raise ValueError(
                f"{model_id}: unknown DiD setting(s): {', '.join(unknown)}. "
                "Declare DiDModelSettings so misspellings fail fast."
            )
        # Pass raw values through so __post_init__ is the single validation/coercion
        # point; pre-coercing here (tuple(...)/bool(...)) would silently reshape
        # misshaped legacy settings instead of failing fast against the strict checks.
        return cls(
            dose=extra.get("dose", False),
            period_varying_dose=extra.get("period_varying_dose", False),
            likelihood=extra.get("likelihood", "beta_binomial"),
            outcomes=extra.get("outcomes", ()),
            waves=extra.get("waves", (0, 1, 2)),
            periods=extra.get("periods", (0, 1)),
            use_child_re=extra.get("use_child_re", True),
            use_age=extra.get("use_age", True),
            use_varying_delta=extra.get("use_varying_delta", False),
            use_intercept_anchor=extra.get("use_intercept_anchor", True),
            tau_t2_prior_sigma=extra.get("tau_t2_prior_sigma"),
            score_mean_link=extra.get("score_mean_link", "logit"),
            arm_gap_t1_prior_sigma=extra.get("arm_gap_t1_prior_sigma"),
            sigma_child_prior_sigma=extra.get("sigma_child_prior_sigma"),
            kappa_prior_family=extra.get(
                "kappa_prior_family", "halfnormal_concentration"
            ),
            kappa_prior_sigma=extra.get("kappa_prior_sigma"),
        )


@dataclass(frozen=True, slots=True)
class DiDRunPlan:
    """Concrete, validated instructions consumed by preparation and modelling."""

    model_id: str
    outcome_symbol: str
    settings_source: str
    dose: bool
    period_varying: bool
    likelihood: str
    off_floor: bool
    outcomes: tuple[str, ...]
    waves: tuple[int, ...]
    periods: tuple[int, ...]
    use_child_re: bool
    use_age: bool
    use_varying_delta: bool
    use_intercept_anchor: bool
    tau_t2_prior_sigma: float | None
    score_mean_link: str
    arm_gap_t1_prior_sigma: float | None
    sigma_child_prior_sigma: float | None
    kappa_prior_family: str
    kappa_prior_sigma: float | None
    #: The one named quantity this fit publishes as its headline, the scale it is
    #: read on and the artefact carrying it (#576 finding 1). The posterior
    #: headline, the prior pushforward, the treatment-prior sweep's items columns
    #: and the release gate's sign-stability clause all read *this* estimand, so a
    #: fit can no longer clear a robustness gate for one quantity while publishing
    #: another. ``focal_term`` stays the swept/power-scaled coefficient.
    focal_estimand: str
    focal_estimand_scale: str
    focal_estimand_artifact: str
    #: #576 finding 2: this fit may not release without its opposite-link twin.
    link_sensitivity_required_for_release: bool
    required_link_companion_model_id: str | None
    # Recorded audit metadata (#394 pillar 4).
    design: str
    estimand: str
    causal_status: str
    analysis_population: str
    missing_data_assumption: str

    @property
    def obs_node(self) -> str:
        return "y_offfloor" if self.off_floor else "y_post"

    @property
    def effect_term(self) -> str:
        """The focal coefficient: the randomised t2 DiD contrast, or the dose term."""
        if self.period_varying:
            return "mu_dose"
        return "beta_dose" if self.dose else "tau_t2"

    @property
    def psense_terms(self) -> tuple[str, ...]:
        """Parameters to power-scale: the focal effect plus variant-defining terms.

        Power-scaling used to cover :attr:`effect_term` alone, which left every term
        that *defines* a variant unmeasured (#390 P2). A reader of DID-007 saw no flag
        on its period-varying dose structure because it was never measured, not because
        it came back clean; likewise DID-013's between-child catch-up scale. Those are
        the places a weak likelihood is most likely, each being informed by far fewer
        observations than the headline.

        Deliberately stops at variant-defining terms. Adding the ordinary nuisance
        scales (``kappa``, ``sigma_child``) would flag across the whole suite at this n
        and bury the rows worth reading. ``sigma_delta`` remains a sensitivity quantity
        whatever it power-scales to — each waitlist deviation is informed by a single t3
        observation, so it cannot identify individual responders.
        """
        terms = [self.effect_term]
        if self.period_varying:
            terms += ["sigma_dose", "beta_dose_phase"]
        if self.use_varying_delta:
            terms.append("sigma_delta")
        return tuple(terms)

    @property
    def run_plan_digest(self) -> str:
        """Canonical digest of the fields that define the fitted equation (#576 finding 6).

        The prior-sensitivity runner rebuilds the *currently registered* declaration
        and compares it with the stored primary through model/outcome identity, the
        data hash, row counts and arm totals. None of those move when the likelihood,
        the intercept anchor, the age adjustment, the random-effect choice or a prior
        width changes, so a primary fitted under an older plan could receive — and be
        released by — a sweep generated under a newer one. This digest closes that:
        it is recorded on the reference and on every sweep cell, and a mismatch fails
        the bundle closed.

        Deliberately over the **modelling** fields only, taken from a fixed key list
        with the same defaults :class:`DiDModelSettings` resolves. Two consequences
        are wanted. A stored plan written before a field existed digests identically
        to a fresh plan that takes that field's default, so existing fits stay
        reproducible without a refit; and a prose revision to ``estimand`` /
        ``causal_status`` — this review makes several — does not invalidate evidence
        for an equation that did not change.
        """
        return did_run_plan_digest(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready run-plan contract for ``config.json``."""
        d = asdict(self)
        # JSON has no tuples; keep the integer wave/period vectors as lists.
        d["waves"] = list(self.waves)
        d["periods"] = list(self.periods)
        d["outcomes"] = list(self.outcomes)
        d["run_plan_digest"] = did_run_plan_digest(d)
        return d

    def prepare_kwargs(self) -> dict[str, Any]:
        """Arguments for ``load_and_prepare`` from the resolved plan.

        Binary models load the t1-t3 ``levels`` panel; dose models keep the P1/P2
        transition frame (``phase_mode="all"``) and add the ``attend`` session
        covariate. Both keep ``require_any_post=False`` (a child observed at only one
        wave still contributes to the arm-by-wave contrasts)."""
        if self.dose:
            return {
                "phase_mode": "all",
                "outcomes": self.outcomes,
                "covariates": ("attend",),
                "pre_required": (),
                "require_any_post": False,
            }
        return {
            "phase_mode": "levels",
            "outcomes": self.outcomes,
            "require_any_post": False,
        }

    def factory_kwargs(self) -> dict[str, Any]:
        """Arguments for ``build_did_model`` for this plan.

        ``period_varying_dose`` receives the *resolved* ``period_varying`` (dose AND
        the flag), matching the former inline behaviour."""
        return {
            "outcome_symbol": self.outcome_symbol,
            "waves": self.waves,
            "periods": self.periods,
            "use_child_re": self.use_child_re,
            "use_age": self.use_age,
            "dose": self.dose,
            "period_varying_dose": self.period_varying,
            "use_varying_delta": self.use_varying_delta,
            "use_intercept_anchor": self.use_intercept_anchor,
            "likelihood": self.likelihood,
            "tau_t2_prior_sigma": self.tau_t2_prior_sigma,
            "score_mean_link": self.score_mean_link,
            "arm_gap_t1_prior_sigma": self.arm_gap_t1_prior_sigma,
            "sigma_child_prior_sigma": self.sigma_child_prior_sigma,
            "kappa_prior_family": self.kappa_prior_family,
            "kappa_prior_sigma": self.kappa_prior_sigma,
        }

    def diagnostic_vars(self) -> list[str]:
        """Variables scanned by summaries and the convergence gate."""
        if not self.dose:
            variables = [
                "alpha_offset" if self.use_intercept_anchor else "alpha",
                "beta_period",
                "arm_gap_t1",
                "tau_t2",
                "arm_gap_t3",
            ]
        else:
            dose_vars = (
                ["mu_dose", "sigma_dose", "beta_dose_phase"]
                if self.period_varying
                else ["beta_dose"]
            )
            variables = [
                "alpha",
                "beta_period",
                "beta_group",
                "theta_treated",
                "gamma_t1",
                *dose_vars,
            ]
        if not self.off_floor:
            variables.append("kappa")
            if self.kappa_prior_family == "halfnormal_inverse_sqrt":
                # ``kappa`` is a Deterministic under the dispersion-scale
                # parameterisation, so name the sampled parameter too: the
                # diagnostics table and the prior-vs-posterior overlay should show
                # what NUTS actually explored, not only its reciprocal-square
                # transform.
                variables.append("inv_sqrt_kappa")
        if self.use_age:
            variables.append("gamma_A")
        if self.use_child_re:
            variables.append("sigma_child")
        if self.use_varying_delta:
            variables.append("sigma_delta")
        return variables

    def _required_robustness_markdown(self) -> str:
        """The release pairing this fit cannot publish without (#576 finding 2)."""
        if not self.link_sensitivity_required_for_release:
            return ""
        return (
            "## Required robustness\n\n"
            "Phoneme blending is response-link sensitive: each item has three "
            "response alternatives, so the ordinary inverse-logit score mean "
            "permits fitted means below chance while the guessing-floor link does "
            "not. Release requires this fit to be reported beside the current "
            f"`{self.required_link_companion_model_id}` fit; neither the "
            "ordinary-logit nor the guessing-floor estimate is sufficient evidence "
            "on its own.\n\n"
        )

    def recipe_markdown(self, *, title: str) -> str:
        """Undergraduate-friendly explanation generated from the resolved plan."""
        # A dose variant fits the P1/P2 transition frame and never reads ``waves``;
        # printing the inherited (0, 1, 2) named an analysis window the recipe's own
        # model does not have (#576 lower-severity 3).
        if self.dose:
            window = "Periods: " + ", ".join(f"P{p + 1}" for p in self.periods)
        else:
            window = "Waves: " + ", ".join(f"t{w + 1}" for w in self.waves)
        return (
            "Note: Generated from the validated difference-in-differences run plan; "
            "template drafted by an LLM-based AI tool (Claude Code/Opus 4.8).\n\n"
            f"# Model recipe: {title}\n\n"
            f"Model ID: `{self.model_id}`.\n\n"
            f"## Design\n\n{self.design}\n\n"
            f"## Estimand\n\n{self.estimand}\n\n"
            f"## Causal status\n\n{self.causal_status}\n\n"
            f"## Analysis population\n\n{self.analysis_population}\n\n"
            f"## Missing data\n\n{self.missing_data_assumption}\n\n"
            "## Terms\n\n"
            f"Outcome: `{self.outcome_symbol}`. {window}. Dose term: "
            f"{self.dose} (period-varying: {self.period_varying}). Child random "
            f"intercept: {self.use_child_re}. Age adjustment: {self.use_age}. "
            f"Score-mean link: {self.score_mean_link}. Dispersion prior: "
            f"{self.kappa_prior_family}.\n\n"
            "## Published estimand\n\n"
            f"This fit's headline is `{self.focal_estimand}`, read on the "
            f"{self.focal_estimand_scale} scale from "
            f"`{self.focal_estimand_artifact}`. The prior pushforward, the "
            "treatment-prior sweep and the release gate all read that same "
            f"quantity; `{self.effect_term}` is the coefficient whose prior is "
            "power-scaled and swept to produce it.\n\n"
            f"{self._required_robustness_markdown()}"
            "## Uncertainty and checks\n\n"
            "The fit reports a posterior distribution; interpret it only after the "
            "convergence gate and posterior-predictive checks pass. The saved "
            "`config.json` contains the same resolved run plan in machine-readable "
            "form.\n"
        )


def declared_did_settings(spec: ModelSpec) -> tuple[DiDModelSettings, str]:
    """Return typed settings and their source, rejecting mixed declarations."""
    settings = spec.model_settings
    if settings is not None:
        if spec.extra:
            raise ValueError(
                f"{spec.model_id}: DiD settings cannot be split between "
                "model_settings and extra"
            )
        if not isinstance(settings, DiDModelSettings):
            raise TypeError(
                f"{spec.model_id}: kind='did' requires DiDModelSettings, got "
                f"{type(settings).__name__}"
            )
        return settings, "typed"
    return (
        DiDModelSettings.from_legacy_extra(spec.extra, model_id=spec.model_id),
        "legacy_extra",
    )


def resolve_did_run_plan(spec: ModelSpec) -> DiDRunPlan:
    """Resolve and validate a difference-in-differences spec before any data are loaded."""
    if spec.kind != "did":
        raise ValueError(f"{spec.model_id}: expected kind 'did', got {spec.kind!r}")
    if not spec.outcome_symbol:
        raise ValueError(f"{spec.model_id}: outcome_symbol is required for a DiD model")

    settings, source = declared_did_settings(spec)
    own = spec.outcome_symbol
    off_floor = settings.likelihood == "bernoulli_offfloor"
    period_varying = settings.dose and settings.period_varying_dose
    # The outcome loads as its own outcome when a spec does not list an explicit set.
    outcomes = settings.outcomes if settings.outcomes else (own,)
    # An explicit ``outcomes`` tuple that omits the focal outcome used to resolve
    # cleanly and fail inside the factory with ``KeyError: Outcome 'W' missing from
    # prepared data`` — after the output directory had been reset and the panel read
    # (#576 lower-severity 4). The focal outcome is the one column the model cannot
    # be built without, so the check belongs before any I/O.
    if own not in outcomes:
        raise ValueError(
            f"{spec.model_id}: the declared outcomes {list(outcomes)} do not include "
            f"the focal outcome {own!r}, so the loader would never read the column "
            "this model fits"
        )
    if settings.score_mean_link != "logit" and own != "B":
        raise ValueError(
            f"{spec.model_id}: the three-choice guessing floor is a property of the "
            f"phoneme-blending items, not of the model; got outcome {own!r}"
        )
    # #576 finding 2. Required from the registered pair *and* from the outcome, so a
    # future graded B arm-by-wave fit outside the pair fails closed at the release
    # gate rather than publishing an unpaired below-chance-permitting estimate.
    link_pair_required = own == "B" and not off_floor and not settings.dose
    link_companion = (
        (
            DID_BLENDING_PRIMARY_MODEL_ID
            if settings.score_mean_link == "three_choice_guessing_floor"
            else DID_BLENDING_COMPANION_MODEL_ID
        )
        if link_pair_required
        else None
    )

    if settings.dose:
        cell_term = (
            "a saturated arm-by-period cell term (theta_treated at the mean treated "
            "dose is the crossover cell contrast (waitlist P2 - waitlist P1) - "
            "(immediate P2 - immediate P1), not a separately supported "
            "current-treatment-presence effect)"
        )
        design = (
            "Waitlist-crossover dose model on the P1/P2 transition rows (sessions are "
            f"interval exposures): {cell_term} with the randomised arm, the shared t1 "
            "baseline and age, and sessions entered centred and scaled among treated "
            "rows."
        )
        estimand = (
            "The published headline is the natural-scale session-dose marginal "
            "averaged over the fitted **treated** rows, each row stepped by the "
            "declared dose contrast at its own period's realised slope "
            "(dose_marginal_summary.csv). The per-period slopes and their "
            "between-period scale are reported beside it; "
            + (
                "mu_dose is the hierarchical centre those slopes are drawn around, "
                "not the realised average, and is the swept/power-scaled coefficient "
                "rather than the published quantity. "
                if period_varying
                else ""
            )
            + "Every one of these is observational, not a randomised contrast."
        )
        causal_status = (
            "Associational: the session-dose marginal and slopes are observational "
            "(sessions are not randomised); the randomised arm enters only as an "
            "adjuster, and the arm-by-period cell terms describe treatment timing "
            "and history rather than an isolated treatment-presence effect. The P2 "
            "slope relates P2 sessions to the t3 outcome conditional on t1; it is "
            "not a P2 gain slope, because the treatment-affected t2 period-start "
            "score and prior P1 dose are deliberately omitted."
        )
    else:
        design = (
            "Waitlist-crossover arm-by-wave levels model on the t1-t3 frame: the arm "
            "gap is estimated separately at each wave (arm_gap_t1 the pre-treatment "
            "balance quantity, tau_t2 the randomised t2 contrast, arm_gap_t3 the "
            "randomised early-start-versus-delayed-start schedule contrast, "
            "delta_crossover = tau_t2 - arm_gap_t3 the change between the two), with "
            "a non-centred child random intercept."
            + (
                " The score mean is mapped onto [1/3, 1] by the three-choice "
                "guessing floor, because each phoneme-blending item has three "
                "response alternatives and an expected score cannot fall below "
                "chance."
                if settings.score_mean_link == "three_choice_guessing_floor"
                else ""
            )
        )
        estimand = (
            "The published headline is the wave-standardised t2 arm gap on the "
            "outcome scale (did_summary.csv's tau_t2_items_*). Its coefficient "
            "tau_t2 is the randomised immediate-treatment-versus-no-treatment "
            "assignment contrast on the fitted available-case sample — the "
            "covariate-adjusted t2 arm-gap level, not the differenced quantity "
            "tau_t2 - arm_gap_t1; the child random intercept and the tight "
            "arm_gap_t1 prior give a partial, prior-weighted baseline adjustment "
            "rather than exact differencing. arm_gap_t3 is the randomised "
            "early-start-versus-delayed-start treatment-schedule contrast and "
            "delta_crossover is the change between those two randomised regime "
            "contrasts."
        )
        causal_status = (
            "tau_t2 is randomised: the effect of assignment to immediate treatment "
            "versus no treatment yet, read at t2 on the available-case sample. "
            "arm_gap_t3 is also identified by the original randomisation, but it is "
            "a different exposure contrast — assignment to the early-start "
            "treatment history versus the delayed-start one — and never a "
            "treated-versus-untreated effect, because both arms are treated by t3. "
            "Latent ability does not confound it: randomised assignment does not "
            "stop being randomised after crossover. What it cannot deliver is a "
            "mechanism — duration, carryover, maturation, ceiling effects and "
            "different taught blocks are inseparable — so delta_crossover is the "
            "change between two randomised regime contrasts and is never evidence "
            "of an identified catch-up process. Any dose term is observational."
        )
    analysis_population = (
        "Available-case children observed across the difference-in-differences waves "
        "(about 53-54 depending on outcome). The randomised interpretation applies to "
        "the t2 contrast on this available-case population, not automatically the "
        "complete randomised cohort."
    )
    missing_data_assumption = (
        "Available-case analysis under ignorable missingness: missing outcomes and "
        "covariates are assumed ignorable given the modelled covariates."
    )

    if settings.dose:
        focal_estimand = "treated_row_dose_marginal"
        focal_estimand_artifact = "dose_marginal_summary.csv"
    elif off_floor:
        focal_estimand = "tau_t2_off_floor_risk_difference"
        focal_estimand_artifact = "did_summary.csv"
    else:
        focal_estimand = "tau_t2_items"
        focal_estimand_artifact = "did_summary.csv"

    return DiDRunPlan(
        model_id=spec.model_id,
        outcome_symbol=own,
        settings_source=source,
        dose=settings.dose,
        period_varying=period_varying,
        likelihood=settings.likelihood,
        off_floor=off_floor,
        outcomes=outcomes,
        waves=settings.waves,
        periods=settings.periods,
        use_child_re=settings.use_child_re,
        use_age=settings.use_age,
        use_varying_delta=settings.use_varying_delta,
        use_intercept_anchor=settings.use_intercept_anchor,
        tau_t2_prior_sigma=settings.tau_t2_prior_sigma,
        score_mean_link=settings.score_mean_link,
        arm_gap_t1_prior_sigma=settings.arm_gap_t1_prior_sigma,
        sigma_child_prior_sigma=settings.sigma_child_prior_sigma,
        kappa_prior_family=settings.kappa_prior_family,
        kappa_prior_sigma=settings.kappa_prior_sigma,
        focal_estimand=focal_estimand,
        # Every DiD headline is a natural-scale (items or risk-difference) marginal,
        # never a bare logit coefficient. The release gate reads this field to decide
        # which sweep column's sign must be stable (#576 finding 1).
        focal_estimand_scale="natural",
        focal_estimand_artifact=focal_estimand_artifact,
        link_sensitivity_required_for_release=link_pair_required,
        required_link_companion_model_id=link_companion,
        design=design,
        estimand=estimand,
        causal_status=causal_status,
        analysis_population=analysis_population,
        missing_data_assumption=missing_data_assumption,
    )
