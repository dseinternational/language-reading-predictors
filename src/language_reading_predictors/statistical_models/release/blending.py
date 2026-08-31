# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The phoneme-blending response-link pair gates.

A ``B`` fit publishes only beside its guessing-floor twin (#608 / #619). ITT
binds through the content-addressed archive; the other seven families through
the lighter stored-artefact check. The gate defaults on the outcome symbol, so
a ``B`` fit in a family with no registered pair fails closed.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping
from language_reading_predictors.statistical_models.release.base import (
    _config_name,
    _load_config,
    _plan,
)

def _blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """Robustness-stage failures for the mandatory phoneme-blending link pair.

    Three families now carry a version of the policy, dispatched from here: the ITT
    archive-grade pair, the level pair (#584 decision 2) and the DiD pair (#576
    finding 2). They differ in evidence *strength*, never in bindingness.

    The registered policy is that neither ``lrp-rli-itt-008`` nor
    ``lrp-rli-itt-108`` may release without the validated trace-backed paired
    bundle, but until 2026-08-20 that was enforced only in the key-findings
    builder and the copied report partial — ``release_decision.json``, the
    artefact whose stated purpose is to combine exactly these policies, said
    ``publishable: true`` for an unpaired B fit (ITT code review, finding 1,
    ``notes/202608201205-itt-code-review-findings.md``). The requirement is
    derived from the module constant (so a stale stored plan cannot bypass it,
    mirroring the itt-010 missingness gate) *and* from the stored plan's
    ``link_sensitivity_required_for_release`` (so a future B-outcome ITT fit
    outside the registered pair fails closed rather than releasing unpaired).
    """
    kind = str(config.get("kind") or "")
    family_gate = _BLENDING_PAIR_GATES.get(kind)
    if family_gate is not None:
        return family_gate(output_dir, config)
    if kind != "itt":
        # Symbol-keyed fail-closed (#608 decision 1, implemented in #619). Every
        # family that registers a ``B`` model has a gate above. A ``B`` fit in a
        # family that does not is a model whose response-link sensitivity nothing
        # can verify -- so it must not publish, rather than slipping through because
        # its ``kind`` was not remembered. This is the direction the policy always
        # stated and the code did not do: before #619 the dispatch returned early
        # for every unlisted kind, so four families published unpaired ``B`` results
        # for months without anything failing.
        if str(config.get("outcome_symbol") or "") == "B":
            return (
                f"{config.get('model_id')} reports a phoneme-blending (B) outcome, "
                f"but the {kind!r} family has no registered response-link pair gate. "
                "Blending is a three-alternative forced-choice test whose expected "
                "score cannot fall below chance, and the #608 policy requires every "
                "B model to be released beside its guessing-floor twin; add the "
                "family's pairing before releasing this fit",
            )
        return ()
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        BLENDING_LINK_MODELS,
        evaluate_local_blending_link_sensitivity,
    )

    model_id = str(config.get("model_id") or "")
    plan = config.get("resolved_run_plan") or {}
    registered = model_id in dict(BLENDING_LINK_MODELS)
    declared = bool(plan.get("link_sensitivity_required_for_release"))
    if not registered and not declared:
        return ()
    if not registered:
        return (
            f"{model_id} declares a mandatory response-link sensitivity pairing, "
            "but no registered blending-link bundle covers it; register the pair "
            "before releasing",
        )
    try:
        status = evaluate_local_blending_link_sensitivity(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the B link-sensitivity pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory trace-backed phoneme-blending link pair "
            "(lrp-rli-itt-008 + lrp-rli-itt-108) is not release-ready: " + reason,
        )
    return ()


def _did_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The DiD family's phoneme-blending pairing (#576 finding 2).

    Same policy as the ITT and level pairs. It did not exist for ``did``, so
    ``lrp-rli-did-003`` — the ordinary-logit fit of a ten-item, three-alternative
    forced-choice test — could publish an unqualified ``B`` headline with no
    guessing-floor companion anywhere. The ITT companion does not cover it: the
    longitudinal random-intercept likelihood lets t1 and t3 data inform the t2
    posterior, so the two fits' response-link sensitivities are not interchangeable.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_did_blending_link_pair,
    )
    from language_reading_predictors.statistical_models.did import (
        DID_BLENDING_COMPANION_MODEL_ID,
        DID_BLENDING_PRIMARY_MODEL_ID,
    )

    try:
        status = evaluate_did_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the DiD B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            f"({DID_BLENDING_PRIMARY_MODEL_ID} + {DID_BLENDING_COMPANION_MODEL_ID}) "
            "is not release-ready: " + reason,
        )
    return ()


def _level_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The level family's phoneme-blending pairing (#584 decision 2).

    Same policy as the ITT pair, one rung down in evidence strength: the level
    check reads both fits' stored artefacts rather than recomputing their estimands
    from trace, because the level family has no content-addressed archive yet. It
    is still binding — a level B fit whose twin is absent, stale, ungated or fitted
    on different rows does not publish — and it fails closed on anything it cannot
    verify, so the weaker apparatus cannot become a weaker *policy*.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_level_blending_link_pair,
    )

    try:
        status = evaluate_level_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the level B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-lf-006 + lrp-rli-lf-106) is not release-ready: " + reason,
        )
    return ()


def _gain_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The gain family's phoneme-blending pairing (#596).

    Same policy and the same evidence tier as the level pair: both fits' stored
    artefacts are read and cross-checked rather than recomputed from trace. The
    gain family needed its own instance because neither the ITT nor the level
    companion covers it — it stacks three period transitions under a shared child
    random intercept and conditions on the own baseline, so it is a different
    likelihood over different rows, and its stored ordinary-link posterior puts
    10.7 % of its mass below the three-choice guessing floor.

    Scope is the **model of record**. ``evaluate_gain_blending_link_pair`` reads
    ``link_sensitivity_required_for_release`` from the fit's own resolved plan, and
    the gain resolver sets that only for the interaction-free graded primary — so
    the treated-only ``lrp-rli-gf-106`` and moderation ``lrp-rli-gf-206`` variants
    return "no link pairing" here rather than failing closed. That exemption is
    recorded and dated in
    ``notes/202608251100-gain-blending-guessing-floor-596.md``; it is the same
    boundary :func:`gate_applies` already draws, and it keeps fail-closed from
    demanding floor twins of variants that were never the published headline.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_gain_blending_link_pair,
    )

    try:
        status = evaluate_gain_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the gain B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-gf-006 + lrp-rli-gf-306) is not release-ready: " + reason,
        )
    return ()


def _aligned_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The aligned family's phoneme-blending pairing (#619).

    Same policy and the same evidence tier as the level, DiD and gain pairs: both
    fits' stored artefacts are read and cross-checked rather than recomputed from
    trace.

    Nothing in this family is randomised, and that is not an exemption. The #608
    decision binds every ``B`` model whether its published quantity is a contrast or
    an association, because the link determines the mapping from the latent scale to
    the reported one and any natural-scale headline inherits it. LRPAL06's published
    ``cohort_marginal.csv`` is exactly such a headline.

    Scope is the model of record: ``resolve_aligned_run_plan`` sets
    ``link_sensitivity_required_for_release`` only for the non-dose primary, so the
    collider-conditioned dose sensitivity returns "no link pairing" here rather than
    failing closed.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_aligned_blending_link_pair,
    )

    try:
        status = evaluate_aligned_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the aligned B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-al-006 + lrp-rli-al-306) is not release-ready: " + reason,
        )
    return ()


def _concurrent_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The concurrent family's phoneme-blending pairing (#619).

    Same policy and evidence tier as the level, DiD, gain and aligned pairs. Two
    features are particular to this family. Its published output is a *table* of
    per-wave marginals rather than a single card, so the pair check verifies the
    identity evidence plus the table's shape rather than comparing one headline
    number. And the link governs blending only as the **outcome**: the six sibling
    models that carry B as a *predictor* take it as a standardised logit covariate,
    not as a score mean, so their plans do not declare the pairing and they return
    "no link pairing" here.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_concurrent_blending_link_pair,
    )

    try:
        status = evaluate_concurrent_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the concurrent B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-ca-007 + lrp-rli-ca-307) is not release-ready: " + reason,
        )
    return ()


def _dose_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The dose family's phoneme-blending pairing (#619).

    Same policy and evidence tier as the level, DiD, gain, aligned and concurrent
    pairs. This is the family #608 used to close the observational-exemption
    argument: the declared focal estimand is the natural-scale treated-row dose
    marginal, published in items by ``dose_marginal_summary.csv``, so it inherits
    the link exactly as a randomised contrast does. That no dose slope is causal
    changes what the number means, not what scale it sits on.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_dose_blending_link_pair,
    )

    try:
        status = evaluate_dose_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the dose B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-dose-084 + lrp-rli-dose-384) is not release-ready: " + reason,
        )
    return ()


def _mediation_blending_pair_release_failures(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, ...]:
    """The mediation family's phoneme-blending pairing (#619).

    Same policy and evidence tier as the other stored-artefact pairs, but the link
    reaches further into this family than any other: every NDE, NIE and total is a
    difference of *simulated outcome means*, so ``score_mean_link`` enters the
    g-formula's counterfactual simulation cell by cell rather than any summary
    afterwards. LRP87's stored posterior also carries the largest below-chance share
    of any registered ``B`` fit (12.1 %).

    Scope is the model of record: ``lrp-rli-med-187`` declares ``companion_of`` and
    reproduces LRP87's numbers under an interventional relabelling, so its plan does
    not declare the pairing and it returns "no link pairing" here.
    """
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_mediation_blending_link_pair,
    )

    try:
        status = evaluate_mediation_blending_link_pair(output_dir, config=config)
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return (f"the mediation B link pair could not be evaluated: {exc}",)
    if status.get("required") and not status.get("ready"):
        reason = str(status.get("reason") or "the paired evidence is stale")
        return (
            "the mandatory phoneme-blending link pair "
            "(lrp-rli-med-087 + lrp-rli-med-387) is not release-ready: " + reason,
        )
    return ()


#: Per-family phoneme-blending pair gates, keyed by ``ModelSpec.kind`` (#619).
#: ``itt`` is deliberately absent: it takes the trace-backed content-addressed
#: archive path inside :func:`_blending_pair_release_failures` rather than the
#: stored-artefact check these seven share. A ``B`` fit whose kind is in neither
#: place fails closed -- see that function.
_BLENDING_PAIR_GATES: dict[str, Callable[[Path, Mapping[str, Any]], tuple[str, ...]]] = {
    "aligned": _aligned_blending_pair_release_failures,
    "concurrent": _concurrent_blending_pair_release_failures,
    "did": _did_blending_pair_release_failures,
    "dose_response": _dose_blending_pair_release_failures,
    "gain_factors": _gain_blending_pair_release_failures,
    "level_factors": _level_blending_pair_release_failures,
    "mediation": _mediation_blending_pair_release_failures,
}


def _joint_blending_scope_note(output_dir: Path, config: Mapping[str, Any]) -> str:
    """Qualifier when a joint fit carrying ``B`` has no release-ready 008/108 bundle
    beside it (2026-08-23 joint audit, finding 12).

    **The recorded policy scope.** The mandatory phoneme-blending response-link
    pairing (``lrp-rli-itt-008`` + ``lrp-rli-itt-108``) governs the *model of
    record* for ``B``: neither of those fits may release without the validated
    trace-backed bundle. ``lrp-rli-itt-012`` also fits ``B``, on the ordinary logit
    mean, and can publish a row for it — but the gate is keyed to ``kind == "itt"``,
    so nothing verified the condition its own findings box asserts in prose. That
    left an unguarded alternate route to a blending treatment claim.

    The resolution is *scope plus verification*, not extension of the withhold. A
    joint ``B`` row is a **secondary structural cross-check**: it is not
    independently release-qualified and cannot supersede or weaken the paired
    008/108 conclusion. Withholding nine valid outcomes because one row's companion
    is stale would destroy sound information to protect a row that is not the model
    of record — the same reasoning the dependence pairing already uses. So the
    check verifies the sibling bundle and, when it is not ready, attaches a note
    saying the joint ``B`` row must not be read as a blending treatment claim at
    all. Fail-closed: anything unverifiable attaches the note with its reason.
    """
    if str(config.get("kind") or "") != "joint":
        return ""
    if "B" not in [str(o) for o in (_plan(config).get("outcomes") or [])]:
        return ""

    def _note(reason: str) -> str:
        return (
            "This joint fit reports an ordinary-logit phoneme-blending (B) effect, "
            "which is a secondary structural cross-check and is not independently "
            "release-qualified. The mandatory response-link bundle "
            "(lrp-rli-itt-008 + lrp-rli-itt-108) that governs the B model of record "
            f"is not release-ready beside it ({reason}), so the B row here must not "
            "be read as a phoneme-blending treatment claim, and it cannot supersede "
            "or weaken the paired 008/108 conclusion."
        )

    from language_reading_predictors.statistical_models.blending_sensitivity import (
        BLENDING_PRIMARY_MODEL_ID,
        evaluate_local_blending_link_sensitivity,
    )

    try:
        directory = Path(output_dir).resolve()
        config_name = str(config.get("config_name") or "") or _config_name(
            directory, str(config.get("model_id") or "")
        )
        if not config_name:
            return _note("this fit's configuration name could not be resolved")
        primary_dir = directory.parent / f"{BLENDING_PRIMARY_MODEL_ID}-{config_name}"
        primary_config = _load_config(primary_dir)
        if not primary_config:
            return _note(f"{BLENDING_PRIMARY_MODEL_ID} has no readable config.json")
        theirs = str(primary_config.get("data_sha256") or "")
        ours = str(config.get("data_sha256") or "")
        if not theirs or not ours or theirs != ours:
            return _note("the bundle was not fitted on the same input data")
        status = evaluate_local_blending_link_sensitivity(
            primary_dir, config=primary_config
        )
        if status.get("required") and not status.get("ready"):
            return _note(str(status.get("reason") or "the paired evidence is stale"))
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return _note(f"the bundle could not be verified: {exc}")
    return ""
