# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP158 - complete-case comparator for LRP58 (letter-sound L -> word reading W).

The revised-DAG confounders that LRP58 fits enter it by the **missing-indicator
method**: hearing (HS) and speech production (SP = ``deapp_c``) are filled to their
column mean and a ``{col}_missing`` flag carries the unknown group as its own
adjustment level. That keeps every child, which matters at n ~ 54. (Phonological
memory, RW = ``erbto``, is **not** in either model's adjustment set — an earlier
version of this docstring said it was, and described a three-confounder complete-case
restriction this model does not perform; #586 finding 5.)

But it is not a free lunch, and the #258 review is right to press on it: mean
imputation plus an indicator **preserves rows without guaranteeing confounding
control**. It assumes the imputed group's confounder effect is captured by a single
intercept shift — i.e. that within the "unknown" stratum the confounder is unrelated
to both exposure and outcome once the flag is in the model. That is an assumption,
not a consequence of the method, and it is doing real work here because the
missingness is not trivial:

- hearing status unknown for **9 of 54 children at every wave** (~17%) under the
  three-valued composite the loader now derives from ``hearing`` / ``earinf``
  (the stored ``hearing_c`` column shows 10: one child recorded as hearing-impaired
  with no ear-infection record was left unknown by its strict OR — see
  ``notes/202608191030-hearing-composite-three-valued-or.md``);
- ``deapp_c`` ~4% of rows.

LRP158 is therefore the honest comparator: **identical to LRP58 in every respect**
except that the mean-imputed rows are dropped (``require_observed``), so both fitted
confounders are genuinely observed. The missingness indicators then go constant and
are dropped, so no vacuous coefficient is estimated. "Identical in every respect" is
now enforced by a paired-contract test rather than asserted: until #586 this model
silently omitted LRP58's ``outcomes=("W", "L")``, six-basis HSGP and tight
``InverseGamma(8, 8)`` lengthscale, so it also differed in loading contract and
functional form — three changes the comparison was never meant to include.

**How to read it.** If the mechanism curve agrees with LRP58's, the imputation is not
driving the result and LRP58 stands as the primary (higher-powered) fit. If they
diverge, the imputation *is* load-bearing and neither fit should be reported without
the other. Note what the comparison can and cannot show either way: complete-casing
selects a **different population**, in which the association may genuinely differ, so
a divergence is not by itself evidence that imputation "drives" LRP58's result. The
comparator is smaller and so has wider intervals **by construction** — that is the
price of the restriction, not a finding.

Same caveats as LRP58: latent general ability is **not** adjusted for and the child
random intercept does not stand in for it, so ``f^L`` is an **adjusted association**,
never a causal effect.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-158",
    kind="mechanism",
    title=(
        "Mechanism model: letter-sound (L) -> word reading (W) - "
        "complete-case comparator (no imputed confounders)"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
        # Matches LRP58: HSGP curve kept, target_accept lifted for boundary steps. A
        # few boundary divergences remain (the HSGP geometry LRP58 also shows);
        # disclosed in the report rather than removed by dropping the curve.
    target_accept=0.999,
    model_settings=MechanismModelSettings(
        # Identical to LRP58 ...
        outcomes=("W", "L"),
        adjust_baseline_symbol="W",
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        use_age_gp=False,
        phase_specific_mechanism=False,
        use_subject_random_intercept=True,
        # ... except that the imputed rows are dropped, so HS and SP are observed.
        require_observed=("hs", "deapp_c"),
        # Matches LRP58's thin-support HSGP reparameterisation. Omitting these left
        # the comparator on the shared defaults (m=10, ell ~ InverseGamma(5, 5)), so
        # it differed from its own baseline in functional form as well as in
        # missing-data policy and could not isolate the imputation question
        # (#586 finding 5). Kept in lockstep by
        # tests/statistical_models/test_mechanism_run_plan.py.
        mech_hsgp_m=6,
        mech_lengthscale_tight=True,
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_mechanism(SPEC, config=config)
