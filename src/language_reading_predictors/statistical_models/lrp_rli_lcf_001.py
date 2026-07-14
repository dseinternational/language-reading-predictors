# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-RLI-LCF-001 - longitudinal correlated-domain-factor model (#313).

The four-wave extension of the cross-sectional ``corr_factor`` CFA (``lrp-rli-mm-001``):
correlated **vocabulary / code / grammar** domain factors measured at every timepoint
over the child x wave panel, delivering the **per-wave latent skill correlation
matrices** and the disattenuated latent slopes derived from them. It is the symmetric,
measurement-error-corrected counterpart to the concurrent regression family
(``lrp-rli-ca-001``, #312): the regression family gives directed, covariate-conditional
observed slopes; this gives the underlying latent correlation web with binomial counting
noise on both sides modelled out.

**Factors.** Vocabulary {R, E, TR, TE}, code {L, B}, grammar {F, T} — taught vocabulary
(TR/TE) folds into the vocabulary domain (Frank sign-off 2026-07-14; TR/TE are observed
at all four waves). Heavily-floored P/N are excluded, as in ``mm-001``.

**Structure.** Scalar-invariant loadings/residuals (the factors mean the same thing at
every t), per-wave within-factor LKJ correlations (the headline), a **trait/state**
across-wave decomposition (PSD by construction, unit factor variance, per-wave-free
within-correlation, compound-symmetry autocorrelation via the shared trait), and the
factor scores **marginalised out** of the Gaussian measurement likelihood (the mm-001
funnel fix — no per-child latent RV). Missing cells are masked, not dropped: one
``MvNormal`` per observed-cell pattern.

**Stance.** A measurement / triangulation model, **not** causal: every latent
correlation and slope is a descriptive association (#115 ID-2). At n ~ 54 a longitudinal
latent model is fragile and prior-dependent - the wide intervals are the honest result,
exactly as ``mm-001`` and the closed LRP66 reported. A prior-sensitivity companion and
the invariance relaxation (configural / AR-across-wave, by LOO) are follow-ups.

See ``notes/202607142330-lrp313-longitudinal-corr-factor.md``.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import (
    fit_longitudinal_corr_factor,
)

SPEC = ModelSpec(
    model_id="lrp-rli-lcf-001",
    kind="long_corr_factor",
    title=(
        "Longitudinal correlated-domain-factor model "
        "(per-wave latent skill correlations)"
    ),
    outcome_symbol="W",
    extra={
        "domains": {
            "vocabulary": ("R", "E", "TR", "TE"),
            "code": ("L", "B"),
            "grammar": ("F", "T"),
        },
        # Small-n latent geometry: even fully marginalised a few boundary divergences
        # can survive at the tier-default target_accept, so lift it (as mm-001 does at
        # 0.999) to clear the strict zero-divergence gate.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_longitudinal_corr_factor(SPEC, config=config)
