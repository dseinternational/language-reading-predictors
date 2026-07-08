# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bayesian statistical models for the RLI study.

LRPITT01-LRPITT11 are the uniform DAG-faithful intention-to-treat (ITT) suite,
one outcome each (issue #119; supersede the deleted LRP52-LRP54/LRP74-LRP75).
The companions supersede the deleted ad-hoc models: LRPITT12 (joint outcome
model), LRPITT13/LRPITT13b/LRPITT14/LRPITT14b (SES robustness) and
LRPITT15/LRPITT15b (taught-vs-not-taught generalisation contrast) replace
LRP55/LRP60/LRP60a/LRP76. LRPITT17-LRPITT24 are general-ability robustness
companions and LRPDID01-LRPDID06 are the waitlist-crossover /
difference-in-differences family. LRP56-LRP58 and LRP71-LRP73 are mechanism
models; LRP59/LRP62/LRP64 are mediation models; LRP77 is the period-resolved
dose-response family; LRPGF/LRPLF/LRPAL are the factor and aligned families; and
LRP65/LRP67 are the adjusted and longitudinal dynamic companions.

Shared conventions are defined in ``preprocessing``, ``priors``, ``hsgp``,
``likelihood`` and ``diagnostics``. Individual model modules (the ``lrp-rli-itt-001`` ...
``lrp-rli-itt-011`` ITT suite, the ``lrp-rli-itt-012`` ... ``lrp-rli-itt-115`` and ``lrp-rli-itt-017`` ...
``lrp-rli-itt-024`` companions, the ``lrp-rli-did-001`` ... ``lrp-rli-did-006`` crossover/DiD family,
and the other ``lrp*`` model modules) are thin wrappers around factories in
``factories.py``.
"""
