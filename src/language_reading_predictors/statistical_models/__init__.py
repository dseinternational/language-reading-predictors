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
difference-in-differences family. LRP56-LRP58 are mechanism models encoding
causal paths implied by the study DAG.

Shared conventions are defined in ``preprocessing``, ``priors``, ``hsgp``,
``likelihood`` and ``diagnostics``. Individual model modules (the ``lrpitt01`` ...
``lrpitt11`` ITT suite, the ``lrpitt12`` ... ``lrpitt15b`` and ``lrpitt17`` ...
``lrpitt24`` companions, the ``lrpdid01`` ... ``lrpdid06`` crossover/DiD family,
and the ``lrp56`` ... ``lrp73`` mechanism/mediation models) are thin wrappers
around factories in ``factories.py``.
"""
