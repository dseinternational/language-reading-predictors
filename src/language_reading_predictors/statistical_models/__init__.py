# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bayesian statistical models for the RLI study.

LRPITT01-LRPITT11 are the uniform DAG-faithful intention-to-treat (ITT) suite,
one outcome each (issue #119; supersede the deleted LRP52-LRP54/LRP74-LRP75).
LRP55 is a joint outcome model across eight tests. LRP56-LRP58 are mechanism
models encoding causal paths implied by the study DAG. LRP60/LRP60a are SES
robustness checks. LRP76 is the taught-vs-not-taught generalisation contrast.

Shared conventions are defined in ``preprocessing``, ``priors``, ``hsgp``,
``likelihood`` and ``diagnostics``. Individual model modules (the ``lrpitt01`` ...
``lrpitt11`` ITT suite, ``lrp55`` ... ``lrp60``, and the mechanism/mediation
models) are thin wrappers around factories in ``factories.py``.
"""
