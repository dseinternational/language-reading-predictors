# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bayesian statistical models for the RLI study (LRP52-LRP58).

LRP52-LRP54 are intention-to-treat (ITT) models for a single outcome.
LRP55 is a joint outcome model across eight tests.
LRP56-LRP58 are mechanism models encoding causal paths implied by the study DAG.

Shared conventions are defined in ``preprocessing``, ``priors``, ``hsgp``,
``likelihood`` and ``diagnostics``. Individual model modules (``lrp52`` ...
``lrp58``) are thin wrappers around factories in ``factories.py``.
"""
