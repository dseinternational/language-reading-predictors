# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which registered statistical models does the #631 ERB quarantine actually change?

A/B the *prepared* analysis frame with the quarantine on and off. The validator is
neutralised in the off arm, because it is part of the same change and would reject
the row the quarantine exists to handle.
"""
from __future__ import annotations
import importlib
import warnings
import sys
import numpy as np

from language_reading_predictors import data_utils
from language_reading_predictors.statistical_models import preprocessing as pp
from language_reading_predictors.statistical_models.definitions import KINDS
from language_reading_predictors.statistical_models.registry import discover_models

# Every loader warns about dropped rows and about the quarantine itself; this probe
# prepares each frame twice, so the noise would bury the result.
warnings.filterwarnings("ignore")

OVERRIDES = {
    "mediation": ("mediation_settings", "resolve_mediation_run_plan"),
    "mediation_multi": ("mediation_settings", "resolve_mediation_multi_run_plan"),
}

def resolver_for(kind: str):
    mod, fn = OVERRIDES.get(kind, (kind, f"resolve_{kind}_run_plan"))
    try:
        return getattr(importlib.import_module(
            f"language_reading_predictors.statistical_models.{mod}"), fn)
    except (ImportError, AttributeError):
        return None

def fingerprint(prep) -> str:
    parts = [f"n_obs={getattr(prep,'n_obs',None)}", f"n_children={getattr(prep,'n_children',None)}"]
    for attr in ("covariates", "post_counts", "pre_logit", "counts"):
        d = getattr(prep, attr, None)
        if not isinstance(d, dict):
            continue
        for name in sorted(d):
            v = np.asarray(d[name], dtype=float)
            parts.append(f"{attr}.{name}:{np.nansum(v):.10f}:{int(np.isnan(v).sum())}")
    return "|".join(parts)

def prepared(model_id, kind):
    resolver = resolver_for(kind)
    if resolver is None:
        return None
    spec = discover_models()[model_id].load().SPEC
    plan = resolver(spec)
    if kind == "aligned":
        return pp.load_and_prepare_aligned(**plan.prepare_kwargs())
    if kind == "horseshoe":
        return pp.load_and_prepare(**plan.rli_prepare_kwargs())
    kwargs = getattr(plan, "prepare_kwargs", None)
    if not callable(kwargs):
        return None
    return pp.load_and_prepare(**kwargs())

def _no_validation(_df) -> None:
    """The off arm must skip the validator: it is part of the same change."""


saved_cells = data_utils.KNOWN_BAD_CELLS
saved_validator = data_utils.validate_erb_consistency

changed, identical, skipped = [], [], []
models = discover_models()
targets = sys.argv[1:] or sorted(models)
for model_id in targets:
    try:
        spec = models[model_id].load().SPEC
    except Exception as exc:
        skipped.append((model_id, f"no SPEC: {type(exc).__name__}"))
        continue
    kind = getattr(spec, "kind", "")
    if kind not in KINDS:
        skipped.append((model_id, f"unknown kind {kind!r}"))
        continue
    try:
        data_utils.KNOWN_BAD_CELLS = saved_cells
        data_utils.validate_erb_consistency = saved_validator
        if hasattr(pp.read_source_csv, "cache_clear"):
            pp.read_source_csv.cache_clear()
        with_q = prepared(model_id, kind)
        if with_q is None:
            skipped.append((model_id, "no prepare_kwargs"))
            continue
        a = fingerprint(with_q)

        data_utils.KNOWN_BAD_CELLS = ()
        data_utils.validate_erb_consistency = _no_validation
        if hasattr(pp.read_source_csv, "cache_clear"):
            pp.read_source_csv.cache_clear()
        b = fingerprint(prepared(model_id, kind))
    except Exception as exc:
        skipped.append((model_id, f"{type(exc).__name__}: {str(exc)[:70]}"))
        continue
    finally:
        data_utils.KNOWN_BAD_CELLS = saved_cells
        data_utils.validate_erb_consistency = saved_validator

    (changed if a != b else identical).append(model_id)

print(f"CHANGED   {len(changed)}")
for m in changed:
    print("   ", m)
print(f"IDENTICAL {len(identical)}")
print(f"SKIPPED   {len(skipped)}")
for m, why in skipped:
    print("   ", m, "-", why)
