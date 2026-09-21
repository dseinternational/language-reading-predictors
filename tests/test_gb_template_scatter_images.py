# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Literal SHAP scatter images in a GB report template must be declared by the model.

Most gradient-boosting templates list their scatter plots from the files found
on disk, so they cannot reference a plot the fit did not write. A template that
names an image literally can: ``lrp-rli-gbg-012`` kept four "selected dependence
pairs" after their ``ShapScatterSpec`` entries had left the model module, and the
rendered report showed four broken images. Neither the fit nor the render fails
on a missing image, so this is the only place the mismatch is caught.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from language_reading_predictors.models.common import ModelConfig
from language_reading_predictors.models.registry import MODELS

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "docs" / "models"

_SCATTER_IMAGE = re.compile(r"shap_scatter_[A-Za-z0-9_]+(?=\.(?:png|svg)\b)")


def _template(config: ModelConfig) -> Path | None:
    """Resolve the template as ``BasePipeline.report`` does: own, then parent's."""
    candidates = [TEMPLATES / config.model_id / "index.qmd"]
    if config.variant_of:
        candidates.append(TEMPLATES / config.variant_of / "index.qmd")
    return next((c for c in candidates if c.exists()), None)


def _declared_scatter_images(config: ModelConfig) -> set[str]:
    """File stems the model's specs write at fit time.

    Restates the naming rule split between ``BasePipeline.shap_scatter_plots``
    (suffix defaults to ``by_<color_by>``) and ``plot_utils.save_shap_scatter_plots``
    (``shap_scatter_<feature>[_<suffix>]``).
    """
    stems: set[str] = set()
    for spec in config.shap_scatter_specs:
        suffix = spec.filename_suffix
        if suffix is None and spec.color_by is not None:
            suffix = f"by_{spec.color_by}"
        tail = f"_{suffix}" if suffix else ""
        for feature in spec.predictors or config.predictor_vars:
            stems.add(f"shap_scatter_{feature}{tail}")
    return stems


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_template_scatter_images_are_declared(model_id: str) -> None:
    config = MODELS[model_id]
    template = _template(config)
    assert template is not None, f"no report template resolves for {model_id}"

    referenced = set(_SCATTER_IMAGE.findall(template.read_text(encoding="utf-8")))
    undeclared = sorted(referenced - _declared_scatter_images(config))

    assert not undeclared, (
        f"{template.relative_to(REPO)} embeds scatter images that no "
        f"ShapScatterSpec on {model_id} produces: {undeclared}. Declare the spec "
        "in the model module or remove the image from the template."
    )


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_scatter_specs_name_fitted_predictors(model_id: str) -> None:
    """A spec naming a feature outside the predictor set raises at fit time."""
    config = MODELS[model_id]
    for spec in config.shap_scatter_specs:
        missing = sorted(set(spec.predictors or ()) - set(config.predictor_vars))
        assert not missing, (
            f"{model_id}: ShapScatterSpec({spec.description!r}) plots {missing}, "
            "which the model does not fit."
        )


def test_the_check_bites_on_a_literal_reference() -> None:
    """Guard the guard: the one template with literal references is really read."""
    config = MODELS["lrp-rli-gbg-012"]
    template = _template(config)
    assert template is not None
    referenced = set(_SCATTER_IMAGE.findall(template.read_text(encoding="utf-8")))
    assert "shap_scatter_age_by_yarclet" in referenced
