# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prior panels must use the same numerical assumptions as the built model."""

import pymc as pm
import pytensor.tensor as pt

from language_reading_predictors.statistical_models import priors
from language_reading_predictors.statistical_models.factories.base import _add_child_random_intercept


def test_two_scales_from_one_constructor_produce_their_actual_densities(monkeypatch, tmp_path):
    with pm.Model() as model:
        priors.gamma_cross_prior().to_pymc("usual")
        priors.gamma_cross_prior(sigma=1).to_pymc("wide")
        priors.gamma_cross_prior(sigma=1).to_pymc("wide_again")
    plotted = {}

    def capture(density, output_dir, name, *, title):
        assert "Prior for " in title
        plotted[name.removeprefix("prior_")] = density
        return str(tmp_path / f"{name}.png")

    monkeypatch.setattr(priors, "plot_and_save", capture)
    paths = priors.save_model_prior_panels(model, str(tmp_path))
    rows = priors.priors_table(model).set_index("parameter")
    assert len(paths) == 2
    assert rows.loc["usual", "panel"] != rows.loc["wide", "panel"]
    assert rows.loc["wide", "panel"] == rows.loc["wide_again", "panel"]
    for name, sigma in (("usual", 0.3), ("wide", 1.0)):
        density = plotted[rows.loc[name, "panel"]]
        assert density.params_dict == {"mu": 0, "sigma": sigma}
        assert priors._dist_from_rv(model[name]) == rows.loc[name, "distribution"]
    assert "Normal(0, 1)" in rows.loc["wide", "rationale"]
    assert "Normal(0, 0.3)" not in rows.loc["wide", "rationale"]


def test_density_identity_depends_on_parameters_not_registration_order():
    with pm.Model() as first:
        priors.gamma_cross_prior(sigma=1).to_pymc("a")
        priors.gamma_cross_prior().to_pymc("b")
    with pm.Model() as second:
        priors.gamma_cross_prior().to_pymc("renamed_b")
        priors.gamma_cross_prior(sigma=1).to_pymc("renamed_a")
    assert priors.used_prior_keys(first) == list(reversed(priors.used_prior_keys(second)))


def test_random_intercept_rationale_uses_its_declared_scale():
    with pm.Model(coords={"child": [0, 1]}) as model:
        _add_child_random_intercept(pt.zeros(2), pt.as_tensor([0, 1]), sigma_prior_sigma=1.0)
    row = priors.priors_table(model).set_index("parameter").loc["sigma_child"]
    assert row["distribution"] == "HalfNormal(1)"
    assert "HalfNormal(1)" in row["rationale"]


def test_model_panel_title_names_the_parameters_without_inventing_a_causal_role():
    with pm.Model() as model:
        priors.tau_prior().to_pymc("beta_association", role="association")
    key = priors.used_prior_keys(model)[0]
    title = priors.model_prior_panel_title(model, key)
    assert "beta_association" in title
    assert "causal" not in title.lower()
