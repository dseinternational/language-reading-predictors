# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figure colours: one colour per role across every figure that shows it.

The arms, the observed values drawn over a model summary and the benefit / harm
bands are each drawn in several modules. These tests keep each role on the same
design-token colour everywhere, and keep the deprecated ``styles.COLOUR_*`` hues
out of the code: they raise a ``DeprecationWarning`` that Python hides outside
``__main__``, so nothing else would notice one coming back.
"""

from __future__ import annotations

import pathlib
import re

from dse_research_utils.plot.styles import CHART_COLOURS, diverging_palette

from language_reading_predictors.statistical_models import (
    arm_overlap,
    diagnostics,
    figure_artifacts,
    ppc_artifacts,
    predicted_scores,
    trajectory_plots,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_each_arm_has_one_colour_in_every_figure():
    control = {arm_overlap._CONTROL_COLOR, predicted_scores._CONTROL_COLOR, trajectory_plots.ARM_COLORS[0]}
    intervention = {
        arm_overlap._INTERVENTION_COLOR,
        predicted_scores._INTERVENTION_COLOR,
        trajectory_plots.ARM_COLORS[1],
    }
    assert control == {CHART_COLOURS[2]}
    assert intervention == {CHART_COLOURS[0]}


def test_observed_values_have_one_colour_distinct_from_the_arms():
    observed = {
        diagnostics._OBSERVED_COLOUR,
        figure_artifacts._OBSERVED_COLOUR,
        ppc_artifacts._OBSERVED_COLOUR,
        trajectory_plots._OBSERVED_COLOR,
    }
    assert observed == {CHART_COLOURS[1]}
    assert observed.isdisjoint(trajectory_plots.ARM_COLORS.values())


def test_icon_array_runs_from_harm_to_benefit_on_the_diverging_scale():
    harm, negligible, benefit = diverging_palette(3)
    assert predicted_scores._HARM_COLOR == harm
    assert predicted_scores._ROPE_COLOR == negligible
    assert predicted_scores._BENEFIT_COLOR == benefit


def test_no_code_uses_the_deprecated_colour_names():
    deprecated = re.compile(r"\bCOLOUR_(?:BLUE|GREEN|ORANGE|PURPLE|RED|YELLOW|DARK_[A-Z]+)\b")
    this_file = pathlib.Path(__file__).resolve()
    offenders = [
        f"{path.relative_to(REPO)}:{number}"
        for folder in ("src", "scripts", "notebooks", "tests")
        for path in sorted((REPO / folder).rglob("*.py"))
        if path.resolve() != this_file
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if deprecated.search(line)
    ]
    assert offenders == []
