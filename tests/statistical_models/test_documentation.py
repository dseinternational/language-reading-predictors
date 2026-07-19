# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Documentation contracts derived from the statistical-model registries."""

from __future__ import annotations

from pathlib import Path

from language_reading_predictors.statistical_models.documentation import (
    assert_registry_count_snapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_registry_count_snapshot_is_current():
    assert_registry_count_snapshot(
        REPOSITORY_ROOT / "docs" / "models" / "registry-counts.json"
    )


def test_inventory_explains_how_the_count_is_guarded():
    inventory = (REPOSITORY_ROOT / "docs" / "models" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "registry-counts.json" in inventory
    assert "check_statistical_documentation.py" in inventory


def test_undergraduate_itt_route_covers_the_phase_4_contract():
    chapter = (
        REPOSITORY_ROOT / "docs" / "report" / "chapters" / "methods-models.qmd"
    ).read_text(encoding="utf-8")

    for required in (
        "lrp-rli-itt-001",
        "Map the symbols to rows, columns, and code",
        "Synthetic example",
        "What partial pooling is—and why it is not in this model",
        "Common misinterpretation checks",
        "Adjusted associations unless",
        "environment-lock.json",
        "#sec-dag",
        "#sec-bridge",
    ):
        assert required in chapter


def test_report_glossary_is_substantive_and_causal_status_is_explicit():
    glossary = (REPOSITORY_ROOT / "docs" / "report" / "glossary.qmd").read_text(
        encoding="utf-8"
    )

    for term in (
        "**Adjusted association.**",
        "**Credible interval.**",
        "**Partial pooling.**",
        "**Posterior-predictive check (PPC).**",
        "**Random intercept.**",
        "**Tail probability.**",
    ):
        assert term in glossary
