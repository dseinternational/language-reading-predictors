# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The strict-typing exemption list can only shrink (#637 stage 4).

``[tool.mypy]`` used to name 36 files explicitly, so a new module went unchecked
until someone remembered to add it — and with ``follow_imports = "skip"`` even a
listed file was checked against ``Any`` for everything it imported. The whole
package is checked now, and the modules that are not yet clean are named in one
exemption list.

An exemption list rots in one direction: a module gets cleaned up as a side effect
of other work and nobody removes its entry, so the list stops describing anything.
This test runs mypy with the exemptions disabled and requires the set of modules
that still fail to be exactly the set declared — an entry that has become clean is
a failure, not a harmless leftover.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

#: One mypy diagnostic, keyed to the file it names. The separator is the *host's*,
#: not always ``/``: mypy prints ``src\language_reading_predictors\x.py`` on
#: Windows, so a ``src/``-only pattern matched nothing there — which left
#: :func:`test_every_exempted_module_still_needs_its_exemption` reporting all 97
#: exemptions as stale and its converse passing vacuously (2026-09-05 review).
_ERROR_LINE = re.compile(r"^(src[/\\][^:]+\.py):\d+: error:")


def _config() -> dict:
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["tool"]["mypy"]


def _declared_exemptions() -> set[str]:
    overrides = _config().get("overrides", [])
    exempt: set[str] = set()
    for override in overrides:
        if not override.get("ignore_errors"):
            continue
        module = override["module"]
        exempt.update([module] if isinstance(module, str) else module)
    return exempt


def _failing_modules() -> set[str]:
    """Modules that still fail the strict flags, with every exemption disabled."""
    # An empty config, so the exemption overrides in ``pyproject.toml`` do not
    # apply and the flags below are the whole policy. Without this mypy reads the
    # project config and reports nothing, which would make this test vacuous.
    empty_config = REPO / ".mypy_coverage_probe.ini"
    empty_config.write_text("[mypy]\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, "-m", "mypy",
            "--config-file", str(empty_config),
            "--python-version", "3.14",
            "--check-untyped-defs",
            "--disallow-incomplete-defs",
            "--disallow-untyped-defs",
            "--follow-imports", "skip",
            "--ignore-missing-imports",
            "--no-implicit-optional",
            "--no-error-summary",
            "--cache-dir", str(REPO / ".mypy_cache_coverage"),
            "src/language_reading_predictors",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    empty_config.unlink(missing_ok=True)
    # 0 = nothing failed the strict flags, 1 = it reported diagnostics. Anything
    # else is mypy failing to run, whose empty stdout would otherwise read as
    # "every module is clean".
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"the strict-flag probe could not run mypy (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    reported = [line for line in result.stdout.splitlines() if ": error:" in line]
    failing: set[str] = set()
    for line in reported:
        match = _ERROR_LINE.match(line)
        if match:
            path = match.group(1).replace("\\", "/")
            failing.add(
                path.removeprefix("src/")
                .removesuffix(".py")
                .replace("/", ".")
                .removesuffix(".__init__")
            )
    # Both callers below compare against this set, and both of them read an empty
    # set as good news — one reports every exemption stale, the other passes
    # having checked nothing. So a parse that attributes no diagnostic to any
    # module is a broken probe, not a clean package.
    if reported and not failing:
        raise AssertionError(
            "mypy reported diagnostics the probe could not attribute to a module, "
            "so the exemption checks would be vacuous; first unparsed line:\n"
            f"{reported[0]}"
        )
    return failing


def test_the_package_is_checked_whole_not_by_a_hand_maintained_file_list():
    config = _config()
    assert config["files"] == ["src/language_reading_predictors"]
    for flag in (
        "check_untyped_defs",
        "disallow_incomplete_defs",
        "disallow_untyped_defs",
        "no_implicit_optional",
    ):
        assert config[flag] is True, flag


def test_mypy_passes_as_configured():
    result = subprocess.run(
        [sys.executable, "-m", "mypy"], cwd=REPO, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-4000:]


@pytest.mark.slow
def test_every_exempted_module_still_needs_its_exemption():
    """No entry may outlive the work it stands for."""
    declared = _declared_exemptions()
    assert declared, "the exemption list is empty — delete the override instead"
    failing = _failing_modules()
    now_clean = sorted(declared - failing)
    assert now_clean == [], (
        "these modules now pass the strict flags; remove them from the mypy "
        f"exemption list in pyproject.toml: {now_clean}"
    )


@pytest.mark.slow
def test_nothing_fails_strict_typing_without_being_declared():
    """The converse: a failing module must be named, not silently tolerated."""
    undeclared = sorted(_failing_modules() - _declared_exemptions())
    assert undeclared == [], (
        "these modules fail the strict flags but are not exempt — fix them or "
        f"add them to the exemption list with a reason: {undeclared}"
    )


def test_registered_model_modules_are_not_exempt():
    """269 of them were unchecked; annotating one delegating call each fixed that.

    Checking them is what caught five specs passing a list where the family
    declares a tuple, and four companions reusing a parent's settings without
    checking the type they narrowed to.
    """
    exempt_models = sorted(m for m in _declared_exemptions() if ".lrp_" in m)
    assert exempt_models == [], exempt_models
