# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply the issue-321 findings-first order to statistical-report templates.

The rewrite is intentionally conservative: it recognises only the shared
statistical-report include contract, validates every candidate before writing
anything, and lists every non-conforming template in one failure. Model-specific
prose is left untouched; the recognised family-result include moves ahead of the
transparency and technical blocks.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "docs" / "models"

_INCLUDE_RE = re.compile(r"^\{\{< include (_partials/_[a-z_]+\.qmd) >\}\}$")
_TOP_SOURCE = {
    "_partials/_header.qmd",
    "_partials/_setup.qmd",
    "_partials/_convergence.qmd",
    "_partials/_gate_badge.qmd",
    "_partials/_key_findings.qmd",
    "_partials/_reading_guide.qmd",
}
_TOP_TARGET = (
    "_partials/_header.qmd",
    "_partials/_setup.qmd",
    "_partials/_gate_badge.qmd",
    "_partials/_key_findings.qmd",
    "_partials/_reading_guide.qmd",
)
_CORE = (
    "_partials/_header.qmd",
    "_partials/_setup.qmd",
    "_partials/_priors.qmd",
    "_partials/_prior_predictive.qmd",
    "_partials/_footer.qmd",
)


class TemplateContractError(ValueError):
    """A statistical report does not match the safe rewrite contract."""


def _include(line: str) -> str | None:
    match = _INCLUDE_RE.fullmatch(line.strip())
    return match.group(1) if match else None


def is_statistical_template(text: str) -> bool:
    """Identify templates using the shared statistical-report scaffolding."""
    includes = {_include(line) for line in text.splitlines()}
    return "_partials/_setup.qmd" in includes


def _single_index(includes: list[str | None], name: str) -> int:
    positions = [i for i, item in enumerate(includes) if item == name]
    if len(positions) != 1:
        raise TemplateContractError(
            f"expected exactly one {name} include; found {len(positions)}"
        )
    return positions[0]


def _validate_common(includes: list[str | None]) -> str:
    for name in _CORE:
        _single_index(includes, name)
    results = [item for item in includes if item and item.startswith("_partials/_results_")]
    if len(results) != 1:
        raise TemplateContractError(
            f"expected exactly one family result include; found {len(results)}"
        )
    return results[0]


def _validate_order(includes: list[str | None], names: tuple[str, ...]) -> None:
    positions = [_single_index(includes, name) for name in names]
    if positions != sorted(positions):
        raise TemplateContractError(
            "managed includes are not in the expected order: " + " -> ".join(names)
        )


def rewrite_template(text: str) -> str:
    """Return one statistical template in the signed-off findings-first order.

    Already-restructured input is returned byte-for-byte. Mixed old/new states,
    duplicated managed includes and unexpected prose inside the old top include
    block are rejected rather than guessed through.
    """
    lines = text.splitlines()
    includes = [_include(line) for line in lines]
    result = _validate_common(includes)

    has_gate = "_partials/_gate_badge.qmd" in includes
    has_technical = "_partials/_technical.qmd" in includes
    has_convergence = "_partials/_convergence.qmd" in includes
    has_diagnostics = "_partials/_diagnostics.qmd" in includes

    if has_gate or has_technical:
        if not (has_gate and has_technical) or has_convergence or has_diagnostics:
            raise TemplateContractError("mixed old/new report scaffolding")
        _validate_order(
            includes,
            (
                *_TOP_TARGET,
                result,
                "_partials/_priors.qmd",
                "_partials/_prior_predictive.qmd",
                "_partials/_technical.qmd",
                "_partials/_footer.qmd",
            ),
        )
        return text

    _single_index(includes, "_partials/_convergence.qmd")
    _single_index(includes, "_partials/_diagnostics.qmd")
    for optional in ("_partials/_key_findings.qmd", "_partials/_reading_guide.qmd"):
        if includes.count(optional) > 1:
            raise TemplateContractError(f"duplicate optional include: {optional}")

    _validate_order(
        includes,
        (
            "_partials/_header.qmd",
            "_partials/_setup.qmd",
            "_partials/_convergence.qmd",
            "_partials/_priors.qmd",
            "_partials/_prior_predictive.qmd",
            "_partials/_diagnostics.qmd",
            result,
            "_partials/_footer.qmd",
        ),
    )

    top_positions = [i for i, item in enumerate(includes) if item in _TOP_SOURCE]
    top_start, top_end = min(top_positions), max(top_positions)
    unexpected = [
        lines[i]
        for i in range(top_start, top_end + 1)
        if lines[i].strip() and includes[i] not in _TOP_SOURCE
    ]
    if unexpected:
        raise TemplateContractError(
            "unexpected content inside top include block: " + repr(unexpected[0])
        )

    top_block = [f"{{{{< include {name} >}}}}" for name in _TOP_TARGET]
    lines = lines[:top_start] + top_block + lines[top_end + 1 :]

    includes = [_include(line) for line in lines]
    bottom_source = {
        result,
        "_partials/_priors.qmd",
        "_partials/_prior_predictive.qmd",
        "_partials/_diagnostics.qmd",
        "_partials/_footer.qmd",
    }
    bottom_positions = [i for i, item in enumerate(includes) if item in bottom_source]
    bottom_start, bottom_end = min(bottom_positions), max(bottom_positions)
    unexpected = [
        lines[i]
        for i in range(bottom_start, bottom_end + 1)
        if lines[i].strip() and includes[i] not in bottom_source
    ]
    if unexpected:
        raise TemplateContractError(
            "unexpected content inside bottom include block: " + repr(unexpected[0])
        )
    bottom_target = (
        result,
        "_partials/_priors.qmd",
        "_partials/_prior_predictive.qmd",
        "_partials/_technical.qmd",
        "_partials/_footer.qmd",
    )
    bottom_block = [f"{{{{< include {name} >}}}}" for name in bottom_target]
    rewritten = lines[:bottom_start] + bottom_block + lines[bottom_end + 1 :]
    result_text = "\n".join(rewritten) + ("\n" if text.endswith("\n") else "")

    final_includes = [_include(line) for line in result_text.splitlines()]
    _validate_order(
        final_includes,
        (*_TOP_TARGET, *bottom_target),
    )
    return result_text


def restructure(*, write: bool) -> int:
    """Validate the inventory atomically, then optionally write all rewrites."""
    rewrites: dict[Path, str] = {}
    errors: list[tuple[Path, str]] = []
    candidates = 0
    for path in sorted(MODELS_DIR.glob("*/index.qmd")):
        text = path.read_text()
        if not is_statistical_template(text):
            continue
        candidates += 1
        try:
            updated = rewrite_template(text)
        except TemplateContractError as exc:
            errors.append((path, str(exc)))
            continue
        if updated != text:
            rewrites[path] = updated

    if not candidates:
        print("No statistical-report templates found.", file=sys.stderr)
        return 1
    if errors:
        print("No files written; non-conforming statistical reports:", file=sys.stderr)
        for path, message in errors:
            print(f"- {path.relative_to(REPO)}: {message}", file=sys.stderr)
        return 1
    if not rewrites:
        print(f"All {candidates} statistical-report templates already match issue #321.")
        return 0
    if not write:
        print(
            f"{len(rewrites)} of {candidates} statistical-report templates need "
            "restructuring; rerun with --write."
        )
        return 1
    for path, updated in rewrites.items():
        path.write_text(updated)
    print(f"Restructured {len(rewrites)} statistical-report templates.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="apply the validated rewrites"
    )
    args = parser.parse_args()
    return restructure(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
