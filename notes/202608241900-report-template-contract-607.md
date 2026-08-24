<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# The statistical-report include contract is the #373 order

- **Date:** 2026-08-24
- **Status:** implementation decision
- **Issue:** #607
- **Supersedes:** the ordering half of [202607152000-findings-first-report-order.md](202607152000-findings-first-report-order.md)

## Decision

The shared statistical-report include order is:

`_header` → `_setup` → `_gate_badge` → `_key_findings` → `_reading_guide` → model prose → `_priors` → `_prior_predictive` → the family result partial → `_technical` → `_footer`

The family result partial sits **below** the prior sections, not above them. `scripts/restructure_statistical_reports.py` and the shared paragraph in `AGENTS.md`, `CLAUDE.md` and `.github/copilot-instructions.md` now say so.

## Why the repository and the tooling disagreed

Two decisions were taken four days apart and only the first reached the tooling.

`c007d08b` (#352, 2026-07-16) established the findings-first scaffolding and put the result partial ahead of the priors. `040161bf` (#373, 2026-07-20) — "box-cull, BARG reproducibility, and priors-before-results reorder" — moved it back below them. #373 is the later decision, every template in the repository follows it, and the prior-predictive check was already argued in the #352 note itself to be "part of the scientific argument about whether the prior can generate plausible outcomes, not merely sampling plumbing", which is the same reasoning that puts it above the results.

The validator kept the #352 order and was never updated, so `rewrite_template` raised `TemplateContractError` for **264 of 264** statistical templates across all 21 families. Nothing caught it because `tests/statistical_models/test_report_restructure.py` exercised the contract only against synthetic fixtures — strings the repository does not contain.

## Why it mattered even though the script is a one-shot migration tool

Nothing in the fit pipeline calls the script, so no artefact was ever wrong. The costs were indirect:

1. The tool could not be used again. A future migration run through it would refuse every input, or "fix" 264 files by reverting #373.
2. `CLAUDE.md` is loaded into agent context every session, so an agent asked to add or check a report would believe results come before priors. That is not hypothetical: the #587 dose-response audit recommended "correcting" five correct templates on the strength of it, and the finding had to be withdrawn during remediation (#606, and the withdrawal is now marked in the audit note itself).

A stale contract that is also the documented one is worse than no contract, because it manufactures false findings that then cost review time to refute.

## What changed

- `scripts/restructure_statistical_reports.py` — the tail order now lives in one `_bottom_target()` helper read by **both** the already-restructured validation path and the legacy migration path, so the two cannot drift from each other again. The migration path's output changes accordingly: a legacy template now lands in the #373 order.
- The shared paragraph in `AGENTS.md`, `CLAUDE.md` and `.github/copilot-instructions.md`, which now states the real order and cites #373 and #607 so the next reader can see it was a decision rather than an accident.
- `tests/statistical_models/test_report_restructure.py` — four new tests: the real `docs/models/*/index.qmd` set matches the documented sequence; `rewrite_template` is a byte-for-byte no-op over all of it; the migration path targets the #373 order; and a #352-ordered template is rejected. The first two assert a floor of 200 templates so an empty glob cannot pass vacuously.

The new tests were mutation-checked: reverting `_bottom_target()` to the #352 order fails them, naming the offending templates.

## Not done here

No report was re-rendered and no fit was touched. Under this resolution the include order is unchanged — the repository was already right.

The alternative resolution, adopting the #352 order, was rejected: it is a 264-file migration plus a full re-render, it reverses a later deliberate decision, and no one has argued the scientific case for putting results above the prior-predictive check since #373 argued the opposite.
