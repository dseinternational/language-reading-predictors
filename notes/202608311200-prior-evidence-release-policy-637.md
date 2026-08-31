<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Unavailable estimand-scale prior evidence qualifies a release; it does not withhold one

- **Date:** 2026-08-31
- **Status:** release-policy decision — implements the open question in #637 stage 1 ("Decide explicitly whether an unavailable required prior check withholds publication or produces a named qualification")
- **Issue:** #637

## The decision

An `unavailable` row in a fit's `prior_pushforward.csv` attaches a **named publication qualification** to the release decision. It does not withhold publication.

The qualification names the estimands whose check could not be computed and says what is thereby unknown: "the estimand-scale prior check is unavailable for `<estimands>`, so this fit's prior influence on the reported scale is unquantified". It is written to `release_decision.json` under `publication_qualification` and rendered by `_gate_badge.qmd`, which since the 2026-08-24 historical-joint review shows qualifications on published fits as well as on development-only ones. It is derived from the stored table, so re-deciding a stored directory at render time reproduces it.

## Why a qualification and not a withhold

The estimand-scale prior pushforward is evidence **about the prior**, not a scientific result. Its absence leaves the posterior, the convergence gate, `priors_table.csv`, the prior-versus-posterior overlay and `psense_summary.csv` all intact and all still checkable. Nothing about the fitted answer becomes less trustworthy; what a reader loses is the ability to judge, on the scale the result is reported in, how much of that answer the prior supplied. That is a real loss and it has to be visible, which is exactly what #381 was about — but it is not grounds to suppress the tables.

Withholding would also be badly targeted. Some families legitimately have no contrast to push a prior through: a single-cohort `aligned` variant writes an `unavailable` row saying so, and that row is correct, informative and permanent. A withholding policy would refuse to publish those fits forever, for stating a true fact about their own design.

## What made the choice available at all

Before this change the qualification could not have been written honestly. Five call sites — `aligned`, `joint`, `dose_response`, `mechanism` and the shared `growth_contrast_pushforward_rows` — wrapped their pushforward computation in `except Exception`, so a `KeyError`, a wrong dimension or a schema defect produced exactly the same `status="unavailable"` row as a genuinely absent prior group. A qualification reading "prior influence unquantified" would have been true of the first case and misleading in the second, where the honest statement is "this fit has a bug".

Those handlers now catch only `PriorEvidenceUnavailable`, raised by `require_prior_evidence` for the two conditions that really are absences: no persisted `prior` group, and a prior group that does not carry the term the check is about. Everything else fails the run. So an `unavailable` row now means one thing, and the qualification means what it says.

## Scope and effect on stored fits

None of the 246 stored `prior_pushforward.csv` files carries an `unavailable` row, so no stored release decision changes. 115 of them predate the labelled schema and have no `status` column at all; those are read as carrying no unavailable rows, which is correct — they were written by the bare-numeric path, which only ever emitted computed values.
