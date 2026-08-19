> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Decision: the Action Picture Test practical-difference thresholds are ratified

**Status: approved 2026-08-18 by the education lead.** This closes the open item flagged in `notes/202608180933-findings-01-itt.md` and the findings overview, and discharges the outstanding half of issue #144 for these two outcomes.

## What was approved

| Outcome                           | Scale fitted            | δ (ratified)                    |
| --------------------------------- | ----------------------- | ------------------------------- |
| APT expressive grammar (EG)       | 37 marks                | **1 mark of 37**                |
| APT expressive information (EI)   | 80 half-marks (doubled) | **2 half-marks — 1 whole mark** |
| APT information comparator (EI40) | 40 whole marks          | **1 mark of 40**                |

The two information values are the same threshold expressed on the two scales the pair is fitted on, so the primary and its denominator comparator are judged against an identical practical difference. The values ratified are the ones the fits already used, so **no refit and no recomputation is required** — `prob_benefit_ge_delta`, `prob_in_rope` and `prob_harm_ge_delta` as published already reflect them.

## How they were derived

By the project's existing rule, applied unchanged: **half the untaught arm's timepoint-1 to timepoint-2 gain, floored at 1 item**. That rule reproduces every previously adopted δ exactly (L 2, W 1, T 1, F 1), which is why it was used rather than a fresh judgement.

## The borderline that needed a decision

Information's rule value is **1.49 marks**, which rounds to 1. Because it sits just under the boundary, 2 marks would have been equally rule-consistent, and the choice is consequential: at δ = 1 the reported probability of a practically-meaningful benefit is 0.214, and a threshold of 2 marks would materially lower it. That is precisely why the value was escalated rather than settled silently. **The decision is 1 whole mark**, the rounded rule value, consistent with the floor applied to W, T and F.

Grammar's rule value did not sit near a boundary and needed no adjudication.

## What ratification does and does not change

It changes the **status** of the magnitude statements, not their content. Before ratification the size statements for these two outcomes were withheld from quotation because their threshold was an unapproved analyst judgement; they may now be quoted on the same footing as the rest of the suite.

It changes nothing about the **direction** statements, which never depended on δ, and nothing about the estimates themselves.

It does not make δ prospective. Like the rest of the suite's thresholds, these were set after the results existed, and the project's convention is to say so wherever they are reported (see `METHODS.md` on threshold provenance). Ratified is not the same as pre-specified, and the notes continue to describe them as post-hoc.

## Related

- `notes/202606251321-lrpitt-suite-design.md` — the deferral these models close.
- `notes/202608180933-findings-01-itt.md` — the ITT findings note carrying the δ reporting.
- `notes/202608180929-full-statistical-refit-2026-08.md` — the run record for the fits.
