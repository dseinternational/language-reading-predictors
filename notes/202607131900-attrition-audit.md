<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Outcome-attrition audit: is an MNAR / informative-attrition sensitivity warranted?

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). Every count below was recomputed from `data/rli_data_long.csv` by `scripts/attrition_audit.py`; re-run it to reproduce. This note is evidence for the issue #230 §3 decision — it does not itself close the item.

Date: 2026-07-13 — relates to issue #230 §3 (measurement and missingness debts).

## The question

Issue #230 §3 flags that the suite's complete-case comparators only handle _covariate_ missingness, not _outcome_ dropout, and asks whether a delta-adjustment / pattern-mixture MNAR sensitivity is needed on the flagship models. That sensitivity only has bite if there is meaningful outcome attrition to be informative about. This audit checks whether there is.

## The evidence

Per outcome: children with a non-missing score at each wave, and attrition across the windows the headline estimands use — the randomised **ITT window** (t1 baseline but no t2 post), the **DiD crossover window** (t2 but no t3), and the **maintenance wave** (t3 but no t4). Cohort n = 54.

| Outcome                  | t1  | t2  | t3  | t4  | ITT-window | DiD-window | t4  |
| ------------------------ | --- | --- | --- | --- | ---------- | ---------- | --- |
| W word reading           | 53  | 53  | 53  | 51  | 0          | 0          | 2   |
| R receptive vocabulary   | 54  | 54  | 54  | 53  | 0          | 0          | 1   |
| E expressive vocabulary  | 54  | 54  | 54  | 53  | 0          | 0          | 1   |
| L letter-sound knowledge | 54  | 54  | 54  | 52  | 0          | 0          | 2   |
| B phoneme blending       | 54  | 54  | 54  | 53  | 0          | 0          | 1   |
| F basic concepts (CELF)  | 54  | 54  | 54  | 52  | 0          | 0          | 2   |
| T receptive grammar      | 54  | 54  | 54  | 53  | 0          | 0          | 1   |
| P phonetic spelling      | 54  | 53  | 53  | 54  | 1          | 0          | 0   |
| N nonword reading        | 50  | 53  | 52  | 52  | 0          | 1          | 1   |

## Reading

Outcome attrition is negligible, and it is absent from exactly the windows that carry the causal claims:

- **The randomised ITT window (t1 → t2) is complete** for all seven graded standardised outcomes (W, R, E, L, B, F, T) — every child with a baseline has a randomised post-score. The single ITT-window gap is one child on phonetic spelling (P), whose primary estimand is the binary off-floor transition, not a graded mean.
- **The DiD crossover window (t1 → t2 → t3) is complete** for every graded outcome. The single DiD-window gap is one child on nonword reading (N), a floored, post-only measure.
- **Only the final maintenance wave (t4) loses anyone**, and at most two children (W, L and F lose 2; the rest lose 0–1). t4 is post-crossover — both arms are treated by then — so it feeds the aligned / level / growth analyses, not the randomised ITT or the DiD contrast.

The nonword t1 count (50, not 54) is not ITT-window attrition: nonword is post-only by convention and its floor-rule model uses an age-only predictor with no t1 baseline (`measures.py`), so the four unobserved t1 nonword scores are never on the estimand's critical path.

## Recommendation (for the team to decide)

A delta-adjustment / pattern-mixture MNAR sensitivity is **not warranted for the primary randomised (ITT) or within-person (DiD) claims**: those windows have no missing outcomes, so there is nothing for a missingness mechanism — MAR or MNAR — to change. Building a δ-sweep there would stress-test a non-existent gap and risks implying an attrition concern the data do not support.

The only place outcome dropout exists at all is t4 (≤ 2 children), which enters the onset-aligned, level-factor and growth-curve analyses. If the team wants a belt-and-braces check, a **t4-only pattern-mixture δ-shift on those later-wave analyses** is the proportionate scope — but at 1–2 of 54 children it is very unlikely to move any conclusion. This note records the evidence; whether it closes issue #230 §3 (with or without the optional t4 check) is the team's call.

## Reproducing this audit

```bash
python scripts/attrition_audit.py
```

Prints the table and writes `attrition_audit.csv` to `output/audit/` (gitignored). Only the script and this note are committed.
