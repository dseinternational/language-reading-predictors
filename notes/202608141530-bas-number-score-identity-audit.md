<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Ricciardelli basnum -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# BAS number-skills score-identity audit

**Status: evidence audit for #338 and #409, 2026-08-14.** This note records why the prepared Byrne/RLM `basnum` column must now be treated as having an unresolved score definition, separately from its already unconfirmed bounded-count denominator. It does not identify the correct score or authorise a likelihood change.

## Evidence

The prepared long extract contains integer `basnum` values from 0 to 60. The cohort's open-access companion paper describes the BAS number-skills values as raw scores and reports group means on the same numerical scale (Byrne et al., 1995, DOI [10.3104/reports.51](https://doi.org/10.3104/reports.51)). The repository has therefore operationally used the observed maximum, 60, as a provisional Beta-Binomial denominator while setting `n_trials_confirmed=False`.

Two independent descriptions of the first-edition BAS Basic Number Skills forms conflict with that interpretation:

- Ricciardelli's University of Adelaide thesis states that BAS Basic Number Skills Test Form B was administered under standard instructions and that Table 6.3 reports each test's maximum possible raw score. The table gives 34 items and a maximum raw score of 34 (Ricciardelli, 1989, [persistent repository record](https://hdl.handle.net/2440/19089)).
- Hatcher, Hulme and Snowling describe BAS Basic Number Skills Test Form C as 34 paper-and-pencil problems and define the score as the number correct (Hatcher et al., 2004, DOI [10.1111/j.1469-7610.2004.00225.x](https://doi.org/10.1111/j.1469-7610.2004.00225.x)).

The agreement across Forms B and C does not prove which form the Byrne cohort received. It does show that the prepared 0–60 column cannot safely be described as the standard item-correct raw score for either documented form. Plausible explanations include a different form, additional items, an ability-score transformation, or an undocumented data conversion; choosing among them without the source record would be speculation.

## Decision

Keep `n_trials=60` only as an operational observed-maximum placeholder so existing diagnostic fits can run. Keep `n_trials_confirmed=False`, add `score_definition_confirmed=False`, and store the evidence conflict in `score_definition_note`. The fit-time publication input contract now emits a separate score-definition blocker. This distinction matters because a confirmed instrument name does not validate the numerical representation supplied to the likelihood.

No current scientific finding is newly withdrawn by this change: every Byrne fit is already withheld by the unresolved 96-versus-97 extract lineage, and fits using `basnum` were additionally withheld by its provisional denominator. The change makes the reason reproducible and prevents a later ceiling-only sign-off from accidentally clearing a score whose form or transformation remains unknown.

The directly affected registered models are `lrp-rlm-hg-008`, `lrp-rlm-mm-001`, `lrp-rlm-adj-001` and `lrp-rlm-hs-001`. Any future concurrent, wider-gain or coupled model using `basnum` inherits the same blocker.

## Resolution needed

Obtain the cohort's test record sheet, administration manual reference, data dictionary or transformation code and answer three questions: which BAS form was administered; whether the stored value is number-correct, an ability score or another composite; and what range is valid for that score. If it is a bounded item count, record the defensible denominator and refit every dependent model. If it is a transformed score, replace count-scale preprocessing and any Beta-Binomial outcome likelihood with a distribution appropriate to that score before refitting. A sensitivity fit cannot substitute for identifying the quantity being modelled.
