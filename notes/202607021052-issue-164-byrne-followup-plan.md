# Byrne reading-language-memory: follow-up analysis plan (issue #164)

<!-- cspell:ignore Byrne MacDonald Buckley readgrp basread basspel bpvs trog woco basdig bassim basnum basmat rlmhg rlmjc xsbr xspg rlm natcen -->

::: {.callout-note}
Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).
:::

::: {.callout-warning}
This note was prepared by an AI tool and may contain mistakes. It is a
**forward-looking plan**, not results: it proposes a sequence of analyses and,
more importantly, surfaces the human / data-owner decisions that must be settled
before most of them can proceed. Verify the provenance and measurement claims
against the primary sources before any of this enters a report.
:::

**Status: plan for discussion (issue #164).** Nothing here is fitted beyond
`rlmhg01`. The job of this note is to lay out the roadmap and pin down what is
blocked on decisions rather than on code.

## Where we are

- PR #163 added the prepared Byrne, MacDonald & Buckley reading-language-memory
  extracts (`data/reading-language-memory/`, 97 children × 5 waves), an audit
  reproduction script (`scripts/replicate_reading_language_memory.py`), and
  `rlmhg01` — a descriptive BAS word-reading growth model for waves 1–3.
- PR #171 (#165 PR 1) brought `rlmhg01` **into the statistical-model package**
  behind an explicit dataset / measure / spec metadata layer (`DatasetSpec`,
  per-study `StudyMeasure`, `LongitudinalPanel`, `kind="historical_growth"`), so
  historical models now share the sampler, convergence gate, output layout and
  report conventions. Every historical model already declares
  `study_id` / `design` / `estimand_type` / `causal_status` in its `config.json`.

So the **infrastructure to add more historical models cheaply now exists**. What
remains is (a) a short list of data-owner / education-lead decisions and (b) the
per-measure and cross-study modelling work those decisions unblock.

## Analysis sequence

1. **`rlmhg01` — BAS word reading, waves 1–3 (DONE).** Complete-case n=76
   (23 Down syndrome / 32 average / 21 reading-matched), Beta-Binomial on the
   bounded count (ceiling 87), descriptive group-by-wave growth. Audit baseline =
   the Table 2 complete-case means.
2. **`rlmhg02`+ — parallel measure-specific growth models.** One historical
   growth model per additional measure: spelling (`basspel`), reading
   comprehension (`woco`), receptive vocabulary (`bpvs`), receptive grammar
   (`trog`), digit recall (`basdig`), similarities/verbal reasoning (`bassim`),
   number skills (`basnum`), and non-verbal reasoning (`basmat`, **wave 3+ only** —
   a shorter panel, handle separately). Same descriptive natural-history framing.
   **Blocked on measure ceilings** (see decisions) — only `basread`'s ceiling is
   confirmed, so only `basread` is registered in `RLM_MEASURES` today.
3. **`rlmjc01` — joint historical growth over a small measure set.** Correlated
   group-by-wave trajectories over, say, reading + a vocabulary + a grammar
   measure, mirroring the joint/correlated stat-model family, to describe how the
   trajectories move together within group.
4. **`xsbr01` — cross-study measurement bridge / calibration.** Link the Byrne BAS
   measures to the current-study (RLI) measures. **Measurement-link assumptions
   must be stated and justified before fitting** (which constructs are claimed
   comparable, on what anchoring, and the invariance assumed). This is a
   measurement diagnostic, not an effect estimate.
5. **`xspg01` — pooled cross-study growth.** Only **after** the bridge assumptions
   in step 4 are explicit and defensible. Pooling before then would blend
   incommensurable scales.

Ids follow the #165 scheme: `rlmhg` = historical growth, `rlmjc` = historical
joint/correlated, `xsbr` = cross-study bridge, `xspg` = cross-study pooled growth.

## Decisions needed before most of this can proceed

These are the gates. They are **human / data-owner / education-lead** calls, not
coding tasks.

1. **Authoritative extract + provenance.** The README flags that a separate raw
   export had 96 rows vs the prepared 97. Confirm the prepared wide/long files are
   the authoritative analysis extract (recommended interim position) and reconcile
   the 96-vs-97 discrepancy before any publication. Document the decision.
2. **Missing variables.** The visual-recall measures needed for the paper's full
   correlation tables are **not** in the prepared extract, so that reproduction is
   partial. Decide: recover them, or record them as explicitly unavailable and
   scope the replication accordingly.
3. **Measure ceilings (gates `rlmhg02`+).** Each bounded-count measure needs its
   confirmed instrument maximum for the Beta-Binomial denominator. Confirm the
   ceilings for `basspel` / `woco` / `bpvs` / `trog` / `basdig` / `bassim` /
   `basnum` / `basmat`, **or** decide those measures use a different likelihood
   (e.g. a Normal / Student-t on the raw score) where a bounded count is not
   appropriate. Until then only `basread` (ceiling 87) can be fitted as-is.
4. **Groups.** Model all three groups jointly per measure (recommended — the
   natural-history contrast `rlmhg01` already uses), or focus primary replication
   on Down syndrome vs reading-matched? The three-group model gives the fuller
   picture; the two-group contrast is the sharper developmental question.
5. **Replication targets.** Which published results are **formal replication
   targets**, which are **descriptive natural-history** estimates we report new,
   and which are **measurement-bridge diagnostics** only? Every historical model
   should be labelled accordingly (the `estimand_type` / `causal_status` metadata
   already supports this).
6. **Cross-study pooling scope.** Whether to attempt the bridge (`xsbr`) and
   pooled (`xspg`) models at all, and if so under what linking assumptions — a
   larger methodological commitment worth deciding before the per-measure work
   fixes the measure set.

## Report labelling (already supported)

Every historical model must state it is **descriptive natural-history evidence,
not an intervention effect** (`readgrp` is a cohort factor). `rlmhg01` does this in
its Overview callout. Bridge models are measurement diagnostics; pooled models
must state their linking assumptions up front. The #165 metadata fields carry this
into `config.json` and the report header, so it is machine-checkable, not just
prose.

## Acceptance-criteria mapping (issue #164)

- _Authoritative extract documented_ → decision 1.
- _Every historical model identifies its audit baseline + complete-case rule_ →
  built into the `historical_growth` pipeline (rlmhg01 writes the observed
  complete-case baseline); repeat per model.
- _Missing variables recovered or listed unavailable_ → decision 2.
- _Pooling states measurement-link assumptions before fitting_ → steps 4–5 +
  decision 6.
- _Reports label historical models as descriptive_ → report labelling above,
  enforced by the metadata layer.

## Suggested next step

Settle decisions 1–3 (extract/provenance, missing variables, measure ceilings)
with the data owner + education lead; that unblocks `rlmhg02`+ (parallel
measure-specific growth), which is then mechanical on the #165 layer. The bridge /
pooled models (decisions 4–6) are a separate, larger methodological track.
