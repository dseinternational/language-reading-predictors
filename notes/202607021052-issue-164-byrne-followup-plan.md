# Byrne reading-language-memory: follow-up analysis plan (issue #164)

<!-- cspell:ignore Byrne MacDonald Buckley readgrp basread basspel bpvs trog woco basdig bassim basnum basmat rlmhg rlmjc xsbr xspg rlm natcen -->

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).
>
> Measurement-gate status substantially updated by a LLM-based AI tool (Codex/GPT-5).

> [!WARNING]
> This note was prepared by an AI tool and may contain mistakes. It is a
> **forward-looking plan**, not results: it proposes a sequence of analyses and,
> more importantly, surfaces the human / data-owner decisions that must be settled
> before most of them can proceed. Verify the provenance and measurement claims
> against the primary sources before any of this enters a report.

**Status: historical roadmap, updated 2026-08-14.** Subsequent work is tracked in #338 and #409; several models below are now implemented. The remaining gates are recorded here so the original roadmap does not misstate the current measurement position.

## Where we are

- PR #163 added the prepared Byrne, MacDonald & Buckley reading-language-memory
  extracts (`data/reading-language-memory/`, 97 children × 5 waves), an audit
  reproduction script (`scripts/replicate_reading_language_memory.py`), and
  `rlmhg01` — a descriptive BAS word-reading growth model for waves 1–3.
- PR #171 (#165 PR 1, **merged 2026-07-02**) brought `rlmhg01` **into the statistical-model package**
  behind an explicit dataset / measure / spec metadata layer (`DatasetSpec`,
  per-study `StudyMeasure`, `LongitudinalPanel`, `kind="historical_growth"`), so
  historical models now share the sampler, convergence gate, output layout and
  report conventions. Every historical model already declares
  `study_id` / `design` / `estimand_type` / `causal_status` in its `config.json`.

So the **infrastructure to add more historical models cheaply now exists on `main`**. What
remains is (a) a short list of data-owner / education-lead decisions and (b) the
per-measure and cross-study modelling work those decisions unblock.

## Analysis sequence

1. **`rlmhg01` — BAS word reading, waves 1–3 (DONE).** Complete-case n=76
   (23 Down syndrome / 32 average / 21 reading-matched), Beta-Binomial on the
   bounded count (ceiling 87 — the _observed_ maximum in the prepared extract, not
   yet checked against the BAS manual; see decision 3), descriptive group-by-wave
   growth. Audit baseline =
   the Table 2 complete-case means.
2. **`rlmhg02`+ — parallel measure-specific growth models.** One historical
   growth model per additional measure: spelling (`basspel`), reading
   comprehension (`woco`), receptive vocabulary (`bpvs`), receptive grammar
   (`trog`), digit recall (`basdig`), similarities/verbal reasoning (`bassim`),
   number skills (`basnum`), and non-verbal reasoning (`basmat`, **wave 3+ only** —
   a shorter panel, handle separately). Same descriptive natural-history framing.
   Measurement readiness is now measure-specific: `basread`, `bpvs`, `trog`,
   `basdig` and `bassim` have confirmed bounds, while `basspel`, `woco` and
   `basnum` retain observed-maximum placeholders and `basmat` retains an
   instrument-identity caveat.
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

**Panel depth / late-wave group attrition.** `rlmhg01` stopping at wave 3
generalises: for `basread`, wave 5 is Down-syndrome-only (average and
reading-matched both drop to zero observed) and wave 4 already thins to 20/25/16,
so any measure whose panel runs past wave 3 loses the between-group contrast at
the final wave(s). Each `rlmhg02`+ model must state its usable wave range per
group; the joint model (`rlmjc01`) and the three-group-vs-two-group framing
(decision 4) both depend on where the group comparison stays estimable. `basmat`
(wave 3+ only) is the extreme case already flagged.

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
3. **Measure ceilings (measure-specific gate).** Confirmed Beta-Binomial
   denominators are `basread=90`, `bpvs=32`, `trog=20`, `basdig=34` and
   `bassim=21`. `basspel=18`, `woco=31` and `basnum=60` remain
   observed-maximum placeholders with `n_trials_confirmed=False`; `basmat`
   retains an instrument-identity caveat. Those unresolved measures need manual
   confirmation or a justified alternative likelihood before publication. They
   do not block the narrow Phase C reverse targets because all four measures in
   that question have confirmed bounds.
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
7. **Random-effects variance structure (`build_historical_growth_model`).** The
   shared `rlmhg01` factory (on #171) uses a single `sigma_subject` and a single
   Beta-Binomial `kappa` pooled across all three groups — i.e. it assumes equal
   between-child spread and equal overdispersion for Down syndrome, average and
   reading-matched children. That homogeneity is doubtful here (DS samples
   typically show wider between-child reading variance, which a single shared SD
   over-shrinks). Agreed in review: index the subject-intercept SD by group
   (`sigma_subject` → `dims="group"`, independent `HalfNormal`s, keeping the
   within-group de-meaning) and consider indexing `kappa` by group; random slopes
   on wave are lower priority with only three waves. It belongs in the #171 factory
   so it propagates to `rlmhg02`+ automatically, and it interacts with decision 4 —
   heterogeneous variances are exactly where the Down-syndrome-vs-reading-matched
   contrast can diverge from the pooled three-group model. Tracked as a follow-up
   on #171.

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

Reconcile the 96-versus-97 extract lineage and decide whether the paper's missing visual-recall variables can be recovered. Confirm the remaining `basspel`, `woco`, `basnum` and `basmat` instrument metadata before publishing models that use them. For the reciprocal reading-language-auditory-memory question, the graph is adopted but both candidate models failed the pre-fit recovery gate (`notes/202608141812-byrne-lcsm-feasibility.md`), so no Phase C model is registered. Bridge and pooled models remain a separate methodological track.
