> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Level-factors #584: the decision-free fixes, 2026-08-23

## What this note records

Issue #584 collects the findings of the follow-up review in `notes/202608231106-level-factors-follow-up-code-review.md`. Its four headline questions are **decisions for Frank**, and three of them cannot be implemented without a refit of all eleven reporting models. This note records what was fixed ahead of those decisions, why each change is safe to make without them, and what is deliberately left open.

Every numerical claim in the review was independently reproduced from the eleven stored `-reporting` traces before any of this was written (see the verification comment on #584). Two of the review's own claims needed correcting, and both corrections are carried into the fixes below.

## Fixed

### Contracts and validation

- **Both arms are now required at every wave, not only at t2** (finding 8). Each wave carries a published arm coefficient — the t1 balance term the changes are measured from, the randomised t2 change, and the post-crossover t3/t4 changes — so a wave with one arm leaves a coefficient determined by its prior and by the other waves while the report still calls the t2 change a t1-to-t2 randomised difference-in-differences. A wave with no fitted rows fails the same check. All eleven current fits contain both arms at all four waves, so nothing in the stored suite is affected.
- **Adjustment-set hygiene is enforced at plan resolution** (lower-severity 4). A repeated adjuster used to reach PyMC and fail there, after the output directory had already been reset; a `_missing` indicator declared without the covariate it flags used to fit a missingness flag with nothing to flag; and `ability_by_time=True` with no ability covariate silently did nothing while the declaration claimed a per-wave ability vector.
- **The eleven registered modules declare typed settings** (lower-severity 3). `METHODS.md` and `docs/models/README.md` say converted families declare immutable typed settings; the level modules were still passing legacy mutable `extra` dictionaries. The legacy path stays for callers that use it, and the resolved plan for every registered model is unchanged — `settings_source` moves from `legacy_extra` to `typed`.
- **The registered suite is pinned by a declarative contract table** (lower-severity 5). The previous coverage test globbed the modules and asserted "at least eleven, each internally consistent". The new table pins each model's outcome, likelihood, ability covariate, adjustment set, group/ability flags and focal term, so an accidental change to a published adjustment set fails a test rather than a report.

### Release integrity

- **Indexed prior-sensitivity evidence is now recomputed from its trace** (finding 3). `_validate_cell_trace` verified only that the _base_ variable existed for an indexed focal term such as `d_grp_time[t2]`, so a row whose focal mean had been edited, whose coordinate named a different wave, or whose `converged` flag was simply untrue attached on hashes alone. `tau_logit_mean` is the mean of the focal coordinate's draws in every family that writes this schema — the items-scale marginal lives in the `items_*` columns — so it is now recomputed from the element the provenance names, the interval must bracket it, the convergence gate is re-run over the cell's own recorded free variables, and the cell's focal term is bound to the primary's stored `resolved_run_plan` so a pre-#552 grid cannot certify a t1-centred primary. For the level family the direction probability is recomputed too: its items-scale marginal adds the same focal draw to every fitted row, so the published `pd` is exactly the focal coordinate's.
- The `gf`, `did` and level sweep **fixtures were declaring `converged=True` over five draws from one chain**, which is the same fail-open in test form. They now sample enough independent draws to face the gate, and their manifest rows are derived from their traces rather than asserted beside them.
- **The `adjusted_robustness` tier is ITT-only** (lower-severity 7). Keying it on the presence of an adjustment set labelled every level- and gain-factor primary that adjusts for a DAG confounder as a robustness comparator. The review said seven level primaries were affected; it is **eight** — LF-002 to LF-006 and LF-009 to LF-011. The withhold policy is uniform across tiers, so no stored release decision changes status.

### Robustness and reporting

- **Power scaling covers the free nuisance scales** (finding 6). The audit passed only `arm_gap_t1` and `d_grp_time`, which establishes focal-term behaviour rather than prior/likelihood robustness. Re-running power scaling over the stored traces reproduces the review's exploratory result exactly: `sigma_child` is flagged in **11 of 11** fits and `kappa` in **8 of 9** graded fits. Those rows now appear in `psense_summary.csv`, where the report already renders the whole table. The gate and the key-findings box still read the focal row only, so a nuisance conflict is disclosed rather than silently blocking a release.
- **The prior pushforward is guarded through `artifacts.guard_optional`** (lower-severity 1), so a failure lands in `artifact_manifest.json` instead of scrolling past in a warning, and **its empirical-Bayes anchor is disclosed** (lower-severity 2): the family was missing from the prior-predictive partial's data-anchored-prior map, and the pushforward section now says that the operating point the prior is pushed through comes from the observed t1 outcomes even though the prior on the contrast does not.
- **Five model reports described smaller models than were fitted** (finding 7). LF-002 to LF-006 omitted every `sum_c gamma_c z(c)` adjustment term and missing indicator from their equations and their causal-labelling caveat. Both are corrected, generated from each model's declared adjustment set. Note that the omission was narrower than the review implies: the rendered results table already listed `gamma_hs` / `gamma_deapp_c` / `gamma_erbto` with the baseline-timing note, so the fitted adjusters were visible to a reader — the equation and the caveat were wrong, not the whole report.
- **The key-findings ability caveat now describes what is computed** (finding 5). It said the headline was "the effect for a child of typical cognitive ability". The transform averages over every fitted t2 profile, keeping each child's own age, ability main effect, adjusters and fitted child intercept, and fixes only the added moderation increment at centred ability. It now says so. (Stored `key_findings.json` files carry the old sentence until they are regenerated.)
- **A pooled `group_by_time=False` plan no longer generates prose about a t2 randomised contrast it does not have** (lower-severity 6).
- **The cross-family synthesis is corrected** (finding 2). `notes/202608182200-findings-by-question.md` claimed to have been re-read from the 2026-08-20 refit but still quoted the pre-#552 LF-001 reading (t1 gap −0.17, t2 gap +0.25 called the randomised contrast, +1.7 words), while its own cross-family table printed the current +2.3. The paragraph now gives the t1-centred quantities from the stored fit and is dated as a correction.

## Left open, and why

These need the decisions #584 asks for, and three of them need a refit of all eleven reporting models:

1. **The natural-scale estimand (finding 1).** Nothing here changes what `level_t2_marginal_effect` computes. One consideration for that decision is recorded on the issue and is worth repeating: the current transform is a monotone per-draw transform of `d_grp_time[t2]`, so the items median, the direction probability and the ROPE all stay sign-coherent with the coefficient the report flags causal. A strict response-scale difference-in-differences is not — it disagrees with the coefficient's sign in about 5% of draws and moves the direction probability across 0.5 for both vocabulary outcomes (R 0.535 to 0.488, E 0.522 to 0.469). Whichever target is chosen, the direction probability has to be recomputed from the same functional, not left on the logit draws.
2. **The LF-006 guessing-floor policy (finding 4).** Adding a `score_mean_link` setting and a registered paired companion, or declaring LF-006 non-headline, is a decision about the family's release contract.
3. **The four-wave model of record (finding 5).** A t1/t2-only comparator is a new fit.
4. **The dispersion-scale prior (finding 6).** Adopting `1/sqrt(kappa)` here would change every graded level posterior.

Two further pieces of finding 3 are also open by design: **release-time revalidation of the attached traces**, and a **content-signed canonical manifest** that cannot be edited independently of its evidence. Both are policy choices about how published evidence is certified rather than repairs to a specific defect, and neither is exercised by any current level release — none of the eleven attaches a sweep at all.

## Verification

- `uv run pytest tests/statistical_models` — passes, including the new contract, validation, tamper and tier tests.
- `uv run ruff check src/`, `npm run format:check`, `npm run spellcheck`.
- No stored fit, report output or release decision was regenerated. The reporting artefacts still carry the pre-fix key-findings sentence and the narrower psense table until the refit that #584's acceptance criteria call for.
