<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Spec (proposal): Tier-1 "decoding-specificity" mini-suite — are letter-sound gains _used for decoding_, or merely _associated_ with reading?

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). **Status: proposal, awaiting sign-off.** This specs three models (1A/1B/1C) that reuse existing factories; nothing is built yet. It follows the same low-build-first discipline as `notes/202607172100-reverse-mediation-wr-ls-direction-spec.md`.

## 1. Why — and the reframing that governs everything

The question "are gains in letter-sound knowledge (`LS`) really being _used for decoding_, or just _associated_ with reading gains through other mechanisms?" is **not** a direction-of-effect question. We already learned (RI-CLPM feasibility → NO-GO, `notes/202607172230-riclpm-direction-plan.md`) that at n ≈ 54 × 4 the within-person cross-lag cannot resolve direction. Piling on more lagged direction models has low expected yield.

The tractable version is a **specificity / dissociation** problem. Letter-sound gains can co-move with reading gains for three _non-decoding_ reasons we must rule out:

1. **shared latent general ability** (`GA`, unmeasured, structurally unblockable in the DAG);
2. **shared teaching dose** — the intervention teaches letters _and_ reading (the `IG → IS → {LS, WR}` witness that already makes the mediation family unidentified for causal NDE/NIE); and
3. **reverse / bidirectional learning** — reading print feeds back into letter knowledge (Roch & Jarrold 2012 found exactly this longitudinal `WR → NW` coupling _in Down syndrome_).

"Really used for decoding" has a signature those three cannot fake: the effect must run through the **alphabetic operation** (grapheme→phoneme conversion + blending), whose behavioural fingerprint is **nonword reading (`N`)** — a string that _cannot_ be read by sight. This matters especially in Down syndrome, where readers are well documented to lean on **sight-word** recognition with comparatively weak **phonological decoding** (Cupples & Iacono 2000); the RLI intervention was designed partly to bring the decoding route online, so "is letter-sound knowledge being converted into decoding" is the study's central mechanistic question, not a peripheral one.

Every strong design below therefore hinges on **nonword reading as the criterion** and on **within-model / matched contrasts** that a pure-confounding account does not predict — rather than trying to resolve one small coefficient away from zero.

## 2. Governing caveats — the honest ceiling (state in every write-up)

- **`GA` is unblockable.** It is latent and a parent of every skill; the child random intercept does **not** stand in for it (a zero-mean, predictor-independent effect cannot block `LS ← GA → Y`). Every association here is an **adjusted association**, never a causal skill→skill effect. The only randomised warrant in the suite is the ITT arm.
- **The mediation family stays ID-2.** The `IG → IS → {LS, PA, NW, WR}` dose witness is untouched by anything below, so 1C is a g-formula _decomposition under stated (cross-world) assumptions_, not an identified natural effect.
- **`N` is heavily floored** — 6 items, ~57–72% at the floor — so `N`-as-outcome is low-powered and `N`-as-early-exposure is near-degenerate. `B` (blending) has a 3-choice guessing floor.
- **No item-level data.** `nonword` is a single 0–6 total and `yarclet` a single 0–32 total; there is **no** per-item nonword scoring or per-grapheme letter-knowledge. The gold-standard proof — do children decode nonwords built from the specific letters _they individually know_ better than nonwords with unknown letters (a within-child, within-item contrast that structurally beats `GA` confounding)? — is **not available** with this dataset. Flag it as the design that _would_ settle the question if item data were ever recovered/collected.
- **The deliverable is triangulation, not one model** (§7). No single fit proves "used for decoding". A _coherent pattern_ across 1A + 1B + 1C + the existing `mech-072/086/087` does; the mirror pattern is honest evidence _against_ the decoding account.

## 3. What already exists (so this is non-duplicative)

`mech-058` `LS→WR`; `mech-072/172` `LS×B→N` (phonics-route interaction, controlled-direct given `B` is a DAG-descendant of `LS`); `mech-090/091` phonological-memory / blending → `WR`; `med-059` arm→`LS`→`WR`; `med-074` arm→`N`→`WR` (floor-limited); `med-086` arm→`LS`→`N`; `med-087` arm→`LS`→`B`; `med-075` sequential arm→`LS`→`B`→`WR`; `med-079` negative-control _mediator_ (grammar). The three designs below are deliberately the gaps: a **matched `LS→N` vs `LS→W` contrast**, a **negative-control _outcome_ panel**, and the **decoding-carrier sequential `LS→N→W`**.

## 4. Design 1A — convergent–discriminant contrast: `LS → N` vs `LS → W`

**Estimand.** On a common scale (standardised exposure, logit-per-SD slope), Δ = β(`LS→N`) − β(`LS→W`). Letter sounds should feed the _pure-decoding_ channel (`N`) at least as strongly as the _mixed_ word channel (`W`, which can be done by sight). Δ ≥ 0 is the decoding-use signature; β(`LS→W`) ≫ β(`LS→N`) says the letter–reading link rides on sight-word / vocabulary routes.

**Why the contrast is informative despite `GA`.** It does _not_ "remove" `GA` confounding — nothing here can. But a pure-`GA`-confounding account (no real decoding mechanism) gives no reason for `LS` to predict `N` _more_ than `W`; if anything `GA` predicts the broader `W` skill at least as much as `N`. A genuine decoding mechanism is the natural explanation for β(`LS→N`) ≥ β(`LS→W`). This is a Campbell–Fiske (1959) convergent/discriminant logic, not an identification claim.

**Build (cheap, reuses everything).** Two matched mechanism models + a cross-model slope contrast:

- `mech-058` (`LS→W`) — **exists**. Adjustment `{G, A, W_pre}` + `adjust_for {hs, hs_missing, attend, deapp_c, deapp_c_missing}`.
- `lrp-rli-mech-096` (`LS→N`, **new**) — `mech-072` _without_ the `B` moderator (a clean total-association `LS→N`, not the controlled-direct one). Matched adjustment `{G, A, N_pre}` + the identical `adjust_for` set; `linear_mechanism=True`, graded Beta-Binomial `N` (the `mech-072` precedent — the floor rule is for _treatment marginals_, not exposure-association slopes), `outcomes=("L","N")`.
- **Contrast:** add to `scripts/compare_statistical_models.py` a standardised-slope Δ = β(`LS→N`) − β(`LS→W`) with the posterior difference (both slopes are logit-per-SD-exposure, hence commensurate across the two item scales), reported mean + 95% CI + P(Δ > 0).

**Optional upgrade (higher value, real new machinery — defer unless the cheap version is promising):** a genuine bivariate model stacking `N` and `W` for each child with a **shared** child random intercept and per-outcome `LS` slopes, so the same stable-trait latent (the best available `GA` proxy) is partialled from both slopes and Δ is a within-model deterministic. New `kind`/factory/pipeline/partial — scoped only if 1A's cheap form warrants it.

## 5. Design 1B — negative-control _outcome_ panel for the `LS` slope

**The cleanest falsification of "it's just `GA`/teaching".** `LS`'s only DAG descendants are `{NW, PA, PS, WR}` (verified against `dag/dag-language-reading.dagitty`, §6). So receptive vocabulary (`R`), expressive vocabulary (`E`), receptive grammar (`T`) and language fundamentals (`F`) are **structurally valid negative-control outcomes**: there is _no_ causal path from `LS` to any of them, and — crucially — their backdoors from `LS` run through the _same_ parent set `{A, GA, HS, IG, IS, SP}` as the reading outcomes. This is exactly the Lipsitch, Tchetgen Tchetgen & Cohen (2010) negative-control-outcome condition: an outcome "known not to be an effect of the primary exposure", "subject to the same source of confounding".

**Prediction.** With the _matched_ adjustment set `{G, A, HS, IS, SP}` (+ own baseline), the `LS` slope should be ≈ 0 for `{R, E, T, F}` and > 0 for the written-code outcomes `{N, W}` (and `P` spelling). If `LS` _also_ predicts the oral-language outcomes, it is behaving like a `GA` / teaching-dose marker and the reading association is suspect — the direct answer to "associated through other mechanisms".

**Build.** A panel of mechanism models, exposure `LS`, matched adjustment (identical conditioning to `mech-058`; only the outcome + own-baseline change) so the panel is a clean like-for-like:

- Positive controls (axis anchors): `LS→W` = `mech-058` (exists); `LS→N` = `mech-096` (from 1A). Optional `LS→P` (spelling; writing uses the code) — off-floor, so a flagged extra, not core.
- Negative controls (**new**): `lrp-rli-mech-097` `LS→R` (rowpvt); `lrp-rli-mech-098` `LS→E` (eowpvt); `lrp-rli-mech-099` `LS→T` (TROG receptive grammar); optional `lrp-rli-mech-100` `LS→F` (CELF basic concepts).

Each: `mechanism_symbol="L"`, `outcome_symbol` = the control, `adjustment=["G","A","<outcome>_pre"]`, `extra={"adjust_baseline_symbol": "<outcome>", "outcomes": ("<outcome>","L"), "adjust_for": ("hs","hs_missing","attend","deapp_c","deapp_c_missing"), "use_age_gp": False, "phase_specific_mechanism": False, "use_subject_random_intercept": True}`.

**Panel output:** a forest of the standardised `LS` slope across `{N, W, P?}` vs `{R, E, T, F}`, in `compare_statistical_models.py` and the findings note. The specificity is the picture, not any single fit.

## 6. Design 1C — decoding-carrier sequential mediation: arm → `LS` → `N` → `WR`

The mediation family's exposure is **always the randomised arm** (`IG`), which is what makes this the well-identified version of "does the letter-sound gain get used for decoding". `med-075` already fits the sequential arm→`LS`→`B`→`WR` (blending as the middle skill). 1C mirrors it with the actual **decoding behaviour** `N` as the carrier — and `N` is the DAG's own designated _"code route (mediator)"_ node (`LS`/`PA` → `NW` → `WR`).

**Build.** `lrp-rli-med-081` (**new**), `kind="mediation_multi"`, mirroring `med-075` with the second mediator swapped `B → N` (and its baseline `B_t1 → N_t1`):

```python
outcome_symbol="W", mechanism_symbol=None,
adjustment=["G","A","E","R","W_pre","L_t1","N_t1",
            "hs","hs_missing","deapp_c","deapp_c_missing","erbto","erbto_missing"],
extra={"mediators": ("L","N"), "order": ("L","N"), "chain": True,
       "outcomes": ("W","N","L","E","R")}
```

`N` enters as a graded Beta-Binomial mediator leg (the `med-074` precedent — off-floor is for terminal `N` outcomes, e.g. `med-086`). **Read-outs:** the joint indirect through the ordered `{LS, N}` block (`NIE_joint` — the total code-route mediation); the `LS → N` coupling `aN_L` (a clearly-positive value is the direct evidence that decoding is downstream of letters _and_ carries the reading effect); per-path splits are exploratory/convention-dependent (as in `med-075`).

**Caveats specific to 1C:** `N` middle mediator is floor-limited (a near-immobile `N` → `NIE ≈ 0` with wide CI is _floor-limited_, not "no route"); direction `NW → WR` is DS-contested (Roch & Jarrold 2012); ID-2 throughout.

## 7. New models + the triangulation truth-table

| ID                 | kind            | exposure → (mediator) → outcome | role                           | status  |
| ------------------ | --------------- | ------------------------------- | ------------------------------ | ------- |
| `lrp-rli-mech-096` | mechanism       | `LS → N`                        | 1A positive / decoding channel | **new** |
| `lrp-rli-mech-097` | mechanism       | `LS → R`                        | 1B negative control            | **new** |
| `lrp-rli-mech-098` | mechanism       | `LS → E`                        | 1B negative control            | **new** |
| `lrp-rli-mech-099` | mechanism       | `LS → T`                        | 1B negative control            | **new** |
| `lrp-rli-mech-100` | mechanism       | `LS → F`                        | 1B negative control (optional) | **new** |
| `lrp-rli-med-081`  | mediation_multi | arm → `LS → N` → `WR`           | 1C decoding carrier            | **new** |
| `lrp-rli-mech-058` | mechanism       | `LS → W`                        | 1A/1B positive anchor          | exists  |

(Confirm the free numbers against `definitions.MODEL_REGISTRY` at build time — `mech` 96–100 and `mediation` 81 are free as of this note.)

**What the pattern means (the deliverable):**

- **Decoding-use supported** if: β(`LS→N`) ≥ β(`LS→W`) [1A]; `LS` ≈ 0 on `{R,E,T,F}` but > 0 on `{N,W}` [1B]; `aN_L` > 0 and `NIE_joint` carries the arm's `WR` gain [1C]; consistent with existing `mech-072` (`L×B` synergy on `N`) and `med-086` (`LS→N`).
- **Decoding-use _not_ supported** if: `LS→W` ≫ `LS→N`; `LS` predicts oral-language controls too; `aN_L`/`NIE_joint` floor-limited to ≈ 0. That is honest evidence the letter–reading link travels by sight-word / vocabulary / `GA` routes.

## 8. Adjustment-set derivations (against `dag/dag-language-reading.dagitty`, revised 2026-07-10)

`LS` parents (arrows in): `{A, GA, HS, IG, IS, SP}`. `LS` descendants: `{NW, PA, PS, WR}` (via `LS→{NW,PA,PS,WR}`, `PA→{NW,WR,PS}`, `NW→WR`). Therefore `{RV, EV, RG, LF}` are non-descendants → valid negative controls.

- **1A/1B mechanism (`LS→Y`):** backdoors run `LS ← {A,GA,HS,IG,IS,SP} → … → Y`. Blocking `{IG(=G), A, HS, IS, SP}` closes every _blockable_ backdoor for **all** the outcomes considered (`W, N, R, E, T, F`) — because each of `W/N/R/E/T/F` is reached from those parents — leaving only `GA` (latent). `attend`=IS, `deapp_c`=SP, `hs`=HS. Own baseline is a pre-treatment precision term (no collider). No member of the set is a collider on an `LS–Y` path. Matched across the panel by construction.
- **1C sequential mediation (arm→`LS`→`N`→`WR`):** exogenous mediator–outcome confounders of the code route are `HS` (`hs`), `SP` (`deapp_c`), `RW` (`erbto`, phonological memory: `RW→NW`); `IS` is a treatment-induced witness and is **not** conditioned (inadmissible for natural effects). Baselines `W_pre, L_t1, N_t1`; `E,R` kept as admissible pre-treatment precision (a parallel whole-word route). Identical structure to `med-075`.

## 9. Build sequence

1. **DAG:** none needed — all edges used (`LS→NW`, `LS→WR`, `LS`↛`{RV,EV,RG,LF}`) are already in the committed contemporaneous DAG. (Contrast with the reverse-mediation spec, which needed a new lagged edge.)
2. **CI adjustment-set guard:** if a contemporaneous-DAG adjustment test exists analogous to `tests/test_lagged_dag_adjustment_sets.py`, add the six new (model → set) rows; otherwise note that the mechanism/mediation sets are asserted only in docstrings and add a guard. **Check before building.**
3. **Models:** write the six modules (`SPEC` + `fit()`); they auto-register via `discover_models()`. Add `_d(...)` rows to `definitions.py`. Add thin `docs/models/{id}/index.qmd` mirroring `mech-058` (1A/1B) and `med-075` (1C); they fall back to `_results_mechanism` / `_results_mediation`.
4. **Fit:** `dev` smoke → `rep-lite` → read convergence gate _before_ interpreting.
5. **Contrast + panel:** extend `compare_statistical_models.py` (1A Δ slope; 1B negative-control forest).
6. **Write-up:** extend `notes/202607161800-findings-mediation.md` (or a new decoding-specificity findings note) with the triangulation table; **lint gate** (`ruff check src/`, `npm run format:check`, `npm run spellcheck`) before commit.

## 10. Repo citation-error flag (separate fix, not part of this build)

While verifying references I found a **wrong DOI in committed content**: `notes/202607172100-reverse-mediation-wr-ls-direction-spec.md` cites Roch & Jarrold (2012) as `doi:10.1002/dys.1433`, but that DOI is Nalavany & Carawan (2011), an unrelated adult-dyslexia/self-esteem paper in _Dyslexia_. The intended source is Roch & Jarrold (2012), _Journal of Communication Disorders_ 45(2):121–128, **doi:10.1016/j.jcomdis.2011.11.001** (PMID 22176835). The `lrp-rli-med-074` docstring cites the same paper by name (no DOI) and is fine. Recommend correcting the reverse-mediation note's DOI in a small separate `docs(notes)` fix.

## 11. References (verified 2026-07-17)

- Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. _Psychological Bulletin_, 56(2), 81–105. [doi:10.1037/h0046016](https://doi.org/10.1037/h0046016)
- Cupples, L., & Iacono, T. (2000). Phonological awareness and oral reading skill in children with Down syndrome. _Journal of Speech, Language, and Hearing Research_, 43(3), 595–608. [doi:10.1044/jslhr.4303.595](https://doi.org/10.1044/jslhr.4303.595)
- Gough, P. B., & Tunmer, W. E. (1986). Decoding, reading, and reading disability. _Remedial and Special Education_, 7(1), 6–10. [doi:10.1177/074193258600700104](https://doi.org/10.1177/074193258600700104)
- Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010). Negative controls: a tool for detecting confounding and bias in observational studies. _Epidemiology_, 21(3), 383–388. [doi:10.1097/EDE.0b013e3181d61eeb](https://doi.org/10.1097/EDE.0b013e3181d61eeb) (PMID 20335814)
- Roch, M., & Jarrold, C. (2012). A follow-up study on word and non-word reading skills in Down syndrome. _Journal of Communication Disorders_, 45(2), 121–128. [doi:10.1016/j.jcomdis.2011.11.001](https://doi.org/10.1016/j.jcomdis.2011.11.001) (PMID 22176835)
