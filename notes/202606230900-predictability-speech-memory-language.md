# Are the speech / verbal-memory / language-sample measures predictable, and what predicts them?

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!IMPORTANT]
> **Reframed (post locked-DAG #115 + ranking shift #116/#118).** The DAG is now
> **locked** and already instantiates these as measurement nodes — `RW` (word + nonword
> repetition, `erbto`) phonological memory and `SP` (`deapp_c`) articulation — so the
> live question is no longer "should we add nodes?" but **"do these behave as the locked
> DAG's `RW`/`SP` nodes predict, and should `RW`/`SP` be split into sub-scores?"** (an
> open-decisions item). The retired hard-selection machinery
> (`uniform_feature_selection.py` + the per-model `_noconstruct` variants) has been
> **removed**; sibling-robustness is now read from `rank_predictors.py`'s
> `ranking_excluding_same_skill.csv` over the full predictor set (the ranking run is
> pending). The original prospective-DAG prose below is retained for provenance.

**Date:** 2026-06-23
**Models:** LRP25–LRP42 (ML / gradient-boosting discovery)
**Status:** exploratory discovery only — evidence for the (now-locked) DAG's open
`RW`/`SP` sub-score-split decision, **not** a DAG edit and **not** a Bayesian/causal
model. See the reframing note above.

## Why this exists

We have modelled 11 outcomes as gradient-boosting discovery targets (LRP01–22).
Among the _spoken_ measures only **DEAP final-consonant accuracy** (`deappfi`,
LRP21/22) was ever an outcome — and it was the cautionary case: predictable only
via its own DEAP sibling, collapsing to ≈ −0.03 once the sibling was removed. The
**verbal-memory (ERB)**, the **other DEAP sub-scores**, and the
**language-sample (LSAM)** measures had only ever been _predictors_, never
outcomes. So "how predictable are they, and where do they sit relative to the
code-based-reading and oral-language clusters?" was unanswered — and it bears on
the _measurement_ side of the shared DAG (v5): are these extra indicators of the
existing general-ability / language structure, self-contained instrument
artefacts, or candidate new constructs (e.g. a phonological-memory node)?

This note gathers the evidence. **The DAG decision is a separate step.**

## What was fit

Twelve **level** models (one per measure) and six **gain** models (ERB and DEAP
sub-scores that have a precomputed `*_gain` column). All on the same machinery as
LRP01–22: `LGBMPipeline`, **GroupKFold by `subject_id`**, the uniform
feature-selection rule (distance-correlation redundancy filter at dcor ≥ 0.70 +
importance noise-floor cut ≤ 0.005; gain baseline force-kept), re-tuned on the
reduced set (Optuna 150-trial MAE, 10-fold, seed 47). Predictor pool = the
standard `Predictors.DEFAULT_LEVEL` / `DEFAULT_GAIN` groups. The FS step is the
committed `scripts/uniform_feature_selection.py` (a reproducible reconstruction of
the method documented in `notes/202606211200-uniform-gb-fs.md`).

### Model id → outcome

| id    | outcome        | kind  | instrument             | n    |
| ----- | -------------- | ----- | ---------------------- | ---- |
| LRP25 | `erbnw_gain`   | gain  | ERB (verbal memory)    | ~147 |
| LRP26 | `erbnw`        | level | ERB                    | ~202 |
| LRP27 | `erbword_gain` | gain  | ERB                    | ~148 |
| LRP28 | `erbword`      | level | ERB                    | ~203 |
| LRP29 | `erbto_gain`   | gain  | ERB (total)            | ~147 |
| LRP30 | `erbto`        | level | ERB (total)            | ~202 |
| LRP31 | `deappin_gain` | gain  | DEAP (articulation)    | ~152 |
| LRP32 | `deappin`      | level | DEAP                   | ~207 |
| LRP33 | `deappvo_gain` | gain  | DEAP                   | ~152 |
| LRP34 | `deappvo`      | level | DEAP                   | ~207 |
| LRP35 | `deappav_gain` | gain  | DEAP (average)         | ~152 |
| LRP36 | `deappav`      | level | DEAP (average)         | ~207 |
| LRP37 | `deapp_c`      | level | DEAP (composite)       | ~207 |
| LRP38 | `lsammlu`      | level | LSAM (language sample) | ~106 |
| LRP39 | `lsammax`      | level | LSAM                   | ~106 |
| LRP40 | `lsamint`      | level | LSAM                   | ~106 |
| LRP41 | `lsamun`       | level | LSAM                   | ~106 |
| LRP42 | `lsamto`       | level | LSAM                   | ~106 |

`deappfi` (LRP21/22) is the pre-existing comparison point and is **not**
re-fit here. The composite **`deapp_c` has no precomputed `*_gain` column** in the
prepared data (unlike `deappav`), so only its level is fit — synthesising a
composite gain would mean editing the shared data pipeline, which is out of scope.
**LSAM is t1–t2 only**, so only level models are fit (a single t1→t2 transition,
n ≈ 53, is too thin for a gain model); these levels are themselves exploratory
(two waves, ≈106 rows).

## Results

Pooled **out-of-fold R²** is the honest metric (51-fold GroupKFold by `subject_id`,
tuned params, scored against each fold's training mean — identical to the
pipeline's `cv_pooled_r2`). "−sib" is the pooled OOF R² after dropping the
same-instrument siblings from the predictor set (the `_noconstruct` check);
"n/a" where no same-instrument sibling is in the default pool. `deappfi`
(LRP22, the pre-existing comparison) is shown for reference.

| model   | outcome         | kind    |     n |   OOF R² | in-sample R² | −sib OOF R² | top predictors (SHAP rank; +/− direction) |
| ------- | --------------- | ------- | ----: | -------: | -----------: | ----------: | ----------------------------------------- |
| LRP26   | `erbnw`         | level   |   202 | **0.76** |         0.84 |    **0.47** | erbword+, aptinfo+, yarcsi+, deappvo+     |
| LRP28   | `erbword`       | level   |   203 | **0.77** |         0.92 |    **0.62** | erbnw+, deappin+, ewrswr+, nonword+       |
| LRP30   | `erbto` (total) | level   |   202 | **0.91** |         0.95 |    **0.48** | erbword+, aptgram+, yarcsi+, deappvo+     |
| LRP32   | `deappin`       | level   |   207 |     0.62 |         0.80 |        0.40 | deappfi+, erbnw+, deappvo+, aptgram+      |
| LRP34   | `deappvo`       | level   |   207 |     0.34 |         0.82 |    **0.04** | deappin+, yarclet−, aptinfo+              |
| LRP36   | `deappav` (avg) | level   |   207 |     0.89 |         0.96 |        0.38 | deappfi+, deappvo+, erbnw+, aptgram+      |
| LRP37   | `deapp_c` (sum) | level   |   207 |     0.93 |         0.98 |        0.30 | deappfi+, deappvo+, erbnw+, eowpvt+       |
| _LRP22_ | _`deappfi`_     | _level_ | _207_ |   _0.55_ |          _–_ |     _−0.03_ | _deappin (collapses without it)_          |
| LRP38   | `lsammlu`       | level   |   106 |     0.52 |         0.71 |         n/a | deappin+, aptinfo+, erbnw+, yarcsi+       |
| LRP39   | `lsammax`       | level   |   106 |     0.26 |         0.48 |         n/a | deappin+, erbnw+, eowpvt+                 |
| LRP40   | `lsamint`       | level   |   106 |     0.48 |         0.76 |         n/a | deappin+, area+, aptinfo+                 |
| LRP41   | `lsamun`        | level   |   106 |     0.58 |         0.81 |         n/a | deappin+, aptinfo+, erbnw+, rowpvt+       |
| LRP42   | `lsamto`        | level   |   106 |     0.38 |         0.76 |         n/a | deappin+, aptinfo+, trog+, age+           |
| LRP25   | `erbnw_gain`    | gain    |   147 |     0.12 |         0.42 |           – | erbnw− (baseline/RTM)                     |
| LRP27   | `erbword_gain`  | gain    |   148 |     0.22 |         0.40 |           – | erbword− (baseline/RTM)                   |
| LRP29   | `erbto_gain`    | gain    |   147 |     0.17 |         0.41 |        0.11 | erbnw−, erbto− (RTM)                      |
| LRP31   | `deappin_gain`  | gain    |   152 |     0.08 |         0.28 |           – | deappin− (baseline/RTM)                   |
| LRP33   | `deappvo_gain`  | gain    |   152 |     0.31 |         0.45 |        0.23 | deappvo−, deappfi+                        |
| LRP35   | `deappav_gain`  | gain    |   152 |    −0.04 |         0.37 |        0.15 | deappvo−, time+ (noise)                   |

Top predictors are by mean |SHAP|; direction is the sign of the SHAP–feature
relationship. Permutation-importance rankings (from the full reporting fit)
agree on the leading one or two predictors. Full per-model artifacts (SHAP
beeswarms, distance-correlation heatmaps, partial-dependence) render from the
`docs/models/lrpNN/` report templates via `fit_model.py <id> --config reporting
--render`; they are gitignored and orthogonal to the numbers above.

## The three questions

### Q1 — Predictability

**Levels are predictable; gains are not.** Every level model clears the
"predictable like the other levels" bar (LRP01–22 levels run ≈ 0.31–0.80). The
DEAP composites are highest — `deapp_c` 0.93, `erbto` 0.91, `deappav` 0.89 — but
that is mechanical (they are sums/averages of components that sit in the
predictor pool; see Q3). Among non-composite measures: verbal memory
`erbnw` 0.76 and `erbword` 0.77 are strongly predictable; articulation
`deappin` 0.62 moderate, `deappvo` 0.34 weak; the language-sample measures
0.26–0.58 (best for lexical-diversity `lsamun` 0.58 and `lsammlu` 0.52, weakest
for `lsammax` 0.26). **All six gain models are near-noise** (−0.04 to 0.31,
baseline/RTM-driven), exactly as every gain model in the suite — there is no
usable signal in the gains and they are not interpreted further.

### Q2 — Cluster membership (the DAG-relevant one)

The two reference clusters are the **code-based reading** cluster
(`Categories.READING`: word reading `ewrswr`, letter sounds `yarclet`, nonword
reading, blending, spelling) and the **oral-language** cluster
(`Categories.LANGUAGE`: expressive/receptive vocabulary `eowpvt`/`rowpvt`,
grammar `aptgram`/`trog`, concepts `celf`). For each new measure: do its top
non-same-instrument predictors come from the reading cluster, the language
cluster, both (i.e. general ability `g`), or does it stand alone?

**None of the new measures stands alone, and none loads on a single cluster —
they all draw on the broad skill set (consistent with a general-ability `g`).**
The recurring cross-instrument predictors are expressive articulation
(`deappin` — it tops _every_ language-sample model), oral language
(`aptinfo`, `aptgram`, `trog`, `celf`, `eowpvt`, `rowpvt`), code-based reading
(`ewrswr`, `nonword`, `yarcsi`, `yarclet`), and verbal memory (`erbnw`).
Concretely:

- **ERB (verbal memory)** is predicted by oral language (`aptinfo`, `aptgram`)
  **and** code-based reading (`yarcsi`, `ewrswr`, `nonword`) **and** articulation
  (`deappin`, `deappvo`) — it sits squarely with `g`, not in one cluster.
- **DEAP (articulation)** is predicted by its own siblings plus verbal memory
  (`erbnw`) and expressive language (`aptgram`) — a narrower, speech-weighted
  loading.
- **LSAM (spontaneous language)** is predicted by articulation (`deappin`) +
  oral language (`aptinfo`, `trog`, `rowpvt`, `eowpvt`) + verbal memory
  (`erbnw`) — the oral-language side of `g`.

No measure is carried by the code-based-reading cluster _alone_, and none is
isolated from it. On the membership question these are indicators of the
existing general-ability / language structure, not a separate construct cluster.

### Q3 — Same-instrument contamination

The same-instrument sibling groups (`Variables.CONSTRUCTS`) are
**`phonological_memory`** = {`erbword`, `erbnw`, `erbto`},
**`articulation`** = {`deappin`, `deappvo`, `deappfi`, `deappav`, `deapp_c`}, and
**`speech_sampling`** = {`lsammlu`, `lsammax`, `lsamint`, `lsamun`, `lsamto`}.

Note one structural fact up front: the five **LSAM** measures are recorded at
t1–t2 only and are therefore in `DEFAULT_EXCLUDED` — they are **absent from the
default predictor pool**. So LSAM measures _cannot_ be carried by same-instrument
siblings; their predictability is necessarily cross-instrument, and no
`_noconstruct` check is needed for them. The ERB and DEAP siblings _are_ in the
pool, so where one tops a model we refit with the same-construct siblings dropped
(`<id>_noconstruct`), exactly as `deappfi`/LRP22 did.

The split is the crux of the DAG question:

- **ERB (verbal memory) is NOT measurement-bound.** Dropping the same-instrument
  ERB sibling leaves substantial out-of-sample skill: `erbnw` 0.76 → **0.47**,
  `erbword` 0.77 → **0.62**, `erbto` 0.91 → **0.48**. The residual is carried by
  language, reading and articulation — i.e. real cross-domain signal, the
  _opposite_ of the `deappfi` collapse.
- **DEAP (articulation) is largely measurement-bound.** `deappvo` collapses
  0.34 → **0.04**, mirroring `deappfi` (0.55 → −0.03): with the same-instrument
  DEAP siblings removed there is essentially no non-articulation predictor. The
  composites and `deappin` retain only a modest residual (`deappin` 0.40,
  `deappav` 0.38, `deapp_c` 0.30) — a weak shared component with verbal memory
  and expressive language, not a strong cross-domain construct.
- **LSAM (spontaneous language):** no same-instrument check is possible (siblings
  excluded from the pool), but by construction its whole R² (0.26–0.58) is
  cross-domain — it is _only_ ever predicted by other instruments.

## Implications for the shared DAG (v5)

For each construct, one of: _already covered_ (predicted mainly by the existing
skill cluster ⇒ another indicator of `g`/language; no new node), _distinct but
measurement-bound_ (predictable only by its own instrument siblings, the
`deappfi` pattern ⇒ a self-contained instrument construct; note it, no causal
node), or _candidate new node_ (coherent cross-domain prediction in **both**
directions ⇒ flag for the DAG review).

**1. Verbal / phonological memory (ERB) — _candidate new node_ (the one to take
forward).** It is strongly predictable (0.76–0.91) **and** keeps 0.47–0.62 once
its own siblings are removed, predicted by language + reading + articulation;
and ERB has itself been a recurring _predictor_ across LRP01–22. That two-way
association with the skill cluster, surviving the same-instrument check, is what
a real construct (phonological short-term memory) looks like — not an instrument
artefact. Flag for the DAG review as a candidate **phonological-memory node**
feeding `g`/skills (or, minimally, an additional reflective indicator of `g`).

**2. Speech-sound accuracy (DEAP) — _distinct but largely measurement-bound_.**
`deappvo` and `deappfi` collapse to ≈ 0 without their DEAP siblings (the
self-contained-instrument pattern); composites and `deappin` retain only a weak
residual (0.30–0.40) shared with verbal memory and expressive language. This is
a self-contained articulation instrument with at most a weak `g`-loading — **not
a strong new-node candidate.** Do not add a causal node; if represented at all,
use a single articulation indicator, not the several correlated sub-scores.

**3. Spontaneous language (LSAM) — _already covered_.** Its predictability is
entirely cross-instrument, led by the oral-language measures plus articulation
and verbal memory — i.e. another indicator of the **existing language / `g`
node**. No new node is warranted; it could optionally be added as a further
language indicator. The data are thin (≈106 rows, t1–t2), so hold this lightly.

## Recommendation

- **Add at most one node: phonological memory (ERB).** It is the only measure
  with a coherent, cross-domain, sibling-robust signal — the case for a
  `phonological-memory` construct (feeding `g`, or as a `g` indicator) is worth
  putting to the DAG review with Frank. **Confirm ERB scoring first** (it is not
  in Burgoyne 2012).
- **Do not add nodes for articulation (DEAP) or spontaneous language (LSAM).**
  DEAP is measurement-bound (keep its sub-scores as predictors, not as a causal
  construct); LSAM is already an indicator of the existing language node and is
  thin.
- **Ignore the gains** — near-noise everywhere, no DAG bearing.
- Net: this round supports **one** candidate measurement-side change (verbal
  memory), pending instrument confirmation; the speech and language-sample
  measures need no structural change.

## Guardrails / caveats

- **Discovery only.** These are cross-validated _associations_, not causal
  effects and not intention-to-treat. Direction (SHAP sign) and magnitude
  (permutation importance) describe prediction, not mechanism. None of these
  measures is shown to _cause_ reading.
- **Power.** ERB/DEAP levels (≈200 rows, 3–4 waves) are reasonable; LSAM (≈106,
  two waves) is thin — an exploratory hint at best. Gain models are near-noise
  across the whole suite (regression-to-the-mean dominates); the value here is
  the **level** predictability and the **cluster membership**, not the gains.
- **Pooled out-of-fold R²** (against each fold's training mean) is the honest
  metric reported, _not_ the per-fold mean R². In-sample R² is shown only to
  expose the overfit gap.
- **Measures not in Burgoyne (2012).** ERB, DEAP and LSAM scoring and maxima are
  unconfirmed (open data-dictionary items). Gradient boosting uses raw scores so
  _fitting_ is unaffected, but _interpretation_ of any candidate new construct
  should be held lightly until the measures are confirmed.
- **FS provenance.** The full-set importance ranking that drives feature
  selection uses one fixed LightGBM config (documented in
  `scripts/uniform_feature_selection.py`), so the exact reduced sets are not
  bit-identical to the lost #102 `uniform_fs.py`; the _algorithm_ is the same and
  every reduced set has 0 residual dcor ≥ 0.70 pairs.
