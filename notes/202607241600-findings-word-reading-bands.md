<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Findings — what word-reading standing goes with: band comparisons, and what a lagged model could add

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Four passes: word-reading halves and quartiles compared across every measure at every wave (model-free, descriptive); an identification analysis of whether a lagged model could test word reading feeding back into letter sounds, blending or nonword reading; and the forward question of whether blending predicts reading. The pass-1 and pass-2 numbers are **raw band contrasts** — not adjusted for anything, so age and latent general ability sit inside every one of them. Passes 3 and 4 report **adjusted associations** from already-fitted models plus machine-checked d-separation results. **Nothing here is causal.** Preliminary — n ≈ 51–53 children, exploratory study.
>
> Supersession warning and 2026-08-08 issue-decision updates added by a LLM-based AI tool (Codex/GPT-5).

> [!WARNING]
> **Superseded in part on 2026-08-02.** The model-free band descriptions remain a dated exploratory snapshot, but any `mech-190` curve, shape or knee statement below is withheld under `notes/202608021625-divergence-qualification-policy.md`; the current reporting fit has divergences and nonlinear shape quantities remain zero-divergence-only.

## Why this note exists

The companion note ([letter sounds and word reading](202607241000-findings-letter-sounds-word-reading-association.md)) asked what predicts word reading. This one turns the question around: **what does word-reading standing itself go with, and what might it feed?** Four passes — the first three in the order requested, the fourth added afterwards:

1. Split the cohort at the within-wave **median** word-reading score at each of t1–t4, and compare the halves on every measure.
2. The same with **quartiles**, plus whether each measure's profile is a gradient or a threshold.
3. What **lagged models** could test whether reading more words predicts later gains in **letter-sound knowledge**, **phoneme awareness (blending)** or **nonword reading**.
4. The companion question in the forward direction, added afterwards: **does phoneme awareness (blending) predict reading outcomes?**

Reproducible assets: [`202607241600-word-reading-band-comparisons.py`](assets/202607241600-word-reading-band-comparisons.py) (passes 1–2 and the prospective extension) and [`202607241600-wr-to-code-lagged-dsep.py`](assets/202607241600-wr-to-code-lagged-dsep.py) (pass 3). CSVs land in `output/notes/202607241600-wr-bands/`. Pass 4 adds no new fits of its own: it reads already-fitted registered models plus the letter-sound probe from the companion note ([`202607241000-ls-wr-association-probe.py`](assets/202607241000-ls-wr-association-probe.py)).

## How to read the numbers

Effect size is **Cliff's delta** with a bootstrap **89 %** interval (house standard, [credible-interval standard](202607172359-credible-interval-standard.md)), not Cohen's _d_. Most of these measures are bounded item counts with heavy floors, where a mean difference over a pooled SD is not interpretable. Cliff's delta is the probability a randomly drawn top-band child outscores a randomly drawn bottom-band child, minus the reverse: **0 = no separation, 1 = complete separation**, and 0.5 means the top-band child wins about three times in four. It handles ties explicitly, which matters more here than anywhere else in the suite.

For the model-based results in passes 3 and 4 the evidence ladder is the suite's: inconclusive / suggestive / moderate / strong / very strong at P ≥ 0.75 / 0.91 / 0.97 / 0.99 ([evidence strength](202606261304-evidence-strength-and-rope-reporting.md)).

## The floor is the first finding, not a caveat

Word reading (`ewrswr`, 79 items) is the splitting variable, and it is heavily floored early:

| Wave | n   | at zero  | median | quartile cuts | children tied _on_ a quartile cut |
| ---- | --- | -------- | ------ | ------------- | --------------------------------- |
| t1   | 53  | **40 %** | 2      | 0 / 2 / 5     | **30 (57 %)**                     |
| t2   | 53  | 15 %     | 5      | 2 / 5 / 13    | 10 (19 %)                         |
| t3   | 53  | 9 %      | 9      | 3 / 9 / 21    | 13 (25 %)                         |
| t4   | 51  | 8 %      | 10     | 6 / 10 / 25   | 5 (10 %)                          |

**The t1 lower-quartile cut is literally zero.** Q1 at t1 is entirely children who read no words, and the 21 zeros spill into Q2 as well, so the Q1/Q2 boundary at t1 is decided by rank tie-breaking among children with identical scores — 57 % of the cohort sits exactly on a cut. Anything the t1 quartile table appears to say about Q1 versus Q2 is an artefact of the tie-break, and the data confirm it: at t1 the Q2-vs-Q1 step is ≈ 0 or slightly negative for **every** measure (letter sounds −0.09, blending −0.05, receptive vocabulary −0.02, taught expressive words −0.14). The median split at t1 is milder but not clean (5 tied, 9 %).

This is why the quartile pass is reported second and read only from t2 onward, and why the whole analysis is anchored on the median split.

## Pass 1 — the median split, all measures, all waves

Cliff's delta, top 50 % versus bottom 50 %, per wave. Sorted by the t4 value; `*` marks a measure whose 89 % interval excludes zero at all four waves.

| Measure                              | t1    | t2    | t3       | t4       |
| ------------------------------------ | ----- | ----- | -------- | -------- |
| letter sounds `yarclet` \*           | 0.79  | 0.88  | 0.72     | **0.82** |
| phonetic spelling `spphon` \*        | 0.40  | 0.62  | 0.65     | **0.80** |
| receptive grammar `trog`             | 0.50  | 0.25  | 0.38     | **0.76** |
| information/vocab `aptinfo` \*       | 0.45  | 0.45  | 0.57     | 0.72     |
| taught expressive words `b1extau` \* | 0.54  | 0.61  | 0.53     | 0.71     |
| blending `blending` \*               | 0.38  | 0.50  | 0.53     | 0.70     |
| taught receptive words `b1retau` \*  | 0.48  | 0.64  | 0.46     | 0.69     |
| expressive vocab `eowpvt` \*         | 0.62  | 0.48  | **0.76** | 0.69     |
| nonword reading `nonword`            | 0.17  | 0.62  | 0.59     | 0.68     |
| language fundamentals `celf` \*      | 0.35  | 0.44  | 0.46     | 0.67     |
| word repetition `erbword` \*         | 0.39  | 0.38  | 0.46     | 0.66     |
| grammar `aptgram`                    | 0.44  | 0.21  | 0.48     | 0.65     |
| receptive vocab `rowpvt` \*          | 0.54  | 0.32  | 0.56     | 0.61     |
| phonological memory `erbto` \*       | 0.39  | 0.40  | 0.45     | 0.54     |
| speech accuracy `deapp_c`            | 0.30  | 0.33  | 0.24     | 0.32     |
| **non-verbal ability `blocks`**      | 0.40  | 0.24  | 0.36     | **0.33** |
| **cumulative sessions `attend_c`**   | 0.00  | 0.31  | 0.25     | **0.12** |
| **age**                              | 0.19  | 0.25  | 0.07     | **0.03** |
| behaviour `behav`                    | −0.09 | 0.04  | −0.15    | −0.08    |
| SDQ                                  | −0.08 | −0.09 | −0.11    | −0.19    |
| ear infections `earinf`              | −0.22 | 0.13  | −0.15    | −0.18    |

Three things worth stating separately.

**Everything in the language-and-literacy domain separates the halves, and the separation grows.** By t4 fourteen measures sit at 0.6 or above — nine of them outside the reading domain. This is not surprising and it is not evidence of anything mechanistic: latent general ability causes both sides of every one of these comparisons, so a positive delta is the expected result and the _ordering_ is the only informative part.

Three of the fourteen are near-duplicates of the splitting variable and are omitted from the table above rather than counted as findings: `yarcewr` (YARC early word reading, correlated 0.95 with `ewrswr`, delta 1.00 at t4), `spraw` (0.64) and `yarcsi` (0.57). A word-reading split separating children on another word-reading measure is a tautology, not a result.

**Age and non-verbal ability are the exceptions, and that is the interesting part.** Age separates the halves not at all by t4 (0.03) and non-verbal ability only modestly (0.33, the weakest of any cognitive or language measure). Whatever distinguishes the strong from the weak word readers in this cohort at the end of the study, it is not that they are older, and block design barely sees it. That is consistent with the companion note's Q1 finding that adjusting for age, hearing and block design left the letter-sound → word-reading slope essentially untouched.

**Behaviour, SDQ, hearing, ear infections and intervention dose are all flat or slightly negative.** The bands are not a behaviour artefact and not a dose artefact — cumulative sessions reach 0.12 at t4 and plain per-window attendance never leaves the noise. Whatever creates the reading gap, it is not that the strong readers turned up more.

### The floor moves the ranking, so read the ranking against it

Three measures climb steeply through the table across waves — nonword reading 0.17 → 0.68, phonetic spelling 0.40 → 0.80, single-word reading `spraw` 0.08 → 0.64. That is mostly the floor receding, not the construct becoming more important. At t1, 72 % of the cohort scored zero on nonword reading and 78 % on phonetic spelling, so both bands are mostly tied at zero and a large delta is arithmetically unavailable. Their **position** in the t1 ranking reflects where the floor sat, and their rise reflects the floor lifting. The same caution does not apply to vocabulary or grammar, which have no floor problem at any wave.

### The bands are durable

Half-split membership agrees across waves 77–88 % of the time, and the underlying continuous score is stable (Spearman 0.76–0.96, rising to 0.96 between t3 and t4). Quartile membership is much less durable — 49 % agreement between t1 and any later wave, reaching 80 % only between t3 and t4 — which is the tie problem again, plus the ordinary fact that four bands of 13 children churn more than two of 26.

## Pass 2 — quartiles

Medians per word-reading quartile at t4 (n = 13, 13, 12, 13):

| Measure                 | Q1 (lowest) | Q2  | Q3    | Q4 (highest) | Shape                    |
| ----------------------- | ----------- | --- | ----- | ------------ | ------------------------ |
| letter sounds (/32)     | 16          | 24  | 28.5  | 30           | rises then flattens      |
| expressive vocab        | 28          | 38  | 44    | 55           | **even gradient**        |
| receptive vocab         | 43          | 48  | 53    | 56           | even gradient            |
| blending (/10)          | 3           | 6   | 8     | 9            | rises then flattens      |
| nonword reading (/6)    | 0           | 1   | 2     | 5            | rises, steepest at top   |
| phonetic spelling (/92) | 0           | 0   | 50.5  | 74           | **threshold at Q2 → Q3** |
| receptive grammar       | 13          | 15  | 21    | 19           | steps then reverses      |
| taught expressive words | 8           | 9   | 14    | 15           | steps at Q2 → Q3         |
| non-verbal ability      | 11          | 15  | 16    | 16           | **step at Q1 → Q2 only** |
| age (months)            | 101         | 98  | 99.5  | 107          | no order                 |
| cumulative sessions     | 161         | 164 | 172.5 | 165          | flat                     |

Splitting each measure's total separation into its three adjacent-quartile steps (in `quartile_monotonicity.csv`) makes the shapes explicit:

- **Expressive vocabulary is the one clean gradient** — steps of 0.41 / 0.42 / 0.37, the most evenly spread of any measure. Every quartile boundary of word reading is also a vocabulary boundary.
- **Non-verbal ability is a floor effect, not a gradient** — steps of 0.42 / 0.10 / −0.07, with 72 % of its separation in the Q1-vs-Q2 step alone. Block design distinguishes the lowest quartile of word readers from everyone else and then stops discriminating entirely. This is a sharper statement of the pass-1 finding, and it is the kind of thing a single top-vs-bottom contrast hides.
- **Phonetic spelling is a genuine threshold** — Q1 and Q2 are both at zero, so its 0.20 / 0.62 / 0.48 step profile is partly the 45 % floor, not a psychological discontinuity. Flagged rather than interpreted.
- **Letter sounds and blending both flatten at the top** — the Q4-vs-Q3 steps (0.40 and 0.04) are the smallest of their three, because Q3 is already at 28.5 of 32 and 8 of 10. Ceiling, not plateau.

At t1 the same table is uninterpretable at its lower boundary for the reason given above, and is reported in the CSVs with the tie diagnostics attached rather than reproduced here.

## Pass 3 — could a lagged model test reading feeding back into the code?

The question asked was whether **learning more words predicts later gains** in letter sounds, blending or nonword reading. Three separate answers are needed: what the descriptive data show, what the DAG says is identifiable, and what has already been fitted.

### First, the descriptive read — and why it cannot settle it

Splitting on word reading at wave _t_ and contrasting each measure's change to _t_+1, residualised on that measure's own level at _t_ (the raw change is regression-to-the-mean plus a ceiling constraint; the residual is the honest version):

| Target          | t1 → t2                  | t2 → t3              | t3 → t4                  | Censoring        |
| --------------- | ------------------------ | -------------------- | ------------------------ | ---------------- |
| letter sounds   | +0.16 [−0.10, +0.41]     | +0.02 [−0.26, +0.27] | −0.02 [−0.27, +0.25]     | **top-censored** |
| blending        | +0.13 [−0.14, +0.38]     | +0.19 [−0.06, +0.45] | **+0.47 [+0.25, +0.68]** | top-censored     |
| nonword reading | **+0.50 [+0.29, +0.71]** | +0.20 [−0.07, +0.46] | +0.24 [−0.05, +0.51]     | bottom-censored  |

**The letter-sound null is censored, not null.** By t3 the top word-reading half sits at a median 28 of 32 letter sounds, with a third at ≥ 30 and 11 % already at the 32 ceiling, against a bottom-half median of 17. The top band physically cannot gain much whatever the truth is, and residualising on the starting level does not repair a hard bound — a linear trend is not a ceiling. So the flat row above is **uninformative about `WR → LS`**, and it is exactly the artefact [LRP-RLI-MED-176](../src/language_reading_predictors/statistical_models/lrp_rli_med_176.py)'s docstring warns of.

**The nonword result is the strongest signal in the note.** Among children scoring zero on nonword reading at t1, **2 of 20 (10 %) in the bottom word-reading half had moved off the floor by t2, against 10 of 16 (62 %) in the top half.** It holds inside both arms (immediate 18 % vs 80 %; wait-list 0 % vs 33 %), and arm is balanced across the bands (14/11 vs 13/12) as randomisation implies — the t1 split is pre-randomisation, so arm cannot confound it. This is the Roch & Jarrold 2012 `WR → NW` pattern ([doi:10.1016/j.jcomdis.2011.11.001](https://doi.org/10.1016/j.jcomdis.2011.11.001)) visible descriptively in this cohort.

**Blending strengthens over time**, reaching +0.47 [+0.25, +0.68] across t3 → t4 _despite_ top-censoring working against it.

### Second, what the DAG permits

Minimal sufficient adjustment sets, searched (not hand-written) against a crossover-aware three-slice unroll of `dag/dag-language-reading-lagged.dagitty`, latent general ability removed throughout because no measured set can block it:

| Coupling             | Direct edge in the DAG?    | Minimal set, transition 1           | Minimal set, transition 2                 | Fittable at n ≈ 54 |
| -------------------- | -------------------------- | ----------------------------------- | ----------------------------------------- | ------------------ |
| `WR → LS`            | **yes** (added 2026-07-17) | **age + hearing + LS + speech (4)** | **age + hearing + arm + LS + speech (5)** | **yes**            |
| `WR → PA` (blending) | yes                        | 8 nodes                             | 11 nodes                                  | no                 |
| `WR → NW`            | **yes** (added 2026-08-08) | 8 nodes                             | 12 nodes                                  | no                 |

Three results follow, and the first is the useful one.

**`WR → LS` has the smallest adjustment set of any reverse coupling in the suite** — four measured nodes on the randomised transition, five once arm has to enter post-crossover. It is smaller than the `WR → TE` set that `lcsm-081` was built around (seven). The reason is structural: in the DAG, letter sounds have exactly one skill parent, speech, so almost nothing needs blocking.

The direction question itself, however, is **already answered** — see the `med-076` / `med-176` contrast in the next section. What no model estimates is the narrower quantity: a **lagged LCSM coupling** `g_W_L` (prior word-reading _level_ → subsequent letter-sound _change_, pooled across all three transitions). `med-176` is a single t2 → t4 mediation window with a different estimand — the share of the _intervention's_ effect on letter sounds that runs through word reading — not a per-transition coupling. The gap is real but much narrower than the cheap adjustment set makes it look; the next section gives the reason, and ["What would be worth building"](#what-would-be-worth-building) draws the conclusion that filling it is optional rather than a priority.

**`WR → PA` needs 8–11 adjusters, confirming the design note's verdict** — now re-derived with letter sounds in the graph, which the original derivation did not have.

**`WR → NW` was added on 2026-08-08 as a provisional working-DAG assumption.** The decision allows a direct visual-analogy or otherwise unrepresented route alongside the mediated `WR_t → {LS, PA}_{t+1} → NW_{t+1}` routes; omitting it would encode the stronger assumption that every reading-to-decoding route is completely represented by the measured six-item nonword task, letter sounds and blending. This is a structural assumption supported by the Roch and Jarrold longitudinal result and the floor-exit pattern above, not a causal result from this cohort. Crucially, adding an outgoing edge from `WR` cannot improve its backdoor identification: outgoing exposure edges are removed when the adjustment graph is formed, so the machine-derived 8-node and 12-node measured sets are unchanged and remain unfittable at n ≈ 54.

### Third, what is already fitted

**All three questions have live, gate-passing answers that predate this note.**

**`med-076` / `med-176`** are the forward and reverse longitudinal mediations, and together they are the suite's existing answer to the direction question. Both passed their gates (0 divergences; R-hat ≤ 1.001; min ESS ≥ 14 600). The natural-indirect-effect through the intermediate measure at t2 on the outcome at t4:

| Direction                     | NIE (probability scale) | Natural units       | P(>0) | Evidence        |
| ----------------------------- | ----------------------- | ------------------- | ----- | --------------- |
| forward `LS → WR` (`med-076`) | +0.037 [+0.014, +0.067] | +2.9 words of 79    | 0.997 | **very strong** |
| reverse `WR → LS` (`med-176`) | +0.014 [+0.001, +0.035] | +0.45 letters of 32 | 0.963 | **moderate**    |

So the reverse route is **not** null: reading feeding back into letter-sound knowledge has moderate evidence behind it, and the letter-sound ceiling did not erase it the way it erased the descriptive contrast. But it is the smaller of the two, and it is **fragile**: `med-176`'s own sensitivity sweep tips the NIE to zero at an unmeasured mediator → outcome confounder worth **30 %** of the fitted coefficient (`robust_over_full_sweep: false`), and latent general ability — which confounds both directions symmetrically and is unblockable by construction — is exactly such a confounder. `med-076` should be read with the same caution, but it has roughly two and a half times the margin.

The other two questions:

**`lcsm-082`** (blending ↔ word reading reciprocal dominance, reporting tier, gate passed: 0 divergences, R-hat 1.002, min ESS 2 557) already estimates the blending coupling:

| Coupling                                  | Median | 89 % CI          | P(>0)  | Evidence       |
| ----------------------------------------- | ------ | ---------------- | ------ | -------------- |
| prior word reading → blending change      | +0.055 | −0.036 to +0.147 | 0.835  | **suggestive** |
| prior blending → word-reading change      | +0.040 | −0.059 to +0.135 | 0.749  | inconclusive   |
| dominance ( \|W→B\| − \|B→W\| )           | +0.081 | −0.173 to +0.397 | 0.688  | inconclusive   |
| prior letter sounds → word-reading change | +0.245 | +0.140 to +0.368 | ≈1.000 | very strong    |
| prior letter sounds → blending change     | +0.155 | +0.042 to +0.284 | 0.987  | strong         |

So the reading → blending question **has been asked and answered**: suggestive, not conclusive, with no evidence that either direction dominates, and with letter sounds outweighing both. There is nothing to build here.

**`lcsm-081`** (reading → taught vocabulary, gate passed) shows the reverse-coupling programme works where it is identifiable: `WR → TR` +0.084 [+0.049, +0.121], P ≈ 1.000 (very strong); `WR → TE` +0.062 [+0.015, +0.112], P = 0.984 (strong).

**`lcsm-091`** settles the literal form of the question. "Learning more words predicts later gains" is a **change-on-change** claim, and 091 is the only model that fits change-on-change terms. Its verdict at this sample size is unambiguous:

| Term                                           | Median | 89 % CI          | P(>0)  |
| ---------------------------------------------- | ------ | ---------------- | ------ |
| prior letter-sound **level** → reading change  | +0.224 | +0.124 to +0.332 | ≈1.000 |
| prior vocabulary **level** → reading change    | +0.248 | +0.061 to +0.443 | 0.983  |
| prior letter-sound **change** → reading change | +0.047 | −0.167 to +0.259 | 0.645  |
| prior vocabulary **change** → reading change   | −0.001 | −0.476 to +0.473 | 0.499  |

The change terms carry intervals two to five times wider than the level terms and sit on zero — the vocabulary one spans nearly a full unit and is centred on exactly nothing. **Change-on-change is not estimable at n ≈ 54**, on the closest available analogue, in a model that otherwise passed its gate cleanly. The question has to be asked in the prior-**level** form to have any power.

## Pass 4 — the forward direction: does phoneme awareness predict reading?

Passes 1–3 ask what word-reading standing goes with and what it might feed. The obvious companion question runs the other way, and it is the one the phonics literature would ask first: **does blending predict reading outcomes?** Six fitted models bear on it, and the answer depends entirely on the adjustment set — and flips between levels and gains.

**Unadjusted, blending tracks reading strongly at every wave.** Pass 1 puts it at Cliff's delta 0.38 / 0.50 / 0.53 / 0.70 across t1–t4, interval excluding zero each time. `lrp-rli-ca-001`'s bivariate per-wave estimates agree: +0.52 / +0.33 / +0.47 / +0.54, every one at P ≥ 0.98.

**Holding letter sounds fixed, it survives.** Per-wave Beta-Binomial regressions of the word-reading count on same-wave letter sounds _and_ blending, plus age, hearing and non-verbal ability (the probe from the companion note; max R-hat 1.0000, min bulk ESS 6 092, zero divergences):

| Wave | blending slope | 89 % CI        | P(>0) | Evidence        | letter sounds, same fit |
| ---- | -------------- | -------------- | ----- | --------------- | ----------------------- |
| t1   | **+0.47**      | +0.21 to +0.71 | 0.997 | **very strong** | +0.56 (P = 0.998)       |
| t2   | +0.17          | −0.03 to +0.37 | 0.916 | **moderate**    | +0.95 (P = 1.000)       |
| t3   | +0.27          | +0.05 to +0.49 | 0.975 | **strong**      | +0.56 (P = 1.000)       |
| t4   | +0.34          | +0.15 to +0.53 | 0.996 | **very strong** | +0.65 (P = 1.000)       |

So blending is **not** merely a proxy for letter-sound knowledge — the obvious sceptical reading, and the data do not support it. What the next table shows is that the redundancy is with something else.

**Holding vocabulary fixed as well, it mostly goes.** `lrp-rli-ca-001` (gate passed) mutually adjusts six same-wave skills — letter sounds, blending, taught receptive and expressive vocabulary, and broad receptive and expressive vocabulary:

| Wave | blending, adjusted | 89 % CI        | P(>0) | letter sounds, same fit |
| ---- | ------------------ | -------------- | ----- | ----------------------- |
| t1   | +0.28              | +0.03 to +0.52 | 0.965 | +0.44 (P = 0.992)       |
| t2   | **−0.02**          | −0.21 to +0.19 | 0.451 | +0.92 (P = 1.000)       |
| t3   | +0.19              | −0.04 to +0.42 | 0.903 | +0.39 (P = 0.993)       |
| t4   | +0.20              | −0.01 to +0.40 | 0.939 | +0.47 (P = 1.000)       |

The shrinkage between the two tables is **vocabulary, not letter sounds** — the only adjusters that differ are the four vocabulary terms. This is a DAG-appropriate adjustment (taught and broad vocabulary are parents of both blending and word reading in the base graph), so it is not over-adjustment; but it does mean blending's apparent contribution to reading _level_ is substantially shared with the vocabulary route rather than being its own.

**On reading _gains_, blending does nothing.** This is where it parts company with letter sounds. `lrp-rli-gf-001` (gate passed, 0 divergences) is the DAG-faithful gain model, adjusting for every parent of word reading:

| Period-baseline term    | Median | 89 % CI          | P(>0) | Evidence         |
| ----------------------- | ------ | ---------------- | ----- | ---------------- |
| taught expressive vocab | +0.125 | −0.011 to +0.259 | 0.929 | moderate         |
| nonword reading         | +0.047 | −0.008 to +0.103 | 0.913 | moderate         |
| letter sounds           | +0.067 | −0.012 to +0.147 | 0.912 | moderate         |
| **blending**            | +0.029 | −0.042 to +0.100 | 0.744 | **inconclusive** |

And there is no evidence it carries any of the intervention effect — though "no evidence of a share" is not "evidence of no share": the interval below still admits roughly ±0.5 words either way. `lrp-rli-med-066` (parallel two-mediator split, gate passed) decomposes the effect on word reading through letter sounds and blending simultaneously: through letter sounds **+1.62 words of 79 [+0.50, +3.21], P = 0.994 (very strong)**; through blending **−0.03 words [−0.62, +0.42], P = 0.423 (inconclusive, centred on zero)**. `lrp-rli-med-075`, which instead routes them sequentially (letter sounds → blending → reading), reproduces this almost exactly (blending −0.03 words, P = 0.419). And `lcsm-082`'s lagged prior-blending → reading-change coupling is +0.040 [−0.059, +0.135], P = 0.749 — inconclusive.

### Two things that temper this

**The one dedicated blending → reading mechanism model is now usable, and it resolves no knee.** `lrp-rli-mech-190` — the only model whose focal exposure is blending with word reading as the outcome, and so the only one fitting that relationship as a flexible curve rather than a single adjusted coefficient — originally had **31 divergences** (R-hat 1.001 and min ESS 2 621 were both fine, so it was a geometry problem, not a length problem). It has since been reparameterised for the thin blending support (fewer HSGP basis functions, a tighter lengthscale prior; [#430](202607241800-repair-mech190-blending-knee.md)) and now passes the reporting-tier gate with **0 divergences**. The repaired curve is **flat and wide** — posterior-mean amplitude ≈ 0.14 on the logit scale against an ≈ 0.45 89% band, near-zero slope throughout — so the shape question _is_ now answerable and the answer is that **no knee is resolved** for blending at this sample size, unlike the letter-sound knee near the top of its range (≈ 29.5 of 32, itself wide and partly manufactured by the bounded scale on a logit link; [skill thresholds](202607171215-findings-skill-thresholds.md)). Read the curve as "shape unresolved", not "shape flat".

**The measure is weak, and that is a live alternative to the substantive reading.** Blending is a **10-item** task administered as three-alternative picture-pointing — target plus an initial-phoneme and a rhyme distracter, so roughly a **one-in-three chance factor**, putting the effective floor near 3.3 rather than 0 — and 19 % of children are at the 10-item ceiling by t4. Letter sounds, by contrast, is 32 items with no chance factor. A shorter, coarser, more compressed measure attenuates its own coefficient toward zero, so **"blending adds nothing to reading growth" and "blending is measured less well than letter sounds" are not separable in these data.** The DAG review note already records the construct half of this: the `PA` node is operationalised by blending alone and so "claims more construct than the single task delivers" ([DAG explanation review](202607101444-dag-explanation-review-draft.md)).

## What we can and cannot say

**Can say.** Word-reading standing goes with almost every language and literacy measure in this cohort, and the separation widens across the study — fourteen measures at Cliff's delta ≥ 0.6 by t4, nine of them outside the reading domain. It does **not** go with age (0.03 at t4), behaviour, hearing, ear infections or intervention dose, and it goes with non-verbal ability only weakly (0.33) and only at the bottom of the distribution — block design separates the lowest word-reading quartile from the rest and then stops. Expressive vocabulary is the one measure that tracks word reading as an even gradient across all four quartiles. Half-split membership is durable (77–88 % wave-to-wave agreement; Spearman 0.76–0.96). Among children who read no nonwords at t1, those in the top word-reading half were far more likely to be reading some by t2 (62 % vs 10 %), within both arms.

**Can also say, from models that predate this note.** Reading feeds back into letter-sound knowledge with **moderate** evidence (`med-176`, NIE +0.45 letters of 32, P = 0.963) — smaller than the forward letter-sounds-to-reading route and not robust to modest unmeasured confounding. How much smaller depends on the scale and the two are not directly comparable: the outcomes are a 32-item and a 79-item test, so the raw ratio (0.45 letters against 2.9 words, about a sixth) overstates the gap, while on the model-native probability scale, and as a share of each test's range, it is closer to **two-fifths**. Reading feeds blending with **suggestive** evidence only (`lcsm-082`, P = 0.84), with no sign that either direction dominates. In the forward direction, blending is a solid concurrent correlate of reading _level_ that survives adjustment for letter sounds (P = 0.92–0.997 across waves) but largely not for vocabulary, and it predicts neither reading _gains_ (`gf-001`, P = 0.744) nor any share of the intervention effect (`med-066`, −0.03 words, P = 0.42).

**Cannot say.** Nothing descriptive here is causal or even adjusted — general ability and age sit inside every band contrast, and the pass-1 ranking is an ordering of raw separations, not of importance. Nothing can be said about `WR → LS` **from the descriptive data**: the letter-sound ceiling censors the top band by t3, which is why the model-based answer above is the one to read. The t1 quartile split says nothing at its lower boundary, where 57 % of children are tied on a cut. Effect-size _rankings_ are partly a ranking of where each measure's floor sits, most acutely for nonword reading and phonetic spelling early on. The adopted `WR → NW` edge does not turn the floor-exit result into an identified effect: its measured adjustment set remains too wide, latent general ability remains unblocked, and the six-item outcome is heavily floored. And the pass-4 null on blending and reading **gains** cannot be separated from the measure: a 10-item three-alternative task with a one-in-three chance floor and 19 % at ceiling by t4 attenuates its own coefficient, so "phoneme awareness does not drive reading growth" and "this blending task is too coarse to show it" fit these data equally well.

## What would be worth building

Less than a first reading of the identification table suggests. Taking the three in priority order:

**First, the nonword DAG question — decided, but the model remains descriptive.** The 2026-08-08 decision adds `WR_t → NW_{t+1}` as a provisional working-DAG assumption. That resolves #428 and removes #433's structural-decision blocker, but it does not create a fittable causal estimand: the 8-node and 12-node measured backdoor sets are unchanged and too wide for this sample. Any promoted model must therefore retain `causal_status="none"` and `estimand_type="descriptive"`; it can formalise the concave association and its uncertainty, not estimate a word-reading effect.

**Second, `WR → LS` — do not build now.** A reverse-coupling LCSM would add a pooled per-transition coupling to `med-176`'s t2 → t4 intervention-effect-share estimand, but no current scientific question requires that extra estimand. It would not repair the important weakness: `med-176` already gives moderate evidence for the reverse direction, but its sensitivity sweep tips at 30 % of the mediator-to-outcome coefficient, and a second model inherits the same unblocked latent-general-ability confounding. #429 should close as not planned; reopen it only if the pooled per-transition coupling becomes an independently specified target rather than a hoped-for robustness check.

**Third, nothing new for blending in the reverse direction.** `lcsm-082` already answers it, has passed its gate, and returns a suggestive-but-inconclusive coupling with no directional dominance. Re-asking it would produce the same number.

**Do not ask it as change-on-change.** `lcsm-091` shows the terms are not estimable at n ≈ 54.

**A repair, not a new model: `mech-190` — done ([#430](202607241800-repair-mech190-blending-knee.md)).** The one direct blending → word-reading fit failed its gate on 31 divergences — a geometry problem (R-hat and ESS were healthy), not a length problem. It has been reparameterised for the thin blending support (fewer HSGP basis functions + a tighter lengthscale prior; the HSGP was already non-centred and `target_accept` already 0.999, so those levers were spent) and now passes at reporting tier with 0 divergences. The shape question is answered: the curve is flat and wide, so **no knee is resolved** for blending — the honest outcome, since the geometry _could_ be fixed, the linear fallback was not needed. The docstring-alias trap flagged here is also fixed: `mech-190` / `mech-191` opened `"LRP91"` / `"LRP92"`, but those aliases resolve to `lcsm-091` and a mediation model; the correct `lrp190` / `lrp191` headers are now in place, and a sweep confirms no other module carries a mismatched alias.

## A maintenance finding

A small one, now repaired. [`202607141030-lagged-dsep-checks.py`](assets/202607141030-lagged-dsep-checks.py) — the asset the reverse-coupling design note points readers at — had a stale reverse-edge list missing the `WR_t → LS_t1` edge added on 2026-07-17. #475 restored that edge and added a mirror assertion; the 2026-08-08 `WR_t → NW_t1` decision updates the DAG, archived assets and test invariant together.

**The CI guard now covers the six-edge list.** `tests/test_lagged_dag_adjustment_sets.py` includes `LS` and `NW`, and `test_unroll_slices_mirror_the_dagitty_template` asserts that every parsed template edge appears in the unroll. A dedicated #428 regression test also verifies the nonword edge and the unchanged 8-node / 12-node minimal measured sets.

The existing reverse-coupling derivations still hold (`WR → TE` invalid without arm, valid with; `WR → TR` valid); the 2026-08-08 sweep additionally records that the new outgoing edge does not narrow `WR → NW`'s backdoor sets.

## Cross-references

- [Letter sounds and word reading](202607241000-findings-letter-sounds-word-reading-association.md) — the companion, asking what predicts word reading rather than what it goes with.
- [Time-lagged model designs](202607141030-time-lagged-model-designs.md) — the reverse-coupling programme and the verified adjustment sets it is built on.
- [Time-lagged DAG](202607131200-time-lagged-dag.md) and `dag/dag-language-reading-lagged.dagitty` — the graph every pass-3 result is read off.
- [Evidence strength and ROPE reporting](202606261304-evidence-strength-and-rope-reporting.md) — the ladder used for the model-based results.
- [DAG explanation review](202607101444-dag-explanation-review-draft.md) — the `PA`-node operationalisation caveat behind pass 4's measurement argument.
- [Skill thresholds](202607171215-findings-skill-thresholds.md) — the letter-sound knee that `mech-190` was repaired to test for blending against.
- [Repair: `mech-190` blending knee-test](202607241800-repair-mech190-blending-knee.md) — the #430 reparameterisation that made pass 4's shape estimate usable.
- Fitted models quoted: `lrp-rli-ca-001`, `lrp-rli-gf-001`, `lrp-rli-lcsm-081`, `lrp-rli-lcsm-082`, `lrp-rli-lcsm-091`, `lrp-rli-med-066`, `lrp-rli-med-075`, `lrp-rli-med-076`, `lrp-rli-med-176`, and `lrp-rli-mech-190` (repaired in #430: reparameterised, reporting-tier gate-passing, no resolved knee).
