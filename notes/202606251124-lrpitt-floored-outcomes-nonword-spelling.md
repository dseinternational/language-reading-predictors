<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Floored ITT outcomes (nonword reading, phonetic spelling): keep with a floor‑rule branch, not exclude

> [!WARNING]
> This note was prepared with an AI tool (Claude Code / Opus 4.8) via a multi‑agent
> deliberation and may contain mistakes. The empirical figures were computed from
> `data/rli_data_long.csv` and should be re‑checked before they enter a report.

Date: 2026-06-25 — relates to issue #119 (the LRPITT ITT suite).

## The question

In planning the uniform ITT suite (LRPITT01–11, issue #119), two outcomes are
heavily **floored** — most children score zero at both baseline and the randomised
post‑wave:

| Outcome                              | symbol | t1 at zero | t2 at zero |
| ------------------------------------ | ------ | ---------- | ---------- |
| Phonetic spelling (`spphon`, 92‑pt)  | `P`    | ~78 %      | ~64 %      |
| Nonword reading (`nonword`, 6 items) | `N`    | ~72 %      | ~64 %      |

(For contrast the clean outcomes move freely: word reading `W` 40 %→15 %, letter
sounds `L` 9 %→0 %, blending `B` 4 %→2 %.)

The floor makes the model's **own‑baseline coupling** `gamma_own` — the term
`gamma_own · logit(y_pre)` — near‑degenerate, because three‑quarters of the
baselines are at the floor and carry almost no information. The question raised:
**does this mean `N` and `P` should be dropped from the ITT and explored later via a
levels/trajectory analysis over all four timepoints?**

## The decision

**Keep both `N` and `P` in the suite.** Do not exclude them. Handle them with a
**arm-blind, pre-model-fitting floor-rule branch**: drop the degenerate `gamma_own`, and report a
**binary off-floor-transition** effect — among children at the floor at baseline — as the _primary_ estimand,
keeping the graded Beta‑Binomial τ only as a flagged, floor‑limited secondary. Add
the four‑wave levels/trajectory model the question proposed, but as a separate,
explicitly **descriptive (non‑ITT)** complement.

## Rationale

The instinct to set these two aside is sound, but the reason given —
"`gamma_own` is degenerate" — is not the deciding factor. Two different problems
were being conflated:

1. **Floored baseline → degenerate `gamma_own`.** This is real but _irrelevant to
   exclusion_. The locked DAG (#115) identifies the ITT effect τ from the **empty
   adjustment set**; `gamma_own` is a **precision term only**, not needed for
   identification. Dropping a noisy precision term costs nothing causal. So a
   degenerate own‑baseline coupling is a reason to **simplify the linear
   predictor** (age‑only), not to drop the outcome.
2. **Floored outcome → little graded movement.** This is the genuine issue, and it
   too is _not_ an exclusion argument — it is an **estimand** argument. With ~64 %
   of post‑scores at zero, a logit‑linear graded τ is leveraged by a handful of
   dispersed tail values rather than by the arm contrast. That is exactly why
   phonetic spelling's τ in the earlier 8‑outcome joint model was +0.09 with the widest CI in the whole fit, (−0.61, +0.82), `P(τ>0)=0.60` — structurally
   uninformative, not a finding. (The earlier note records this under the old negative-=-benefit convention as −0.09; sign flipped here per #117.) The LRP72 phonics‑route note documents the same
   "link‑saturation" hazard on a floored count. The honest response is to **reframe
   the estimand**, not delete the outcome — excluding `N`/`P` would silently bias
   the suite toward the outcomes that happen to be less floored (a selective‑
   reporting risk), and a floor‑limited null is itself a legitimate pre‑specified
   ITT result.

**Where the randomised signal actually lives.** The t1→t2 contrast is real and
correctly signed once you look at _coming off the floor_ rather than graded scores
(computed from the trial data; immediate = group 1, the arm treated in period 1):

| Outcome               | immediate arm off‑floor by t2 | wait‑list control |
| --------------------- | ----------------------------- | ----------------- |
| Nonword `N`           | 10 / 21 (48 %)                | 2 / 15 (13 %)     |
| Phonetic spelling `P` | 7 / 24 (29 %)                 | 2 / 17 (12 %)     |

The at‑risk denominators differ by arm, so this is modelled as a **risk contrast**
(a Bernoulli/logistic τ on `Pr(score > 0 at t2 | score = 0 at t1)`, fit on the baseline-floored at-risk set), with the repo's `G = 2 - group` coding (positive = benefit; PR #117), so that `G = 1` is the immediate-intervention arm and a **positive** τ still means the intervention helps.

## Wait-list crossover replication (triangulation)

A follow-up asked what happened to the wait-list group when it **first received the
intervention in period 2** (t2 -> t3) -- a within-design mirror of the immediate
group's randomised period-1 window. Off-floor movement among children still at the
floor at the start of each group's first treated period (script reproduces the
period-1 figures above exactly, as a correctness check):

|                                                 | Nonword `N`     | Phonetic spelling `P` |
| ----------------------------------------------- | --------------- | --------------------- |
| Immediate -- period 1 (treated, **randomised**) | 10/21 (48 %)    | 7/24 (29 %)           |
| Wait -- period 1 (untreated control)            | 2/15 (13 %)     | 2/17 (12 %)           |
| **Wait -- period 2 (first treated, crossover)** | **6/18 (33 %)** | **2/16 (12 %)**       |
| Immediate -- period 2 (maintenance)             | 7/15 (47 %)     | 4/18 (22 %)           |

**`N` replicates cleanly.** The wait group barely moved off the floor while
untreated (13 %) but jumped to 33 % off-floor movement _when it crossed over to
intervention_ -- ~2.5x, timed to treatment onset and mirroring the immediate group's
own first-treated period (48 %). The effect **follows the treatment across the
crossover** -- the signature of a genuine intervention effect -- and corroborates the
randomised period-1 contrast for nonword beyond the single window.

**`P` does not replicate.** The wait group moved off the floor at 12 % whether
untreated _or_ treated -- no crossover bump. The period-1 immediate-vs-wait gap
(29 % vs 12 %) fails to reproduce at crossover, so `P`'s off-floor signal is fragile
and may be partly noise.

**Caveats.** The within-group pre/post (wait P1 -> P2) is **not randomised** --
crossing over coincides with ~6 more months of maturation, so treatment onset and
ageing are confounded here; the clean randomised estimate remains the period-1
contrast, and the crossover is **triangulation**, not a second randomised estimate.
Nor is it a pure on/off switch: the immediate group keeps clearing the floor in
maintenance/post periods, partly because each period's at-risk set is just the
still-floored children (denominator depletion) and partly continued consolidation.

**Implication.** This sharpens the `N`-vs-`P` asymmetry (below) and argues that the
descriptive complement should be framed as an **intervention-aligned (treatment-on)
period** analysis -- pooling each group's _first treated period_ (immediate P1 + wait
P2) -- rather than a fixed-calendar four-wave trajectory, which is the cleaner way to
let the post-t2 data speak while respecting the crossover. This dovetails with the
period-resolved / intervention-aligned analysis already proposed in issue #104.

## The agreed specification (arm-blind floor-rule branch)

A rule fixed **arm-blind and before any treatment model is fitted** (data-driven on the floor rate, not pre-registered) — see the estimand-shopping guard below: any outcome with **≥ 40 % of post‑scores at zero at
t2** (currently only `P` and `N`):

1. **drops `gamma_own`** and uses an **age‑only** precision predictor
   `alpha + tau·G + gamma_A·age`;
2. reports a **binary off-floor-transition τ as PRIMARY**, among children at the floor at baseline (`Pr(score > 0 at t2 | score = 0 at t1)`, on the baseline-floored at-risk set) — a conditional/subgroup ITT effect;
3. retains the graded Beta‑Binomial τ as a **flagged, floor‑limited SECONDARY**,
   read only beside the per‑arm mover table, under a "detection‑limited outcome"
   banner;
4. carries a **proportion‑at‑zero posterior‑predictive check** and the per‑arm
   off‑floor mover counts as diagnostics.

`N` lands on the age‑only predictor for two compounding reasons: the 72 % baseline
floor, _and_ the repo's existing convention that `nonword` is "post‑only; not used
as a baseline" (`measures.py`). Note the convention is a modelling choice, not a
data absence — `nonword` _is_ populated at t1 (50 non‑missing, 14 non‑zero) — so the
age‑only spec is justified by **degeneracy + convention**, not by impossibility.

## The four‑wave levels analysis: complement, not substitute

The floor genuinely eases later — off‑floor rises to `N` ~60 % and `P` ~52 % by t4 —
so a four‑wave levels/growth model is the only place the graded scale finally has
variance worth modelling. **But after t2 the wait‑list arm crosses over**, so the
t2→t4 window is **not randomised**. Such a model answers "does this skill grow under
treatment‑plus‑maturation," confounded by crossover and by whatever predicts late
emergence — **not** "does the intervention move this skill." It is therefore added
to issue #119 as a clearly‑labelled **descriptive** deliverable, complementing the
randomised off‑floor ITT and never sold as one. It does not rescue the randomised
question; it answers a different one.

**Preferred form -- intervention-aligned.** Frame the descriptive complement as an
**intervention-aligned (treatment-on) period** model -- pooling each group's _first
treated period_ (immediate P1 + wait P2) -- rather than a fixed-calendar trajectory.
The wait-list crossover above shows why this is the more honest descriptive window
(see the crossover-replication section, and issue #104).

## The one genuine objection, and the guard

The strongest cross‑lens objection is **estimand‑shopping**: the mover counts above
already hint at the result, so "switch to a binary estimand because of the floor" is
only credible if the floor rule is **written into issue #119 and applied arm‑blind
before any model is fitted**. That is now a hard **arm-blind, pre-model-fitting** gate in the issue — data-driven on the floor rate, not a protocol pre-registration.

**Fallback.** If that discipline cannot be guaranteed, the safer default is the
_uniformity‑purist_ option: keep all 11 models identically specified, drop
`gamma_own`, and honestly report `N`/`P`'s wide graded τ under a floor banner.
Bespoke per‑outcome specifications are themselves a researcher degree of freedom, so
the floor‑rule branch earns its keep only if it is fixed in advance.

## Per‑measure note

`N` and `P` are not symmetric. `N` (6 items) has the cleaner arm separation
(48 % vs 13 %) and a coarse scale where "off‑floor" is nearly the whole story; `P`
(92 points) has the weaker contrast and the long graded tail that most tempts an
over‑read graded τ. If either is the weaker candidate for a _graded_ effect it is
`P`; the measurement‑sensitivity audit already singled `P` out (for exclusion from
the LRP62 reading‑route composite) — honoured here as a floor banner, not a suite
exclusion.

## Provenance

Reached via a multi‑agent deliberation: an evidence pass (repo floor/ceiling audit,
prior `P` result, the `SPPHON_NONE`/`NONWORD_NONE` indicators, and a fresh empirical
probe of floor and mover structure across all four waves by arm), a four‑lens
judgement panel (causal identification, measurement/likelihood, reporting integrity,
decision‑relevance), and an adversarial synthesis + verification. The verifier
corrected one overstatement (that `N`'s baseline "cannot be formed" — it can, it is
merely degenerate) and confirmed the empirical figures; both fixes are folded in
above.

## Related

- Sign convention: PR **#117** (`G = 2 - group`, positive τ = intervention benefit; merged 2026-06-25). This note uses the post-#117 convention; the empirical off-floor counts are computed from the raw `group` column and are unaffected by the recode.
- `notes/202606231600-dag-revision-consolidated.md` — locked DAG; ITT identified by
  the empty adjustment set (own baseline / age are precision only).
  evidence for `P`, `N` and others.
  floored‑count hazard precedent (graded effect not identifiable on a floored
  count).
  8‑outcome joint model.
