<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Design lessons for future studies — what the walls in this study tell us

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). A meta-note, not a findings note: it records where our analyses **could not cleanly answer a question**, diagnoses _why_, and turns each wall into a concrete recommendation for the design of future studies (sample size, waves/duration, measures, randomisation structure). Living document — extend as new walls are hit.

## 1. Why this note exists

A study's most durable contribution is often not its point estimates but its map of _what a study of this shape can and cannot learn_. This analysis programme repeatedly ran into questions it could not answer cleanly — the letter-sounds↔word-reading direction, the causal role of any single skill, several mechanism decompositions. Each "we can't answer that" is information about how the _next_ study should be built. This note collects those lessons so the next protocol inherits them.

## 2. The central idea: three kinds of wall, three different levers

The failures we hit are **not all the same kind**, and the fix differs by kind. Conflating them wastes resources — the single most expensive mistake would be to throw sample size at a wall that sample size cannot move.

| Wall                  | Symptom                                                                                                                   | What fixes it                                      | This study's examples                                                                                                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Power / precision** | Model is well-behaved, calibrated, unbiased — but intervals are too wide / power ≈ 0                                      | **More information**: larger n and/or more waves   | RI-CLPM direction (power **0.00** at n≈54×4); most mechanism/interaction estimands land "suggestive", not "strong"                                                                               |
| **Identification**    | No amount of n or statistical cleverness helps; the causal quantity is not a function of the observable data distribution | **A different _design_** (randomisation structure) | "Which component causes the reading gain?" (the `IG→IS→{everything}` teaching-dose witness); every skill→skill "effect" (unblockable latent general ability, `GA`); direction from a short panel |
| **Measurement**       | Information is destroyed _at collection_ and cannot be recovered analytically                                             | **Different _measures_ / data capture**            | Floors (nonword reading 57–72% floored; spelling; blending's guessing floor); single-indicator constructs (no disattenuation); no item-level responses retained                                  |

The diagnostic question for any future "we couldn't answer X" is therefore: **is this a power, identification, or measurement wall?** The DAG plus a pre-fit simulation usually tells you _before_ fielding — which is the cheapest possible time to learn it.

## 3. Sample size

**What n ≈ 54 (of 57 randomised) _was_ enough for:** the primary total intention-to-treat effect on the main outcome (word reading) — the quantity the trial was powered for, and it delivered a usable randomised estimate.

**What it was _not_ enough for** — every _secondary/mechanistic_ question:

- **Within-person direction.** The RI-CLPM feasibility study (`notes/202607172230-riclpm-direction-plan.md` §12) is the clearest evidence: with a genuinely forward-dominant truth (`δ − β` = +0.10), across 60 simulated datasets at the true design, **not one** 90% interval cleared zero (power 0.00), even though the model sampled cleanly and its intervals were calibrated (coverage 1.00). This is a pure power wall — the data simply do not contain enough _within-person_ cross-lag information at this n.
- **Mechanism / mediation decompositions** land wide; most direction and interaction reads sit at "suggestive" on the evidence ladder rather than "strong".
- **Interactions / moderation** (e.g. the letter-sound × blending synergy on decoding) stay suggestive.
- **Heterogeneity / subgroups** are effectively out of reach.

**Lesson.** Power the study for the _mechanistic and secondary_ estimands you actually care about, not only the primary ITT. Mechanism, mediation, interaction and direction estimands need a **multiple** of the primary-effect n — realistically the low-to-mid hundreds for this class of bounded/floored longitudinal outcome, but the target should be _derived, not guessed_. The RI-CLPM feasibility harness (`notes/assets/202607172230-riclpm-feasibility.py`) is already parameterised by n and waves; running it across an n-grid produces a power/precision curve that sets the target for a specific estimand. **Recommendation: make a pre-fit design analysis of this kind a standard step.**

## 4. Number of waves and study duration

Four waves is the _minimum_ for a random-intercept cross-lagged model (identification needs ≥3; stable cross-lags want ≥4–5, and the "first-wave problem" degrades estimates when waves are few — Mulder & Hamaker 2021). For **direction / dynamics** questions, _within-person_ information scales with the number of waves, so **adding waves is more efficient than adding people**; for **between-person effects and precision**, adding people helps more. Match the wave count to the question:

- Direction / cross-lagged dynamics → **≥ 5–6 waves**.
- Duration also lets floored skills accumulate signal (nonword reading moved from 28% → 60% of children off the floor across t1–t4) — though more time cannot rescue a badly-floored _instrument_ (§6).

## 5. Study design and randomisation structure — the deepest lesson

**A bundled intervention can estimate its total effect but cannot identify which of its parts causes that effect — regardless of sample size or statistical method.** The RLI intervention teaches letter sounds, vocabulary and reading together, so in the DAG the randomised arm reaches every skill through a shared teaching-dose path (`IG → IS → {LS, PA, NW, WR, …}`). That path is a _treatment-induced common cause_ of any candidate mediator and the outcome — the "witness" that makes the mediation family unidentified for causal natural direct/indirect effects. This is an **identification wall**: no n and no cleverness converts a bundled trial into a component-level causal answer.

The design that _does_ answer "which component works" is a **componential / factorial optimisation trial** (the Multiphase Optimization Strategy, MOST — Collins 2018), which randomises intervention components **independently** so each component's effect is estimated free of the others. Related levers:

- **Randomise dose / intensity**, not only assignment — this turns the dose-response question from a collider-prone _association_ into a causal contrast.
- **Weigh the waitlist-crossover trade-off explicitly.** The crossover is ethical and gives a clean randomised contrast at t2, but (a) collapses to _association_ at every post-crossover wave, and (b) creates the very treatment-induced dose confounder that breaks mediation. A parallel-arm design, or a stepped-wedge with more randomised windows, trades some efficiency/ethics for **durable randomised contrasts across more waves** — worth it if later-wave causal effects or mechanism are primary aims.

**If the aim is mechanism, the design must be chosen for mechanism from the start.** Retrofitting mechanism onto a trial built to estimate a total effect is exactly where we spent the most effort for the least identifiable return.

## 6. Measure selection — often the cheapest high-value change

- **Floors.** A 6-item nonword test with 57–72% of children at zero is near-useless as a _graded_ measure and forces a coarse off-the-floor binary. Choose or build instruments with adequate floor _for this population_: more, easier, finely-graded items and basal rules. The DS literature is explicit that decoding is a relative weakness (Cupples & Iacono 2000), so decoding measures in particular must be floor-calibrated for DS or they throw away the signal the study most wants.
- **Ceilings.** Letter-sound knowledge (32 items) shows some ceiling; give headroom so growth is not censored at the top.
- **Guessing floors.** Blending is a 3-alternative forced choice (chance ≈ 33%); more alternatives, more items, or an open-response format raises the usable range.
- **Retain item-level responses — the single cheapest high-value change.** Only totals were kept (`nonword` 0–6, `yarclet` 0–32). Keeping _which_ graphemes each child knows and _which_ nonwords they read would unlock a within-child, within-item contrast — do children decode nonwords built from the letters _they personally know_ better than nonwords with unknown letters? — that is **robust to general-ability confounding by construction** (it is a within-child comparison). That is the one analysis that structurally beats the `GA` problem, and it is impossible here purely because item responses were not retained. It costs essentially nothing extra at collection.
- **Multiple indicators per construct (≥ 3).** Most constructs are measured by a single instrument, so measurement error is unmodelled and structural coefficients are attenuated. Three-plus indicators per latent construct support measurement models that disattenuate and separate measurement from structure; the current correlated-factor / measurement-model family is indicator-starved.
- **A proper general-ability battery.** `GA` is the unblockable confounder behind _every_ skill→skill association in this study. A rich, multi-indicator non-verbal-ability measure would let a future study condition on an _estimate_ of `GA` and move some skill→skill associations closer to adjusted-causal. (Measured `GA` is still an imperfect proxy — but far better than a latent nuisance a random intercept cannot absorb.)
- **Design negative controls in.** Collect outcomes and exposures known to be null under the causal hypothesis (Lipsitch, Tchetgen Tchetgen & Cohen 2010), so residual confounding can be _detected_ even observationally. The Tier-1 mini-suite (`notes/202607172330-tier1-decoding-specificity-spec.md`) retrofits negative-control outcomes; a future study should plan them from the outset.

## 7. Missingness and attrition

54 of 57 randomised children entered the analytic set; complete-case requirements dropped further rows, and two confounders needed missingness indicators plus imputation. At small n **every lost case is expensive** and can flip a "suggestive" read. Recommendations: strong retention protocols; **collect all baseline confounders completely** (especially the exogenous ones — hearing, speech, phonological memory — that anchor the adjustment sets); consider **planned-missingness designs** to field a broad measure battery without over-testing any one child; and pre-register a principled imputation model rather than defaulting to complete-case.

## 8. Analysis planning

- **DAG-first pays off.** Committing the causal DAG up front is what let us _classify_ each wall as power vs identification vs measurement — the classification in §2 is only possible because the graph is explicit. Pre-register the DAG, the estimands, the adjustment sets, and the go/no-go gates.
- **Simulate before fielding.** A pre-fit design analysis / simulation-based calibration across the estimands that matter (not just the primary ITT) would have predicted the RI-CLPM power wall _before_ any data were collected. Making feasibility simulation a standard gate turns "we couldn't answer X" into a documented, principled stop rather than a post-hoc disappointment.

## 9. If you could change three things (prioritised)

1. **Design for mechanism if mechanism matters** — a componential/factorial (MOST) or dose-randomised design. Only this makes "which component causes the gain" _identifiable_. (Removes the identification wall.)
2. **Fix the measures** — floor/ceiling-appropriate instruments, **item-level retention**, ≥3 indicators per construct, and a real general-ability battery. (Removes the measurement wall; item-level retention is near-free and uniquely defeats `GA` confounding.)
3. **Power the secondary estimands** — size and wave-count set by simulation for the mechanism/interaction/direction questions (likely low-hundreds n and ≥5–6 waves for direction), not just the primary ITT. (Removes the power wall.)

The **order is the point**: spending money on (3) to attack a wall that is really (1) buys nothing. Diagnose the wall first.

## 10. Cross-references

- RI-CLPM power wall + the feasibility harness: `notes/202607172230-riclpm-direction-plan.md`.
- The identification walls (teaching-dose witness, `GA`, mediation ID-2) in full: `notes/202607172000-adjustment-set-review-full-suite.md`, `notes/202607142340-lrp264-mediation-adjustment-dsep.md`, and the DAG revision record `notes/202607101100-dag-revision-team-decisions.md`.
- The specificity designs that partly work around the measurement wall: `notes/202607172330-tier1-decoding-specificity-spec.md`.
- Direction contrast reads that stand in for the un-fittable cross-lag: `notes/202607161800-findings-mediation.md`.

## 11. References (verified 2026-07-17)

- Collins, L. M. (2018). _Optimization of Behavioral, Biobehavioral, and Biomedical Interventions: The Multiphase Optimization Strategy (MOST)_. Springer. [doi:10.1007/978-3-319-72206-1](https://doi.org/10.1007/978-3-319-72206-1)
- Cupples, L., & Iacono, T. (2000). Phonological awareness and oral reading skill in children with Down syndrome. _Journal of Speech, Language, and Hearing Research_, 43(3), 595–608. [doi:10.1044/jslhr.4303.595](https://doi.org/10.1044/jslhr.4303.595)
- Hamaker, E. L., Kuiper, R. M., & Grasman, R. P. P. P. (2015). A critique of the cross-lagged panel model. _Psychological Methods_, 20(1), 102–116. [doi:10.1037/a0038889](https://doi.org/10.1037/a0038889)
- Lipsitch, M., Tchetgen Tchetgen, E., & Cohen, T. (2010). Negative controls: a tool for detecting confounding and bias in observational studies. _Epidemiology_, 21(3), 383–388. [doi:10.1097/EDE.0b013e3181d61eeb](https://doi.org/10.1097/EDE.0b013e3181d61eeb)
- Mulder, J. D., & Hamaker, E. L. (2021). Three extensions of the random intercept cross-lagged panel model. _Structural Equation Modeling_, 28(4), 638–648. [doi:10.1080/10705511.2020.1784738](https://doi.org/10.1080/10705511.2020.1784738)
