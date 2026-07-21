# Cross-model summary of findings — full statistical-model suite (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This note reads the whole suite together. It is the companion synthesis to the per-family notes (01–18) and the index/reading guide (`notes/202607210900-findings-00-index-and-reading-guide.md`), reporting the findings from the full re-fit of every Bayesian statistical model in the study (179 models, production `reporting` configuration, fit 2026-07-20/21). Read the index first for the study design, the outcome measures, the 89% credible-interval / tail-probability / evidence-ladder conventions, and — crucially — the rule that **only a contrast anchored in randomisation is causal**; everything else is an adjusted association describing _who progresses_, not a lever. All data and models are preliminary and the sample is small (~54 children), so throughout we lead with the interval, not the point.

## The one-paragraph answer

The reading and phonics intervention produces **clear, replicated, randomised benefits on the code-related and directly-taught skills** — letter-sound knowledge, phoneme blending, word reading, and taught vocabulary — and **no detectable movement on broad standardised vocabulary**. The word-reading gain is **mediated by letter-sound knowledge**, not by a vocabulary route. The finding is robust to adjustment for general ability, socio-economic status and study site, and it is recovered independently by three different randomised-effect designs. The many associational models describe a coherent developmental scaffold — letter-sounds appear to lead word reading, receptive vocabulary appears to lead expressive — without licensing causal claims.

## 1. The randomised headline, and its triangulation

Three families estimate the intervention's causal effect through different randomised designs, and they **agree**. The table gives each outcome's intention-to-treat effect (ITT `τ`, note 01), the difference-in-differences t2 contrast (`τ_t2`, note 05), and the gain-factor on-intervention marginal (`β_trt`, note 03), on the items scale where available, with the ITT tail probability and evidence label.

| Outcome                        | ITT `τ` (items)       | ITT P(helps) | Evidence         | DiD `τ_t2`        | Gain-factor `β_trt` | Verdict             |
| ------------------------------ | --------------------- | ------------ | ---------------- | ----------------- | ------------------- | ------------------- |
| **LS** letter-sound knowledge  | **+3.5** [+1.7, +5.3] | 99.9%        | **very strong**  | +3.5 [+1.2, +5.8] | +3.3 [+1.6, +5.0]   | benefit, replicated |
| **WR** word reading            | **+2.4** [+0.7, +4.1] | 99%          | **strong**       | +2.2 [−0.3, +4.7] | +2.6 [+0.9, +4.3]   | benefit, replicated |
| **PA** phoneme blending        | **+1.0** [+0.2, +1.7] | 98%          | **strong**       | +0.9 [+0.1, +1.7] | +0.8 [+0.1, +1.6]   | benefit, replicated |
| **TE** taught expressive vocab | **+1.5** [+0.4, +2.7] | 98%          | **strong**       | +1.5 [+0.0, +3.0] | +1.2 [+0.0, +2.3]   | benefit             |
| **TR** taught receptive vocab  | **+1.4** [+0.2, +2.5] | 97%          | **moderate**     | +1.2 [−0.3, +2.7] | +1.1 [−0.1, +2.2]   | benefit, softer     |
| **LF** basic concepts          | +0.9 [−0.3, +2.0]     | 89%          | suggestive       | +0.6 [−0.5, +1.8] | +1.1 [+0.1, +2.0]   | leans positive      |
| **RG** receptive grammar       | +0.7 [−0.8, +2.1]     | 76%          | suggestive       | —                 | +0.8 [−0.5, +2.2]   | leans positive      |
| **RV** receptive vocab (std)   | +0.2 [−3.7, +4.3]     | 54%          | **inconclusive** | −0.1 [−5.1, +5.0] | −1.4 [−5.3, +2.4]   | flat / negligible   |
| **EV** expressive vocab (std)  | +0.2 [−3.1, +3.5]     | 53%          | **inconclusive** | +0.8 [−4.0, +5.5] | +1.1 [−2.1, +4.3]   | flat / negligible   |

The gradient is the substance: **very strong** on letter-sound knowledge, **strong** on word reading, phoneme blending and taught expressive vocabulary, **moderate** on taught receptive vocabulary, tapering through **suggestive** on the broader language measures (basic concepts, grammar), and **inconclusive** on the two broad standardised vocabulary tests. Three designs that make different assumptions — ITT (baseline as a precision term), difference-in-differences (within-person crossover), and gain-factor ANCOVA (period-stacked change) — land on the same ordering and nearly the same magnitudes. The joint multivariate fit (note 02) reproduces the per-outcome `τ`s, and its taught-versus-not-taught generalisation contrasts show the benefit concentrated on the specific words taught. This is the strongest form of evidence a single small study can offer: a consistent randomised signal, triangulated.

Two honest qualifications carry across all three families. First, **direction is not size**: even where the direction is very strongly resolved, the probability the benefit clears the project's pre-agreed minimally-important difference is often only moderate (e.g. phoneme blending is 98% positive but the items magnitude is genuinely uncertain on a 10-item test), and these δ thresholds were set after an initial results review, so they are read beside the threshold-sensitivity analysis. Second, these are **modified, available-case** analyses of ~54 children (fewer for the floored and subset models), so point estimates are on average magnitude-inflated — the interval is the honest summary.

## 2. The mechanism: word reading improves _through_ letter-sounds

The mediation family (note 08, g-formula decomposition) asks _how_ the word-reading benefit arises. The answer is consistent and specific: the natural indirect effect **through letter-sound knowledge is positive and entirely above zero** (e.g. `med-059`, NIE +1.7 items [+0.6, +3.2]), while the routes through **nonword reading** (`med-074`, NIE ≈ 0 [−0.4, +0.5]) and, as a **negative-control**, through **receptive grammar** (`med-079`, NIE ≈ 0 [−0.2, +0.6]) are flat. In the two-mediator models the **letter-sound route dominates the vocabulary route**. This is exactly the pattern a genuine decoding-specific benefit predicts, and it dovetails with the flat standardised-vocabulary `τ`. The important caveat: mediation is a **decomposition under untestable no-unmeasured-confounding assumptions**, so the routes are strong associations consistent with a decoding mechanism, not proof of one; the total effect it decomposes is, however, the randomised `τ`.

The mechanism family (note 07) adds the shape of these couplings — letter-sound knowledge is very strongly associated with word reading, nonword reading, and the broader language measures — but every mechanism slope is an **adjusted association**, never "letter-sounds drive reading".

## 3. Reconciling the apparent negatives on standardised vocabulary

Two families show a _negative_ blip on standardised vocabulary that must not be misread as harm. The **level-factors** family (note 04) reports a t2 contrast of −3.8 items on receptive vocabulary (`lf-002`, 92% probably negative) and −2.4 on expressive (`lf-003`); the **aligned per-protocol** family (note 06) shows −3.1 on expressive (`al-003`). Neither is evidence the intervention lowered vocabulary. The level-factors model fits an _absolute level_ and does **not** difference out each child's baseline, so a negative t2 gap most plausibly reflects residual baseline imbalance carried forward on a noisy 170-item test; the aligned cohort contrast is **not randomised** (it is confounded by age-at-onset and cohort timing). The baseline-differencing randomised families — ITT, difference-in-differences, gain-factors — all put standardised vocabulary at essentially zero, and they are the authoritative read. The lesson is methodological: on the broadest, noisiest measures, the unadjusted-level and per-protocol views are the least reliable, and they reconcile with "no effect", not "harm".

## 4. Robustness

The ITT headline is stable under every stress test in note 01. Adjusting `τ` for **general (block-design) ability** (`itt-017–024`) moves nothing: letter-sound knowledge stays +3.5 (very strong), word reading +2.2 (strong). Adjusting for **socio-economic status** on the matched complete-case subset (`itt-013/113` against unadjusted comparators `itt-014/114`) leaves both anchor effects intact — the slight softening of a label there is the halved sample, not the covariate. Adjusting for **study site** (`itt-027/028`) if anything sharpens the effects. The headline is not an artefact of ability, background or site imbalance.

## 5. The developmental scaffold (associational families)

The associational families paint a coherent, and mutually consistent, picture of skill structure — all as adjusted associations, confounded by latent general ability, never causal:

- **Concurrent associations** (note 11): at any wave, letter-sound knowledge and word reading are very strongly coupled (`ca-001`: +8.5 word-reading items per letter-sound unit at t2), as are the vocabulary measures with the taught sets.
- **Latent change-score** (note 10): earlier **letter-sound knowledge predicts later word-reading change** (`lcsm-082/091`, very strong), and within vocabulary **receptive knowledge leads expressive** (`lcsm-081/181`, very strong) — a temporal ordering consistent with the mediation story. Each crossover-aware model's randomised window-1 contrast is positive, echoing the ITT.
- **Growth curves** (note 15): trajectories are jointly structured, with baseline ability associated with grammar growth (very strong association).
- **Horseshoe selection** (note 13): sparse Bayesian variable selection independently surfaces the **letter-sound ↔ word-reading ↔ expressive-vocabulary** axis as the predictors that escape shrinkage, cross-checking the gradient-boosting layer.
- **Correlated-factor measurement** (note 14): the latent skill domains are strongly and coherently correlated (vocabulary–grammar ≈ 0.80, vocabulary–code ≈ 0.74 in the trial cohort; reading/language/memory/ability all 0.79–0.93 in the Byrne cohort). These correlations are robust; the models' _structural_ legs are held pending reparameterisation (see §7).
- **Survival** (note 17): on the heavily floored skills, the chance of coming off the floor is _suggestively_ higher in the treated arm for nonword reading (hazard ratio 1.35, 80% positive) and _inconclusive_ for phonetic spelling — prognostic associations, not randomised effects.
- **Adjusted baseline predictors** (note 12) and the **historical Byrne cohort** (note 18) provide standing-predictor and natural-history context: in the Byrne reading-language-memory cohort, all measured domains show very strong natural growth across waves and correlate strongly — a non-randomised developmental backdrop to the trial.

## 6. Dose-response

Attendance dose (note 09) is associated with more word-reading progress, most clearly in the **randomised first period** (`dose-077`: period-1 slope 99.8% positive; `dose-277`: +1.3 words per standard-deviation of sessions, very strong). But **dose is a partial collider** — attendance is itself shaped by ability and early response — so conditioning on it can induce bias, and this family is a sensitivity view of associations, never a causal dose-response. The four period-resolved dose models are divergence-only gate-review fits (see §7).

## 7. Convergence and what to trust

Of 179 models, **162 pass the convergence gate**; the 17 flagged for review split cleanly (details in each note and the index):

- **13 divergence-only** (`did-007`, four `dose-*`, `hs-001`, six `mech-*`): R-hat ≤ ~1.003, effective sample sizes in the thousands, at most 0.09% divergent draws — well inside the ≤1% working guidance. **Usable with a note.**
- **4 correlated-factor measurement models** (`mm-001/002/101`, `rlm-mm-001`): these fail R-hat/ESS on a latent-factor funnel geometry. Their **domain correlations are robust and reported**; their **structural coefficients are held** pending a non-centred reparameterisation. This is the one place a structural number should not yet be quoted.

No fit failed to sample and every one produced a report; the gate flags are about interpretive caution, not broken models.

## 8. Bottom line

For children with Down syndrome in this study, the intervention **causally improves the code-related and directly-taught skills it targets — letter-sound knowledge most strongly, then word reading, phoneme blending and taught vocabulary — and the word-reading gain runs through letter-sound learning.** It does **not** move broad standardised vocabulary over this window (an inconclusive, probably-negligible effect, not a demonstrated absence). The result is randomised, replicated across three designs, robust to ability/SES/site adjustment, and coherent with the developmental scaffold the associational models describe. The caveats are the ones that attach to any small, preliminary study: ~54 children, wide intervals, available-case estimands, post-hoc importance thresholds, and — for every non-randomised quantity — residual confounding by latent general ability. Read the direction with confidence where the evidence is labelled strong or very strong; read the magnitudes, and every association, with the interval in hand.
