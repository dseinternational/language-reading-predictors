> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).
>
> The comparator-equivalence, per-wave reporting and Δ-interpretation corrections for [#591](https://github.com/dseinternational/language-reading-predictors/issues/591) were made by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `joint_mechanism` family — is the letter-sound route specific to decoding?

**Read `findings-00-overview` first**, and ideally the `mechanism` note, which this family refines. Nothing here is causal.

## The data

**RLI trial only**, and the two models use it differently — which turns out to be the whole story.

- **`jm-001`** uses a **levels** design: one row per child at a single timepoint, run once per wave, with 53 children at waves 1–3 and 52 at wave 4. It asks who _is_ higher.
- **`jm-002`** uses a **transition ANCOVA** design: all period transitions stacked, 53 children contributing 153 rows, each row carrying each outcome's own starting score. It asks whose post-period outcome is higher conditional on that starting score; it does not regress a literal change score on a change in letter sounds.

## What the model is for

The `mechanism` family estimates one exposure-outcome slope per model. To ask whether letter-sound knowledge is _more_ tied to one outcome than another, the dependence between slopes must be represented. A joint posterior that models the shared children is one way to do that; merely subtracting factorised marginal fits does not provide a dependence-aware interval.

This family fits word reading and nonword reading **together** and reports the contrast Δ = slope(letter sounds → nonword) − slope(letter sounds → word reading).

The logic is a convergent/discriminant argument. Nonword reading can _only_ be done by decoding — the words are invented, so sight recognition is unavailable. Word reading can be done either by decoding or by recognising familiar words. If letter-sound knowledge is specifically a decoding skill, it should track nonword reading **more** strongly, giving Δ > 0.

The two models also differ in machinery. `jm-001` uses a Binomial likelihood with a correlated residual between outcomes; `jm-002` uses a Beta-Binomial with a correlated child-level random intercept. Both need somewhere to put the extra variation, and each puts it in a different place — see the note below on why that matters.

## What was found — and the two models disagree

| Model    | Design                      | Δ = slope(→nonword) − slope(→word)     | 89% range      | Reading                                        |
| -------- | --------------------------- | -------------------------------------- | -------------- | ---------------------------------------------- |
| `jm-002` | transition (post given pre) | **+0.81**                              | +0.50 to +1.14 | letter sounds track **nonword** reading more   |
| `jm-001` | levels (cross-sectional)    | **−0.47, −0.17, −0.26, +0.06** (t1–t4) | see below      | mixed; the sign does not hold across the waves |

**These point in opposite directions, and that should not be smoothed over.** Both are well-converged fits of the same children.

`jm-001` is reported as **all four waves**, not as one. An earlier version of this note quoted its timepoint-1 estimate (−0.48) as "`jm-001`'s Δ"; that was the wave whose posterior sat furthest from 0.5, chosen after the fits were seen, and both the model report and the key-findings box now report the whole set instead (#591). The 89% intervals are −0.97 to +0.04, −0.61 to +0.25, −0.68 to +0.15 and −0.30 to +0.41. Note also that the levels design re-standardises the exposure within each wave (SD 1.59, 1.38, 1.39, 1.44 logits), so "per SD" is a slightly different raw increment at each timepoint.

Three considerations explain the disagreement and why the baseline-conditional transition estimand is the more relevant one for the stated question; they do not turn it into a causal result.

**They answer different questions.** A levels model compares children at one moment, so its slope absorbs stable differences that make a child score highly on both measures at once, including possible general ability. A transition model conditions each post-period outcome on its own pre-period score, which can reduce that stable component without turning the outcome into a literal change score. For the narrower question of whether letter-sound level is more strongly associated with post-period nonword than word reading after baseline adjustment, the transition estimand is the relevant one.

**The factorised numerical cross-check agrees with `jm-002`.** Computing the same contrast from two separate single-outcome mechanism fits gives **+0.78** (89% +0.475 to +1.099), close to `jm-002`'s +0.81. The comparison table correctly marks that product-of-marginals calculation as _not_ an identified posterior contrast and `jm-002` as identified within its joint posterior. The agreement is reassuring only as a numerical cross-check: the fits use the same data and closely related likelihood, prior and adjustment assumptions, so they are not independent evidence.

**And the two are not fitted on the same rows or the same unit** (#591). `jm-002` requires both outcome baselines on every retained transition and standardises the exposure once over that joint union (153 rows, SD 1.412); `mech-096` keeps its own 152 nonword rows (SD 1.386) and `mech-101` its own 156 word-reading rows (SD 1.434). One SD is therefore a slightly different raw increment in each fit. The +0.81 versus +0.78 gap is **not** a measurement of what the working-independence assumption cost — it mixes a dependence change with a sample and scale change — and `scripts/compare_statistical_models.py` now publishes that reconciliation with an explicit `comparable` verdict beside both rows.

**`jm-001`'s intervals include zero at every wave** and its estimate moves from −0.47 at timepoint 1 through −0.17 and −0.26 to +0.06 at timepoint 4, so it is not a stable finding even on its own terms.

The defensible conclusion: **within the baseline-conditional transition model, letter-sound knowledge is more tightly associated with nonword decoding than with word reading.** That is compatible with a decoding-specific channel, but it does not identify one. Nor does the contrast's sign by itself rule the common factor out: with `LS`, `N` and `W` all loading on one latent general ability, the two latent-scale slopes stay proportional to their loadings, so Δ is proportional to the loading _difference_ with no causal letter-sound route at all. Residual confounding, unequal loadings, differing measurement properties (79 items against 6, with no cross-instrument invariance imposed) and the severe nonword-reading floor all remain alternative explanations for the contrast.

## A caveat about `jm-002`'s machinery

`jm-002` needs two devices for extra variation: a child-level random intercept and a Beta-Binomial dispersion parameter. Both absorb "the data are more spread out than a simple model allows", and with only three rows per child they are hard to tell apart. For **nonword reading** — a 6-item, heavily floored measure — they are genuinely hard to separate: the two parameters have a posterior correlation of 0.34, and the child-level spread has an 89% interval running from 0.03 to 0.66, i.e. from almost nothing to substantial.

This model initially failed the convergence gate for exactly this reason and was refitted with far more simulation. That fixed the computation (effective sample size rose from 256 to 7,596) but, as expected, left the underlying uncertainty essentially unchanged. **The Δ contrast has a comparatively tight posterior conditional on this model; the nonword variance components are not well determined.** Power-scaling still flags `beta_N`, the Δ contrast and the residual correlation in `jm-002` for potential prior–data conflict; `jm-001` flags its nonword slope and Δ contrast as potential strong-prior/weak-likelihood cases. Those warnings qualify the scientific robustness even though they do not block publication. Do not quote the variance components.

## What these models cannot tell you

**Neither slope is causal.** Both remain vulnerable to residual confounding; the non-zero negative-control outcomes in the `mechanism` family show that part of every letter-sound association is non-specific, with general ability one plausible explanation — although those same models put the nonword excess well above the oral-language controls.

**A specific channel is not a teaching claim.** That letter sounds track decoding does not establish that teaching more letter sounds produces proportionally more decoding.

**Nonword reading is thinly measured.** Six items with a floor limits what any model can extract, and it bounds the precision of everything involving that outcome.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable. `jm-001` (per-wave levels, constructed against the `concurrent` family's parameterisation) and `jm-002` (phase-stacked transitions, constructed against `mech-096`/`mech-101`). "Constructed against" is not "nested with": `jm-001` conditions on the **latent** held-fixed outcome in a logistic-normal Binomial model where `ca-011` conditions on an **observed** count in a Beta-Binomial one with mean-imputed predictors, and `jm-002` differs from its marginals in fitted rows and exposure scale as above. Each model's `config.json` and report now carry that statement explicitly.

Both models also report a quantity whose name is misleading if read literally. `share_retained` is a **conditional-to-marginal slope ratio**, not a bounded pathway share: it is unbounded, can be negative under suppression or exceed one under amplification, and identifies no mediated fraction. It is published with a prespecified denominator-stability rule, with the posterior probability of each of those three cases, and without a mean (#591).
