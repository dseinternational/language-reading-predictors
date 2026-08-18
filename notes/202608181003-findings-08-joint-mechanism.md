> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

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

| Model    | Design                      | Δ = slope(→nonword) − slope(→word) | 89% range      | Reading                                      |
| -------- | --------------------------- | ---------------------------------- | -------------- | -------------------------------------------- |
| `jm-002` | transition (post given pre) | **+0.81**                          | +0.50 to +1.13 | letter sounds track **nonword** reading more |
| `jm-001` | levels (cross-sectional)    | **−0.49**                          | −0.98 to +0.02 | letter sounds track **word** reading more    |

**These point in opposite directions, and that should not be smoothed over.** Both are well-converged fits of the same children.

Three considerations explain the disagreement and why the baseline-conditional transition estimand is the more relevant one for the stated question; they do not turn it into a causal result.

**They answer different questions.** A levels model compares children at one moment, so its slope absorbs stable differences that make a child score highly on both measures at once, including possible general ability. A transition model conditions each post-period outcome on its own pre-period score, which can reduce that stable component without turning the outcome into a literal change score. For the narrower question of whether letter-sound level is more strongly associated with post-period nonword than word reading after baseline adjustment, the transition estimand is the relevant one.

**The factorised numerical cross-check agrees with `jm-002`.** Computing the same contrast from two separate single-outcome mechanism fits gives **+0.78** (89% +0.475 to +1.099), close to `jm-002`'s +0.81. The comparison table correctly marks that product-of-marginals calculation as _not_ an identified posterior contrast and `jm-002` as identified within its joint posterior. The agreement is reassuring only as a numerical cross-check: the fits use the same data and closely related likelihood, prior and adjustment assumptions, so they are not independent evidence.

**`jm-001`'s interval includes zero narrowly** (upper limit +0.02) and its estimate varies across waves from −0.49 to +0.06, so it is not a stable finding even on its own terms.

The defensible conclusion: **within the baseline-conditional transition model, letter-sound knowledge is more tightly associated with nonword decoding than with word reading.** That is compatible with a decoding-specific channel, but it does not identify one: residual confounding, differing measurement properties and the severe nonword-reading floor remain alternative explanations for the contrast.

## A caveat about `jm-002`'s machinery

`jm-002` needs two devices for extra variation: a child-level random intercept and a Beta-Binomial dispersion parameter. Both absorb "the data are more spread out than a simple model allows", and with only three rows per child they are hard to tell apart. For **nonword reading** — a 6-item, heavily floored measure — they are genuinely hard to separate: the two parameters have a posterior correlation of 0.36, and the child-level spread has an 89% interval running from 0.03 to 0.66, i.e. from almost nothing to substantial.

This model initially failed the convergence gate for exactly this reason and was refitted with far more simulation. That fixed the computation (effective sample size rose from 256 to 7,596) but, as expected, left the underlying uncertainty essentially unchanged. **The Δ contrast has a comparatively tight posterior conditional on this model; the nonword variance components are not well determined.** Power-scaling still flags `beta_N`, the Δ contrast and the residual correlation in `jm-002` for potential prior–data conflict; `jm-001` flags its nonword slope and Δ contrast as potential strong-prior/weak-likelihood cases. Those warnings qualify the scientific robustness even though they do not block publication. Do not quote the variance components.

## What these models cannot tell you

**Neither slope is causal.** Both remain vulnerable to residual confounding; the positive negative-control outcomes in the `mechanism` family show that letter-sound associations are not specific, with general ability one plausible explanation.

**A specific channel is not a teaching claim.** That letter sounds track decoding does not establish that teaching more letter sounds produces proportionally more decoding.

**Nonword reading is thinly measured.** Six items with a floor limits what any model can extract, and it bounds the precision of everything involving that outcome.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable. `jm-001` (per-wave levels, matched to the `concurrent` family's parameterisation) and `jm-002` (phase-stacked transitions, matched to `mech-096`/`mech-101`).
