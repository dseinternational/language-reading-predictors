> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Findings: the `joint_mechanism` family — is the letter-sound route specific to decoding?

**Read `findings-00-overview` first**, and ideally the `mechanism` note, which this family refines. Nothing here is causal.

## The data

**RLI trial only**, and the two models use it differently — which turns out to be the whole story.

- **`jm-001`** uses a **levels** design: one row per child at a single timepoint, run once per wave, 53 children. It asks who _is_ higher.
- **`jm-002`** uses a **transition** design: all period transitions stacked, 53 children contributing 153 rows, each row carrying the child's own starting score. It asks who _gains_ more.

## What the model is for

The `mechanism` family estimates one exposure-outcome slope per model. To ask whether letter-sound knowledge is _more_ tied to one outcome than another, you need both slopes from a single posterior — otherwise the comparison ignores that the same children produced both, and the difference has no honest interval.

This family fits word reading and nonword reading **together** and reports the contrast Δ = slope(letter sounds → nonword) − slope(letter sounds → word reading).

The logic is a convergent/discriminant argument. Nonword reading can _only_ be done by decoding — the words are invented, so sight recognition is unavailable. Word reading can be done either by decoding or by recognising familiar words. If letter-sound knowledge is specifically a decoding skill, it should track nonword reading **more** strongly, giving Δ > 0.

The two models also differ in machinery. `jm-001` uses a Binomial likelihood with a correlated residual between outcomes; `jm-002` uses a Beta-Binomial with a correlated child-level random intercept. Both need somewhere to put the extra variation, and each puts it in a different place — see the note below on why that matters.

## What was found — and the two models disagree

| Model    | Design                          | Δ = slope(→nonword) − slope(→word) | 89% range      | Reading                                      |
| -------- | ------------------------------- | ---------------------------------- | -------------- | -------------------------------------------- |
| `jm-002` | transition (conditional change) | **+0.81**                          | +0.50 to +1.13 | letter sounds track **nonword** reading more |
| `jm-001` | levels (cross-sectional)        | **−0.49**                          | −0.98 to +0.02 | letter sounds track **word** reading more    |

**These point in opposite directions, and that should not be smoothed over.** Both are well-converged fits of the same children.

Three things resolve which to believe.

**They answer different questions.** A levels model compares children at one moment, so its slope absorbs everything stable that makes a child score highly on both measures at once — chiefly general ability. A transition model conditions on where each child started and asks about subsequent change, which removes much of that stable component. For a question about a _channel_ — does knowing letter sounds help you decode — the change-based question is the relevant one.

**The independent estimate agrees with `jm-002`.** Computing the same contrast from two separate single-outcome mechanism models gives **+0.78** (89% +0.475 to +1.099), close to `jm-002`'s +0.81. The project's own comparison table marks the separate-model version as _not_ formally identified (it multiplies two marginals rather than using a joint posterior) and marks `jm-002` as identified — but the two agreeing is meaningful, because they share no fitting machinery.

**`jm-001`'s interval nearly includes zero** (upper limit +0.02) and its estimate varies across waves from −0.49 to +0.06, so it is not a stable finding even on its own terms.

The defensible conclusion: **letter-sound knowledge is more tightly linked to nonword decoding than to word reading, when the question is asked about change rather than about standing differences.** That is what a decoding-specific channel predicts, and it is the one result in the mechanism area that a general-ability confound does not readily explain — a confound lifting everything together would not make one outcome separate by this margin.

## A caveat about `jm-002`'s machinery

`jm-002` needs two devices for extra variation: a child-level random intercept and a Beta-Binomial dispersion parameter. Both absorb "the data are more spread out than a simple model allows", and with only three rows per child they are hard to tell apart. For **nonword reading** — a 6-item, heavily floored measure — they are genuinely hard to separate: the two parameters have a posterior correlation of 0.36, and the child-level spread has an 89% interval running from 0.03 to 0.66, i.e. from almost nothing to substantial.

This model initially failed the convergence gate for exactly this reason and was refitted with far more simulation. That fixed the computation (effective sample size rose from 256 to 7,596) but, as expected, left the underlying uncertainty essentially unchanged. **The Δ contrast is well determined; the nonword variance components are not.** Do not quote the latter.

## What these models cannot tell you

**Neither slope is causal.** Both inherit the general-ability confound documented in the `mechanism` note, where the negative-control outcomes came out clearly positive.

**A specific channel is not a teaching claim.** That letter sounds track decoding does not establish that teaching more letter sounds produces proportionally more decoding.

**Nonword reading is thinly measured.** Six items with a floor limits what any model can extract, and it bounds the precision of everything involving that outcome.

## Model inventory

Both models pass the convergence gate with zero divergences and are publishable. `jm-001` (per-wave levels, matched to the `concurrent` family's parameterisation) and `jm-002` (phase-stacked transitions, matched to `mech-096`/`mech-101`).
