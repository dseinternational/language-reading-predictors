# Difference-in-differences (waitlist-crossover) findings (2026-07-21)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

This is family note 04 in the set of per-family notes from the full 2026-07-21 re-fit of every Bayesian statistical model in the study (production `reporting` configuration, 6 chains × 6000 draws, 89% credible intervals). Read it alongside the shared index and reading guide, `notes/202607210900-findings-00-index-and-reading-guide.md`, which explains the study, the outcome measures and their item maxima, and the house rules for reporting a posterior. The **difference-in-differences (DiD)** family has **fourteen** models: eleven binary arm-by-wave models (one per skill, plus one word-reading variant) and three session-dose companions. **Thirteen pass the convergence gate cleanly; one — did-007 — is a divergence-only gate-review fit that is usable with the caveat noted below.** All data and models are preliminary.

## What this family probes

The trial is a **waitlist-crossover randomised design**: about 54 children with Down syndrome were randomly assigned either to start a reading and phonics intervention immediately or to start it after a wait, and were tested at four timepoints. The DiD family exploits that crossover structure directly. Each binary model jointly fits the bounded post-scores at three waves (t1, t2, t3) for one skill, on a proportion-correct scale (a Beta-Binomial likelihood with a logit link — that is, the model works in log-odds and we translate back to items), and estimates a separate **immediate-minus-waiting-list arm gap at each wave** plus a child random intercept. That random intercept partially pools stable between-child differences; it does **not** stand in for latent general ability and is not an exact fixed-effect control.

What it adds over the other families is that it reads the intervention effect _through the within-person crossover trajectory_ — the immediate arm pulls ahead at t2, then the waiting-list arm catches up once it too receives the intervention — rather than through a single post-baseline comparison (the ITT family) or a period-stacked gain model (the gain-factors family). It is a **longitudinal sensitivity analysis alongside the randomised ITT, not an independent experiment**: because it re-uses the same children and measurements, agreement between its t2 contrast and its ITT sibling is a parameterisation cross-check, not a replication.

The four wave-contrasts mean different things, and telling them apart is the whole point. Only the t2 contrast is randomised and causal; the rest are descriptive **associations**:

- The **pre-treatment (t1) arm gap** is a balance check — both arms are still untreated at t1, so it should sit near zero. It is not an effect.
- The **t2 immediate-minus-waiting-list contrast** (`tau_t2`) is the first post-baseline wave, both arms are still as-randomised (the waiting-list arm has not yet crossed over), so this is the **clean randomised (causal) contrast**. A positive value favours the intervention. This is one of only three genuinely causal quantities in the whole suite.
- The **t3 arm gap** compares two _already-treated_ arms with different exposure histories — an **association**, not a randomised effect.
- The **catch-up quantity** (`delta_crossover`, the t2 contrast minus the t3 gap) measures how much of the immediate arm's advantage has closed by t3 once the waiting-list arm has also been taught. It is a descriptive **association**, not a second treatment effect.

Three dose companions (did-006, did-007, did-107) reframe the model around session attendance, separating _being currently treated_ (a treatment-presence term) from _treated-centred session intensity_ (the dose slope), while adjusting for arm, period, the shared pre-randomisation t1 outcome and t1 age. Session count is **not randomised** — it reflects attendance and roll-out — so every dose coefficient is an **observational association**, never a proven lever.

## How to read these numbers

Full rules are in the index note; briefly: the point estimate is the posterior **median**; the uncertainty is the **89% equal-tailed credible interval** (the house standard — "there is an 89% posterior probability the value lies in this range"), with the inner **50%** interval as the "most-of-the-mass" band where the ground truth gives it; direction is the **tail probability** P(contrast > 0), read directly and never as a p-value. The evidence ladder attaches fixed labels to that probability — **inconclusive** (< 0.75), **suggestive** (≥ 0.75), **moderate** (≥ 0.91), **strong** (≥ 0.97), **very strong** (≥ 0.99) — describing the strength of evidence for a _directional_ claim, never the size of the effect. Direction and size are separate: a high probability says an effect is _probably positive_, not _large_. Effects are modelled on a proportion-correct scale and translated to **items** (probability × the test maximum), because "+3.5 of 32 letter sounds" is easier to grasp than a log-odds coefficient. For the two heavily-floored skills (PS, NW) the translated quantity is an **off-floor risk difference** — a change in the probability of _being_ off the floor at that wave, reported in percentage points (pp).

**Causal versus association for this family:** only `tau_t2` is causal (randomisation licenses it). The t1 gap is a balance quantity; the t3 gap, the catch-up quantity, and every dose coefficient are **adjusted associations** describing _who progresses_, not levers you can pull. Reading an association as if it were causal is the Table-2 fallacy. Residual confounding by latent general ability remains for all of them, and with roughly 54 children any estimate that just clears a threshold is on average magnitude-inflated (the winner's curse), so we lead with the interval, not the point.

## Per-model findings

### Binary arm-by-wave models (the randomised t2 contrast is CAUSAL)

The table gives each model's headline `tau_t2`, the causal t2 randomised contrast, as its items translation with the 89% credible interval, the tail probability and its evidence label. Item columns are per each test's maximum; PS and NW are off-floor risk differences in percentage points. "N" is the number of children. Rows are ordered strongest to weakest by evidence for a directional effect.

| ID      | Outcome (max items)                        | N   | τ_t2 headline — median [89% CI] | Tail probability | Evidence     | Gate |
| ------- | ------------------------------------------ | --- | ------------------------------- | ---------------- | ------------ | ---- |
| did-002 | LS — letter-sound knowledge (32)           | 54  | +3.5 items [+1.2, +5.8]         | P(>0) = 0.99     | very strong  | pass |
| did-003 | PA — phoneme blending (10)                 | 54  | +0.9 items [+0.1, +1.7]         | P(>0) = 0.96     | moderate     | pass |
| did-004 | TE — taught expressive vocab, block 1 (24) | 54  | +1.5 items [+0.0, +3.0]         | P(>0) = 0.95     | moderate     | pass |
| did-001 | WR — word reading (79)                     | 53  | +2.2 items [−0.3, +4.7]         | P(>0) = 0.92     | moderate     | pass |
| did-013 | WR — word reading, catch-up variant (79)   | 53  | +2.2 items [−0.3, +4.6]         | P(>0) = 0.92     | moderate     | pass |
| did-008 | TR — taught receptive vocab, block 1 (24)  | 54  | +1.2 items [−0.3, +2.7]         | P(>0) = 0.90     | suggestive   | pass |
| did-010 | LF — basic concepts, CELF (18)             | 54  | +0.6 items [−0.5, +1.8]         | P(>0) = 0.81     | suggestive   | pass |
| did-012 | NW — nonword reading (off-floor)           | 53  | +6 pp [−6, +18]                 | P(>0) = 0.79     | suggestive   | pass |
| did-011 | PS — phonetic spelling, SPPHON (off-floor) | 54  | +2 pp [−7, +12]                 | P(>0) = 0.65     | inconclusive | pass |
| did-009 | EV — standardised expressive vocab (170)   | 54  | +0.8 items [−4.0, +5.5]         | P(>0) = 0.61     | inconclusive | pass |
| did-005 | RV — standardised receptive vocab (170)    | 54  | −0.1 items [−5.1, +5.0]         | P(<0) = 0.51     | inconclusive | pass |

**Letter-sound knowledge, LS (did-002).** The strongest DiD result. At t2 — the randomised comparison — children in the immediate-intervention group scored about **+3.5 of 32 letter sounds** higher than the waiting-list children (89% credible range +1.2 to +5.8), with a 99% probability the true effect is positive: _very strong_ evidence of benefit. After the waiting-list children started the intervention, the gap between the arms narrowed by about 2.2 items — a descriptive catch-up association, not a second randomised effect, consistent with real catch-up once both arms are taught.

**Phoneme blending, PA (did-003).** The randomised t2 contrast is about **+0.9 of 10** (89% +0.1 to +1.7), with a 96% probability of being positive — _moderate_ causal evidence of benefit. The post-crossover gap narrowed by about 0.5 items (association).

**Taught expressive vocabulary, block 1, TE (did-004).** The randomised t2 contrast is about **+1.5 of 24** taught expressive words (89% +0.0 to +3.0), 95% probability positive — _moderate_ causal benefit. The gap then narrowed by about 1.8 items (association) — much of the t2 lead closes by t3, which is exactly what a directly-taught-word effect should do once both arms have been taught the same words.

**Word reading, WR (did-001).** The randomised t2 contrast is about **+2.2 of 79 words** (89% −0.3 to +4.7), 92% probability positive — _moderate_ causal benefit; the 89% interval just includes zero, so this is a clear-direction but not decisive signal. The post-crossover gap narrowed by only about 0.2 items (association) — the immediate arm largely holds its lead through to t3.

**Word reading, catch-up-heterogeneity variant, WR (did-013).** This is a variant of did-001 that adds an exploratory child-specific catch-up term; its overall estimand is labelled associational apart from the fixed t2 contrast. Its randomised t2 contrast is essentially identical to did-001 — about **+2.2 of 79** (89% −0.3 to +4.6), 92% probability positive, _moderate_ causal benefit — confirming the headline is stable to the extra structure. The child-level heterogeneity it estimates is an association, not an individual-causal-effect average; the ground-truth key-findings did not surface a numeric catch-up figure for this variant, so it is not quoted here.

**Taught receptive vocabulary, block 1, TR (did-008).** The randomised t2 contrast is about **+1.2 of 24** (89% −0.3 to +2.7), 90% probability positive — _suggestive_ causal benefit. The gap narrowed by about 0.8 items (association); the arms have largely converged by t3.

**Basic concepts, LF (did-010).** The randomised t2 contrast is about **+0.6 of 18** (89% −0.5 to +1.8), 81% probability positive — _suggestive_ causal benefit. The gap narrowed by about 0.5 items (association).

**Nonword reading, NW (did-012) — heavily floored, off-floor scale.** Being in the immediate-intervention group changed the chance of scoring above zero by about **+6 percentage points** relative to the waiting list (89% −6 to +18 pp), 79% probability positive — _suggestive_ evidence that the intervention raises the chance of coming off the floor. The off-floor gap then narrowed by about 2.9 pp (association). Floored outcomes are low-information; read the direction, not the magnitude.

**Phonetic spelling, PS (did-011) — heavily floored, off-floor scale.** Being in the immediate-intervention group changed the off-floor chance by about **+2 percentage points** (89% −7 to +12 pp), 65% probability positive — _inconclusive_. Here the off-floor gap _widened_ by about 1.4 pp (association) rather than closing. Exploratory and low-information.

**Standardised expressive vocabulary, EV (did-009).** The randomised t2 contrast is about **+0.8 of 170** (89% −4.0 to +5.5), 61% probability positive — _inconclusive_. The gap narrowed by about 1.1 items (association). A flat, uninformative result on this broad vocabulary test, not evidence of "no effect".

**Standardised receptive vocabulary, RV (did-005).** The randomised t2 contrast is about **−0.1 of 170** (89% −5.1 to +5.0) — essentially zero — with a 51% probability the true effect is _negative_: _inconclusive_ evidence of harm, i.e. no detectable signal in either direction. The gap widened by about 1.2 items (association). Flat and uninformative, as for EV.

### Session-dose companions (all coefficients are ASSOCIATIONS — dose is not randomised)

These three re-analyse attendance rather than assignment. None reports a causal treatment-effect headline: as the shared companion key-finding puts it, "this companion model estimates how outcomes vary with the amount of intervention received; that dose relationship is an observational association, not a randomised comparison, so no causal treatment-effect headline is reported."

- **Word reading, pooled dose (did-006).** WR outcome, 105 observations from 53 children over two phases. **Gate: pass.** The model separates being currently treated from treated-centred session intensity and reports the dose slope as an observational association. The ground-truth key-findings file surfaced only the generic companion statement above and not the numeric dose slope, so no dose figure is quoted here; the substantive point is that any session-intensity gradient it shows is observational, not a proven dose effect.

- **Letter-sound knowledge, period-resolved dose (did-007).** LS outcome, 107 observations from 54 children over two phases. **Gate: review — divergence-only.** This fit registers 14 divergent transitions out of 36,000 draws (about 0.04%, well under the ≤ 1% guidance); R-hat (maximum 1.001), effective sample size (minimum about 7,250) and BFMI all pass, so it is **usable with the divergence caveat noted**, and its clean-passing pooled comparator did-107 reproduces the same structure. Its `key_findings.json` was not generated, so the following come from the model's own `did_summary.csv` and are on the **log-odds scale**, all **observational associations**:
  - _Treatment-presence term_ (being currently treated): median +0.42 log-odds (89% +0.09 to +0.76; inner 50% +0.28 to +0.56), 98% probability positive — a _strong_ association. For letter-sounds the signal sits mainly in _being treated_ rather than in the intensive session margin.
  - _Transition-period term_: median +0.28 (89% +0.03 to +0.53; 50% +0.18 to +0.39), 97% probability positive — _moderate_; reflects general improvement across periods.
  - _Post-crossover arm term_: median +0.18 (89% −0.12 to +0.49; 50% +0.06 to +0.31), 83% probability positive — _suggestive_ association.
  - _Baseline-outcome persistence_ (the shared pre-randomisation t1 score): median +0.64 (89% +0.53 to +0.75; 50% +0.59 to +0.68), essentially certain to be positive (probability ≈ 1.00) — _very strong_; children's t1 letter-sound level strongly predicts their later level, as expected.
    The dose slope itself is described in the model's audit note as "an observational intensive-margin association"; a numeric value for it was not surfaced in the ground truth, so it is not quoted.

- **Letter-sound knowledge, pooled-slope comparator (did-107).** LS outcome, 107 observations from 54 children over two phases; the no-period-variation comparator to did-007. **Gate: pass.** It fits a single pooled dose slope instead of period-resolved slopes and reports the same generic companion finding. Its role is to confirm that the period-resolved fit adds no material structure; the ground truth surfaced only the companion statement and not the numeric slope, so no dose figure is quoted here. Because did-107 passes the gate cleanly and mirrors did-007's design, the letter-sound dose conclusion does not hinge on the flagged did-007 fit.

## What to take away

Read together, the DiD family gives a **coherent, code-focused crossover picture** that stands as a longitudinal sensitivity analysis alongside the randomised ITT. On the clean randomised t2 contrast the evidence is strongest for the code-related and directly-taught skills — _very strong_ for letter-sound knowledge (LS, about +3.5 of 32), _moderate_ for phoneme blending (PA), taught expressive vocabulary (TE) and word reading (WR), and _suggestive_ for taught receptive vocabulary (TR), basic concepts (LF) and off-floor nonword reading (NW). It tapers to _inconclusive_ on the heavily-floored phonetic spelling (PS) and on the two broad standardised vocabulary tests, receptive (RV) and expressive (EV), where the contrasts sit essentially on zero — a flat, no-detectable-signal result, not a demonstrated absence of effect. The **crossover dynamics are internally consistent**: the immediate arm pulls ahead at t2 on the taught and code skills, and by t3 the waiting-list arm has partly (LS, PA, TR) or largely (TE) caught up once it too is taught — visible in the positive catch-up quantities and shrinking t3 gaps, all of which are descriptive associations. The dose companions add an observational layer only: for letter-sounds (did-007/did-107) the association is carried by treatment presence rather than session intensity, and both are associations, not levers.

Two cautions frame all of this. First, only the t2 contrast is causal; the t1 gap is a balance check and the t3 gap, the catch-up quantities and every dose coefficient are adjusted associations subject to residual confounding by latent general ability — do not read them as levers. Second, with roughly 54 children the intervals are wide and point estimates that just clear a threshold are on average inflated, so the honest read is the interval and the direction, especially for the floored skills (PS, NW) and the flat broad-vocabulary results (RV, EV).
