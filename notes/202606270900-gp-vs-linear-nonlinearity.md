# What difference do Gaussian processes make? GP-vs-linear exploration (2026-06-27)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

**Question.** The gradient-boosting (GB) explore step suggests almost every
predictor→outcome relationship is non-linear. Do Hilbert-space Gaussian processes
(HSGPs) in the Bayesian models therefore capture something the linear predictors
miss — e.g. faster word-reading gains once letter sounds / vocabulary clear a
threshold? This note records the prior art and a fresh head-to-head on the current
models.

## Short answer

**No — at this sample size GPs do not improve the models, and can actively hurt.**
Replacing or augmenting the linear terms with GPs (a) leaves the causal estimands
unchanged, (b) does not improve out-of-sample fit (LOO ties or prefers linear), and
(c) reintroduces the Neal's-funnel divergences. For **every** baseline predictor of
word reading (letter sounds, expressive vocab, receptive vocab) the GP is strictly
worse: each is a **credible *linear*** positive effect (CrI excludes zero) that the
default GP `f_mech` dilutes into an uncertain curve — so the non-parametric model
actually *hides* real, linear signal at this sample size.

## Where non-linearity already lives (recap)

- **The link is always non-linear.** Linear predictor → logit → Beta-Binomial, so
  effects are non-linear in probability already; AME reporting evaluates each child
  at its own point on that curve.
- **HSGPs are wired in** (`hsgp.py`, 20-basis ExpQuad): ITT age & own-baseline,
  joint age, and the mechanism dose-response `f_mech`. `f_mech` is the **only GP on
  by default** (≈159 stacked period-observations make it identifiable); the ITT/joint
  GPs default **off**. Interactions are linear-only.
- **GB's job is to find shapes, not estimate them** — and a flexible learner on ~54
  points paints non-linearity onto noise (the project's GB replication found the gain
  models "near-noise / baseline-driven, permutation-importance noise-dominated").

## Prior art (LRP52 reporting-config sensitivity, notes 202604181445)

GP-on vs off on the ITT word-reading model: τ statistically identical (−0.409 vs
−0.406); GP-on gave 1.2–2.6 % divergences (η-amplitude → basis-weight funnel); **LOO
preferred linear** (ΔELPD 1.33, dse 0.51; stacking weight 1.00 vs 0.00); both GP
amplitude posteriors' 95 % CrIs touched zero (not identified). Conclusion at the
time: 53 children × 1 post-score cannot identify a 20-basis HSGP → GPs off for
ITT/joint.

## Fresh head-to-head (current models, reporting config; `experiment_gp.py`)

### The predictors of word reading: GP `f_mech` dose-response vs a linear slope
For each baseline skill → word-reading relationship, the GP (the default `f_mech`,
20-basis HSGP) is compared to a single linear slope on identical data. The linear
model wins **every** comparison (LOO stacking weight 1.00), the GP amplitude's lower
95% bound **touches zero in every case** (no identifiable curvature), and the GP
reintroduces divergences:

| relationship | linear slope `beta_mech` (95% CrI) | GP amplitude `f_mech__eta` | LOO weight (linear / GP) | ΔELPD (GP−linear) | GP div. |
|---|---|---|---|---|---|
| **Expressive vocab → W** (lrp57) | **+0.262 [0.124, 0.399]** ✶ | 0.306 [0.036, 0.706] | **1.00 / 0.00** | −3.0 (dse 1.9) | 25 |
| **Letter sounds → W** (lrp58) | **+0.204 [0.083, 0.325]** ✶ | 0.240 [0.016, 0.641] | **1.00 / 0.00** | −2.0 (dse 1.5) | 34 |
| **Receptive vocab → W** (lrp56) | **+0.161 [0.030, 0.294]** ✶ | 0.174 [0.011, 0.551] | **1.00 / 0.00** | −1.0 (dse 1.5) | 95 |

✶ = the linear slope's 95% CrI excludes zero. **All three baseline predictors are
*credibly, positively* associated with word reading when modelled linearly** — but
the default GP `f_mech` *dilutes each real signal into an uncertain curve whose band
spans zero* (which is exactly why the main reporting findings reported these couplings
as "suggestive, not established"). Here the GP is not merely unhelpful: it **costs
power** to detect relationships that are real and adequately linear, and it diverges.

### ITT word reading: residual own-baseline GP (added *on top of* the linear term)
| | τ (logit, 95% CrI) | divergences | LOO weight | ΔELPD vs best |
|---|---|---|---|---|
| **linear (lrpitt10)** | **+0.355 [0.040, 0.672]** | **0** | **1.00** | 0.0 |
| + residual own-baseline GP | +0.364 [0.046, 0.690] | 17 | 0.00 | −0.2 (dse 0.5) |

- The residual-GP amplitude `f_ypre__eta` = 0.149 **[0.007, 0.481]** — **touches zero:
  no residual curvature beyond the linear own-baseline term.**
- τ is unchanged (+0.355 → +0.364); the GP adds nothing to LOO (essentially tied) and
  brings back divergences. (An earlier *replace* variant — GP instead of linear age +
  baseline — gave the same verdict with 392 divergences and a large `f_ypre` amplitude
  that was just the GP carrying the ≈linear effect, not curvature.)

## Interpretation

1. **The data cannot support extra flexibility at this n.** Replacing linear with GP
   doesn't change the answer, doesn't predict better, and brings back the funnel.
2. **GB "non-linearity everywhere" is discovery on noise**, not a signal the Bayesian
   models are missing. The genuine non-linearity (the link) is already modelled.
3. **For the small-n mechanism questions, *linear is preferable to the default GP*** —
   it both fits as well or better (LOO) and is more powerful (recovers the credible
   L→W slope the GP misses). Worth reconsidering whether `f_mech` should default to a
   linear slope for these families, or at least be reported alongside it.
4. **The "joint threshold" hypothesis** (faster word learning once *both* letter sounds
   and vocabulary clear a threshold) would need a 2-D GP surface / tensor product —
   the most data-hungry option of all, and not worth attempting when 1-D GPs already
   fail to resolve curvature at this n.

## Caveats / follow-ups

- All conclusions are bounded by n ≈ 53 (ITT) / ≈ 159 (stacked mechanism); they say
  the data *cannot resolve* non-linearity, not that none exists.
- Two valid comparison framings give the same verdict: "linear vs flexible for the same
  predictor" (this note) and "residual curvature beyond linear" (LRP52 add-on).
- Concrete follow-up: consider a `linear_mechanism` default (or dual report) for the
  mechanism family; keep ITT/joint GPs off; revisit GPs only if n grows materially.
