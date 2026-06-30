# Bayesian moderated nonlinear factor analysis for pooling disparate language, reading and memory measures

<!-- cspell:ignore nu lambda eta varepsilon beta gamma alpha psi exp neq qquad quad operatorname logistic mid Rightarrow GPT Coxe Zulauf McCurdy Pettit crosswalk crosswalks -->

::: {.callout-note}
Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).
Substantially revised by an LLM-based AI tool (Codex/GPT-5).
:::

::: {.callout-warning}
This note was prepared by an AI tool and may contain mistakes. It advances
statistical proposals; treat the methodological and statistical claims as a
starting point for review, not as settled fact. Verify the cited papers against
the primary sources before any claim enters the report.
:::

**Status: draft for discussion.** This is a forward-looking methodology note
proposing — not reporting — a prospective measurement approach. It develops the
idea of using **Bayesian moderated nonlinear factor analysis (MNLFA)** to place
measures from disparate instruments, studies and populations onto common latent
skills, so that this project and others' data can be combined into more
informative models of how language, reading and memory skills interact. Nothing
here has been fitted; the note's job is to explain the method from the ground up,
say honestly what it would and would not buy us, and propose a staged way to
test feasibility.

_File paths below are relative to `src/language_reading_predictors/`, except those
under `scripts/`, `notes/` or `docs/`._

## 1. Purpose and the three questions

The project owner has asked us to explore three connected questions. They share a
single root: our scientific reach is currently limited by the fact that the most
interesting data live in _different instruments and different datasets_, and we
have no principled way to put them on one ruler.

- **Q1 — Could parent-report checklist vocabulary (the vocabulary-growth, "VG",
  CDI style: a parent ticks which of N words their child understands or says,
  recorded as a count "k of N") be combined with the broader range of _direct_
  assessments in this project (LRP: tests a researcher administers face to
  face — reading, grammar, picture vocabulary, repetition)?**
- **Q2 (the key question) — Could the same methodology let us incorporate _other
  researchers'_ datasets, which use varying measures of language, reading and
  memory, to yield more informative models of how different skills interact?**
- **Q3 (implicit) — Does combining measures this way de-attenuate and enrich our
  models of how skills interact (mediation and moderation), and — just as
  important — what does it _not_ fix?**

The short answers, defended below, are: **Q1 yes, with a co-administration
caveat; Q2 yes in principle and this is exactly what the method was built for,
but strictly conditional on having bridging/anchor data or credible external
calibration; Q3 yes for measurement-induced attenuation, but it does nothing for
causal confounding, temporal ordering or genuine construct mismatch.** The rest
of the note earns those answers and connects the proposal to machinery we already
run.

## 2. The problem: you cannot just pool the scores

Suppose we want one analysis of _how early language relates to later reading in
children with Down syndrome_, using as much of the world's data as we can find.
We have VG checklists; we have the LRP direct battery; other groups have their
own studies with their own tests, often in another language. The naïve move is to
stack the raw scores — line up everyone's "vocabulary number" in one column. This
is almost always wrong, and seeing _why_ motivates everything that follows.

- **Different rulers.** A score of 40 on a 50-word parent checklist is not the
  same quantity as 40 correct on the 170-item EOWPVT expressive naming test
  (`eowpvt`). Even two tests that both claim to measure "vocabulary" carve up the
  skill at different points.
- **Different difficulty and different content.** A checklist of common early
  words mostly separates children at the low end; a demanding naming test mostly
  separates children further along. The _same_ child looks different depending on
  which test they happened to take — and which test they took is confounded with
  which study they were in.
- **A raw total throws away how informative each item is.** Summing ticks treats
  every word as equally diagnostic. Some words reveal far more about vocabulary
  than others; sum and mean scores ignore this, whereas modern psychometric
  scoring weights items by how much they actually reveal (Gottfredson et al.
  2019).
- **Floors and ceilings.** Several of our measures lack usable range in this
  population. Our own measurement-sensitivity audit
  (`notes/202606171000-measurement-sensitivity-audit.md`;
  `scripts/measurement_audit.py`) flags phonetic spelling (`spphon`) and nonword
  reading (`nonword`) as detection-limited/floored, and the floor-handling suite
  (`statistical_models/floor.py`) already treats them specially. A pooling
  argument must respect that, not paper over it.
- **The groups genuinely differ.** Children with Down syndrome, typically
  developing children, different ages and different test languages may genuinely
  differ in the skill — and _that difference is the thing we want to estimate_.
  We must not let it be contaminated by, or confused with, artefacts of the
  measuring instrument.

So the harmonisation problem is: _how do we place scores from disparate
instruments, studies, languages and ages onto one comparable scale for an
underlying skill — without erasing the real differences we care about, and
without being fooled by measurement?_ This is precisely the problem that
**integrative data analysis** (IDA — pooling several studies' participant-level
data into one analysis) was built to confront (Curran and Hussong 2009; Bauer and
Hussong 2009), and MNLFA is its measurement engine (Curran, McGinley, Bauer et
al. 2014).

## 3. A primer, building up to MNLFA

Readers unfamiliar with factor analysis, item response theory or Bayesian latent
models should be able to follow this section. We build the name "Bayesian
moderated nonlinear factor analysis" one word at a time, intuition first.

### 3.1 Latent variables and factor analysis: the skill is hidden; tests are noisy windows

The first move is to stop treating a test score _as_ the skill and start treating
it as _evidence about_ the skill. Think of "expressive vocabulary" as a real but
unobservable quantity inside each child — a **latent variable** (or **factor**),
written η ("eta"). We never see η. We see **indicators**: the child's responses
on actual tests. Factor analysis says each indicator is a noisy, biased window
onto η, with three numbers per indicator. For a continuous, standardised score
_y_:

$$y = \nu + \lambda\,\eta + \varepsilon$$

- **ν (intercept)** — where the indicator sits when the skill is at its reference
  level: the test's _baseline easiness_. Two tests of the same skill differ here
  simply because one is scored more generously.
- **λ (loading)** — how strongly the indicator tracks the skill, the "volume
  knob". A high loading is a sharp, discriminating window; a low loading is a
  foggy one.
- **ε (residual)** — everything in this test that is _not_ the shared skill:
  item quirks, guessing, a bad day. Its variance is how much fog is on the glass.

The idea that makes pooling possible: **several different indicators can load on
the same factor.** A checklist count, a naming score and a sentence-repetition
score can each be its own (ν, λ, ε) view of _one_ expressive-language η. We then
compare children on the estimated η — the common skill behind all of them —
rather than comparing checklist numbers to naming numbers. The factor model is a
principled translator between instruments. We can also have several correlated
factors at once — a vocabulary factor, a "code" factor for letter-sounds and
blending, a grammar factor — which is exactly what LRPMM01 already does within one
study (its three correlated factors are vocabulary, measured by _both_ the
receptive and expressive picture-vocabulary indicators; code; and grammar).

### 3.2 Measurement invariance and DIF: the translation must behave the same way for everyone

The catch the whole field organises around: comparing children's η is only fair
if _the windows behave the same way for everyone_. If a test has a different
intercept or loading in one group than another, a difference in the raw indicator
no longer cleanly reflects a difference in skill — part of it reflects the
_instrument_ behaving differently. We would be measuring with a ruler whose
markings change depending on who holds it.

The property we need is **measurement invariance**: the link from skill to
indicator (the νs and λs, ideally the residual variances) is the same across the
groups we compare. Meredith (1993) gave the classic hierarchy — _weak_ (equal
loadings), _strong_ (equal loadings and intercepts), _strict_ (also equal
residuals) — and showed you need at least strong invariance before comparing
latent means is fair; Millsap (2011) is the standard treatment. Strong invariance
is the bar this note actually needs — it is what licenses comparing latent means;
strict invariance is stronger than required here.

When invariance fails for an item it shows **differential item functioning
(DIF)**: it functions differently across groups even for children at the same
true skill.

- **Intercept (uniform) DIF** — the item is systematically easier/harder in one
  study/language/group at _every_ skill level. A vocabulary word common in
  British English but rare in another language's checklist: equally-skilled
  children tick it at different rates because of the word, not the skill.
- **Loading (non-uniform) DIF** — the item _discriminates_ differently. An item
  that separates high from low skill well in typically developing children but
  sits near the floor for the Down syndrome sample, so it barely discriminates
  there.

Ignore DIF and pool anyway, and the artefacts leak straight into the science:
apparent group differences that are really translation differences, distorted
trajectories, spurious correlations. The honest options are (a) discard
non-comparable items, wasting hard-won data, or (b) _model_ the
non-comparability. MNLFA takes route (b) — that is what "moderated" buys.

### 3.3 "Moderated": let the measurement parameters depend on covariates

Traditional invariance testing handles one categorical grouping at a time and
asks a yes/no question. Our reality is several covariates at once — study, age
(continuous), diagnostic group, test language — with partial, graded
non-invariance. The **moderated** in MNLFA lets the measurement parameters
themselves be functions of covariates **x**:

$$\nu_j(x) = \nu_{j0} + \beta_j\,x \qquad (\beta_j \neq 0 \;\Rightarrow\; \text{intercept DIF})$$

$$\lambda_j(x) = \lambda_{j0} + \gamma_j\,x \qquad (\gamma_j \neq 0 \;\Rightarrow\; \text{loading DIF})$$

and — crucially — lets the latent variable's own mean and variance depend on
covariates too:

$$\alpha(x) = \alpha_0 + c_\alpha\,x \quad(\text{mean}), \qquad \psi(x) = \exp(\psi_0 + c_\psi\,x) \quad(\text{variance; } \exp \text{ keeps it positive})$$

The division of labour is the whole point. The **measurement** block
(ν_j(x), λ_j(x)) is where DIF lives — non-invariance is no longer a
disqualification but a _parameter we estimate_. The **structural** block
(α(x), ψ(x)) is where the _science_ lives — α(x) is how true skill shifts with
age, group and study (the developmental trajectory and the group contrast we
actually care about); ψ(x) is how its spread changes. Because measurement DIF and
true group differences are now _separate parameters in one model_, MNLFA
estimates genuine skill differences _while_ accounting for instruments behaving
differently. Bauer (2017) shows formally that MNLFA subsumes and unifies the two
older invariance frameworks — multiple-groups (every parameter may vary, but only across
one categorical grouping) and MIMIC (covariates may be continuous or categorical,
but in its standard form DIF enters only as covariate effects on item intercepts
and on the latent mean, not on loadings) — recovering both as special cases while
removing both restrictions.

This is where MNLFA meets our stack: age already enters our models as a smooth
covariate; here it moderates both the latent mean (the growth curve, α(age)) and,
if needed, item behaviour. The "study" covariate is how two researchers'
different instruments live in one model without pretending they are the same
instrument.

### 3.4 "Nonlinear": two senses, both relevant

**Sense A — a nonlinear link, because real indicators are not continuous.** The
clean line _y = ν + λη + ε_ assumes a continuous, roughly normal score. But CDI
data are _binary at the item level_ (does the child say this word: yes/no) and
_count-shaped in aggregate_ (k of N). Reading and memory tasks are often
correct/incorrect items. For a binary item we pass η through an S-shaped
(logistic) link:

$$P(\text{item } j = 1 \mid \eta, x) = \operatorname{logistic}\!\big(\lambda_j(x)\,\eta - \nu_j(x)\big)$$

If you know **item response theory (IRT)**, this _is_ IRT: the loading λ is the
item's _discrimination_ and ν is its negative intercept. (We write the link as a
subtraction here, rather than the addition of Section 3.1, so that a larger ν
reads as a _harder_ item — the conventional IRT direction.) The item's
_difficulty_ — the skill level η at which a correct response becomes 50% likely —
is ν/λ, and equals ν only for an item whose loading is exactly 1. The pay-off is
unification: the linear factor model and IRT are not rival traditions but two
link choices in one family. Continuous indicators get an identity link, binary/
ordinal indicators get logistic/cumulative links, counts get a count link (Bauer
and Hussong 2009; Curran et al. 2014; Curran, Hussong, Cai et al. 2008). This
matters enormously for us, because **our existing VG measurement model is already
in this family**, but the strength of that claim depends on what we have. **If
VG/CDI responses are available at item level**, each word can be a Bernoulli IRT
indicator with its own difficulty, discrimination and possible DIF. **If we only
have aggregate `k of N` checklist totals**, the Beta-Binomial likelihood is a
bounded-count, overdispersion-robust indicator of the latent skill, not an
item-level IRT model: it cannot learn word difficulties, discriminations,
item-level DIF or anchor-item invariance. It is still in the same broad
latent-measurement family, but it is a weaker bridge. Moving to MNLFA is therefore
a _generalisation of what we already do_, not a rewrite: keep the Beta-Binomial
leg for checklist totals where that is all we have; use item-level links where
item responses are available; let direct-assessment items or totals enter through
their own links onto the _same_ latent skills.

**Sense B — nonlinear, smooth moderation.** Section 3.3 wrote moderation as
linear in **x** (α_0 + a·age). But a vocabulary trajectory over age is not a
straight line — it accelerates then decelerates. We want to _learn_ the curve,
not impose it. This is where our existing **Hilbert-space Gaussian process
(HSGP)** machinery slots in:

$$\alpha(\text{age}) = \alpha_0 + f(\text{age}), \qquad f \sim \text{smooth Gaussian-process prior (HSGP-approximated)}$$

The same trick can let an item's difficulty drift _smoothly_ with age rather than
jumping. So "nonlinear" in MNLFA, read through our stack, means nonlinear links
for non-continuous indicators (Sense A) _and_ smooth GP-based moderation of the
latent trajectory and, where warranted, the item parameters (Sense B). The HSGP
curves we already fit become the moderation functions of an MNLFA.

### 3.5 The assembled generative model

Putting the rungs together, here is the data-generating story for each child _i_
and each item _j_ they were administered — read top to bottom as "how the data
came to be":

1. **Covariates.** Child _i_ has known covariates **x**_i: study, age, group
   (Down syndrome / typically developing), test language.
2. **True skill.** η_i ~ Normal( α(**x**_i), ψ(**x**_i) ), where α carries the
   HSGP age-trajectory and the group/study shifts and ψ lets the spread vary.
3. **Item behaviour.** Each administered item _j_ has covariate-dependent
   measurement parameters ν_j(**x**_i), λ_j(**x**_i) — the same word/test item
   may be easier/harder or sharper/blunter by study, language and age (modelled
   DIF).
4. **Response**, generated from η_i through the link appropriate to the observed
   indicator: checklist item tick → Bernoulli/logistic when item-level responses
   exist; checklist total → Beta-Binomial(k of N) with overdispersion (our
   existing bounded-count leg, but not item-level DIF); binary direct item →
   Bernoulli/logistic (a 2-parameter IRT item, richer still once its parameters
   are covariate-moderated); ordinal rating → cumulative/graded link; continuous
   score → identity with residual ε.

Two ingredients make this fit our setting. First, the **anchoring/bridging**
precondition: identifiability requires that _something_ be shared across studies —
overlapping measures (the same or a translated instrument in more than one study)
or overlapping people/items — so the model can tell genuine skill differences
apart from instrument differences. Without at least some anchor indicators/items
whose measurement behaviour is constrained or assumed sufficiently invariant, the
latent scale is not pinned down, and "study differences" and "skill differences"
are not separately estimable. This is the substantive entry ticket for any
third-party dataset, not a software detail. Second, once fitted, each child gets
not a point score but a **posterior distribution** over η_i — a calibrated
statement of how much we know about that child's skill given the indicators they
took, with wider and more model-dependent posteriors when a skill is only weakly
measured. That distribution is the bridge to the downstream science (Section
3.6c).

### 3.6 Why Bayesian

Everything above can be fitted by maximum likelihood, and the MNLFA literature
largely does. We propose to fit it _Bayesianly_, on our existing PyMC/nutpie
stack. Each reason solves a concrete problem this project has.

**(a) Priors as partial pooling — "approximate invariance", freeing only the DIF
the data demand.** Section 3.3 let _every_ item's intercept and loading drift with
_every_ covariate — a huge number of DIF parameters, most of which should be near
zero (most items probably _are_ roughly invariant). Estimating them all freely
overfits; testing them one-by-one is fragile and does not scale. The Bayesian
answer is a _shrinkage prior_ that pulls DIF parameters towards zero unless the
data insist — **approximate, rather than exact, invariance**. A sparsity-inducing
**horseshoe** prior (Carvalho, Polson and Scott 2010) keeps almost all DIF tightly
at zero (clean, comparable items) while letting a genuinely biased item escape to
its true value. This is the Bayesian counterpart of regularised DIF detection for
MNLFA (Bauer, Belzak and Cole 2019; Belzak and Bauer 2020) and of the
**alignment** philosophy — estimate group factor means/variances _without_
exact invariance, tolerating small misfit (Asparouhov and Muthén 2014). (Horseshoe
shrinkage and alignment are mechanically different — one a sparsity prior, the
other a post-hoc rotation to a simplicity criterion — but they share the goal of
comparable latent scales without insisting on exact invariance.) In our hands:
pool studies aggressively, but let the data carve out the specific items that
genuinely behave differently in another language or group, automatically, without
a brittle sequence of significance tests. Items the prior frees as DIF are
_selected_ by that procedure, so they should be reported as exploratory, not as
confirmed non-invariance.

**(b) Full-posterior uncertainty in latent scores, propagated downstream — the
de-attenuation point.** This matters most for the scientific questions. In a
two-step "score then regress" workflow you estimate each child's η as a point,
then feed those points into a downstream model of how skills interact (mediation,
moderation, the language → reading relationships LRP cares about). But the ηs are
_estimated_, not observed; treating noisy estimates as exact values biases and
distorts the very interaction and mediation effects we are trying to quantify —
classically _towards_ zero for a single error-laden predictor, but in either
direction once several correlated latent predictors and their interactions are
involved, which is exactly the mediation/moderation structure we care about. The
Bayesian route fits the measurement model and the structural model _jointly_, so
the _whole posterior_ of each η — its uncertainty, not just its mean — flows into
the downstream estimates. LRPMM01 already does measurement-error-corrected factor
→ gain slopes for exactly this reason; doing it in one joint model means
measurement-error correction and uncertainty propagation are handled coherently
rather than bolted on. This is not automatic: de-attenuation depends on the
measurement model and anchors being defensible, and propagating uncertainty can
widen intervals when the bridge is weak.

**(c) Graceful behaviour at small n.** This sample is ~54 children. Maximum-
likelihood factor models with many DIF parameters are exactly where ML
misbehaves — non-convergence, boundary estimates (negative variances, loadings
pinned at extremes), wildly uncertain estimates reported as precise.
Weakly-informative priors regularise these away: variances stay positive by
construction, implausible values are gently down-weighted, and where the small
sample cannot determine a parameter the posterior _says so_ (it stays wide)
rather than the optimiser inventing false certainty. The Bayesian IRT/factor
literature documents this stabilising behaviour and its workflow — prior
predictive checks, posterior predictive checks, prior sensitivity (Bürkner 2021).

**(d) Natural fusion with the stack we already run.** None of this needs a new
toolchain. MNLFA's aggregate-count measurement leg for checklists _is_ our
Beta-Binomial bounded-count likelihood with overdispersion; its smooth latent
age-trajectory and possible smooth item moderation are our HSGP-over-age; its
sampler is the nutpie NUTS we already use; its priors and partial pooling are the
same idioms as the existing LRP correlated-factor and latent-change-score models.
Bayesian MNLFA is best understood not as importing a foreign method but as
_naming and generalising the measurement model the two projects have already
converged on_.

## 4. What this buys us, by question

### Q1 — Combine CDI checklists with LRP direct measures

Yes. Both become _typed indicators_ of shared latent language factors: the CDI
"words understood" count is a Beta-Binomial(k of N) indicator of a _receptive
vocabulary_ η; the CDI "words spoken/signed" count an indicator of _expressive_
η; the direct receptive (`rowpvt`) and expressive (`eowpvt`) picture-vocabulary
tests, TROG-2 grammar (`trog`), the Action Picture Test, and the taught/not-taught
block tests load on the same factors through their own links. "Instrument" enters
as a covariate, with DIF _modelled_ rather than assumed away.

::: {.callout-important title="The precondition for Q1"}
The precondition for Q1 is **bridging information linking checklist and direct
measures**. The cleanest bridge is co-administration: some children with _both_
a CDI count and a direct vocabulary score, so the model can learn how the two
instruments map onto the same η. Without any such overlap — and the
Burgoyne (2012) RCT sample did not collect parent CDI checklists — the
checklist scale and the direct scale are only weakly tied (through shared
covariates and strong prior assumptions), and the harmonisation rests on
assumptions we cannot check. This is a data-collection implication, not just a
modelling choice.
:::

### Q2 — Incorporate other researchers' datasets (the key question)

This is the central use case MNLFA was built for: **integrative data analysis**
across studies whose instruments are _not identical_. The logic is the same as
Q1, scaled up. Different teams measured the _same underlying skill_ with
_different tests_ — receptive vocabulary via BPVS in one study, ROWPVT or PPVT in
another; verbal short-term memory via forward digit span in one, nonword
repetition in another. We stop treating the test score as the quantity of
interest and treat each test as a _noisy indicator of a latent skill on a common
scale_. If two instruments load on the same latent factor, they can sit on one
ruler _even when no child took both_ — but only through a measurement link; a
common factor _label_ is not enough on its own.

Linkage can come from several sources, roughly from strongest to weakest:

1. **Common items or measures** — the same item or instrument appears in more than
   one study, so a subset of indicators can be constrained as anchors.
2. **Common respondents / co-administration** — some children completed both
   instruments, allowing the model to learn the scale relationship directly.
3. **Equivalent item-content clusters** — items are not identical but are judged
   to measure the same behaviour or skill. These are useful but weaker than true
   common items, because their equivalence is a substantive assumption.
4. **External calibration, crosswalks or random-equivalent groups** — a separate
   validation sample, published crosswalk, or defensible common-distribution
   assumption supplies the link. This is calibration evidence, not a link learned
   from the pooled studies alone.

Under a connected linking design, **no study need administer every measure**.
Provided the _web_ of instruments is connected (every study links to the pooled
set, and the union of overlaps forms one connected graph), the latent _scale_ can
be shared even though each child contributes only the items their study used. But
a child's factor value for a skill they did not actually measure is mostly
imputed from covariates, factor correlations and priors; it should not be
described as measured with the same evidential weight as an observed skill.
Missing indicators are handled by the likelihood — full-information maximum
likelihood (FIML) in the frequentist tradition; in our Bayesian/PyMC setting,
naturally, by masked likelihood legs over only the observed indicators, the _same
masking pattern LRP67 already uses_.

The hard constraint, stated plainly: **at least one trustworthy shared anchor
indicator per skill must connect each study to a common core, and some indicators
must be constrained or assumed invariant to identify DIF on the rest.** You
cannot let every parameter of every item vary with study and still recover where
cohorts truly sit. Sparse overlap also limits what can be checked: with one
anchor, its invariance is an assumption; several independent anchors are needed
before the data can meaningfully challenge that assumption. Content-equivalent
clusters and external crosswalks can help, but they should be labelled as
calibration assumptions unless they are backed by common-person or common-item
data. The harmonisation literature is explicit that **sparse overlap is the
binding limitation** (Howe et al. 2024): when only a handful of items bridge
studies, or anchors are themselves contaminated by DIF, harmonisation degrades and
anchor selection becomes the crux. The canonical worked templates are the
depression-measure harmonisation tutorial of Zhao et al. (2022) and the
alcohol/substance-use IDA consortia, automated in the aMNLFA R package
(Gottfredson et al. 2019). Pooling cohorts that used _different depression
scales_ onto one latent metric is structurally identical to pooling cohorts that
used _different vocabulary/memory/reading tests_.

Two further practical choices for the note's record:

- **One-stage vs two-stage.** Two-stage (the aMNLFA workflow) fits the
  measurement model, exports factor scores, then runs the substantive model —
  simpler, but treats scores as known and under-propagates uncertainty unless
  plausible values / multiple imputation are carried through carefully. One-stage
  / simultaneous (Bauer 2017) estimates measurement and structural models
  _jointly_ so factor uncertainty flows through. The one-stage route is preferable
  for our scientific purpose but heavier and more fragile at this sample size; in
  PyMC, latent skills, harmonised item parameters and the mediation/moderation
  regressions can all be nodes in one graph. This is the de-attenuation mechanism
  of Q3, conditional on a defensible measurement model.
- **Pooled vs federated.** Classic IDA needs item-level data physically pooled in
  one place — often impossible with third-party data for governance/consent
  reasons. The **federated** alternative is not a plug-in replacement: for a
  hierarchical Bayesian MNLFA with DIF and sparse overlap, simple sufficient
  statistics usually will not exist. A federated route would need a separate
  design — for example site-local likelihood contributions, secure shared
  computation, or carefully combined posterior/plausible-value summaries — and
  would trade efficiency and diagnostic transparency for not moving raw data. It
  should be flagged early in any third-party conversation, not assumed solved.

### Q3 — De-attenuation and enrichment, and the explicit non-fixes

Joint Bayesian fitting propagates latent-score uncertainty into the
mediation/moderation estimates. Where the measurement model is right and the
anchors are strong, this can reduce _measurement-induced_ attenuation and may
improve precision; where the bridge is weak, it should instead expose the
uncertainty by widening the posterior. State plainly what it does **not** do:

- It does **not** remove **causal confounding.** If an unmeasured common cause
  drives both predictor and outcome, sharper measurement gives a more precise
  estimate of a still-confounded association — and a more _precise_ confounded
  estimate is arguably more dangerous than a noisy one, because it is easier to
  over-interpret. MNLFA improves the _measurement_
  leg of the LRP causal apparatus (the locked DAG, the mediation g-formula, the
  moderation analyses); the identification assumptions that license a causal
  reading remain entirely that apparatus's job. Per the project's standing
  honesty convention, only the randomised ITT term is causal; everything
  observational stays an _adjusted association_, never "X drives Y".
- It does **not** establish **temporal ordering.** Better-measured cross-sectional
  associations are still cross-sectional. The CLPM cautions in
  `notes/202606201117-longitudinal-modelling-stance-clpm.md` still apply.
- It does **not** rescue a **construct mismatch.** If two tests labelled
  "vocabulary" actually tap different things, forcing them onto one η does not
  make them commensurate — it averages over a real difference. Anchoring assumes
  the anchors measure the _same_ construct; that is a substantive claim to defend,
  not a free lunch. The first model to try may be one factor, but the first model
  to trust may be multidimensional, bifactor, second-order, or include
  instrument-specific method factors if the instruments share a broad domain but
  differ in content, response format or administration mode.
- It does **not** conjure information that floored measures never captured.
  De-attenuation cleans up measurement _noise_; it cannot recover discrimination
  a detection-limited test never had. The existing floor apparatus
  (`statistical_models/floor.py`: off-floor Bernoulli primary estimands,
  probability-scale ROPEs for `spphon`/`nonword`) must be respected within the
  measurement model, not bypassed by it.

## 4b. Memory specifically: the clearest case for cross-dataset harmonisation

Memory is both the most scientifically central gap and the cleanest test case for
Q2, for two reasons.

**The DS phenotype makes a correlated, multi-construct memory model essential.**
Down syndrome shows a _specific, disproportionate deficit in verbal short-term
memory_ (the phonological loop), against relatively _spared visuospatial_
short-term memory (Jarrold, Baddeley and Hewes 1999; Jarrold, Baddeley and
Phillips 2002; Purser and Jarrold 2005; Baddeley and Jarrold 2007). The deficit is
specific to verbal material and not attributable to hearing or speech-motor
difficulty (Jarrold et al. 2002), and is a capacity limitation rather than rapid
decay (Purser and Jarrold 2005). Crucially, verbal short-term memory is
mechanistically implicated in _learning_: the phonological loop is a
language-learning device that stores novel word-forms while lexical records are
built (Baddeley, Gathercole and Papagno 1998), so a verbal-memory bottleneck
throttles vocabulary growth — and, downstream, reading. Nonword repetition (the
purest phonological-memory marker, with word repetition as a speech/perceptual
control; Laws 1998) predicts later vocabulary and grammar in DS over a five-year
follow-up (Laws and Gunn 2004), and DS reading deficits track vocabulary and
phonological awareness, not decoding per se (Næss et al. 2011; Næss 2016) —
echoing Burgoyne et al. (2012), where receptive language, not phoneme awareness,
predicted reading growth. This is why the right object is a _correlated
multi-construct latent model_ — verbal STM distinct from a relatively spared
visuospatial STM, with the DS-vs-TD contrast expressed as a factor-mean
difference (a moderation MNLFA handles natively), and memory → vocabulary →
reading entering as error-corrected _paths_, which is the science.

**Memory is exactly where our own coverage is thinnest — and where pooling pays
off.** LRP's memory coverage is verbal-only: the Early Repetition Battery word
(`erbword`), nonword (`erbnw`) and total (`erbto`) repetition, characterised in
the repo as indexing verbal/phonological short-term memory. There is **no digit
span, no Corsi/visuospatial span, and no explicit working-memory task** anywhere
in `data_variables.py` or `measures.py`; and the ERB measures currently live only
in the exploratory gradient-boosting layer (LRP25-30), not the Bayesian
`measures.py` MEASURES dict. So a third-party dataset that carries forward digit
span, a nonword-repetition variant, or Corsi/visual span could do two things at
once: provide an **anchor** (nonword repetition is a plausible shared indicator
bridging studies — though it is itself phonologically and linguistically loaded
and can behave differently across languages and task versions, so its invariance
as an anchor must be _tested_, not assumed) and _extend_ the latent space. A
digit-span study and a nonword-repetition study can both inform the same
verbal-STM trajectory only if they are connected by shared anchors or by enough
cross-construct overlap; where a child has no memory indicator, their memory
factor is imputed from the model, not directly measured. A Corsi study adds a
separate, weakly-correlated visuospatial factor that is more language-neutral but
taps the distinct, relatively spared construct. That is harmonising
_different memory tests onto one verbal-STM metric_ — structurally the depression-
harmonisation template — and it is what would finally let memory enter the
interaction/mediation models that LRP alone cannot support. (The exploratory
LRP25-30 work exists precisely to decide whether the shared DAG needs a
verbal/phonological STM _node_; MNLFA pooling is how that node could be measured
well enough to use.)

## 5. Relationship to our existing models and shared backbone

MNLFA is not a departure; it is the convergent generalisation of three things we
already have.

| Existing piece                                                                                          | What it already does                                                                                                                                                                                                                                                    | What MNLFA generalises                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LRPMM01** (`statistical_models/factories.py::build_correlated_factor_model`; note `202606291700-...`) | Correlated domain factors {vocabulary (receptive + expressive indicators), code, grammar} with an LKJ correlation, positive loadings, communalities (each indicator's variance share explained by its factor), and **measurement-error-corrected factor → gain slopes** | The _within-study_ multi-factor measurement model. MNLFA adds moderated (study/age/group/language) item parameters and pools _across_ studies/instruments.                                     |
| **LRP67** (`...::build_lcsm_model`; note `202606191100-...`)                                            | Latent logit true-score `x_m[i,t]` per measure/wave; **masked Beta-Binomial measurement leg** whose `kappa` is "measurement overdispersion, distinct from process noise"                                                                                                | Already _separates a measurement layer from a structural layer_ and _masks over observed indicators only_. That is the seed of an MNLFA measurement model and the missing-by-design mechanism. |
| **Measurement-sensitivity audit** (`scripts/measurement_audit.py`; note `202606171000-...`)             | Descriptive per-measure floor/ceiling, range used, dispersion, mover fraction; flags detection-limited measures; "cannot estimate reliability from single summary scores"                                                                                               | The exact limitation an MNLFA item/latent layer is _designed_ to address — reliability and DIF become estimable once items load on a latent skill.                                             |
| **HSGP + Beta-Binomial + nutpie stack**                                                                 | Smooth age effects; overdispersion-robust k-of-N likelihood; the sampler and workflow                                                                                                                                                                                   | Become, respectively, the _moderation functions_, the _checklist measurement link_, and the _estimation engine_ of an MNLFA. No new toolchain.                                                 |

In short: the measurement leg already exists (LRP67), the multi-factor structure
already exists (LRPMM01), the smooth moderation already exists (HSGP), and the
audit already tells us why we need the latent layer. MNLFA assembles them and lets
"study/instrument" become a modelled covariate.

## 6. Preconditions, risks and limitations

- **Construct mapping comes before modelling.** Build an item/indicator map before
  fitting anything: classify each indicator as core, peripheral, common,
  near-common, unique or non-comparable for the intended latent skill. Similar
  test labels are not enough.
- **Bridging data is the linchpin.** No anchor/overlap, no harmonisation —
  full stop. For Q1 this means co-administration of CDI and direct measures; for
  Q2 it means at least one trustworthy shared indicator, common-respondent bridge
  or credible external calibration per skill connecting each cohort to a common
  core. This must be checked _before_ any modelling effort.
- **Identification under sparse overlap.** With few bridging items, or anchors
  themselves carrying DIF, estimates become unstable and prior-dependent. At
  n ~ 54 within a single cohort, even LRPMM01 is "fragile and prior-dependent";
  cross-study pooling does not magically remove that fragility, though it can add
  information where overlap is genuine. And with only minimal anchors, the
  invariance of the anchors _themselves_ cannot be checked from the data — it is
  an assumption; only over-identification (several independent anchors) makes even
  partial testing possible.
- **Partial-invariance modelling is a judgement, not an automation.** Shrinkage/
  horseshoe priors free only the DIF the data demand, but _which_ anchors to
  trust is a substantive call; regularised anchor selection (Belzak and Bauer
  2020) helps but does not remove the need for psychometric judgement.
- **Complexity vs interpretability.** An MNLFA over several cohorts and constructs
  has many more parameters and failure modes than our current per-outcome models.
  It demands prior predictive checks, simulation-based identifiability checks, and
  careful reporting, or it will be harder to defend, not easier.
- **Method effects can masquerade as construct differences.** Parent report vs
  direct assessment, receptive vs expressive response demands, spoken vs pointing
  responses, and language or administration mode may need explicit method factors
  or instrument effects, not just DIF terms.
- **Causal assumptions unchanged.** Repeating Q3: better measurement does not
  identify causes. The DAG and design carry that load.
- **Governance of third-party data.** Consent, data-sharing agreements and ethics
  may forbid pooling raw item-level data at all; a federated route may be the
  only feasible one, but it is a separate methodological design with efficiency,
  diagnostic and implementation costs. Treat governance as a first-class
  precondition alongside the statistical ones.

## 7. A staged, low-risk feasibility plan

Each stage has an explicit go/no-go gate; we do not advance until the gate passes.

1. **Build the construct and linkage map.** For each proposed latent skill, list
   every candidate indicator and classify it as core, peripheral, common,
   near-common, unique or non-comparable; separately record the linking source
   (common item, common respondent/co-administration, content-equivalent cluster,
   external calibration/crosswalk, or common-distribution assumption). **Gate:**
   the indicator graph is connected for each skill, and the non-comparable items
   are excluded or assigned to a different factor/method factor.
2. **Pilot a 2-construct MNLFA on existing LRP data only.** Reuse LRP67's masked
   Beta-Binomial machinery; build a _vocabulary_ factor (R, E) and a _code_ factor
   (L, B) over the existing direct measures, with age as a smooth moderator and
   _group fixed_ (single cohort). Compare a one-factor version against
   multidimensional or method-factor alternatives where the posterior predictive
   checks suggest construct or mode mismatch. **Gate:** does it recover
   LRPMM01-comparable loadings/communalities and sample cleanly (good R-hat, no
   divergences) at n ~ 54? If not, stop — the joint model is too fragile here.
3. **Simulation-based identifiability and DIF-recovery check.** Simulate data
   with known item parameters, a known anchor structure and planted DIF; confirm
   the horseshoe/shrinkage prior recovers the planted DIF and shrinks the rest,
   and that sparse overlap degrades it in the way the literature predicts.
   **Gate:** acceptable recovery and calibrated uncertainty under realistic
   overlap; quantify how much overlap we actually need.
4. **Add a within-project bridge: CDI ↔ direct vocabulary.** Only if/when a small
   co-administration sample exists (or can be collected), test whether checklist
   counts and a direct vocabulary test harmonise onto one η. **Gate:** the
   instrument DIF is modest and the bridged η is stable; otherwise treat Q1 as
   blocked on data collection and say so.
5. **Bring in a single, well-chosen third-party dataset** — ideally one carrying a
   memory anchor (nonword repetition or digit span) plus at least one shared
   language/reading indicator. Use a federated design only if governance requires
   it and the estimation strategy has been specified.
   **Gate:** the anchor genuinely connects the web (connected-graph check),
   posterior predictive checks reproduce item/score distributions by study and
   instrument, estimates are stable, and the added cohort _narrows_ (not inflates)
   the key interaction posteriors. Validate the harmonised factors against
   external criteria where available, known DS-vs-comparison contrasts, and
   longitudinal patterns before scaling to several cohorts.

At every stage, prior predictive checks, posterior predictive checks and prior
sensitivity are mandatory, and findings are reported as exploratory triangulation,
not inferential headlines. Any anchor choice, content-equivalent cluster or
external calibration must get a sensitivity analysis.

## 8. Open questions for discussion

- **Is the co-administration data for Q1 worth collecting?** Without it, the
  CDI ↔ direct bridge rests on untestable assumptions. Is a small co-administered
  sample feasible within our reach?
- **Which third-party cohorts realistically have bridging measures and shareable
  data?** Q2 is only as good as the available anchors and the governance route.
  We need a concrete shortlist before committing modelling effort.
- **One-stage joint model, or two-stage with multiply-imputed factor scores
  ("plausible values")?** One-stage is preferable in principle and natural in
  PyMC, but heavier and more fragile at our n. Do we accept the complexity, or
  start two-stage and migrate?
- **How aggressive should the DIF-shrinkage prior be, and how do we choose
  anchors?** Horseshoe vs regularised-horseshoe vs alignment-style; manual vs
  regularised anchor selection. This is a design decision with real consequences.
- **When is one latent factor too simple?** Decide in advance what evidence would
  push us from a single shared factor to multidimensional, bifactor, second-order
  or instrument-method-factor models.
- **Where does memory enter the locked DAG?** If MNLFA gives us a usable verbal-
  STM latent, does the DAG gain a memory node, and does that change any
  pre-registered estimands? (Cross-reference
  `notes/202606231600-dag-revision-consolidated.md`.)
- **What is the minimum acceptable evidence to graduate this from a note to a
  model in the `measures.py`/factories layer?** Define the bar now.

## 9. References

MNLFA / integrative data analysis / measurement invariance:

- Asparouhov, T., and Muthén, B. (2014). Multiple-group factor analysis
  alignment. _Structural Equation Modeling_, 21(4), 495–508.
  doi:10.1080/10705511.2014.919210
- Bauer, D. J. (2017). A more general model for testing measurement invariance and
  differential item functioning. _Psychological Methods_, 22(3), 507–526.
  doi:10.1037/met0000077
- Bauer, D. J., Belzak, W. C. M., and Cole, V. T. (2019). Simplifying the
  assessment of measurement invariance over multiple background variables: using
  regularized moderated nonlinear factor analysis to detect differential item
  functioning. _Structural Equation Modeling: A Multidisciplinary Journal_,
  27(1), 43–55. doi:10.1080/10705511.2019.1642754
- Bauer, D. J., and Hussong, A. M. (2009). Psychometric approaches for developing
  commensurate measures across independent studies: traditional and new models.
  _Psychological Methods_, 14(2), 101–125. doi:10.1037/a0015583
- Belzak, W. C. M., and Bauer, D. J. (2020). Improving the assessment of
  measurement invariance: using regularization to select anchor items and
  identify differential item functioning. _Psychological Methods_, 25(6),
  673–690. doi:10.1037/met0000253
- Bürkner, P.-C. (2021). Bayesian item response modeling in R with brms and Stan.
  _Journal of Statistical Software_, 100(5), 1–54. doi:10.18637/jss.v100.i05
- Carvalho, C. M., Polson, N. G., and Scott, J. G. (2010). The horseshoe estimator
  for sparse signals. _Biometrika_, 97(2), 465–480. doi:10.1093/biomet/asq017
- Curran, P. J., and Hussong, A. M. (2009). Integrative data analysis: the
  simultaneous analysis of multiple data sets. _Psychological Methods_, 14(2),
  81–100. doi:10.1037/a0015914
- Curran, P. J., Hussong, A. M., Cai, L., Huang, W., Chassin, L., Sher, K. J., and
  Zucker, R. A. (2008). Pooling data from multiple longitudinal studies: the role
  of item response theory in integrative data analysis. _Developmental
  Psychology_, 44(2), 365–380. doi:10.1037/0012-1649.44.2.365
- Curran, P. J., McGinley, J. S., Bauer, D. J., Hussong, A. M., Burns, A.,
  Chassin, L., Sher, K., and Zucker, R. (2014). A moderated nonlinear factor model
  for the development of commensurate measures in integrative data analysis.
  _Multivariate Behavioral Research_, 49(3), 214–231.
  doi:10.1080/00273171.2014.889594
- Gottfredson, N. C., Cole, V. T., Giordano, M. L., Bauer, D. J., Hussong, A. M.,
  and Ennett, S. T. (2019). Simplifying the implementation of modern scale scoring
  methods with an automated R package: automated moderated nonlinear factor
  analysis (aMNLFA). _Addictive Behaviors_, 94, 65–73.
  doi:10.1016/j.addbeh.2018.10.031
- Meredith, W. (1993). Measurement invariance, factor analysis and factorial
  invariance. _Psychometrika_, 58(4), 525–543. doi:10.1007/BF02294825
- Millsap, R. E. (2011). _Statistical Approaches to Measurement Invariance._
  Routledge. ISBN 978-1-84872-819-6.
- Zhao, X., Coxe, S., Sibley, M. H., Zulauf-McCurdy, C., and Pettit, J. W.
  (2022). Harmonizing depression measures across studies: a tutorial for data
  harmonization. _Prevention Science_, 24(8), 1569–1580.
  doi:10.1007/s11121-022-01381-5
- Howe, G. W., Dagne, G. A., Valido, A., Espelage, D. L., Abram, K. M., Brown,
  C. H., and Gallo, C. (2024). The impact of sparse datasets when harmonizing data
  from studies with different measures of the same construct. _Prevention
  Science_, 25(6), 989–1002. doi:10.1007/s11121-024-01704-8 (sparse-overlap
  limitation).

Down syndrome phenotype — memory, language and reading:

- Baddeley, A., Gathercole, S., and Papagno, C. (1998). The phonological loop as a
  language learning device. _Psychological Review_, 105(1), 158–173.
  doi:10.1037/0033-295X.105.1.158
- Baddeley, A., and Jarrold, C. (2007). Working memory and Down syndrome. _Journal
  of Intellectual Disability Research_, 51(12), 925–931.
  doi:10.1111/j.1365-2788.2007.00979.x
- Burgoyne, K., Duff, F. J., Clarke, P. J., Buckley, S., Snowling, M. J., and
  Hulme, C. (2012). Efficacy of a reading and language intervention for children
  with Down syndrome: a randomized controlled trial. _Journal of Child Psychology
  and Psychiatry_, 53(10), 1044–1053. doi:10.1111/j.1469-7610.2012.02557.x
- Jarrold, C., Baddeley, A. D., and Hewes, A. K. (1999). Genetically dissociated
  components of working memory: evidence from Down's and Williams syndrome.
  _Neuropsychologia_, 37(6), 637–651. doi:10.1016/S0028-3932(98)00128-6
- Jarrold, C., Baddeley, A. D., and Phillips, C. E. (2002). Verbal short-term
  memory in Down syndrome: a problem of memory, audition, or speech? _Journal of
  Speech, Language, and Hearing Research_, 45(3), 531–544.
  doi:10.1044/1092-4388(2002/042)
- Laws, G. (1998). The use of nonword repetition as a test of phonological memory
  in children with Down syndrome. _Journal of Child Psychology and Psychiatry_,
  39(8), 1119–1130. doi:10.1111/1469-7610.00416
- Laws, G., and Gunn, D. (2004). Phonological memory as a predictor of language
  comprehension in Down syndrome: a five-year follow-up study. _Journal of Child
  Psychology and Psychiatry_, 45(2), 326–337.
  doi:10.1111/j.1469-7610.2004.00224.x
- Næss, K.-A. B. (2016). Development of phonological awareness in Down syndrome: a
  meta-analysis and empirical study. _Developmental Psychology_, 52(2), 177–190.
  doi:10.1037/a0039840
- Næss, K.-A. B., Melby-Lervåg, M., Hulme, C., and Lyster, S.-A. H. (2011).
  Reading skills in children with Down syndrome: a meta-analytic review.
  _Research in Developmental Disabilities_, 33(2), 737–747.
  doi:10.1016/j.ridd.2011.09.019
- Polišenská, K., and Kapalková, S. (2013). Language profiles in children with
  Down syndrome and children with language impairment: implications for early
  intervention. _Research in Developmental Disabilities_, 35(2), 373–382.
  doi:10.1016/j.ridd.2013.11.022
- Purser, H. R. M., and Jarrold, C. (2005). Impaired verbal short-term memory in
  Down syndrome reflects a capacity limitation rather than atypically rapid
  forgetting. _Journal of Experimental Child Psychology_, 91(1), 1–23.
  doi:10.1016/j.jecp.2005.01.002

::: {.callout-note title="A nuance to keep honest"}
A nuance to keep honest (per Polišenská and Kapalková 2013): state the DS
oral-language profile as _receptive vocabulary a relative strength, expressive
language and grammar/morphosyntax a relative weakness_ — strongest as a
DS-versus-other-syndrome and DS-versus-mental-age contrast — rather than an
absolute grammar collapse; grammar can track lexical level once vocabulary is
matched.
:::
