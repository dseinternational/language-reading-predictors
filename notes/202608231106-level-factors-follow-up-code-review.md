> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Level-factors follow-up code review (functional + statistical), 2026-08-23

## Scope and conclusion

This is a documentation-only follow-up to
`notes/202608201500-level-factors-code-review.md`. It reviews all eleven registered
`level_factors` models (`lrp-rli-lf-001`–`011`) across:

- their declarations and resolved run plans in `level_factors.py`;
- levels-mode preprocessing and the realised analysis rows;
- `factories.py::build_level_factors_model`;
- `pipelines/level_factors.py` and the shared primary-fit stages;
- natural-scale reporting, key findings, release decisions and prior-sensitivity
  attachment;
- the static model reports and cross-family findings notes;
- the registered tests; and
- all eleven stored `-reporting` traces and fit artefacts.

The PyMC construction is largely correct. No sign, shape, coordinate, phase-index,
Beta-Binomial parameterisation or posterior-vector-labelling error was found. The
main unresolved issue is that the reported items/risk headline is not the quantity
the prose says it is. There are also a stale headline synthesis, a fail-open path in
the prior-sensitivity evidence, a material phoneme-blending link sensitivity, and
several modelling and robustness limitations that need explicit decisions.

**Recommendation:** do not treat the present level-factor natural-scale cards as a
settled response-scale difference-in-differences until finding 1 is resolved. The
conditional logit coefficient `d_grp_time[t2]` remains a well-defined fitted
quantity, subject to the longitudinal model and missing-data assumptions below.

No source, model, report or stored fit is changed in this review.

## Finding 1 (statistical/reporting, major): the natural-scale headline is not the claimed response-scale difference-in-differences

At centred ability, let:

- `c = arm_gap_t1`;
- `d = d_grp_time[t2]`;
- `z_it` be the arm-free part of the predictor; and
- `s(x) = inverse_logit(x)`.

The fitted group contribution implies

```text
eta_i,t1(g) = z_i,t1 + g c
eta_i,t2(g) = z_i,t2 + g (c + d).
```

`reporting.level_t2_marginal_effect` instead computes, over the observed t2 rows,

```text
mean_i { s(z_i,t2 + G_i c + d) - s(z_i,t2 + G_i c) }.
```

The implementation removes `d + gamma_grp_ability * ability` only from rows whose
observed group is immediate, leaves `arm_gap_t1` in those rows, and then adds `d` to
every row (`reporting.py`, lines 4184–4206). Consequently, control profiles are
evaluated around `z_i,t2`, while immediate-arm profiles are evaluated around
`z_i,t2 + c`. This is a hybrid average over observed-arm operating points. It is
not any of the following:

1. a conventional t2 average marginal effect that sets the complete group
   contribution to control and intervention for every profile;
2. a response-scale difference-in-differences,
   `[p(I,t2) - p(W,t2)] - [p(I,t1) - p(W,t1)]`; or
3. the full t2 arm contrast, which contains `c + d` (and any coherent treatment of
   the group-by-ability interaction), rather than `d` alone.

`d_grp_time[t2]` itself is a conditional **logit-scale change in the arm gap** — a
ratio of odds ratios. Because the inverse logit is nonlinear, that coefficient has
no unique risk-difference or items-scale translation. Nevertheless, the current
transform drives `rope_summary.csv`, `prior_pushforward.csv`, the ROPE plot and the
findings-first headline (`pipelines/level_factors.py`, lines 220–285), while the run
plan and reports call it a difference-in-differences of adjusted levels
(`level_factors.py`, lines 519–564; `reporting.py`, lines 5522–5539).

### Numerical consequence in the stored traces

The table contrasts the current posterior mean with two alternative functionals
computed from the same stored draws. The response-scale DiD uses common t1/t2
profiles with the moderation increment held at centred ability. The full t2 column
is the model-standardised complete t2 arm gap; it is not asserted to be the chosen
replacement, but shows why the target matters.

| Outcome                      | Current card | Response-scale DiD | Full t2 arm contrast |
| ---------------------------- | -----------: | -----------------: | -------------------: |
| Word reading                 |  +2.30 items |              +2.37 |                +2.53 |
| Receptive vocabulary         |  +0.23 items |              -0.08 |                -4.50 |
| Expressive vocabulary        |  +0.14 items |              -0.20 |                -3.06 |
| Letter sounds                |  +2.84 items |              +2.83 |                +3.25 |
| Taught expressive vocabulary |  +1.30 items |              +1.08 |                +0.31 |

The current transform is numerically close to an immediate-arm/ATT-like shift in
these traces (maximum difference about 0.03 items). A genuine response-scale DiD
moves some near-zero medians by about 0.3 items but does not change an 89% interval
conclusion. The full t2 contrast differs materially for vocabulary because it
answers a different question and retains the fitted baseline arm gap.

### Proposed fix and acceptance tests

Frank should choose one primary quantity:

- retain `d_grp_time[t2]` and report it only as the conditional logit-scale DiD;
- define a genuine marginal response-scale DiD using t1 and t2 counterfactual
  predictions on one stated population; or
- define the randomised t2 marginal arm contrast, preferably in a model whose t2
  group-by-ability term is identified from the randomised window.

Then:

1. implement the chosen functional once and reuse it for posterior summary, ROPE,
   prior pushforward, plot and key findings;
2. state the standardisation population, child-effect convention and treatment of
   effect modification in the resolved run plan;
3. add synthetic counterexamples in which logit DiD, probability DiD and t2 arm
   difference are deliberately unequal; and
4. regenerate all eleven reports after the definition is fixed.

## Finding 2 (scientific reporting, major): the main synthesis quotes the superseded LF-001 result and estimand

`notes/202608182200-findings-by-question.md` says at line 8 that it was reread from
the 2026-08-20 refit and incorporates the #552 level-factor reparameterisation. Its
line 167 still gives the pre-#552 reading: t1 arm gap -0.17, t2 arm gap +0.25 called
the randomised contrast, and +1.7 words.

The current family note (`notes/202608180936-findings-04-level-factors.md`, lines
28–58) gives:

- `arm_gap_t1` about +0.033;
- causal `d_grp_time[t2]` about +0.347;
- derived full t2 gap about +0.380; and
- the current natural-scale card about +2.3 words.

The same synthesis already prints +2.3 in its cross-family outcome table, so it is
internally inconsistent.

**Proposed fix:** replace the LF-001 paragraph after finding 1 settles the
natural-scale target, then scan every later note/report for pre-#552 uses of
`b_grp_time[1]` as the causal term. Keep genuinely historical design notes
unchanged, but label their old estimand explicitly.

## Finding 3 (release integrity, major but latent): indexed prior-sensitivity evidence is not reproduced from its trace

`sensitivity._validate_cell_trace` documents and implements a weaker rule for an
indexed focal term such as `d_grp_time[t2]`: it verifies only that the base variable
`d_grp_time` exists. It recomputes divergences but does **not** recompute or verify:

- the indexed coordinate or focal draws;
- `tau_logit_mean`, its interval, direction probability or items/risk summary;
- R-hat, effective sample sizes or BFMI; or
- the manifest's `converged` claim.

`attach_outcome_bundle` trusts the CSV `converged` and `tau_logit_mean` values when
checking convergence and sign stability. The post-attachment release check
`release._standard_sweep_evidence` verifies file presence and hashes but never
reopens the traces; it again trusts the CSV values.

The following mutation was reproduced with the existing test helpers and attached
successfully:

```python
rows["tau_logit_mean"] = 999.0
attach_outcome_bundle(...)
```

The fixture itself uses one chain and five draws while declaring `converged=True`,
and its success test accepts the bundle. Primary hashes catch changed primary
files, and `trace_sha256` catches changed cell-trace bytes, but editing only the CSV
leaves those hashes valid.

This has not corrupted the eleven current LF release decisions: none has an
attached `tau_prior_sensitivity.csv`, and all currently release as `clear` or
`prior_data_conflict`. It is a fail-open path for a future prior-dominant result.

The primary-reference binding is also incomplete. It checks broad identity, total
rows and overall arm counts, but not the stored `resolved_run_plan`, focal vector
and coordinate, t2 counts or `fitted_data_identity.digest`. A semantically stale
trace or a changed fitted row set with the same totals can therefore pass.

### Proposed fix and acceptance tests

1. Parse the indexed focal term and recompute the complete row summary from the
   trace, including coordinate label, direction and intervals.
2. Re-run the full convergence gate from the cell trace, rather than trusting its
   CSV row.
3. Bind the cell to the primary's resolved plan and fitted-data identity, including
   wave-specific arm counts.
4. Revalidate traces at release time, or content-sign a canonical manifest that
   cannot be edited independently of its evidence.
5. Add tests for same-sign focal tampering, false convergence, wrong coordinate,
   stale parameterisation and changed row membership.

## Finding 4 (statistical, moderate and material): LF-006 lacks the known one-in-three guessing-floor link sensitivity

`LevelFactorsModelSettings` has no score-mean-link setting. LF-006 therefore uses
the ordinary Beta-Binomial inverse-logit mean hard-coded at `factories.py`, lines
5028–5034. The level family has no equivalent of the ITT-008/108 paired ordinary
and three-choice-guessing-floor fits, and the LF-006 report carries no comparable
qualification.

The stored LF-006 posterior makes the issue material:

- 24 of 215 rows have posterior-mean expected proportions below one third;
- 13.7% of all row-by-draw expected proportions are below one third;
- at t2, 8 of 54 row means and 16.0% of row-by-draw mass are below one third; and
- the current ordinary-link mean card is about +0.644 items.

Applying `mu = 1/3 + (2/3) * inverse_logit(eta)` to the same latent draws would
mechanically rescale that diagnostic contrast to about +0.429 items. This is not a
valid substitute for a refit: the posterior, empirical-Bayes t1 anchor and
latent-scale prior pushforward would also change.

The explicit paired-release implementation is ITT-only, and the METHODS wording
appears within the ITT section, so this is not an unambiguous violation of the
current LF software contract. The literal wording says "any headline B
interpretation", however, and the measurement issue is family-independent.

**Proposed fix:** either add `score_mean_link` to the level settings/run
plan/factory/reporting contract and register a paired LF-006 companion under the
same convergence/reporting gates, or mark LF-006 explicitly non-headline and
require readers to interpret it only beside the ITT link pair.

## Finding 5 (statistical limitation, moderate): the quantity called the clean t2 contrast borrows from post-crossover waves

Under the t1-centred parameterisation,

```text
b_grp_time[t] = arm_gap_t1 + d_grp_time[t],  t in {t2, t3, t4}.
```

The t3/t4 likelihood therefore informs the shared `arm_gap_t1` through the priors
on `d_grp_time[t3/t4]`, and `d_grp_time[t2]` trades off against that anchor. The
shared child random intercept, dispersion and time-invariant group-by-ability term
also borrow information across all four waves. In the stored traces, posterior
correlations between `arm_gap_t1` and `d_grp_time[t2]` range from approximately
-0.07 to -0.44.

This is not a PyMC bug. It means that randomisation underlies the window, but the
reported coefficient additionally depends on a longitudinal working model and on
post-crossover observations. Calling it simply the "clean randomised contrast"
overstates that separation.

The one `gamma_grp_ability` coefficient is also shared across all waves. It
combines pre-randomisation arm-by-ability slope imbalance, possible t2 effect
modification and post-crossover associations. The natural-scale card deliberately
omits it, which avoids folding a non-randomised mixed coefficient into the causal
claim. Its separate summary row is not clean treatment moderation.

Finally, the key-findings sentence calls the card an effect "for a child of typical
cognitive ability" (`reporting.py`, lines 5548–5554). The implementation averages
over every fitted t2 profile, retaining row-specific age, ability main effect,
adjusters and fitted child intercept; it fixes only the added moderation increment
at centred ability. The report partial correctly says this is not one typical
child.

### Proposed fix

- Fit or summarise a t1/t2-only causal comparator, and report its difference from
  the four-wave fit.
- Consider separating the t3/t4 raw gaps from the t1 balance parameter.
- If effect modification remains a target, use a t1-centred or wave-specific
  group-by-ability interaction so the t2 term is identified from the randomised
  window.
- Replace "child of typical ability" with the exact standardised-profile-average
  definition selected under finding 1.

## Finding 6 (statistical robustness, moderate): nuisance-prior conflicts are omitted from the family power-scaling audit

The pipeline passes only `arm_gap_t1` and `d_grp_time` to `run_psense`
(`pipelines/level_factors.py`, lines 111–139). An exploratory power-scaling audit
of the stored traces found:

- `sigma_child` flagged for potential prior-data conflict in all eleven fits; and
- `kappa` flagged in eight of the nine graded-score fits.

The direct posterior values support that diagnostic:

- `sigma_child ~ HalfNormal(0.5)`, while its word-reading posterior median is about
  1.39 and its phonetic-spelling median about 1.67; and
- `kappa ~ HalfNormal(50)` has a 99th percentile near 129, while receptive- and
  expressive-vocabulary posterior medians are about 170 and 198.

`priors.kappa_prior` already documents that the HalfNormal(50) shape suppresses the
near-Binomial region for high-denominator outcomes. Exploratory wider-prior refits
did not change the current focal conclusions, but did change posterior dispersion
and uncertainty. The present release audit therefore establishes focal-term
power-scaling behaviour, not full likelihood/prior robustness.

**Proposed fix:** run and report power scaling over all diagnostically material
free parameters; add registered `kappa` and child-SD sensitivity axes; and decide
whether the dispersion-scale `1/sqrt(kappa)` prior adopted by the historical/RLM
families should extend to these high-denominator RLI level fits. A nuisance conflict
need not automatically block publication, but it should be visible.

## Finding 7 (scientific documentation, moderate): five model equations omit covariates that are fitted

The static equations for LF-002–006 omit every `sum_c gamma_c z(c)` adjustment term
and missing indicator, although the specifications fit:

| Model  | Fitted background terms omitted from the report equation |
| ------ | -------------------------------------------------------- |
| LF-002 | hearing and phonological memory, with missing indicators |
| LF-003 | hearing, speech and phonological memory, with indicators |
| LF-004 | hearing and speech, with indicators                      |
| LF-005 | phonological memory and its indicator                    |
| LF-006 | hearing, speech and phonological memory, with indicators |

The affected equations are `docs/models/lrp-rli-lf-002/index.qmd`, lines 29–30;
`lf-003`, lines 29–30; `lf-004`, lines 29–30; `lf-005`, lines 40–44; and `lf-006`,
lines 29–30. LF-001/007/008 correctly have no such adjusters, and LF-009/010/011
correctly document theirs.

The fitted code is correct; five reports describe materially smaller models than
were fitted.

**Proposed fix:** generate the equation/adjustment sentence from the resolved run
plan, or add a specification-to-report contract test that pins every registered
model's background terms and missing indicators.

## Finding 8 (validation and missingness, moderate/low): baseline support is not checked and the DiD populations differ for P and N

`LevelFactorsRunPlan.validate_prepared` requires both randomised arms among observed
t2 rows but does not require both arms at t1. The factory requires only at least one
t1 outcome. A synthetic panel with only controls observed at t1 and both arms at t2
passes validation; `arm_gap_t1` is then determined by its prior and later waves but
`d_grp_time[t2]` is still labelled a t1-to-t2 randomised change. An empty interior
wave can likewise leave a prior-only published coefficient.

All current fits contain both arms at t1 and t2, so the validation defect is latent.
Two current outcomes nevertheless use different child compositions across the two
waves:

- LF-005/P: t1 has 54 children (26 control, 28 immediate), t2 has 53 (25, 28);
  one control is t1-only.
- LF-011/N: t1 has 50 children (23 control, 27 immediate), t2 has 53 (25, 28);
  three children are t2-only.

The longitudinal generalised linear mixed model can use an incomplete panel under
its outcome model and missing-at-random assumptions. Randomisation alone does not
repair observation depending jointly on arm and potential outcomes, nor does it
make two arm gaps measured in different children a design-only DiD.

**Proposed fix:** require both arms at t1 and t2 for a t1-centred causal term,
require support for every published wave coefficient, and add balanced t1/t2 plus
missingness/weighting sensitivities for P and N.

## Lower-severity functional and audit findings

1. `level_prior_pushforward` failures are caught by an ad hoc `try/except` in the
   family pipeline. The fit prints a warning but `artifact_manifest.json` contains
   no skipped/error record. Use `artifacts.guard_optional` and test an injected
   failure.
2. The prior pushforward describes what the model implied "before seeing data",
   although `alpha_anchor` is computed from the observed t1 outcome. It is an
   empirical-Bayes, data-conditioned prior check; the priors table discloses this,
   but the pushforward prose should too.
3. All eleven modules still declare mutable legacy `extra={...}` dictionaries,
   despite `METHODS.md` and `docs/models/README.md` saying converted modules declare
   immutable typed settings. Strict translation catches unknown keys, so this is
   not a current numerical defect. Migrate the registered modules and add a test
   that every registered LF specification is typed.
4. Settings validation permits duplicate adjusters, a missing indicator without
   its paired base term, and `ability_by_time=True` without an ability covariate
   when `group_ability=False`. Duplicate terms fail only later during PyMC model
   construction. Reject these contracts during plan resolution.
5. Registered-suite coverage dynamically globs modules, asserts only `len >= 11`,
   and checks self-consistency. It does not pin the exact ID/outcome/ability/
   adjustment/likelihood map or forbid an unintended extra registration. Add a
   declarative expected-contract table.
6. A supported but unregistered pooled `group_by_time=False` plan correctly has no
   causal term, while generated estimand/causal prose and the shared results partial
   still claim a t2 randomised coefficient. Branch the prose on the resolved focal
   term.
7. `release._model_tier` labels any fit with `adjust_for` as
   `adjusted_robustness`, although its documentation describes adjusted ITT
   comparators. Seven LF primaries therefore receive that metadata label. Current
   release policy is uniform across these tiers, so no decision changes.

## Verified sound

- All eleven registered ID/outcome/ceiling/likelihood mappings agree with
  `measures.py` and the data schema.
- Every model uses baseline block-design ability. The registered background sets
  match the revised DAG outcome by outcome.
- Speech and phonological-memory terms are routed to the pre-randomisation t1
  measurement, hearing remains contemporaneous, and evolving measure-skill parents
  are correctly excluded rather than conditioning on treatment-affected mediators.
- Positive group coefficients favour the immediate-intervention arm.
- `phase`/`post_phase` coordinates and
  `b_grp_time = [arm_gap_t1, arm_gap_t1 + d_grp_time]` are dimensionally correct.
- The exact zero-sum wave intercept removes the former intercept/time translation
  ridge; the arm-blind Haldane-smoothed empirical-Bayes anchor uses t1 only and is
  disclosed.
- The ordinary Beta-Binomial mean/concentration construction is correct. P and N
  correctly fit binary off-floor **status at each wave**, not a floor-exit
  transition.
- Posterior vector coordinates, factor-summary roles, prior inventory and focal
  release-term naming propagate consistently.
- Every realised current-data plan validates with both arms at t2. Row/child counts
  are W 210/53, R 215/54, E 215/54, L 214/54, P 213/54, B 215/54, F 214/54,
  T 215/54, TR 215/54, TE 215/54 and N 207/53. LF-011 correctly drops only the
  constant `erbto_missing` indicator.

## Verification and limitations of this review

A 663-test cross-section passed, covering the factories, reporting, key findings,
level settings, prior-sensitivity attachment, release decisions, prior inventory,
registry, pipeline boundaries and model identifiers. Targeted levels-mode
preprocessing selectors also passed. The only warnings were established SHAP
deprecations and Numba object-mode notices.

Every numerical finding above was recomputed from the stored reporting traces. The
traces record source commit `4e924948ef388debd84e14f45958ab37d62c3db8`, an
ancestor of the current checkout that includes the #552 t1-centred equation. This
review did not run eleven new production MCMC fits, so the numerical comparisons
are trace audits, not a current-HEAD refit.

## Proposed implementation sequence

1. Decide and implement the natural-scale t2 estimand (finding 1).
2. Correct the stale synthesis and five model equations.
3. Decide and implement the LF-006 response-link policy.
4. Harden prior-sensitivity trace validation and primary/row identity binding.
5. Add a t1/t2-only comparator and nuisance-prior/missingness sensitivities.
6. Close the lower-severity plan, manifest and test-contract gaps.
7. Refit all eleven reporting models, regenerate the reports and re-read the
   cross-family syntheses from the regenerated artefacts.

## Decisions requested from Frank

1. Which natural-scale causal quantity should the level family headline: conditional
   logit DiD, marginal response-scale DiD, or full randomised t2 arm contrast?
2. Should LF-006 receive its own required guessing-floor companion, or be declared
   non-headline and interpreted only beside the ITT link pair?
3. Should the four-wave model remain the model of record for the t2 claim, with a
   t1/t2-only sensitivity, or should post-crossover gaps be decoupled structurally?
4. Should the dispersion-scale `1/sqrt(kappa)` prior and broader child-SD checks be
   extended to this family?
