# LRPITT suite — design decisions (issue #119)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

Date: 2026-06-25

## Scope

Replaces the ad-hoc ITT models (LRP52/53/54/74/75) with a uniform, DAG-faithful
suite **LRPITT01–11** (11 RCT-phase outcomes) plus integer-numbered companions
**LRPITT12–15b** (joint, SES, generalisation). Built in three commits on the
`fix/group-coding-positive-benefit` foundation branch:

- **PR1** — factory/floor/pipeline/reporting foundation + tests.
- **PR2** — the 11 single-outcome modules + standalone reports + registry; deletes
  LRP52/53/54/74/75.
- **PR3** — companions LRPITT12 (joint), LRPITT13/13b/14/14b (SES), LRPITT15/15b
  (generalisation); migrates and deletes LRP55/60/60a/76.

## Key decisions

### DAG-faithful adjustment (identification)

Under the locked DAG (#115, `notes/202606231600-dag-revision-consolidated.md`),
`IG` is a randomised root and latent general ability does not touch it, so the ITT
effect `τ` is **identified by the empty adjustment set**. The uniform linear
predictor is therefore

    eta_i = alpha + tau·G_i + gamma_own·logit*(y_pre_i) + gamma_A·A_std_i

with the own baseline and **linear age as precision terms only** (a dedicated
`gamma_age_prior` = Normal(0, 0.3)), **no cross-baseline conditioning** (dropped
the `sum_{k≠own} gamma_k` term that LRP52–54 carried; prior fits found it not
credibly non-zero, `notes/202604181600-lrp52-58-findings.md`), and **HSGPs off**
by default (`notes/202604181445-lrp52-gp-sensitivity.md`). Attendance/sessions are
never conditioned on (a DAG collider).

### Sign convention

Adopt the foundation branch's **positive = benefit** (`G = 2 − group`; `G = 1` is
the immediate-intervention arm), so a **positive τ means the intervention raised
the outcome**. This overrides the (stale) "negative τ" wording in the issue body,
which predates the sign flip bundled with the locked DAG.

### Floored outcomes — pre-specified floor rule (P, N)

Phonetic spelling (P, ~78% at zero at t2) and nonword reading (N, ~72%) are
heavily floored at the _outcome_, so a graded Beta-Binomial τ is leveraged by a
few dispersed tail values rather than by the arm contrast. A rule fixed **before
fitting** and applied **arm-blind** (`floor.is_floored`, threshold ≥ 40% at zero
at t2) handles them: drop the degenerate own baseline (age-only predictor), report
a **binary "off-floor at t2" estimand** (`Pr(post > 0)`, Bernoulli/logistic τ) as
**PRIMARY**, and retain the graded Beta-Binomial τ only as a flagged,
**detection-limited SECONDARY** read beside per-arm off-floor mover counts. P and N
are treated **identically** (the rule removes per-outcome researcher choice).
Diagnostics: a proportion-at-zero PPC + the mover table. This implements the
issue's floor-rule branch (the full deliberation is in
`notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md`). The
"uniformity-purist" fallback was **not** adopted.

Word reading (W, ~40% floored at _baseline_ but not at the t2 post-score) does not
trip the rule and stays a graded own-baseline model.

### N post-only handling

Nonword is the repo's "post-only" measure and has four missing t1 baselines. The
new `pre_required` parameter of `load_and_prepare` exempts N's baseline from the
complete-case mask (so those four children are kept), while the GROUP/AGE and
post-presence checks still apply; the factory's `use_own_baseline=False` path never
indexes the (unused) N baseline.

### Joint companion (LRPITT12) — N excluded, P graded

The joint's `pre_logit` is a **dense** outcome×baseline matrix that cannot hold N's
missing/degenerate baseline without injecting NaNs or polluting the cross-baseline
block, so **N is excluded** from the joint (its effect is read from the LRPITT11
off-floor model). P **is** included as a graded outcome (its floored baseline
simply shrinks `gamma_own[P]`). LKJ residual correlation and the age GP are off by
default. A per-outcome own-baseline mask to admit N is deferred.

### Companions

- **SES robustness** (LRPITT13/13b adjusted for W/L; LRPITT14/14b matched
  unadjusted complete-case comparators) — adjusted-vs-comparator isolates the SES
  adjustment with the sample held fixed.
- **Generalisation contrast** — taught vs not-taught, both expressive
  (`τ[TE] − τ[UE]`, LRPITT15) and receptive (`τ[TR] − τ[UR]`, LRPITT15b); positive
  ⇒ the programme moved taught words more than untaught (limited transfer).

### Part B plumbing (models deferred)

The factory gained a linear **τ-moderator path** (`tau_moderator_symbol`,
`gamma_tau_mod`/`gamma_tau_int`, regularising Normal(0, 0.3), with a nested
no-interaction baseline for LOO) so the Part B HTE models are unblocked. The actual
HTE and CACE/dose models are **deferred** to a later milestone.

## Deferred / follow-ups

- Part B response-to-intervention: HTE moderation by pre-randomisation traits; and
  a CACE/instrumental-variable dose analysis (IS is a DAG collider — randomisation
  as instrument, complier effect with heavy caveats, never "more sessions cause
  more gain").
- **LRPITT16** — descriptive floored-outcome trajectory complement (4-wave,
  intervention-aligned periods, explicitly **non-ITT**; connects to #104).
- Expressive information (aptinfo) / expressive grammar (aptgram) ITTs (new
  measures, confirmed APT ceilings, aptinfo half-mark fix).
- Confirm the UR/UE (= 12) item ceilings against the data dictionary.
- Joint-with-N via a per-outcome own-baseline mask in `build_joint_model`.
- Report-boilerplate dedup (#82). _(The `hdi_prob → ci_prob` rename, #101, is done in this PR.)_
