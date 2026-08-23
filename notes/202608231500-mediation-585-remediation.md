> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Mediation remediation for #585, 2026-08-23

## What was verified before anything was changed

Every release-blocking finding in #585 was re-derived from the code, from freshly built PyMC graphs and from the stored `reporting` artefacts rather than taken from the review note. All five reproduce; three are wrong in the released artefacts rather than only on paper. The verification, including the two places where the review overstates its case, is recorded on the issue itself.

The decisive evidence, for the record:

- **Leg split.** MED-059 fits `a_{A,E,G,L,R,deapp_c,hs}` on the mediator leg and `b_{A,E,G,GM,M,R,W,deapp_c,hs}` on the outcome leg: no `a_W`, no `b_L`. The same split holds in all eight W-outcome models and in all four two-mediator models, whose second mediator law also lacked the first mediator's baseline.
- **MED-060.** `E` and `R` were declared, never loaded, silently filtered, and absent from both `effective_confounders` and `dropped_confounders` in the released `config.json`.
- **Off-floor sample.** Relaxing `pre_required` to the baselines the pre-#585 likelihood used moved MED-060, MED-086 and MED-186 from 50 to 53 children.
- **MED-092 positivity.** Support by period was 28 treated / 25 untreated in period 1, then 53 / 0 and 52 / 0. The released `key_findings.json` headlined the all-period `+3.1 items`; the supported period-1 contrast in the same directory was `+2.56`.
- **Interventional aliases.** MED-078's `mediation_summary.csv` matched MED-059's to every printed digit.
- **Row counts.** `config.json` recorded 53 / 53 / 158 for MED-076 / MED-176 / MED-092 against fitted `obs_id` sizes of 51 / 52 / 157.

## Decisions taken

### 1. One common baseline vector, conditioned on by every leg

The review asked for leg sets _derived_ against the lagged DAG rather than a global set applied mechanically. Doing that derivation lands on the common set anyway for these models, and the reason is worth stating because it is not the usual "adjust for everything" reflex.

The outcome leg must block the mediator-to-outcome backdoors, and `tests/test_lagged_dag_adjustment_sets.py` already certifies that the declared union does so — which means the outcome leg was missing a certified member, the mediator baseline. The mediator leg is a different argument: the g-formula evaluates `E[Y | g, m, C]` at a child's own covariates while drawing `m` from `P(m | g', C)`, so the two laws must condition on the _same_ `C` or the composition is not the declared functional at all. A reduced mediator law is admissible only if the omitted covariate is conditionally irrelevant to the mediator, and `WR_t -> LS_t1` in the lagged DAG says baseline word reading is not. So both legs take the union: the outcome baseline, the mediator baseline(s) and every bounded-measure confounder.

The added terms are named `a_base_<symbol>` / `b_base_<symbol>` (and `a<mediator>_base_<symbol>` in the two-mediator legs) rather than folded into the existing `a_<symbol>` / `b_<symbol>` names. That is deliberate: `b_W` is hard-coded as the _outcome's_ own-baseline coefficient regardless of the outcome symbol, so in the reverse-direction models MED-176 / MED-276 — where the mediator _is_ word reading — the natural name would have collided with an unrelated coefficient. Prefixing makes collisions impossible by construction and leaves every pre-existing coefficient name untouched.

A composite mediator (MED-062) enters the outcome leg as one composite baseline term matching the mediator leg's `a_comp`, not as its route symbols separately.

### 2. Off-floor baselines are modelled, not dropped — so the sample rule stops being stricter than the likelihood

The review is careful that the 50-versus-53 gap diagnoses a contract mismatch without establishing that the three children belong in the corrected sample, and it asks for the baseline model to be settled _before_ the sample is. Settling it resolves the gap in the other direction.

A floored measure's baseline logit is a near-degenerate spike, so the graded `Normal(1, 0.25)` autoregressive term genuinely does not belong. But the project already has the right functional form for exactly this case — the binary off-floor-at-baseline indicator with `gamma_own_offfloor ~ Normal(0, 1)`, used by the off-floor ITT, DiD and gain-factor models. Adopting it here means the off-floor outcome leg (`b_own_offfloor`) and the off-floor second-mediator leg (`a<M>_own_offfloor`) both model the baseline they were already implicitly requiring. The fitted samples stay at 50; what changes is that they are now _justified_ by the likelihood rather than an artefact of a loader default.

`pre_required` is then resolved from the leg terms and threaded through every preparation path, so a measure loaded for some other reason can no longer exclude a child. That does relax the rule for models loading the default eight-outcome ITT set.

### 3. MED-092's headline moves to the randomised window

`T = (G == 1) | (phase >= 1)`, so after the crossover there are no untreated rows and no adjustment can create positivity in those cells. The period-1 decomposition — already computed, already saved, simply not the headline — becomes `mediation_summary.csv`; the all-period average becomes `mediation_summary_all_periods.csv` under an explicit extrapolation label; and `period_treatment_support.csv` records the per-period arm counts so the distinction is checkable rather than asserted. The tipping analysis now sweeps the primary estimand rather than the extrapolated one.

### 4. The interventional companions are relabelled, not retired

The review proposed retiring the IDs or relabelling. Retirement is not what the evidence supports: the code and reports already stated that the flag is an interpretive relabelling producing identical numbers. What they did _not_ say is the thing that matters — that a well-defined estimand is not an identified one. An interventional effect is identified under an exposure-induced mediator–outcome confounder only if that confounder is measured and integrated over, within its treated distribution in the outcome leg and its control distribution in the mediator leg. Intervention dose is in neither leg, so this functional identifies the interventional target only under the same condition the natural version needs. The reports now say so, and the temporal-ordering sensitivity — previously skipped for these three "because it is a natural-effect construction" — now runs for them, since they are the same fitted model as their parents and were carrying strictly less evidence.

### 5. The sensitivity artefact is renamed, and the calibration formula is derived rather than asserted

"Bayesian E-value analogue" is gone. An E-value is a specific minimum-strength measure on the risk-ratio scale; this is a one-directional bias model on a single logit coefficient that holds the mediator law and every other term fixed. The docstrings and the report partial now say what it is, note that a single global δ moves an opposite-signed draw away from zero, and frame δ\* as a bound on robustness.

On the named-`IS` calibration the review overstates: it calls `delta_IS ≈ |β(IS→M) · β(IS→Y)|` "not the general omitted-variable-bias formula" and "not scale-invariant". Both are true of the general case and neither is true here, because the module already puts both slopes on one-standard-deviation scales, which makes `Var(M) = 1` and the bare product exactly the linear bias. That derivation is now written down instead of being left implicit, along with the approximation that genuinely does remain — transporting a linear bias onto a logit coefficient — and a linear-Gaussian recovery test. The "could account for" phrasing is replaced by scenario-conditional wording, and the docstring states that the two-mediator calibration is a different, descriptive construction that must not inherit this interpretation.

### 6. Diagnostics

Per-leg posterior-predictive coverage: the shared stage only ever sent the last observation node to the PPC writer, so mediator fit — the leg the indirect effect is built from — appeared in no released coverage summary. Each leg is now routed by its own likelihood, and `ppc_coverage_markdown` already pooled across nodes correctly, so the rendered sentence needed no change. In-sample predictive fit is a specification check, reported and never used to certify the decomposition.

Derived-estimand gating: bulk/tail ESS and MCSE were already computed per row but never gated, so a fit could pass the all-free-RV gate with unusable g-formula draws. `gate_derived_estimands` adds a pre-specified check — ESS ≥ 400 and MCSE ≤ 5% of the reported interval half-width — into `diagnostics_summary.json`, which the release gate picks up because it fails closed on unrecognised non-`True` checks. `proportion_mediated` is deliberately exempt: it is a ratio that is unstable by construction whenever the total can cross zero.

## Bounded impact of the leg-set change: MED-059, old versus new

The review recommends invalidating all 19 artefact sets without showing how far the numbers move. This measures it. Both arms use the same code, the same 53 prepared rows, the same `reporting` preset (6 chains x 6000 draws, `target_accept` 0.95, seed 47) and the same g-formula settings; the only difference is whether the mediator leg carries `a_base_W` and the outcome leg `b_base_L`.

**Validity check.** The old arm reproduces the released `lrp-rli-med-059-reporting` artefact to every printed digit (total 2.0612, 89% -0.1759 to 4.2268, P(>0) = 0.9313). So the comparison isolates the leg-set change, and the rest of the new code path is behaviour-preserving on this model. Both arms sampled cleanly: 0 divergences, max R-hat 1.0004 / 1.0003, minimum bulk ESS 14,941 / 13,217.

| Quantity            | Old (released)                               | New                                          | Shift | Shift as share of old 89% half-width |
| ------------------- | -------------------------------------------- | -------------------------------------------- | ----- | ------------------------------------ |
| Total               | +2.06 items, 89% -0.18 to +4.23, P(>0) 0.931 | +2.32 items, 89% +0.15 to +4.53, P(>0) 0.954 | +0.26 | 12%                                  |
| NDE                 | +0.17, 89% -1.80 to +2.15, P(>0) 0.555       | +0.14, 89% -1.79 to +2.15, P(>0) 0.544       | -0.04 | 2%                                   |
| NIE                 | +1.79, 89% +0.66 to +3.37, P(>0) 0.997       | +2.10, 89% +0.86 to +3.72, P(>0) 0.999       | +0.30 | 22%                                  |
| Proportion mediated | 0.83, 89% -0.73 to +3.24                     | 0.89, 89% +0.19 to +3.01                     | +0.06 | 3%                                   |

**These shifts are small against the posterior but far larger than Monte-Carlo noise** — `|shift| / MCSE` is roughly 1,600 for the total and 2,800 for the NIE. That distinction matters: the numbers are not interchangeable, and regenerating summaries over the old traces would not reproduce them. One qualitative reading does change: the total effect's 89% interval stops including zero.

**Where the movement comes from.** `a_base_W`, the outcome baseline missing from the mediator law, is a real signal at 0.234 +/- 0.077 on the logit scale — about three posterior SDs from zero, which is what the lagged DAG's `WR_t -> LS_t1` edge predicts. Adding it reallocates variance inside the mediator leg: the own-baseline `a_L` falls 0.635 -> 0.519 (-1.34 old SDs) and the randomised `a_G` rises 0.624 -> 0.746 (+0.60). `b_base_L`, the mediator baseline added to the outcome law, is inert at 0.016 +/- 0.087, and the outcome leg barely moves (`b_M` -0.06 old SDs, `b0` -0.07). That is exactly the asymmetry the DAG derivation implies: baseline letter sounds reach word reading only through the mediator, so conditioning on the mediator already blocks that path, whereas baseline word reading genuinely confounds the mediator law. The outcome-leg term is a contract fix; the mediator-leg term is the substantive one.

**No fitted sample changes anywhere.** Checked across all 19 registered models against their stored traces: every model keeps its stored fitted row count (53 for most, 50 for MED-060/074/086/186, 51 for MED-076, 52 for MED-176, 157 for MED-092). Relaxing `pre_required` adds no rows because 53 was already the ITT-phase maximum, and the off-floor models stay at 50 because they now model the baseline they were already requiring. The refit is therefore driven purely by the likelihood change, not by sample membership.

**What this settles.** "Refit and re-report", not "withhold from interpretation as probably wrong". Direction, rough magnitude and the mediated-share conclusion are all stable on MED-059; what moves is a modest, one-directional shift that nonetheless changes one interval's zero-crossing status. The existing artefacts should be treated as provisional pending refit rather than as findings withdrawn.

**MED-059 is a lower bound, not a representative case.** Its outcome-leg addition is inert and its sample is unchanged. The off-floor models (MED-060/086/186) gain a _strongly prognostic own-baseline_ term — the project's own floor-rule calibration notes period-1 control children moving off the floor at 2/17 when at-floor versus 7/8 when off-floor — so their decompositions should move more. MED-092 changes its headline window on top of the leg change, a shift of about 18% in the total (+2.66 versus +3.25 items on a dev fit). Those three should be checked individually as the refit batch proceeds rather than assumed to behave like MED-059.

## What is deliberately not in this change

**No refits.** Every code change above alters a likelihood, so all 19 stored artefact sets are pre-remediation and must be refitted at the `reporting` preset before any mediation number is interpreted — the comparison above shows the numbers move well beyond Monte-Carlo noise, so regenerating summaries over the stored traces is not an option. That is a long sweep and, per the repo's convention for review-fix batches, is tracked separately from the code PR.

**No LOO.** `compute_loo=False` remains hard-coded. The review is right that omitting predictive validation should be an explicit decision rather than a claim that mediation models have no predictive unit, and right that `MED-092` would need a marginal leave-child-out target rather than a sum of conditional nodes. That design work is not attempted here.

**No parallel-mediator dependence model.** Finding 7's substance — that the product-of-marginals draw is a stochastic intervention unless conditional independence holds, and that the chain components are sequential allocations rather than order-invariant path effects — is a naming and modelling question that outlives this batch.
