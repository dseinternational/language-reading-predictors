> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Verifying the mechanism audit (#581 / #586) and implementing its fixes, 2026-08-23

## Purpose

`notes/202608231106-mechanism-statistical-model-audit.md` recorded an independent audit of the `mechanism` family and #586 tracked its remediation. This note records (a) an independent re-verification of every claim in that audit against the code, the current data and the 41 stored `reporting` fits, (b) two places where the audit was imprecise, (c) one finding it materially understated, and (d) what was implemented in response.

## Verification outcome

Every one of the audit's thirteen numbered findings and all five dormant-configuration defects were reproduced. The audit's own verification figures were also reproduced exactly: 41 registered specifications, all passing their computational gate, zero divergences, maximum R-hat 1.0042, minimum effective sample size 1,886, minimum chain BFMI 0.663.

Reproduced quantitatively:

- **mech-191's population.** 55 missing `attend` values, of which 54 are t4 cells that can never be a transition's pre row and one is a relevant t2 cell. The fitted frame held 156 rows from 53 children with 28 zero-session rows; in period 1 all 25 fitted wait-list rows sat at zero and no immediate-arm row did, with only seven fitted rows anywhere between 1 and 30 sessions. The module docstring's stated mechanism — that the missing values "are dropped by the factory's exposure keep-mask" — is wrong: the loader records an absent session count as a zero, so the keep-mask dropped nothing.
- **Functional forms.** 21 linear, 20 HSGP, of which 16 fit `InverseGamma(8, 8)` and 4 the `InverseGamma(5, 5)` default. `prior_ell.svg` in every HSGP fit carried the literal label `InverseGamma(alpha=3, beta=1)`.
- **Unused complete-casing.** Exactly mech-063 and mech-163 change, 151 → 155 rows each; every other registered model's fitted rows are unchanged.
- **mech-158.** Its resolved plan differed from mech-058's in loading contract (8 pre-score requirements vs 2), HSGP basis (10 vs 6) and lengthscale prior (`InverseGamma(5, 5)` vs `InverseGamma(8, 8)`) as well as missing-data policy; 128 rows / 44 children against 156 / 53.
- **Prior sensitivity.** All 20 stored HSGP fits have **both** focal GP hyperparameters power-scaling flagged, not "amplitude or lengthscale" — the audit understated this.
- **Negative control.** The ability-adjusted L → F coefficient is +0.249 (89% 0.121 to 0.379), against the L → W anchor's +0.248 (0.151 to 0.345).

## Where the audit was imprecise

1. **Finding 7 is latent, not manifest.** No stored fit pairs an inconclusive `gamma_int` with settled items-scale evidence, so no published report currently carries the false confirmation. The ordering defect is real and was fixed; the issue text should have said the exposure was latent.
2. **The L → W anchor quoted at +0.248 is mech-101, which is _not_ ability-adjusted.** The like-for-like ability-adjusted comparator is mech-201 at +0.244 (0.147 to 0.343). The argument is unchanged, and slightly stronger, with mech-201.
3. A first pass of this verification wrongly reported "controlled direct effect" as a phantom finding. That was a search error: the phrase is hyphenated in the codebase (`controlled-direct`) and appears in six model modules and four report templates. The audit was right.

## Where the audit understated the problem: the knee is boundary-pinned

The audit's finding 1 argues that the criterion _could_ certify a knee that does not exist, illustrating with a hypothetical straight line. The stored fits fail it concretely, and by a sharper mechanism.

Recomputing the binned slopes for mech-058 — the family's flagship letter-sound → word-reading readiness result:

| bin centre (LS items)                | 8.5 | 17.5   | 22.0   | 25.0   | 28.0   | 31.0       |
| ------------------------------------ | --- | ------ | ------ | ------ | ------ | ---------- |
| median between-bin slope             |     | 0.0229 | 0.0226 | 0.0283 | 0.0408 | **0.0592** |
| share of draws picking this interval |     | 0.059  | 0.007  | 0.033  | 0.169  | **0.733**  |

The curve accelerates monotonically to the **last** interval and 73% of draws put the maximum there. Across all 13 letter-sound HSGP fits the knee median is 29.5–29.75 on an observed range of 2–32 against a 32-item ceiling, and the knee median equals its own 89% upper limit in every one: the interval is right-censored by the edge of the data. The published "reading rises fastest around ~30 letter sounds" is where the data stop, not where the curve bends.

Under the corrected criterion — net rise, an interior winning interval, and moderate evidence that the slope above exceeds the slope below — **none of the 20 stored HSGP mechanism fits has a well-defined knee**. Eighteen are boundary-pinned; the rest fail the curvature check.

Two further defects not in the audit:

- mech-191's rendered report printed the literal string `nan` for its below-knee slope (its steepest interval was the lowest one, so there was no "below" set to average).
- The readiness prose hard-coded letter-sound examples and count units in every report, including the sessions model, where the exposure is neither a count nor letter sounds.

## Decisions taken

Three decisions were signed off before implementation (Frank, 2026-08-23):

1. **mech-191**: restrict to on-intervention periods and refit, rather than retaining the zero-anchored association and correcting the prose. The fitted result becomes an intensive-margin association among treated periods and no longer borrows the randomised zero-dose anchor. The restricted frame is 128 rows from 52 children over 10–94 sessions.
2. **Refits**: land the code and prose now, refit as a separate tracked batch. Four models are affected — mech-063, mech-163 (row contract), mech-158 (matched to mech-058) and mech-191 (population).
3. **Batch C**: the three genuinely new analyses the audit proposes — a Mundlak between/within decomposition, a phase-stability sensitivity and a near-Binomial-capable dispersion prior — are deferred to their own issues. They are new modelling with design choices, not corrections.

## What the corrected criterion is

`reporting._readiness_knee` now reports `scale="latent_logit"` and three qualification diagnostics beside the location, with `knee_well_defined` their conjunction:

- `increasing_frac` > 0.9 — the curve rises at all (the pre-existing check);
- `boundary_pinned` — whether the modal steepest interval is the first or last, in which case the location is bounded by the end of the observed range rather than identified within it;
- `prob_slope_above_gt_below` ≥ 0.91 — the shared evidence ladder's "moderate" rung (10:1 odds) on the local slope contrast. For a straight line this sits near 0.5 whatever the net rise, which is precisely what `increasing_frac` cannot detect.

`steepest_interval_share` reports selection stability. A fit written before this criterion carries no verdict column, and the report fails closed on it rather than inheriting the old claim.

The items-scale question is separate and deliberately not answered here: `d E[y] / dx` carries an inverse-link `p(1 − p)` factor and can peak elsewhere, which a regression test now demonstrates. Computing it under a declared reference population is Batch C work.

## Verification of the implementation

- Full test suite passes (exit 0), with 26 new tests across `test_readiness.py`, `test_key_findings.py`, `test_prior_inventory.py` and `test_mechanism_run_plan.py`.
- All 41 registered specifications still resolve; the new rejections catch none of them.
- 37 of 41 stored fits were regenerated from their stored traces without resampling, via `scripts/regenerate_mechanism_artefacts.py`. Every `f_mech__ell` row now reconciles its distribution, rationale and density panel; the four awaiting refit are reported as `needs refit` and the script exits non-zero.
- mech-058 (tight HSGP), mech-096 (linear) and the moderated fits were re-rendered and read end to end. The linear reports no longer claim an HSGP curve or an `InverseGamma(5, 5)` lengthscale, and mech-058 now states plainly that its steepest interval is not qualified as a knee, with both reasons and the prior-sensitivity qualification.
