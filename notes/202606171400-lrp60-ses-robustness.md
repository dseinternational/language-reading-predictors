# LRP60 SES robustness — matched complete-case comparator (LRP52 / LRP60a / LRP60)

> [!WARNING]
> AI-assisted analysis. Numbers reproducible from `fit_statistical_model.py`
> (reporting config); interpretation should be reviewed by the study team.

> [!NOTE]
> **Sign convention superseded.** These results (e.g. "τ = −0.62 is beneficial")
> were computed under the original `G = group − 1` coding, where _negative_ τ =
> intervention benefit. The repo has since standardised on `G = 2 − group`, so
> **positive τ = benefit** (see METHODS.md); the magnitudes are unchanged, the sign flips.

Date: 2026-06-17

## Context

LRP60 adjusts the word-reading ITT for SES (`mumedupost16`, `dadedupost16`,
`agebooks`). Requesting those covariates triggers complete-case dropping in
`load_and_prepare`, so LRP60 runs on **33 of 54** children. A naive LRP52-vs-LRP60
comparison then confounds two things: did **SES adjustment** move the effect, or did
**dropping 21 children** move it? LRP52 (unadjusted) keeps the full sample, so it is
not an apples-to-apples comparator.

## Method

Decouple "restrict to complete cases" from "adjust for". A new
`restrict_complete` argument to `load_and_prepare` applies the existing
complete-case mask to a set of columns **without** adding them to
`prepared.covariates` (so the factory adds no `gamma_*` terms). This lets us fit the
**unadjusted** ITT on the **exact same 33-child subset** LRP60 uses (**LRP60a**),
isolating the adjustment from the sample change.

- Three fits, reporting config (6 chains × 6000 draws), phase-0 randomised window,
  word-reading outcome `W`. All other settings identical across LRP60a/LRP60.
- **Sign convention: negative τ = intervention benefit** (the ITT models' `tau` is
  the control-minus-intervention contrast). `tau` is on the logit scale; HDI is the
  94% highest-density interval; P(benefit) = P(τ < 0).
- Verified: LRP60a and LRP60 land on the **identical** 33 children; LRP60a carries
  **no** SES coefficients; LRP52's full sample is **53** (one child drops on a
  missing pre-score, not 54).

## Results

| Fit        | Sample              | Adjusts SES | τ (logit) | 94% HDI        | P(benefit) | divs | R-hat |
| ---------- | ------------------- | ----------- | --------: | -------------- | ---------: | ---: | ----: |
| **LRP52**  | full, n=53          | no          | **−0.40** | [−0.75, −0.06] |       0.99 |    0 |  1.00 |
| **LRP60a** | complete-case, n=33 | no          | **−0.55** | [−1.00, −0.08] |       0.99 |    0 |  1.00 |
| **LRP60**  | complete-case, n=33 | yes         | **−0.62** | [−1.10, −0.12] |       0.99 |    0 |  1.00 |

(min ESS_bulk ≈ 14k across fits.)

## Verdict — the word-reading effect is robust

- **Selection (LRP60a vs LRP52):** the 33-child subset shows a _slightly larger_
  benefit (−0.55 vs −0.40), credibly beneficial in both, with heavily overlapping
  HDIs. Dropping the 21 children does **not** explain the effect away — if anything
  the complete-case subset is marginally stronger.
- **SES adjustment (LRP60 vs LRP60a, sample fixed):** adjusting moves τ from −0.55
  to −0.62 — _slightly more_ beneficial, not less. SES adjustment does **not** erode
  the effect.
- **τ stays clearly beneficial (P(benefit) ≈ 0.99) across all three fits.** Neither
  the sample change nor SES adjustment is responsible for the word-reading effect.

**On the sign of LRP60.** On the consistent internal convention used by every ITT
model (negative = benefit), LRP60's τ = **−0.62 is beneficial**. Earlier secondhand
notes of a "+0.62" reflected the same magnitude reported in the flipped
intervention-helps-positive direction — **not** a sign flip; there is no reversal.

**Regression check.** LRP60 reproduces unchanged: when `restrict_complete` is empty
(its default), `extra_cols == list(covariates)`, so the adjusted run's data prep and
model are byte-identical to before; the fit still uses n=33 with the three SES
coefficients present.

## Caveats

- n is small (33 in the complete-case fits); HDIs are wide. This is a **robustness
  verdict, not a new headline**, and claims are restricted to the randomised phase-0
  window as for the other ITT models.
- LRP60a removes the confound between adjustment and sample by _holding the sample
  fixed_; it does not recover the dropped children.

## Step 2 (Bayesian imputation, full sample) — not built

Step 2 would keep all 54 by imputing the missing SES covariates inside the model
(a **missing-at-random** assumption). It was **not built**: Step 1 is already
decisive — the complete-case subset is, if anything, _more_ beneficial than the full
sample, so the selection concern does not threaten the verdict and a full-sample
imputed estimate would very likely land between LRP52 and LRP60, still beneficial.
It remains available as a flagged MAR sensitivity check (`lrp60b`) if the team wants
the full-sample number.

## Reproduce

```
python scripts/fit_statistical_model.py lrp52   --config reporting
python scripts/fit_statistical_model.py lrp60a  --config reporting
python scripts/fit_statistical_model.py lrp60   --config reporting
# tau_summary.csv in each output/statistical_models/models/<id>-reporting/
```
