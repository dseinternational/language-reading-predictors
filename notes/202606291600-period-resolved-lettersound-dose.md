# Period-resolved letter-sound (L) session-dose readout on the DiD machinery

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

**Issue:** #135 (salvaged from the closed PR #108 / LRP86). **Models:** `lrpdid07`
(period-varying), `lrpdid07base` (pooled comparator).

## Question

Does the **letter-sound (L) session-dose slope vary by period?** `lrpdid06` gives
the word-reading (W) session dose-response with a single pooled `beta_dose`; the
letter-sound analogue with a period-resolved slope was the one genuinely-new
question in the now-closed LRP86 (#108). This is a **bounded check, not a new
family**.

## Design

A small extension of the difference-in-differences family, **not** a new backbone:

- **Identification** reuses `lrpdid02` (L treatment effect): the phase-stacked
  waitlist-crossover frame (P1 = t1->t2, P2 = t2->t3, both arms), each child their
  own control (child random intercept), the immediate arm (treated in both
  periods) anchoring the time/maturation trend.
- **Dose pattern** reuses `lrpdid06`: the binary treated indicator is replaced by
  the standardised per-period intervention-session count (`attend`).
- **New:** the dose slope is **resolved by period** with partial-pooled
  per-period slopes `beta_dose_phase[p] = mu_dose + sigma_dose * z_p` (non-centred),
  added to `build_did_model` behind `period_varying_dose` (default off, so
  `lrpdid06` is unchanged). The pooled `period_varying_dose=False` model
  (`lrpdid07base`) is its nested comparator.
- **Decision:** a nested PSIS-LOO of `lrpdid07` (period-varying) vs `lrpdid07base`
  (pooled), wired into `compare_statistical_models.py`
  (`did_dose_loo_compare.csv`). If LOO prefers the pooled model, quote the pooled
  slope (`dose_overall` / `mu_dose`) and stop.
- Beta-Binomial on the logit scale (the suite convention), conditional change
  (own baseline + age as precision terms), `N = 32` for `yarclet`.

## DAG compliance (#115)

The per-period session count is the **exposure** `IS`, **not** a conditioning
variable: this is the **ID-3 observational dose-response**, estimable but
**confounded by `GA -> IS`**, to be reported as an **adjusted association** -
never "more sessions cause more gain". Crucially the DiD family adjusts only
`{IG, A}` and **never conditions on cumulative sessions** (`attend_cumul`, the
`IS` collider) - the conditioning that closed PR #108 (the locked DAG: "never
condition on `IS` ... would bias even the ITT"; ID-3 prescribes `{IG, A}`). So
this readout sidesteps that problem by construction. The only causal contrast
remains the randomised one (the ITT `lrpitt07` / the DiD arm term `lrpdid02`).

## Expectation (honest)

The most likely outcome is **"no period variation at this n"**: the word-reading
analogue (LRP77, #107) found the dose slope **constant across periods** (LOO
preferred the pooled model), and LRP86's own period-varying L fit carried a
Pareto-k warning with LOO preferring the pooled slope. A clean convergence to
"pooled preferred" is a useful, publishable negative result that retires the case
for a richer period-interaction dose model on L. Sized and framed accordingly.

## References

- `notes/202606260702-did-crossover-design.md` - the merged DiD family design.
- `notes/202606231600-dag-revision-consolidated.md` - locked DAG, ID-3 (dose-response
  identification; `IS` collider rule).
  this mirrors.
- `lrpdid02` (L treatment effect), `lrpdid06` (W session dose-response) - the
  machinery extended.
