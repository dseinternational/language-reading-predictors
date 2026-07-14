<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Time-lagged models: concrete designs for the reverse-coupling suite

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5). For team sign-off before implementation (#250; models #229, mediation follow-on #264).

## Aim and status

The model plan (`notes/202607131400-time-lagged-dag-model-plan.md`) set the phases; this note turns them into concrete model specifications ready to pre-register. It also resolves — with verified derivations rather than judgement calls — the two sub-decisions the plan left "for the meeting": **which reverse edges are pre-specified versus exploratory**, and **how to handle the waitlist crossover**. The lagged DAG is the adopted Option A structure (`dag/dag-language-reading-lagged.dagitty`, 2026-07-13); nothing here changes the graph.

Everything below that says `[VALID]` or `[NOT-VALID]` was checked mechanically with networkx d-separation runs against the `.dagitty` file and against a crossover-aware **three-slice unroll** of the same structure (the two-slice template cannot represent the arm-specific intervention windows, so the three-slice graph is where the crossover consequences appear). The script is preserved at [`assets/202607141030-lagged-dsep-checks.py`](assets/202607141030-lagged-dsep-checks.py) and reruns from the repo; the implementation PR should promote it to a test so the derivations cannot silently go stale when the DAG is next revised. All checks remove latent general ability `GA` first — no measured set can block it, so each check asks the honest question ("`GA` aside, does this set block every backdoor?") and every coupling below is labelled an **adjusted association** accordingly, per the suite convention.

## What the data will support (survey, 2026-07-14)

The four team-directed reverse edges are `WR → TE`, `WR → TR`, `WR → PA` and `WR → RW`. Data support for the corresponding measures:

| Edge target                  | Measure / column               | Waves observed (of 54/54/54/53 rows) | Floors / ceilings                                           | Verdict                  |
| ---------------------------- | ------------------------------ | ------------------------------------ | ----------------------------------------------------------- | ------------------------ |
| `TE` taught expressive vocab | `TE` / `b1extau` (n = 24)      | 54 / 54 / 54 / 53                    | none (means 5 → 10, no cell at 0 or 24 beyond one child)    | strong                   |
| `TR` taught receptive vocab  | `TR` / `b1retau` (n = 24)      | 54 / 54 / 54 / 53                    | none (means 12 → 17)                                        | strong                   |
| `PA` blending                | `B` / `blending` (n = 10)      | 54 / 54 / 54 / 53                    | mild late ceiling (14–21 % at 10 by t3–t4, immediate arm)   | usable, Beta-Binomial    |
| `RW` phonological memory     | no registered measure (`erb*`) | `erbto`: 53 / 51 / 51 / 47           | denominators undocumented; one `erbword` value out of range | **deferred** (see below) |

Word reading `W` (`ewrswr`, n = 79) is already a modelled LCSM process; its heavy t1 floor (35–43 % at 0) is handled by the latent + masked Beta-Binomial machinery as in `lrp-rli-lcsm-067`.

**`W → RW` is deferred**, on three independent grounds: (a) the ERB columns have no registered `Measure` and no documented test ceiling, and the `n_trials_confirmed` discipline in `measures.py` exists precisely to stop us guessing a Beta-Binomial denominator; (b) `erbword` contains an out-of-range value (28 at group 2, t4, where every other cell tops out at 18) that needs a data-quality check before the column is trusted as an outcome; (c) the identification result below puts it in the same unfittable class as `W → PA` anyway. `erbto` keeps its existing role as the standardised `rw` adjuster covariate (#245). Deferral is until the ERB scoring is documented and the stray value resolved, not a judgement that the edge is uninteresting.

## The intervention structure the models must carry

From the session records (`attend`, which counts sessions in the interval following each wave): the immediate arm (group 1) has sessions in all three windows (means ≈ 73, 65, 52), the waitlist arm (group 2) in windows 2–3 (0, ≈ 73, ≈ 56). So **only window 1 (t1 → t2) is a randomised contrast**; in window 2 both arms are on the programme (immediate on block 2, waitlist crossing over to block 1 — the words `TE`/`TR` measure); window 3 is continued delivery for both arms. Two verified consequences:

1. **Arm is a confounder of every coupling on transitions 2 and 3.** On the three-slice unroll, the set that identifies `WR_t → TE_{t+1}` on transition 1 fails on transition 2 until `IG` is added (`[NOT-VALID]` without, `[VALID]` with). Intuition: window-1 teaching raises the immediate arm's reading at t2 (the exposure), and window-2 teaching raises the waitlist arm's block-1 taught-vocabulary change over t2 → t3 (the outcome) — omit arm and the coupling absorbs a spurious negative component. Any model that pools couplings across transitions therefore **must** condition on arm × window. This settles the plan's crossover sub-decision, and it is not optional polish: for the taught-vocabulary targets the crossover effect is first-order, because the waitlist arm's block-1 catch-up happens exactly in window 2.
2. **Sessions (`IS`) stay out of the change equations.** Dose is the locked DAG's collider (ID-3): with `GA` latent, conditioning on sessions reopens the ability backdoor. Arm × window indicators derive from randomised `IG` plus design timing, so they block the intervention backdoor without touching the collider. Same rationale as `lrp-rli-lcsm-067` omitting dose; dose-response coupling variants are explicitly out of scope.

Concretely: each measure's change equation gets **arm × window change intercepts** `a_m[g, w]` (2 arms × 3 transitions = 6 cells per measure, each cell informed by 26–28 children), replacing the single pooled `a_change`. A free by-product: the window-1 cell contrast `a_TE[imm, w1] − a_TE[wait, w1]` is the randomised latent ITT contrast on taught-vocabulary change, directly comparable in direction and rough magnitude with `lrp-rli-itt-002`'s τ — a built-in consistency check between the new family and the ITT suite.

## Which reverse edges are pre-specified: the identification result

Minimal measured adjustment sets read off the graph and verified on the three-slice unroll (transition 2, the worst case; `GA` aside throughout):

| Estimand          | Verified sufficient measured set                                                  | Fittable at n ≈ 54?                                                                           |
| ----------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `WR_t → TE_{t+1}` | `{TE_t, TR_t, RW_t, SP_t, age_t, HS, IG}`                                         | **yes** — two modelled processes + four covariates                                            |
| `WR_t → TR_{t+1}` | `{TR_t, RW_t, age_t, HS, IG}` (the `TE`-model superset is also valid)             | **yes** — same model                                                                          |
| `WR_t → PA_{t+1}` | needs ≈ 13 nodes incl. `EV_t, RV_t, TE_t, TR_t, LS_t`, floored `NW_t`, `WR_{t−1}` | **no** — an `081`-sized set is `[NOT-VALID]`; the full set is far beyond the parameter budget |
| `WR_t → RW_{t+1}` | same wide set as `PA`                                                             | **no** (and the measure is unregistered)                                                      |

The asymmetry has a readable cause: `TE_{t+1}` and `TR_{t+1}` have few parents (the taught-vocabulary measures sit at the top of the within-wave cascade), so their backdoors channel through a handful of nodes — and `rw` (`erbto`), `sp` (`deapp_c`) and `hs` are exactly the standardised adjuster covariates the suite already carries (#244/#245). `PA_{t+1}` sits mid-cascade with many parents, several of them floored or two waves back, so blocking it measured-node by measured-node is hopeless at this sample size — and the reverse edge `WR → RW` itself is one of the culprits (it makes `WR_{t−1}` a confounder of `WR_t → PA_{t+1}` through phonological memory).

**Recommendation (decision 1 of 2 for sign-off):** pre-specify `WR → TE` and `WR → TR` — both delivered by one model whose conditioning set exactly matches the verified sets above; demote `WR → PA` to a clearly-flagged exploratory companion (the reciprocal-dominance question is still worth a look, but its couplings cannot be dressed as anything better than broadly-confounded associations); defer `WR → RW` entirely. This replaces the plan-note sketch of "add `TE` alongside W/L/E": the derivation says the confounders of the reverse couplings are `TR`/`RW`/`SP`/`HS`/age/arm, **not** letter-sounds and broad vocabulary, so `L`/`E` processes would spend scarce parameters without closing any backdoor. `lrp-rli-lcsm-067` already owns the forward L/E → reading-change story; the new model does not re-deliver it.

One covariate-timing point to record, since #258 (P1) established the opposite rule for the contemporaneous graph: in the **lagged** graph the licensed adjusters for a change equation are the **prior-wave** states (the parents at the prior wave), so `rw`/`sp` enter at t−1 here — different graph, different licence; both are the observed-parent sets their respective graphs name.

## Model specifications

All are `ModelSpec.kind == "lcsm"` — the McArdle latent change-score scaffold of `build_lcsm_model` generalised, not a new family. Shared machinery: latent logit true-scores per measure with non-centred initial statuses, change-score recursion with per-measure process noise, masked Beta-Binomial observation (no row dropping), couplings pooled across the three transitions, and the established fallback ladder for sampling trouble (tighten priors → shared process noise → no process noise). Priors follow the reconciled suite scales: couplings and covariate slopes `Normal(0, 0.3)` (the shared association scale, prior critical review 2026-07-07 rec. 3), self-feedback `Normal(−0.3, 0.2)`, arm × window change intercepts `Normal(0, 1.5)` pending the prior-predictive check.

### `lrp-rli-lcsm-081` — reading → taught vocabulary (the headline reverse coupling)

The founding RLI hypothesis as a direct test: does a child's prior-wave word-reading standing predict their subsequent change in the block-1 **taught** vocabulary measures, over and above the taught measures' own carry-over, the intervention windows, and everything measured the lagged DAG names?

- **Processes:** `TE`, `TR`, `W` (all four waves, denominators 24/24/79).
- **Change equations** (transitions w = 1..3, child i; `x` are latent logit levels at the prior wave):

  ```text
  Delta_TE[i,w] = a_TE[arm_i, w] + b_TE * x_TE + g_WTE * x_W + c_TR * x_TR
                + d_age * age[i, w-1] + b_hs * hs_i + b_hsm * hs_missing_i
                + b_rw * rw[i, w-1] + b_rwm * rw_missing[i, w-1]
                + b_sp * sp[i, w-1] + b_spm * sp_missing[i, w-1]  (+ process noise)

  Delta_TR[i,w] = a_TR[arm_i, w] + b_TR * x_TR + g_WTR * x_W
                + d_age * age[i, w-1] + (same hs / rw / sp covariate block)  (+ process noise)

  Delta_W[i,w]  = a_W[arm_i, w] + b_W * x_W + d_age * age[i, w-1]      (+ process noise)
  ```

  `g_WTE` and `g_WTR` are the headline coefficients (prior-**level** → subsequent **change**, the same specification-1 form as `lcsm-067` — not change-on-change). With the `TR` process plus the `hs`/`rw`/`sp` covariates, the conditioning set for each headline coupling **equals** the verified sufficient measured set above — the first model in the suite whose adjustment set is read off a graph and machine-checked end to end. The `W` equation is deliberately bare: forward couplings into reading are `lcsm-067`'s estimands, not this model's.

- **Covariates:** `hs`/`hs_missing` time-invariant (the 2026-07-13 hearing decision: one noisy parent-reported baseline, labelled as such); `rw` = standardised `erbto` and `sp` = standardised `deapp_c` at the **prior wave**, filled with per-wave missing indicators exactly as `add_missing_indicator_covariates` does for the transition-stacked families, so no child-wave is dropped.
- **Estimand labels:** `g_WTE`, `g_WTR` are **adjusted associations** (latent `GA` is unblockable; the LCSM's independent initial statuses absorb only part of the stable between-child differences, and the RI-CLPM that would separate them is not estimable at n ≈ 54 — same honesty box as `lcsm-067`/#273). The window-1 arm contrasts `a_m[imm, 1] − a_m[wait, 1]` are randomised latent ITT contrasts, reported as consistency checks against `lrp-rli-itt-001/002`, not as new headline effects.
- **Comparator `lrp-rli-lcsm-181`** (the +100 variant convention): identical minus `g_WTE`/`g_WTR`. PSIS-LOO of 081 vs 181 is the plan's "does the reverse edge earn its place" readout.
- **Deliverables:** posterior mean, 95 % credible interval and `P(g > 0)` for both couplings; the LOO comparison; the ITT triangulation table. **Pre-specified reading:** the RLI hypothesis is supported in this cohort if both couplings are positive with `P(g > 0) ≥ 0.9` and 081 is not LOO-worse than 181; weaker patterns are reported as-is with no post-hoc re-labelling.
- **Parameter budget:** ≈ 45–51 structural parameters (3 processes × 12 + 3 couplings + 6–12 covariate slopes, depending on whether the `hs`/`rw`/`sp` block is shared across the two taught-vocabulary equations — sharing is the recommended default at this n) against ≈ 640 observed child-wave cells — the heaviest LCSM yet (067 has ≈ 25), hence dev-tier gate first, `rep-lite` before `reporting`, and the fallback ladder documented in the spec rather than improvised.

### `lrp-rli-lcsm-082` — blending ↔ word reading, reciprocal dominance (exploratory)

The cross-lagged dominance question the DAG note names: does prior reading predict blending change more strongly than prior blending predicts reading change?

- **Processes:** `W`, `B`, `L` (letter-sounds retained from 067: it is a parent of both and the one cheap measured confounder both directions share).
- **Couplings:** `Delta_W` gets `g_BW * x_B + g_LW * x_L`; `Delta_B` gets `g_WB * x_W + g_LB * x_L`; both target equations carry the arm × window intercepts and the `hs`/`rw`/`sp` covariate block (`B` is blending = `PA`, an `HS`/`SP`/`RW` child); `L`'s own equation stays bare (self + age + arm × window).
- **Readout:** the dominance contrast on comparable scales — per posterior draw, standardise each coupling by the model's own latent scales (`g* = g · sd(x_source) / sd(Delta_target)`) and report the posterior of `|g*_WB| − |g*_BW|` with `P(|g*_WB| > |g*_BW|)`.
- **Label:** **exploratory, association-only, both directions** — the verified minimal set for either direction needs ≈ 13 nodes including floored nonword and two-waves-back reading, so no fittable version of this model approaches the measured blocking set. The mild blending ceiling (14–21 % at 10 in late waves) is absorbed by the Beta-Binomial. Fit only after 081 clears its diagnostics gate (same machinery, similar weight); a hypothesis-generator for a future cohort, not an evidence claim.

### Phase 2 (#264) — what the worked mediation derivation already shows

The three-slice machinery gives #264 its method, and the `TE` worked example lands three usable facts: (a) baseline `E`/`R` (t1 values) are **admissible** mediator–outcome adjusters — they precede treatment on the unroll, so #259's descendant argument fails exactly as the 2026-07-12 interim decision anticipated — but they are **not required** (the set stays `[VALID]` without them); (b) the family's all-baseline conditioning style does **not** strictly block the mediator–outcome backdoors — the graph wants contemporaneous non-descendant states (`TR_t2`, `RW_t2`, `SP_t2`, `LS_t2`); and (c) several of those (`TR`, `LS`) are treatment-affected, so cross-world natural effects stay unidentified and the family's existing interventional-effects framing remains the right one. Practical recommendation to carry into #264: **keep `E`/`R` in** (admissible precision terms; removing them buys no validity), and make #264's deliverable the per-mediator derivation table over all eleven MED models using this script's machinery, refitting only if any set actually changes. No new model is designed here.

### Out of scope (recorded as decisions, not omissions)

- **Lagged change-on-change couplings** (#229 specification 2): two usable transitions at n ≈ 54; revisit as an exploratory 081 variant only if 081 samples cleanly, as its own signed-off spec.
- **RI-CLPM / free cross-lagged system:** stays parked (not estimable; docstring of 067 records it).
- **Dose-response coupling variants:** excluded on ID-3 (sessions collider), as above.
- **`W → RW`:** deferred as above, pending ERB documentation and the `erbword` data-quality check.

## Implementation notes (for the build PR, not for sign-off)

- `build_lcsm_model` generalises: `couplings: dict[target_symbol, tuple[source_symbols, ...]]` replaces the single `reading_symbol` (default `{"W": ("L", "E")}` keeps `lcsm-067` byte-identical); `arm_window_intercepts: bool` swaps `a_change[outcome]` for `a_change[arm, trans, outcome]`; an optional per-target covariate block (time-invariant and prior-wave per-wave covariates with missing indicators).
- `WavePanel` / `load_wave_panel` currently carry no `group` and no per-wave covariates — add `group` (per child) and a `wave_covariates` dict (`erbto`, `deapp_c` standardised + indicator, reusing the `add_missing_indicator_covariates` policy), plus `hs`/`hs_missing` via the existing `add_hearing_status`.
- `fit_lcsm` generalises the coupling table to per-target tables, adds the window-1 ITT-contrast table, and keeps the existing diagnostics/LOO flow; registry entries in `definitions.MODEL_REGISTRY` and thin reports under `docs/models/lrp-rli-lcsm-081/` (`_results_lcsm` partial, extended for multiple coupling tables).
- Promote `assets/202607141030-lagged-dsep-checks.py` to a pytest (`tests/`) so the adjustment-set claims are CI-checked against the `.dagitty` file.

## For sign-off

1. **Reverse-edge pre-specification:** `W → TE` + `W → TR` pre-specified in `lcsm-081`; `W → PA` exploratory in `lcsm-082`; `W → RW` deferred. (Recommended above, with the derivations as the reason.)
2. **Crossover handling:** arm × window change intercepts in every process's change equation, sessions excluded. (Required by the transition-2 derivation, so this is sign-off of the mechanism, not of whether.)

The per-edge one-line justifications the DAG note still owes (`WR → TE`/`TR`/`PA`/`RW`) should land with this sign-off, and 081's spec is written so a dev-tier fit can start the day it is agreed.
