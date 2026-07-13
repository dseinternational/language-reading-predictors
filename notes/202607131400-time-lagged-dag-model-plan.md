# Time-lagged DAG and models: plan for discussion

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8). For discussion before implementation (#250; models #229 / #264).

## Aim

Agree a time-lagged DAG and a realistic model plan before building anything. The DAG makes change over time and the reading-to-language feedback explicit. The models are the small, pre-specified set our sample size can actually support.

## Where we already are

- **Draft lagged DAG (#288):** two options, A recommended, with figures and a note.
- **`lrp-rli-lcsm-067` already exists.** It unrolls the question over all four waves: reading change is coupled to a child's prior-wave letter-sound and vocabulary standing, with the coupling pooled across the three transitions (t1→t2, t2→t3, t3→t4) as a deliberate constraint at n≈54.
- **A full RI-CLPM was judged not estimable at this sample size and never built.** The LRP67 docstring records it as the dropped companion; there is no RI-CLPM model in the suite. Growth and correlated-factor models do exist.

So the lagged DAG is not starting from zero. It is the causal structure LRP67 already assumes, written down so we can review it and justify any additions.

## The binding constraint: n ≈ 54, four waves

This shaped every longitudinal choice so far and it shapes this one. At n≈54 we cannot fit a free system of within-wave and cross-wave effects. We fit a few pre-specified couplings, pooled across waves, with informative priors and a Beta-Binomial likelihood that handles the early-wave floors. A full RI-CLPM would not estimate on these grounds, which is why LRP67 uses the parsimonious LCSM instead.

The consequence for this discussion: **the A-vs-B question is about which edges and adjustment sets we can justify, not about fitting the whole graph.** Whatever we choose, the fitted models stay parsimonious.

## Decision 1: DAG structure, A vs B

- **Option A (recommended):** the base DAG copied at each wave, joined by carry-over and the reverse reading-to-language edges. Keeps the within-wave cascade.
- **Option B (current draft):** pure-lagged; no within-wave skill edges, every effect acts across a wave.

Figures and the full explanation are in #288 (`notes/202607131300-time-lagged-dag-options.md`). Note that LRP67's coupling is already lagged (change depends on prior-wave levels), so at the model level we are close to B in practice regardless; A is the more faithful description of what we believe, and it is what justifies adding a within-wave path later if the sample ever allows.

## Decision 2: the open sub-decisions

- **Reverse edges: which to pre-specify.** `WR → EV` (reading to expressive vocabulary, the print-exposure / RLI headline) is the one to test first. `WR → PA` gives the reciprocal blending test. `WR → RW` (reading to phonological memory) is the most tentative and easy to drop.
- **Is hearing time-varying?** The DAG currently treats hearing (`HS`) as a fixed, measured-once cause. This is exactly what Frank's hearing literature review should settle: if hearing in DS under 12 changes materially over this age range, `HS` should be time-varying, not a stable root.
- **Waitlist crossover.** The intervention's active window is arm-specific (immediate arm t1→t2, waitlist arm t2→t3). The two-slice template hides this; any fitted model must apply the intervention to the correct window per arm.
- **Age and ability.** Age enters per wave; general ability is latent and time-invariant.

## Models the lagged DAG justifies (mapped to issues)

- **Coupling extensions of LRP67 (#229).** Formalise the existing LCSM structure and add one pre-specified reverse coupling: does prior-wave reading predict later vocabulary change, over and above the forward path? This is the RLI hypothesis as a direct test. Pooled coupling, three transitions, same as LRP67.
- **Temporal mediation (#264).** With waves explicit, baseline vocabulary is a clean pre-treatment confounder, and dose → skill → reading can be traced across waves. Interventional estimands where dose is a treatment-induced confounder, as in the current mediation family.
- **Not RI-CLPM / free cross-lagged.** Stays parked unless the sample grows or a very reduced two-variable version is clearly justified.

## Proposed order

- **Phase 0 (now):** settle the DAG. Needs the A-vs-B call and the sub-decisions, at a short meeting. Blocked on the hearing review for the `HS` decision. This note plus the #288 figures are the input.
- **Phase 1:** one pre-specified reverse-coupling extension of LRP67 (reading → vocabulary change), the headline RLI test. Dev-tier fit, diagnostics, report. Decides whether the reverse edge earns its place.
- **Phase 2:** temporal mediation (#264) using the lagged adjustment sets.
- **Phase 3:** crossover-aware structure (arm and period) if Phases 1–2 hold up.

Each phase is a pre-registered spec, a dev-tier fit, then a report, the same workflow as the rest of the suite.

## For the meeting

1. Option A or B.
2. Which reverse edges are pre-specified versus exploratory.
3. How to handle the waitlist crossover.
4. The hearing review result, which sets whether `HS` is time-varying.
