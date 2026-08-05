<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 5).

# Decision: a diagnosed robustness gap withholds the key-findings causal headline (#392 P1)

**Decided 2026-08-05.** Option **A — evidence-bound withhold** of the three offered on #392, taken by Frank Buckley. This note records the policy, the reasoning, the sub-decisions taken alongside it, and the practical consequence, because a release rule that changes what a report is allowed to say is exactly the kind of decision a future reader will want to interrogate rather than infer from the code.

## The defect

`generate_key_findings` checked the sampling-quality gate and nothing else. Two robustness gaps the suite already diagnoses could therefore sit unresolved while the findings-first box published an unqualified cause-and-effect sentence:

- a power-scaling **prior-data conflict** on `tau`, which `_emit_itt_extras` computes for every ITT fit and then no release path consulted;
- a **P/N floor-rule** model whose required six-cell treatment-prior grid was absent, incomplete, or not provenance-aligned.

The second is the sharper failure, because the machinery to catch it already existed. `_results_floored.qmd` refuses to release under precisely those conditions — but it renders _after_ the key-findings box. A reader met an apparently released headline and then, further down, a callout stating that the release gate had failed. The box and the prose contradicted each other, and the box wins the reader's attention.

## The policy

A fit that passes the sampling gate is additionally classified on its causal term's power-scaling statistics, at ArviZ's default flag threshold `t` = 0.05:

| Class               | Condition                      | ArviZ diagnosis                          | Action                                  |
| ------------------- | ------------------------------ | ---------------------------------------- | --------------------------------------- |
| clear               | otherwise                      | `✓`                                      | release                                 |
| prior-data conflict | `prior >= t` and `lik >= t`    | potential prior-data conflict            | release **with a note**                 |
| prior-dominant      | `prior > t > lik`              | potential strong prior / weak likelihood | **withhold** without validated evidence |
| unavailable         | no unique, parseable `tau` row | —                                        | **withhold**                            |

The three substantive classes reproduce `arviz_stats.psense_summary`'s own predicate comparison for comparison, so a fit's release class and the psense table printed in its own report can never disagree. That is worth doing deliberately rather than inventing a rule from the numbers, and the first draft of this gate shows why: it used "flag whenever either statistic exceeds the threshold", which gets the healthiest case exactly backwards. A posterior sensitive to the **likelihood** and insensitive to the prior is the ideal — the data are driving the result and the prior is doing nothing. Kallioinen et al. (2024) classify on prior sensitivity and consult the likelihood only to separate a conflict from a prior-dominated posterior. The error surfaced immediately on a real `lrp-rli-itt-001` dev fit (prior 0.015, likelihood 0.092, ArviZ `✓`), which the draft rule would have shipped with a spurious "the prior attenuates this" note; a grid test now pins the agreement across both sides of the threshold.

Floor-rule models take a separate branch that mirrors `_results_floored.qmd` exactly: the grid is required when `tau_psense_status` returns `conflict` or `unavailable`, and a required-but-not-ready grid withholds. Both gates call the same `evaluate_floor_sensitivity`, which recomputes convergence, effects and provenance from the content-addressed traces rather than trusting the CSV, so the two cannot drift apart.

A withhold sets a new `robustness_unresolved` status: no sentences, a reason naming the unresolved evidence, and an `evidence` field naming what would lift it. `_key_findings.qmd` renders it as a `callout-important` and sets `_scientific_results_released = False`, so the result tables and figures are suppressed too — the same treatment a failed sampling gate gets.

### Why a conflict is not automatically a withhold

This is the part most likely to look inconsistent from outside, so it is worth stating directly. The suite's effect priors are zero-centred and deliberately conservative. A prior-data conflict on such a prior means the prior is **pulling a real effect towards zero**, not that it is manufacturing one — and in that class the likelihood is moving the posterior too, so the data still have a say. Withholding there would suppress findings for being _understated_, which is not a reader-protection argument. The case that genuinely warrants a gate is the one where the prior moves the posterior and the data do not, so the direction of the reported effect is not established by the data alone — ArviZ's "potential strong prior / weak likelihood".

The released-with-note branch says so in the box: the size is best read as a lower bound, the direction is the more reliable part.

### Why "unmeasured" withholds

#381's central meta-finding was that a report with no psense table shows no flags, and a reader cannot tell "measured, no concern" from "never measured". Treating an absent or ambiguous diagnosis as a pass would rebuild that ambiguity inside the release gate itself. `tau_psense_status` was already fail-closed on this reasoning for the floor gate; the same standard applies here.

The practical cost is low and the repair needs no refit. Power-scaling runs before report finalisation in the ITT pipeline, so a fit produced today always has its psense row. An older stored fit that lacks one is repaired by `scripts/regenerate_psense.py` followed by `scripts/regenerate_key_findings.py`, and the withhold reason distinguishes "never measured" from "the prior out-works the data" so a reader knows which they are looking at.

Duplicate `tau` rows count as unavailable rather than first-wins. A gate that silently picks one of two disagreeing diagnoses is worse than one that reports it cannot tell.

## Sub-decisions taken alongside the ruling

**Tiering: uniform.** The withhold applies equally to base ITT models, adjusted-robustness models and outcomes outside the standard 44-cell sweep. This was the default offered alongside option A. The graded alternative raised earlier in the discussion — withhold for primary, qualify-never-withhold for adjusted robustness — is a one-line change to `_WITHHOLD_TIERS` in `statistical_models/release.py`, and the tier a fit was judged in is recorded on every decision so the audit trail survives either choice.

**Evidence that lifts a withhold.** For floor-rule outcomes, the provenance-validated six-cell grid the existing gate already recomputes. For ordinary ITT, a `tau_prior_sensitivity.csv` sweep meeting all three clauses of the stated bar — present in the fit's output directory, computed from the same trace and commit as the posterior, and showing the sign of the effect is stable across the grid. Concretely the file must be readable and carry the standard sweep's full column set (so a hand-rolled CSV of the same name cannot pass), have rows for this fit's outcome across at least two prior scales, have every cell converged, record `primary_config_sha256` / `primary_trace_sha256` matching this directory's own `config.json` and `trace.nc`, and keep one sign for `tau_logit_mean` throughout. A lifted withhold becomes `qualify` — the finding ships, labelled prior-informed and exploratory.

Sign stability is the bar rather than interval width, because a conservative prior is _expected_ to move the magnitude; only a direction flip means the reported effect was the prior's doing.

The per-fit checks deliberately do not call `evaluate_standard_sensitivity`. That evaluator measures a sweep against the complete 44-cell cross-outcome grid, so its `ready` can only be true for the sweep-level artefact that lives outside any single fit's directory — calling it here would withhold unconditionally. The checks above are the per-fit subset of the same idea. (The first implementation checked only that the file existed, which would have let an empty file lift a withhold; caught in review on #475.)

**P/N released-case wording.** When a floor model does release, its causal sentence now names the post-hoc, data-adaptive, baseline-floor subgroup and states that the result is neither the effect for all randomised children nor the effect for children already off the floor. Previously these emitted the ordinary ITT sentence, which understated both the narrower population and the post-hoc selection.

**A gate that cannot be evaluated withholds.** If the gate itself raises, the decision is a withhold naming the exception, not a silent pass. Degrading to "no gating" would reinstate the original defect precisely when something unexpected is wrong. It costs no data — every CSV is still written — and is loud.

## What this does _not_ change

No estimand, prior, likelihood, analysis population or sampling setting moves. Nothing is refitted. The gate reads artefacts already in the output directory, so it can be applied to stored fits by regenerating their key findings.

The expected practical consequence on the current suite is **small, and concentrated where it should be**. `notes/202607261700-psense-coverage-backfill.md` measured every ITT headline `tau`: 28 clear, 15 potential prior-data conflict, and **zero** in the prior-dominant class. Under this policy the 15 are released with an attenuation note, not withheld. The models this actually bites are the P/N floor primaries missing their grid — which is the case #392 raised.

## Scope, and what is deliberately left

ITT only, including the floored P/N primaries — the family #392 reviewed. The same rule was proposed to mirror onto `did`'s `tau_t2` and `gain_factors`' `beta_trt`; `ReleaseDecision` and the classifier are family-agnostic so that mirroring is wiring rather than redesign, but it is left as a follow-up so this change stays reviewable against the issue's own scope. A `gain_factors` fit with no psense at all still releases today, and there is a test asserting that, so the change cannot silently gate the rest of the suite.

## Related

#392 (the review and the three options), #381 (power-scaling coverage, and the measured-versus-unmeasured distinction this rests on), #382 (the sensitivity refits that would supply lifting evidence), #464 (why a note must not be appended past the five-sentence cap — it silently drops the causal sentence), `notes/202607261700-psense-coverage-backfill.md` (the per-family measurement quoted above).
