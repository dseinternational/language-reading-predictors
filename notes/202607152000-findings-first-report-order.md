<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Findings-first statistical-report order

- **Date:** 2026-07-15
- **Status:** implementation decision
- **Issue:** #321

## Decision

Use the same findings-first order for every statistical-model report: header and shared setup; a compact sampling-quality gate badge; the gate-interlocked key-findings box; the collapsed reading guide; model-specific overview and model prose; the family results partial; priors; the prior-predictive check; a collapsed Technical checks section containing the full convergence banner, sampling diagnostics and analyst-grade posterior-predictive views; then the footer.

The gate badge sits above the key findings. A passing fit gives the reader a one-line green verdict before any claim. A failed, missing or inconsistent gate fails closed: the badge is red, names the failed checks where available, and the key-findings renderer withholds result sentences and shows its own warning. The full convergence detail remains present inside Technical checks, so changing default visibility does not weaken the diagnostic gate.

The prior-predictive check stays visible after results. This was signed off on 2026-07-14 because it is part of the scientific argument about whether the prior can generate plausible outcomes, not merely sampling plumbing. The full posterior-predictive and sampler diagnostics move into the fold. Headings nested inside that fold may drop out of the page table of contents; that accepted trade-off keeps the default table of contents focused on the scientific narrative.

## Implementation guardrails

The one-off `scripts/restructure_statistical_reports.py` rewrite recognises only the shared statistical-report include contract. It validates the complete inventory before writing, lists every non-conforming report and makes no changes if any template fails the contract. It replaces only the managed include blocks: model-specific prose remains unchanged, while the recognised family-result include moves ahead of the priors and technical blocks. The direct `_convergence` and `_diagnostics` includes are retained through the shared collapsed `_technical` partial rather than deleted.

Verification combines unit tests for the pass/fail badge and the key-findings interlock, an inventory test for the include order of every statistical template, an idempotence and prose-preservation test for the rewrite utility, and a Quarto-rendered failed-gate fixture. The rendered fixture must show the red badge and withheld-findings banner, omit a deliberately planted finding sentence, keep both nested technical partials in the HTML, and render the technical callout as collapsible.
