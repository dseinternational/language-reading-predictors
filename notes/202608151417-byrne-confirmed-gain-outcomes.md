> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne confirmed-input grammar and memory gains (#409 D1)

## Decision

Complete the confirmed-input portion of the wider-outcome D1 matrix with `lrp-rlm-adj-004` plus its `lrp-rlm-hs-003` regularised ranking cross-check for TROG receptive-grammar gain, and `lrp-rlm-adj-005` for BAS digit-recall gain. Each adjusted model has its bivariate and prior-width companions. All three registered models use the paper-compatible waves 1→3, condition the wave-3 outcome on its own wave-1 baseline, and use the same four other confirmed measures plus age as wave-1 predictors.

This scope follows measurement integrity rather than estimated association strength. TROG and BAS digit recall have confirmed instrument identities and score ceilings, while BAS spelling and word completion still use provisional observed-maximum ceilings. The latter outcomes therefore remain outside a bounded-count analysis until those ceilings are documented. The confirmed-input D1 scope now covers BPVS receptive vocabulary, TROG receptive grammar and BAS digit recall; it does not complete the wider provisional-input matrix originally sketched in #409. A proposed BAS digit-recall horseshoe was fitted but not registered: the higher-information `rep-lite` fit retained four divergences at `target_accept=0.99`, two at 0.995 and one at 0.999. The clean short test-tier run at 0.995 did not override that evidence. Horseshoe rankings are zero-divergence-only under the project policy, so retaining the adjusted model without a forced cross-check is the more defensible result.

## Analysis frames and estimands

The TROG models use 69 complete cases from the prepared 97-child extract: 21 children with Down syndrome, 30 average readers and 18 reading-matched children. The BAS digit-recall adjusted model uses 71: 22, 30 and 19 children respectively. No predictor is imputed. The standardised predictor matrices have modest condition numbers (4.44 for TROG and 4.55 for digit recall), although their strongest absolute pairwise correlations are 0.76 and 0.77; regularisation remains important, but it does not make individual conditional slopes independent pieces of evidence.

The own baseline is part of each linear predictor, so these are post-score-given-baseline estimands rather than literal difference-score analyses. Every quantity is a descriptive association. The historical groups were not randomised, complete-case selection can distort the fitted samples, and reading-matched children were selected on BAS word-reading level. Because `basread` is a focal predictor in both outcomes, its pooled slope and the other coefficients conditional on it carry a specific selection caveat that nuisance group dummies cannot remove.

## Release boundary

Confirmed measurement inputs remove one avoidable blocker but do not resolve the cohort source-provenance mismatch: the prepared extract contains 97 participants while the separate raw export contains 96. Scientific result text remains fail-closed until that source reconciliation is documented.

Both adjusted test-tier primary fits passed the computational gate with zero divergences. TROG had maximum R-hat 1.0023 and minimum effective sample size 3,176; digit recall had maximum R-hat 1.0017 and minimum effective sample size 3,623. All seven bivariate and alternative-prior-width refits per outcome passed their own convergence checks. Child-level PSIS-LOO had no Pareto-k value above 0.70 (maxima 0.397 and 0.538 respectively).

The TROG horseshoe required a smaller integrator step: its `rep-lite` fit had two divergences at the inherited `target_accept=0.99`, then passed at 0.999 with zero divergences, maximum R-hat 1.0010 and minimum effective sample size 7,469. Its maximum child-level Pareto-k was 0.389. The registered default records 0.999. These diagnostic passes establish computational adequacy at the sampled tiers; they are not publication clearance, and no posterior scientific findings are reported while the input gate remains unresolved.

Reference: Byrne, A., MacDonald, J., & Buckley, S. (2002). Reading, language and memory skills: a comparative longitudinal study of children with Down syndrome and their typically developing peers. _British Journal of Educational Psychology, 72_(4), 513–529. https://doi.org/10.1348/00070990260377497
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald basread basdig bpvs bassim trog -->
