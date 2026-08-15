> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne receptive-vocabulary gain over confirmed inputs (#409 D1)

## Decision

Register one matched pair for the first wider-outcome slice: `lrp-rlm-adj-003`, a mutually adjusted model with bivariate and prior-width companions, and `lrp-rlm-hs-002`, its regularised-horseshoe ranking cross-check. Both model wave-3 BPVS receptive vocabulary conditional on wave-1 BPVS and the same wave-1 BAS word reading, receptive grammar, verbal memory, verbal reasoning and age predictors over the paper-compatible waves 1→3. Every measurement input has a confirmed instrument identity and ceiling; the models therefore fail fast if a future edit introduces a provisional input.

## Analysis frame and estimand

The complete-case frame contains 71 of the prepared extract's 97 children: 22 with Down syndrome, 30 average readers and 19 reading-matched children. Twenty-six are excluded because at least one required outcome or predictor value is missing. No predictor is imputed. The own BPVS baseline is part of the linear predictor, so this is a post-score-given-baseline estimand rather than a literal difference-score analysis. The adjusted model estimates per-SD conditional associations under independent Normal slope priors and translates them to the 32-item BPVS scale; the horseshoe ranks the matched slopes by posterior `P(|beta| > 0.1)` under shared global-local shrinkage.

All quantities are descriptive associations. The historical reading groups were not randomised, complete-case selection can distort the fitted sample, and the reading-matched group was selected on BAS word-reading level. Because `basread` is itself a focal predictor, its pooled slope and the other coefficients conditional on it carry a specific selection caveat that nuisance group dummies cannot remove. The matched pair is intended for method-level triangulation, not for causal or selection-free claims.

## Why not claim temporal measurement invariance next

The confirmed five-measure battery (`basread`, `bpvs`, `trog`, `basdig`, `bassim`) is not sufficient for a defensible multi-domain invariance model: only the language domain has two indicators, while reading, memory and ability would each be single-indicator factors with fixed reliability. Expanding to the eight-measure common battery would introduce three provisional ceilings (`basspel`, `woco`, `basnum`). Separate one-wave factor fits would also be a descriptive stability screen, not a formal longitudinal invariance test, because the same children recur across waves and the cross-wave dependence must be represented. Phase C measurement-invariance work is therefore deferred pending a better identified and confirmed measurement design; it should not be presented as resolved by mechanically refitting the existing wave-3 factor model.

## Release boundary

Confirmed measurement inputs remove one avoidable blocker but do not resolve the cohort source-provenance mismatch: the prepared extract contains 97 participants while the separate raw export contains 96. Scientific result text remains fail-closed until that source reconciliation is documented.

The adjusted model's test-tier primary fit passed the computational gate with zero divergences, maximum R-hat 1.0014 and minimum effective sample size 3,792. All five bivariate and both alternative-prior-width refits also had zero divergences and passed their own convergence checks. Child-level PSIS-LOO showed no Pareto-k value above 0.70 (maximum 0.437). The test tier is diagnostic-only and is not itself publication-eligible.

The horseshoe's first test-tier fit had two divergences despite `target_accept=0.99` and was rejected. A higher-information `rep-lite` fit then passed with zero divergences, maximum R-hat 1.0008 and minimum effective sample size 5,742; no child-level Pareto-k value exceeded 0.70 (maximum 0.429). This resolves the sampled funnel at the eligible tier, but release remains `inputs_unresolved` because of the 96/97 source mismatch. No posterior scientific findings are reported while that input gate remains unresolved.

Reference: Byrne, A., MacDonald, J., & Buckley, S. (2002). Reading, language and memory skills: a comparative longitudinal study of children with Down syndrome and their typically developing peers. _British Journal of Educational Psychology, 72_(4), 513–529. https://doi.org/10.1348/00070990260377497
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald basread basdig bpvs trog -->
