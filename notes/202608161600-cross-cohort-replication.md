> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Matched cross-cohort replication: RLI and Byrne

## Decision

Issue #409 asked for side-by-side replication of three associations across the RLI intervention study and the Byrne, MacDonald and Buckley historical cohort: age with reading gain, verbal memory with reading gain, and the stable vocabulary–reading correlation. The existing fitted models do not provide a defensible like-for-like forest because their outcomes, adjustment sets and estimands differ. The replication therefore re-estimates each question with one common exploratory estimator in both cohorts. It does not pool the cohorts or treat either association as causal.

For the two gain-framed questions, the common estimator is ordinary least squares for standardised later word reading, conditional on standardised baseline word reading, baseline age, baseline verbal memory and study-group indicators. Bounded reading scores receive a Haldane–Anscombe corrected logit before standardisation. RLI uses time 1 to time 4; Byrne uses wave 1 to wave 3, the prespecified common follow-up window. Calling this “gain-framed” means that baseline reading is held constant; it is not a raw change-score regression.

For the stable correlation, the estimator takes children complete for reading and receptive vocabulary at waves 1–3, removes each study-group-by-wave mean from both logit-transformed measures, averages those residuals within child, and computes Pearson's correlation across children. This is a matched descriptive stable-level estimate, not the latent correlation from either cohort's structural model.

Uncertainty uses 2,000 non-parametric bootstrap resamples of children, stratified by study group. The reported ranges are the project's standard 89% equal-tailed percentile intervals.

## Findings

Baseline age has the same negative direction in both cohorts. In RLI, one within-study standard deviation higher baseline age is associated with 0.18 standard deviations lower later word reading after adjustment (coefficient −0.18, 89% interval −0.32 to −0.02; 51 children). In Byrne, the corresponding estimate is −0.28 (−0.38 to −0.20; 76 children). This systematically reproduces the earlier age finding, but differences in instruments, follow-up duration and cohort composition mean the two magnitudes are not directly comparable.

Verbal memory does not give a resolved association in either cohort. The RLI Early Repetition Battery estimate is +0.01 (−0.19 to +0.22; 51 children); the Byrne BAS recall-of-digits estimate is +0.08 (−0.06 to +0.25; 76 children). Both ranges include zero. The non-negative point estimates are compatible with a weak positive association, but the data do not distinguish that from no association.

The stable receptive-vocabulary–word-reading correlation replicates closely. It is +0.66 in RLI (89% interval +0.53 to +0.77; 53 children) and +0.66 in Byrne (+0.54 to +0.75; 72 children). This says that children who are stably higher in receptive vocabulary within their study group also tend to be stably higher in word reading. It does not establish direction, mechanism or causality.

## Limits and release status

The shared estimator removes a major comparability problem, but it cannot make the studies identical. RLI uses the 79-item word-reading composite, ROWPVT receptive vocabulary and Early Repetition Battery total across times 1–4; Byrne uses BAS word reading, the BPVS short form and BAS recall of digits across waves 1–3. The group indicators also serve different roles: randomised intervention arm in RLI and observational reader group in Byrne. Complete-case selection differs by question and cohort. Standardisation makes directions and relative within-study associations readable on one type of scale; it does not justify a pooled estimate.

The Byrne source lineage was reconciled later on 2026-08-16 (`notes/202608161340-byrne-source-provenance-reconciliation.md`). The prepared 97-participant extract exactly matches the checksum-pinned 97-case SPSS source on all 52 shared non-identifying fields; the separate 96-row CSV is an incomplete derivative missing one Down-syndrome participant. Fresh tables and figures therefore record `comparison_publication_ready = true` for the input contract and retain the source-file digest and reconciliation note. That flag clears the source and selected-measure inputs only: it does not make the associations causal, poolable across instruments or immune to complete-case selection.

## Reproduction

Run `python scripts/exploratory/cross_cohort_replication.py`. The script writes the full audit table and three separate forest plots under `output/exploratory/cross_cohort/`; separating regression coefficients from correlations avoids putting quantities with different meanings on one axis.

## Reference

Byrne, A., MacDonald, J. and Buckley, S. (2002). Reading, language and memory skills: A comparative longitudinal study of children with Down syndrome and their mainstream peers. _British Journal of Educational Psychology_, 72(4), 513–529. https://doi.org/10.1348/00070990260377497
