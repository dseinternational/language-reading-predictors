> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Retirement of the pooled-moderation command

The September 2026 code review found that `scripts/pooled_moderation.py` reads `gamma_int_trt_own` from GF-001 through GF-008. These primary models no longer estimate that interaction. The approved implementation retires this command and its unused fitting module. The command now explains the decision and exits without reading model output or fitting a pool. Existing output files are retained as historical artefacts.

Changing the inputs to GF-201 through GF-208 would not be a complete repair. In GF-205, the baseline moderator is a binary off-floor indicator. The graded models use a standardised baseline logit. Their coefficients therefore describe different baseline contrasts. The outcomes also come from the same children. A replacement must state a common target and account for that dependence. It must justify how it uses the per-model posterior distributions, check convergence and bind each input to the current model specification.

The earlier analysis and its rationale remain in [the July 2026 pooling note](202607131700-lrp228-item9-pooled-moderation.md). This retirement makes no new claim about intervention moderation and does not reinterpret an archived numerical result as a current estimate.
