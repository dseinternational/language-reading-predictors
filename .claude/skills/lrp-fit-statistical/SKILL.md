---
name: lrp-fit-statistical
description: Fit, check and render the Bayesian statistical models. Use the project methods and full refit runbook for interpretation, sensitivity evidence, resumption and publication.
---

> [!NOTE]
> Clarity and currency edits by a LLM-based AI tool (Codex/GPT-6).

> Available-case modified ITT terminology updated by a LLM-based AI tool (Codex/GPT-5).
>
> Divergent-transition gate guidance, credible-interval standard, causal-term guidance, architecture, model/family counts and sweep figures corrected against `METHODS.md`, the registry and stored artefacts by a LLM-based AI tool (Claude Code/Opus 5).
>
> Substantially corrected by a LLM-based AI tool (Codex/GPT-5).

# Fit statistical models

Use PyMC models to estimate the study's specified contrasts and associations with uncertainty. [METHODS.md](../../../METHODS.md) governs interpretation; [the catalogue](../../../docs/models/README.md) lists registered models. Read the [full refit runbook](../../../docs/runbooks/full-statistical-model-refit.md) before a publication run.

## Fit and render

From the repository root:

```bash
uv sync --locked
uv run python scripts/fit_statistical_model.py lrp-rli-itt-001 --config dev
uv run python scripts/fit_statistical_model.py lrp-rli-itt-001 --config reporting --render
uv run python scripts/fit_statistical_model.py all --config reporting --render
```

Use `--output-dir` or `DSE_LRP_OUTPUT_DIR` to select the output root; the command option takes precedence. Publication runs use a fresh root and a committed, clean checkout. Individual fits publish a completed staging directory, so a failed refit preserves the previous successful directory.

Sampling presets come from the locked shared library. `rep-lite` uses 4 chains × 4,000 draws; `reporting` uses 6 × 6,000. Both have a default acceptance target of 0.95. A command override takes precedence over the model default, then the preset. Read the actual settings in `config.json`.

Allow several hours for a full sweep. Mediation integration and hierarchical models can dominate elapsed time. Do not infer duration from the model count. The `all --render` command renders after fitting the full set; the runbook explains how to render completed fits after an interruption.

For resumable runs, use `scripts/run_refit_sweep.py` and inspect its `--help`. Resume checks require matching source, data, environment and sampling settings. A successful process exit does not establish convergence or publication eligibility. Do not skip fits merely because a trace or HTML file exists.

## Check before interpreting

Each fit writes to `output/statistical_models/models/<model_id>-<config>/` unless redirected. Read `config.json`, `diagnostics_summary.json`, `release_decision.json`, the family result tables and the rendered report together.

A clean computational pass requires zero divergences, R-hat ≤ 1.01, bulk and tail effective sample size ≥ 400, and minimum per-chain BFMI ≥ 0.3. Use `statistical_models.sampling_quality.sampling_quality(...)` for unrounded diagnostics. Do not recreate the gate from rounded `az.summary()` output.

Investigate divergences before changing sampling settings. More draws can improve precision when chains mix, but cannot repair biased exploration. A low divergence percentage is insufficient for release. The narrow qualification policy in `METHODS.md` requires a reviewed trace and named estimand; causal, mediation and latent-structure results require zero divergences.

Passing convergence is only one release requirement. Check input provenance, required artefacts, sensitivity refits and paired phoneme-blending fits. The runbook specifies the required evidence bundles and final re-evaluation.

## Interpret and report

Report the named estimand's median, inner 50% and outer 89% equal-tailed credible intervals, and its tail probability. Calculate natural-scale effects within each posterior draw before summarising. The 89% level is a reporting convention; it does not guarantee adequate simulation precision.

Keep causal language tied to the design. ITT, the crossover t2 arm gap, period-1-standardised primary gain-factor contrasts and the level-factor t2 change have the stated randomisation and available-case assumptions. Later crossover and level-factor arm contrasts compare randomised treatment schedules; they do not isolate a mechanism. Skill, dose and moderation coefficients remain adjusted associations. A child random intercept does not remove confounding by unmeasured ability.

Use the stored evidence label for the named direction or practical threshold. A label describes evidence, not effect size. State prior dependence, uncertainty and missing-data restrictions. Consult the catalogue for the family-specific tables rather than applying one conversion to every model.

## Publish and record

Follow the runbook's upload procedure within the user's authorised scope. Verify the destination, credentials and trace-inclusion choice on the current host. Do not assume that a managed identity or a previous user's login has write access. Public research uploads are anonymously readable; private archive destinations are separate.

Record the configuration, source and data identities, convergence and release results, remaining failures, output location and any published URL in a dated note. Apply the AI-authorship label and run the repository's required checks before a commit or pull request.
