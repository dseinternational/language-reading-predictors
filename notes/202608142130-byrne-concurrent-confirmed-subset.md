<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald basread basspel basnum basmat basdig bassim bpvs trog woco -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Byrne concurrent models: confirmed-measure subset

**Decision for #338/#409, 2026-08-14: register two per-wave concurrent models using only the five RLM measures whose bounded-count denominators and instrument identities are confirmed.** `lrp-rlm-ca-001` makes BAS word reading (`basread`) focal; `lrp-rlm-ca-002` makes BPVS receptive vocabulary (`bpvs`) focal. Both use the remaining measures from `{basread, bpvs, trog, basdig, bassim}` as same-wave predictors over waves 1–4, with age and pooled three-group nuisance adjustment.

## Why this subset

The Byrne umbrella plan lists concurrent ports as the strongest unbuilt internal analysis, but a direct port of the full paper battery would make a Beta-Binomial likelihood depend on unverified scale limits. BAS spelling (`basspel`), BAS number skills (`basnum`) and WORD comprehension (`woco`) therefore remain excluded because their item ceilings are provisional. `basmat` remains excluded because its instrument identity is unresolved and no wave-1 score exists. The resolver enforces this boundary: any RLM concurrent specification naming a measure without both a confirmed denominator and a confirmed instrument identity fails before data are loaded or an output directory is reset.

This restriction changes the scientific question. The models are not a full multivariable reproduction of the paper's correlation matrix; they are a safer conditional description of the confirmed reading, language, auditory-memory and verbal-similarities measures. `bassim` is not treated as a complete measure of general ability, so the associations remain potentially confounded by general ability and other common causes.

## Estimand and modelling choices

Each wave is a separate between-child Beta-Binomial regression. The focal score is conditioned on within-wave standardised Haldane logits of the other four skills, standardised age and two dummy variables for the three reading groups. The largest group at a wave is the nuisance reference. A raw numeric `readgrp` slope is forbidden because the codes are categories, not ordered equal intervals.

Each mutually adjusted predictor slope is paired with a single-skill refit omitting age, group and the other skills, matching the existing concurrent-family comparison. The two coefficients answer different conditional questions; their difference is not shared variance and not evidence of mediation. All quantities are descriptive associations. Same-wave measurement supplies no temporal ordering, the historical cohort supplies no randomised intervention, and the reading-matched group was selected on reading level.

Rows missing the focal outcome are excluded separately at each wave. Missing predictors are retained and mean-imputed to zero after within-wave logit standardisation, with observed and imputed counts published. This is a pragmatic descriptive policy rather than a missing-at-random correction; it can distort coefficients in either direction. Wave 4 is reported as an attrition-sensitive extension beyond the paper's audited waves 1–3.

## Release boundary

The measures in these fits clear the instrument-level gate, but the dataset-level 96-versus-97-participant provenance discrepancy remains unresolved. The existing non-RLI publication-input contract therefore withholds scientific publication at the input stage even when computation, required artefacts and all subfit diagnostics pass. Resolving source provenance can later change that stored release decision only after the fit is regenerated under the authoritative input record; old fits do not silently inherit a later sign-off.
