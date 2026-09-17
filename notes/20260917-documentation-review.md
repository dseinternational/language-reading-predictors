> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Documentation review, 17 September 2026

## Scope and approach

This review started from commit `b62fd6ce093c2cb81e5c842cf96ff11ffa6e4d95`. The inventory contained 738 tracked prose documents, including Markdown and Quarto files, excluding the spelling dictionary. It covered project guides, contributor instructions, local workflow skills, data descriptions, report templates, learning material and dated notes.

The review checked repetition, supersession, local links, changing facts and interpretation. Current instructions were compared with the registry, source code, data dimensions and command interfaces. Repeated report prose was reviewed across model families. Historical notes were assessed for a continuing role as decisions, evidence or source records. This was not a fresh validation of every scientific assertion in the archive.

## Removals and retained history

The cleanup removed 117 superseded notes. These included repeated July and August findings series, earlier run summaries, the retired feature-selection inventory, completed pipeline-migration logs, superseded report-layout plans and a retired pooled-moderation analysis. The September findings series remains, with an explicit warning that it describes an earlier rebuild.

Scientific decisions, original-source audits and distinct methodological reviews remain. The older technical-report plan now retains its approved author decisions. The later proposal remains a proposal. The withdrawn measurement-model convergence exception now states the current rule without presenting its earlier mistaken explanation as guidance.

References to removed evidence now link to the exact file before this cleanup. They do not silently substitute a newer analysis for the evidence originally cited. The [notes guide](README.md) explains how to retrieve removed files from Git history.

## Main corrections

- The [documentation guide](../docs/README.md) separates current instructions, report templates, the unfinished integrated report and dated findings.
- [METHODS.md](../METHODS.md), the contributor instructions and the [priors guide](../docs/models/PRIORS.md) now use shorter explanations. The priors guide describes construction-time descriptors instead of retired name-based lookup rules.
- The [refit runbook](../docs/runbooks/full-statistical-model-refit.md) uses the supported resumption script and explains staged publication. File existence alone no longer stands in for fit-identity checks.
- Edits to 112 model templates remove repeated explanations, stale predictor counts, obsolete links and incorrect descriptions of fitted terms or comparison windows. The text distinguishes prediction, adjusted association and randomised contrasts.
- The integrated report's evidence table is labelled as unimplemented. Its example no longer displays a fitted estimate through a helper that checks only convergence rather than the complete publication decision.
- Data guides reflect the added Object Assembly column and its derivation. Workflow skills omit stale machine details and unsupported numerical rules for accepting tuned models.

The review also corrected statements that were too definite. A subtest correlation alone does not identify reliability or the fraction of confounding removed. A small screening-regression change cannot establish what perfect measurement would do. The [ability-composite decision](202609071900-nonverbal-ability-composite.md) retains the approved companions and adds these qualifications. The cited measurement-model assumptions were checked against Eisinga et al. (2013), DOI [10.1007/s00038-012-0416-3](https://doi.org/10.1007/s00038-012-0416-3). The methods guide also no longer guarantees that a dispersion-prior change preserves coefficient signs or treats a baseline coefficient below one as sufficient evidence of regression to the mean.

## Verification and limits

The documentation checks cover all 276 registered statistical reports and their required section order. The focused documentation suite passed 23 tests, including synthetic-data report rendering. All 31 gain-factor contract tests also passed. Python source lint, Markdown formatting and spelling checks passed. The local-link audit found no missing file targets after repairs; it checks file existence, not every heading anchor or external website. Historical Git links were checked against the named commits.

The three contributor instruction files remain identical. Model equations, fitted-data files and model-page executable chunks were not changed. Small supporting changes let the formatter ignore deleted tracked files, update one renamed test heading and remove a test whose sole subject was a deleted historical note.

No models were refitted. Historical estimates, intervals and release decisions were not recomputed. Existing numerical findings therefore remain dated evidence, and this review does not certify them for publication. Existing literature citations were retained unless the surrounding claim required correction; the review was not a full external-source audit.
