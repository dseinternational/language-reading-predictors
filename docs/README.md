> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Documentation guide

All analyses are preliminary. Start with the document that matches your question.

| Question                                              | Read                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| What does the project study, and how do I install it? | [Project introduction](../README.md)                                                             |
| How are the analyses designed and interpreted?        | [Methods and reporting rules](../METHODS.md)                                                     |
| How does one Bayesian model work?                     | [Worked example with synthetic data](learning/itt-model-walkthrough.md)                          |
| Which analyses are registered?                        | [Model catalogue](models/README.md)                                                              |
| What do the priors mean?                              | [Priors guide](models/PRIORS.md)                                                                 |
| Which causal assumptions do the models use?           | [DAG guide](../dag/README.md)                                                                    |
| Where did the data come from?                         | [RLI data](../data/readme.md) and [historical cohort](../data/reading-language-memory/README.md) |
| How do I refit and publish the model reports?         | [Refit runbook](runbooks/full-statistical-model-refit.md)                                        |
| Why was a method chosen or changed?                   | [Notes and decision records](../notes/README.md)                                                 |
| How is the code organised?                            | [Contributor instructions](../AGENTS.md)                                                         |

## Reports and dated findings

`docs/models/<model_id>/index.qmd` files are report templates. A fit copies its template and shared sections into the output directory, alongside the data needed to render it. Read the resulting report with its configuration, diagnostics and publication decision. A template alone does not establish that a result is ready to report.

The [integrated study report](report/index.qmd) remains a draft. Several chapters are placeholders, and its data helper does not yet enforce every publication requirement. Use the fitted model reports for results while the integrated report is completed.

The September findings notes summarise a specific earlier rebuild. Later changes can make those numbers or release statuses stale. The [notes guide](../notes/README.md) explains which records are current guidance and which are historical evidence.

## Maintaining the documentation

Keep scientific rules in `METHODS.md`, model-specific detail in the catalogue and templates, and commands in the runbook. A dated note should record a decision, its evidence or a reproducible run. Avoid copying changing model counts, full prior inventories or machine-specific credentials into several guides.

When a note is superseded, retain any decision or evidence still needed to explain the current analysis. Remove duplicate findings and completed plans once their useful content has a clear home. Historical citations can link to the exact version in Git history so a newer result is not mistaken for the evidence originally cited.
