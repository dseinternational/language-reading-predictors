> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Documentation and source-comment review, 21 October 2026

The review started from `8ded6d90`. The repository-wide inventory covered 627 Markdown and Quarto files and 840 source, graph, style, bibliography and configuration files. This includes the dated notes used as decision records, local workflow instructions, report templates, notebook Markdown comments and Python documentation strings. The Word reference template contains no document text. Published-paper text, licences, historical numerical evidence and generated graph structures were retained.

Automated screening checked file references, repeated prose, source comments and documentation strings. Close review followed claims about prediction, treatment timing, confounding, mediation, score units and publication readiness back to `METHODS.md`, current settings and construction code, and the relevant later decisions. This is a documentation review, not a new mathematical audit of every algorithm or a systematic literature review.

## Corrections

- **Boosting.** Removed copied target-distribution snapshots, unsupported claims that rankings are robust to parameter choices, blanket predictions of weak performance, obsolete variant plans and repeated accounts of retired selection work. Templates now direct readers to fitted performance and ranking evidence. The taught-vocabulary summaries name their actual total-score exclusions. Source documentation uses canonical IDs and corrects APT companion references and September Huber-tuning provenance. Same-skill comparisons are identified as separate ranking-script refits, with their own output root and performance; gain targets have no curated exclusion comparison.
- **Measurement.** DEAP component scores are percentages; `deapp_c` sums three components on a 0–300 scale. The final-consonant documentation distinguishes vowels from voicing. Unconfirmed not-taught vocabulary denominators and the word-repetition quarantine warning remain.
- **Timing.** Concurrent reports distinguish baseline, t2's waiting-list comparison and later post-crossover waves. The lagged-DAG comments acknowledge unequal assessment intervals and label within-wave timescales as assumptions. The notes index calls the t2 calculations timing scenarios, consistently with the methods.
- **Confounding.** A child random intercept does not establish control for latent general ability. Stability or attenuation after adding ability proxies does not quantify confounding removed. Dose documentation states that child means use fitted positive-attendance rows and retains the conditional interpretation of the presence coefficient.
- **Mediation.** Assignment continues to randomise treatment schedules after crossover. It does not randomise a skill mediator or identify natural indirect effects in this DAG. Temporal ordering and forward/reverse decompositions do not establish causal direction. A t3/t4 comparison changes follow-up time as well as ceiling support. The older reverse-mediation decision record now states these limits and removes its obsolete proposal status, unchecked implementation list and causal-direction interpretation table.
- **Likelihoods.** The July working-likelihood note no longer claims that randomisation protects the sign and size of a misspecified model's marginal effect. Rasch sufficiency is stated conditionally on the measurement model and calibration, rather than as proof that totals lose no relevant information. Historical probes remain evidence for their own fits.
- **Guides and comments.** The integrated report points to completed fitted reports and complete publication decisions. The prior guide describes actual constructor arguments, including overrides. The runbook distinguishes the formatter's tracked-file selection from CSpell's path scan and gives the PowerShell activation command. Comments describe the chosen missing-outcome policy and numerical stabilisation without claiming impossibility or a floating-point guarantee.

The September code review now links its implementation and follow-up records. Other scientific decisions, source audits and distinct run records remain; a dated review's open-work list is evidence about its revision, not an instruction to repeat completed work.

The Roch and Jarrold citation was checked against the [primary article abstract](https://pubmed.ncbi.nlm.nih.gov/22176835/) (DOI [10.1016/j.jcomdis.2011.11.001](https://doi.org/10.1016/j.jcomdis.2011.11.001)). It reports a longitudinal predictive association with nonword reading, not an identified effect on letter-sound knowledge. No new literature-based causal claim was added.

## Verification

The locked Python and Node environments were installed. The existing LightGBM installation lacked its Windows DLL; reinstalling the same locked version repaired it. Sandboxed Quarto subprocesses stalled, so the complete selected suite was rerun with Windows subprocess and temporary-cache access.

The documentation and reporting suite passed **524 tests**, including synthetic Quarto renders, registry documentation, report ordering, publication-display checks, fonts, data reads and boosting definitions. All **15 additional ranking tests** in `tests/test_rank_predictors.py` and `tests/test_cluster_ranking.py` passed after checking the sibling-exclusion workflow:

```bash
uv run python -m pytest tests/statistical_models/test_documentation.py tests/statistical_models/test_report_restructure.py tests/statistical_models/test_mechanism_report_forms.py tests/statistical_models/test_review_reporting.py tests/statistical_models/test_report_extras.py tests/test_gb_reporting.py tests/test_gb_template_scatter_images.py tests/test_report_fonts.py tests/test_report_data_reads.py tests/test_models.py tests/test_borrowed_params.py
```

Ruff, Markdown formatting and British-English spelling checks passed. The registry-count check passed, all 277 statistical-report templates retain their required include order, and the local Markdown-link audit found no missing file targets.

Python syntax-tree comparisons permit only comments, documentation strings and explanatory `notes` metadata to change. All 873 fenced blocks and the displayed equation in the 71 changed Quarto files match the base revision. Graph statements, model equations, settings, parameter values, data, dependency locks and the three synchronised instruction files are unchanged. No models were refitted or stored scientific results recomputed. Existing provenance and publication checks still govern saved-fit reuse.

## CI follow-up, 22 October 2026

The [first full CI run](https://github.com/dseinternational/language-reading-predictors/actions/runs/39646294893) passed 4,277 tests, skipped 15 and failed one documentation check. The dose-response prose described latent-ability confounding but no longer named the `GA -> IS` path explicitly, as `test_dag_contradicting_prose_is_gone_from_every_dose_module` requires. The local suite above did not include that test.

The follow-up names all three attendance parents (`A -> IS`, `GA -> IS`, `IG -> IS`) in the shortened `dose-077` explanation. It also removes an outdated test comment about prose that had already been deleted. The test assertions and executable model code are unchanged. All 51 tests in `tests/statistical_models/test_dose_response_estimand.py` pass locally after the correction. There were no review comments or inline threads when checked.
