<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Technical report — plan and outline (structure-first draft)

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8). Structure and rationale
> below were assembled with a multi-agent pass over `METHODS.md`, the model suite,
> the Quarto scaffolding, the findings notes, and external Bayesian-reporting
> guidance. Empirical claims are placeholders to be filled from the model CSVs at
> drafting time; reading-science and method citations must have DOIs verified per
> `METHODS.md` before they enter the report.

Date: 2026-06-26 — **Status: PROPOSAL** (author decisions of 2026-06-26 incorporated;
see §7. Ready to scaffold `docs/report/` on the next pass.)

## 1. Purpose

A plan for the study's detailed **technical report** (`docs/report/`, a Quarto book).
The report describes what we did, what we found, and how to replicate it, in enough
detail for a methods-literate reader to audit the work.

A **separate secondary publication** (future, out of scope here) will summarise the
findings and practical implications for a broad audience of families and educators.
The technical report therefore aims at **researchers** — including the many who know
p-values and confidence intervals but are newer to Bayesian methods — and at serious
practitioners willing to engage with the detail. It must still read clearly and expand
its jargon, but it does **not** carry a families-first reading layer; that job belongs
to the secondary publication, which this report's executive summary and per-outcome
evidence table are designed to *seed*.

This note fixes the *structure* only. It does not write chapters or scaffold files.

## 2. Audience and delivery

**Primary audience: the methods-literate reader**, assumed fluent in frequentist
statistics and newer to Bayesian methods (the `METHODS.md` "science undergraduate"
anchor). Secondary: practitioners and researchers auditing the analysis against a
reporting checklist.

Because the canonical output is **print (PDF)**, delivery is print-appropriate rather
than interactive:

1. **Lead with the finding, then the numbers.** Each results section opens with one
   short plain-language paragraph (the finding and how sure we are), then states the
   formal result; derivations, diagnostics and code go to **footnotes and appendices**,
   not folded callouts (print has no folding). This is good technical-report writing,
   not a system of audience tiers.
2. **Causal-status label** on every results-section header —
   `[CAUSAL]` / `[ROBUSTNESS]` / `[ASSOCIATION]` (a bracketed prose label; it renders
   in print and needs no infrastructure). This makes "only the randomised τ is causal"
   a property of the table of contents: a reader cannot reach a mechanism slope without
   crossing a labelled boundary.
3. **One mandated result sentence-shape**, single-sourced from the model CSVs
   (`rope_summary.csv` / `tau_summary.csv`): items-scale median **+ 95% credible
   interval + P(τ>0) + P(benefit ≥ δ) + evidence-ladder label** — never a bare point
   or a bare "significant/not". This is already the `METHODS.md` convention; the report
   operationalises it.
4. **No inline code.** With no code-fold in PDF, methods are described in prose and
   maths; full, runnable code stays in the repository and is pointed to from the
   reproducibility appendix. Figures and tables (auto-generated from the CSVs) carry the
   quantitative content.

## 3. Best-practice basis

`METHODS.md` already commits us to most of the discipline (median + 95% CrI, tail
probabilities not p-values, ROPE/δ for practical significance, convergence-before-
interpretation, no verbal evidence labels that connote size, DOIs on every citation).
The report is largely an *operationalisation* of `METHODS.md`, cross-checked against
three external guidelines:

- **Gelman, Vehtari, Simpson, et al. (2020), *Bayesian Workflow*.** arXiv:2011.01808,
  DOI `10.48550/arXiv.2011.01808`. **Preprint — no peer-reviewed journal version; cite
  as a preprint.** Source for the workflow process (priors → prior-predictive →
  computation/diagnostics → posterior-predictive → comparison/LOO → sensitivity →
  transparent model-building history).
- **Kruschke (2021), Bayesian Analysis Reporting Guidelines (BARG).** *Nature Human
  Behaviour* 5(10), 1282–1291, DOI `10.1038/s41562-021-01177-7`. The explicit reporting
  checklist (model/likelihood/priors; computation & convergence including R-hat, ESS, MCSE;
  posterior summary + ROPE decision; predictive checks; reproducibility).
- **Depaoli & van de Schoot (2017), WAMBS checklist.** *Psychological Methods* 22(2),
  240–261, DOI `10.1037/met0000065`; WAMBS-v2 tutorial chapter (2020),
  DOI `10.4324/9780429273872-4` (verify page range). The to-do checkpoints, including
  convergence stability and prior/posterior-predictive checks.

The three are complementary (workflow process / reporting checklist / to-do
checkpoints), no conflicts. **Verify exact checklist wording and the WAMBS page range
against the primary articles before quoting verbatim.**

Five checklist items that need *explicit* placement (none is currently obvious and all
are cheap to add) — handled in the outline below:

| Item | Where it now lives |
|---|---|
| Interval type — **equal-tailed** (resolved; see Phase 0 status) | Ch 3 (bridge); diagnostic-table HDI noted in Ch 6 |
| Monte Carlo standard error (MCSE) | Ch 6 (computation) |
| Convergence stability (unchanged after doubling) | Ch 6 (computation) |
| Bayes factors not used — tail probabilities instead (N/A note) | Ch 4 (methods) |
| Prior sensitivity / robustness | Ch 10.5 (robustness), flagged future-work where not yet run |
| Fake-data / SBC validation status | Ch 4 (methods), one line |

## 4. Recommended outline

**Organizing principle:** an **estimand / DAG-first spine** — causal discipline first,
then results partitioned into causal / robustness / association rings, with **workflow
completeness** consolidated into two diagnostic chapters rather than scattered. Source
artifacts are named so chapters reuse what exists rather than reinvent.

### Part 0 — Front matter
- **Preface** (`index.qmd`) — reader contract; the causal-status label legend; the
  preliminary-status banner; the AI-authorship Quarto callout. *Reuses:* `CLAUDE.md`
  labelling rule.
- **Executive summary** — the report's own abstract-plus: the question, the design, the
  headline randomised findings with uncertainty, the main caveats. Written plainly; it
  also **seeds the secondary publication**. *Reuses:* findings notes; `tau_forest.png`,
  `rope_summary.csv`.
- **One-row-per-outcome evidence table** — the at-a-glance spine: one row per outcome ×
  {ITT τ, DiD, evidence label}. Also a seed for the secondary publication. *Reuses:*
  `tau_summary.csv`, `did_summary.csv`, `rope_summary.csv`.

### Part I — Question, design, causal frame
- **Ch 1. The study and the questions (as estimands)** — RLI scope (language **and**
  reading, not vocabulary only), the waitlist-crossover design, sample size, the
  questions stated as estimands. *Reuses:* `intro.qmd` (currently TODO); `METHODS.md`.
- **Ch 2. The causal model: the locked DAG** — DAG figure + node legend; the empty-
  adjustment-set identification of the ITT; the three traps (dose is a collider — never
  condition on it; skill→skill couplings are latent-ability-confounded associations;
  grammar/expressive language and speech have no modelled path to word reading).
  *Reuses:* `notes/202606231600-dag-revision-consolidated.md` (**the DAG figure must be
  authored — biggest missing asset**).
- **Ch 3. A reader's guide to Bayesian results (the bridge)** — the key chapter for the
  frequentist-leaning audience. Why Bayesian; how to read a posterior; the **five
  frequentist→Bayesian translations** (CrI ≠ CI; tail probability ≠ p-value; ROPE
  decision ≠ NHST; no "significance"; evidence can *support* a null); the `p ≈ 2(1−pd)`
  identity *with the warning that a pd threshold is a p-value in disguise*; direction vs
  magnitude + ROPE/δ + evidence ladder; Type-S/Type-M (winner's curse) at this sample
  size; **state the HDI-vs-equal-tailed choice here**. *Reuses:* `METHODS.md` glossary;
  `notes/202606261304-evidence-strength-and-rope-reporting.md`.

### Part II — Methods and the exploratory step
- **Ch 4. Methods** — Beta-Binomial / logit likelihood and items-scale reporting; the
  shared prior panel and its justification; sign convention (positive τ helps); the floor
  rule for floored outcomes; estimation, convergence and comparison thresholds;
  GroupKFold; the **"Bayes factors not used" N/A note** and a **one-line fake-data/SBC
  validation status**. Code is described, not shown; it lives in the repo. *Reuses:*
  `priors.py`, prior plots, `floor.py`, `METHODS.md`.
- **Ch 5. Step 1 — gradient-boosting discovery, and its honest limits** — short. Role =
  predictor screen, *not* inference; the regression-to-the-mean / baseline-driven
  negative; read SHAP for direction only; no causal claims; what it licensed forward.
  Headline negative in the chapter; detailed SHAP/permutation tables go to an appendix.
  *Reuses:* `notes/202606201500-gb-replication-findings.md`; SHAP figures.

### Part III — Did the models behave? (diagnostics gate)
- **Ch 6. Computation and convergence** — sampler / software / versions / seeds; R-hat
  ≈ 1.00, bulk + tail ESS, < 1% divergences; **MCSE**; **stability check**; one worked
  example plus a roll-up table across all reported models. *Reuses:* `config.json`,
  diagnostics CSVs, trace/energy plots; `compare_statistical_models.py` roll-ups.
- **Ch 7. Predictive checks and fit** — prior-predictive (priors cover without
  determining) and posterior-predictive overlays; the floor diagnostic that justifies
  the binary estimand for floored outcomes. *Reuses:* prior/posterior predictive plots,
  proportion-at-zero diagnostics.

### Part IV — The causal core `[CAUSAL]`
- **Ch 8. The headline ITT effects** — one section per outcome (word reading;
  letter-sounds; blending; receptive/expressive vocabulary; grammar/expressive language;
  floored outcomes — phonetic spelling, nonword reading). Each section: plain lead →
  formal numbers (the mandated sentence-shape) → detail in footnotes. One headline model
  per outcome; the rest link to the App B catalogue. **Read the vocabulary results
  carefully** — a near-zero transfer effect is a substantive finding, not a measurement
  failure and not "the model ignores vocabulary." *Reuses:* `tau_summary.csv`,
  `rope_summary.csv`, off-floor mover counts, the measurement-sensitivity audit.
  *(Map outcome → LRPITT id against the registry at drafting time.)*
- **Ch 9. Consistency across outcomes (joint view)** `[CAUSAL]` — joint-model forest;
  the joint τ reproduce the single-outcome τ; cross-outcome contrasts; the taught-vs-
  untaught generalisation contrast (limited transfer). *Reuses:* `tau_forest.png`,
  ITT-vs-joint consistency CSVs.

### Part V — Does the headline survive? `[ROBUSTNESS]`
- **Ch 10. Robustness and within-person replication** — waitlist-crossover DiD
  (triangulation, not independent replication — it shares the untreated cell); general-
  ability adjustment; SES + matched comparators (complete-case caveat); dose as a flagged
  sensitivity only; **prior sensitivity** (re-fit under broader priors; flag future-work
  where not yet run). *Reuses:* `did_summary.csv`, robustness `tau_summary.csv`.

### Part VI — Outer rings `[ASSOCIATION]` (standing caveat)
- **Ch 11. Alternative estimation views** — gain-factors (the treatment term reproduces
  τ — the only causal coefficient); level-factors (clean only at t2); onset-aligned
  per-protocol (no causal term; age-at-onset confound). *Reuses:* `factor_summary.csv`,
  `cohort_marginal.csv`.
- **Ch 12. Mechanism and moderation** — mechanism slopes ("associated with, after
  adjusting for stable level" — never "drives"); moderation models + nested PSIS-LOO.
  *Reuses:* `mechanism_curve.csv`, interaction summaries, LOO-compare CSVs.
- **Ch 13. Mediation (triangulation only)** — g-formula NDE/NIE, **not** point-
  identified; temporal-ordering sensitivity; route-compatible, not route-isolating.
  *Reuses:* `mediation_summary.csv` and its sensitivity variant.

### Part VII — Comparison and synthesis
- **Ch 14. Model comparison and build history** — PSIS-LOO with `elpd_diff` vs SE and
  Pareto-k; the **iterative model-building history** (dropped cross-baselines, retired
  Random Forest, floor-rule provenance) for multiplicity transparency *in the main text*,
  not buried. *Reuses:* LOO CSVs; model-selection notes.
- **Ch 15. What it all means** — the causal core restated; the associational ring
  re-filed as *hypotheses to test*, not findings; the practical-implications thread that
  the secondary publication will expand. *Reuses:* findings notes.
- **Ch 16. Limitations and future work** — sample size / winner's curse; latent-ability
  only partially repaired; mediation not identified; provisional δ; DiD shared cell; the
  deferred DAG decisions; reporting-plumbing debts. *Reuses:* the DAG note's open-
  decisions register; findings notes.

### Part VIII — Back matter
- **App A. Glossary** — `METHODS.md` glossary verbatim, plus the gaps (CI vs CrI,
  SESOI/MID, NUTS/nutpie, R-hat, ESS, divergences, predictive checks, elpd, Beta-
  Binomial, logit, collider, empty adjustment set, waitlist-crossover, DiD, floor rule).
- **App B. Full model catalogue** — roll-up of all eight families / ~90 models with
  estimand + causal status, linking to the existing `docs/models/{id}/index.qmd`
  companions. *Re-read the CSVs into the report; do not transclude the per-model qmd
  (front-matter / relative-path conflict).*
- **App C. Priors, ROPE/δ registry, evidence ladder, floor rule.**
- **App D. Reproducibility and data availability** — repo + persistent identifier,
  versions/seeds, archived `trace.nc`, fit commands, CC BY 4.0 docs / AGPL code.
- **References** (`references.qmd`, `::: {#refs}`) — populate the empty `references.bib`:
  Burgoyne et al. 2012, Kruschke 2021, Gelman et al. 2020 (preprint), Depaoli & van de
  Schoot 2017, LightGBM/SHAP, and the reading-science frameworks (verify DOIs).

## 5. Why this structure meets the brief

- **Readable without audience tiers** — every results section leads with the finding and its
  uncertainty in plain words before the formal numbers; the Bayesian bridge (Ch 3) sits
  before any result, serving the frequentist-leaning audience directly.
- **Checklist-complete** — diagnostics, comparison and sensitivity own dedicated
  sections (Ch 6, 7, 14, 10.5); the five easily-missed items are explicitly placed (§3).
- **Only-τ-is-causal is structural** — the `[CAUSAL]`/`[ROBUSTNESS]`/`[ASSOCIATION]`
  label partitions Parts IV / V / VI.
- **Reuses existing work** — per-model `docs/models/*` pages, family CSVs, the DAG note,
  and `METHODS.md` feed named chapters; little is reinvented.
- **Seeds the secondary publication** — the executive summary and evidence table are
  written so a broad-audience companion can lift them directly.

## 6. Sequencing / phasing plan (drafting order)

- **Phase 0 — unblock the build.** Make **PDF the canonical format** in `_quarto.yml`
  (keep HTML as a convenience build but depend on nothing interactive; drop the DOCX
  target unless a stakeholder needs it — there is no `docs/template.docx` yet). Widen the
  title/abstract (and `book.output-file`, and the hardcoded title in `footer.tex`) from
  *vocabulary* to *language and reading*. Populate `references.bib` with the verified
  method DOIs + Burgoyne. (Resolved: the effect CSVs use **equal-tailed** intervals —
  `reporting.py` — matching METHODS.md, so Ch 3 declares ETI.)
- **Phase 1 — harvest what exists.** Auto-generate from CSVs via shared code chunks: the
  evidence table, the convergence roll-up, the τ forest, every numbers panel. Lift the
  glossary and priors appendix from `METHODS.md` / `priors.py`. The per-model
  `docs/models/*` pages feed App B and the footnote-level detail.
- **Phase 2 — hand-write the discipline spine.** Ch 2 (DAG + identification, **author the
  DAG figure**), Ch 3 (the bridge), Ch 5 (GB honest negative). The highest-value new prose.
- **Phase 3 — hand-write the narrative.** The plain leads, the vocabulary-null section,
  and Ch 15 synthesis — paraphrasing the *same* label strings the number panels use, so
  the prose never outruns the qualified number.
- **Phase 4 — fill `index.qmd` / `intro.qmd` last**, once the chapter set has settled.

**Auto-generated** (code chunks re-reading CSVs): evidence table, all numbers panels,
convergence/LOO roll-ups, forests, predictive-check figures, App B catalogue.
**Hand-written:** DAG figure + Ch 2/3/5, all plain leads, the vocabulary section,
synthesis, limitations, and the five gap-closing additions.

## 7. Author decisions (resolved 2026-06-26) and remaining minor forks

**Resolved:**

1. **Document architecture →** the technical report stays a *technical* report (primary
   audience: researchers / methods-literate readers). A **separate secondary
   publication** will summarise findings and practical implications for families and
   educators. Consequence: the three-tier audience-callout machinery is dropped; the
   executive summary + evidence table are designed to seed the secondary publication.
2. **Per-model depth →** one headline model per outcome in the main text + a roll-up
   table; the full ~90-model catalogue links from App B.
3. **Canonical output format →** PDF / print parity. Consequence: no folded callouts, no
   inline code-fold; depth goes to footnotes/appendices and code stays in the repo.

**Minor forks resolved by recommendation (revisit if wanted):**

4. **GB step →** a short honest-negative chapter (Ch 5), not an appendix — it is load-
   bearing for "only τ is causal"; detailed SHAP/permutation tables go to an appendix.
5. **Causal-status label →** a bracketed prose label (`[CAUSAL]` etc.), not a styled
   component — zero infrastructure and prints cleanly.

## Phase 0 status (done 2026-06-26)

- **Output location / gitignore** — already satisfied: `_quarto.yml` sets
  `output-dir: ../../output/report` and `output/` is in the root `.gitignore`. No change.
- **Format** — PDF made canonical (listed first); HTML kept as a convenience build;
  DOCX removed (no `docs/template.docx` existed — it would have failed the build).
- **Title / scope** — book `title` widened to *language and reading*, `subtitle` added,
  `output-file` → `language-reading-development-report`, `abstract` rewritten (scope-
  correct; headline-findings sentence left as a marked placeholder); the stray top-level
  `abstract: "TODO"` removed; the hardcoded title in `footer.tex` widened.
- **Bibliography** — `references.bib` seeded with Burgoyne 2012, Kruschke 2021 (BARG),
  Gelman et al. 2020 (Bayesian Workflow — preprint), Depaoli & van de Schoot 2017 (WAMBS),
  LightGBM and SHAP. DOIs verified for the first four; the NeurIPS papers carry none.
- **Interval convention** — resolved to **equal-tailed**: the effect CSVs use central
  quantiles (`reporting.py`); only the per-parameter diagnostic table uses HDI
  (`diagnostics.py`).

Done since Phase 0: scaffolded the full chapter `.qmd` set (7 Parts + lettered
appendices, each with the AI-authorship callout, causal-status badges, and per-section
source-artifact pointers); removed the superseded `intro.qmd`; **authored the DAG
figure** as a native Quarto `{dot}` block in Ch 2 (`fig-dag`, with the node-key table) —
renders in both PDF and HTML; **drafted Ch 3 (the Bayesian bridge)** and **Ch 5 (the
gradient-boosting honest-negative)** in full prose, and seeded `references.bib` with the
estimation / evidence-and-design-analysis citations (Makowski 2019, Gelman & Carlin 2014,
Kruschke 2018, Lakens 2018, Amrhein 2019 — DOIs verified).

Still open: reading-science citations (simple view of reading, self-teaching, lexical
restructuring — verified DOIs); the results-chapter prose (Ch 8 onward); and wiring the
auto-generated tables/figures (evidence table, τ forest, convergence roll-up, per-outcome
panels) — these need the ITT suite fit, as only `lrpgf01` / `lrplf01` dev CSVs are
present in this worktree.

## 8. Provenance

Built from a multi-agent pass: parallel readers digested `METHODS.md`, the
`statistical_models/` suite (factories, pipeline, priors, reporting, a spec per family),
the `docs/report` + `docs/models` Quarto scaffolding, and the findings notes; one agent
synthesized the external reporting guidance (Bayesian Workflow / BARG / WAMBS) with DOI
verification; three agents proposed outlines from different organizing principles
(audience-layered, workflow-structured, estimand/DAG-first); a final agent scored them
against the checklist and the audiences and produced the hybrid above. Author decisions
of 2026-06-26 then narrowed it to a technical-report-only, print-first design.

## Related notes

- `notes/202606231600-dag-revision-consolidated.md` — the locked DAG the report is built on.
- `notes/202606261304-evidence-strength-and-rope-reporting.md` — the evidence ladder / ROPE layer.
- `notes/202606201500-gb-replication-findings.md` — the GB honest-negative for Ch 5.
- `notes/202606171000-measurement-sensitivity-audit.md` — floor/ceiling evidence for the floored outcomes.
- `METHODS.md` — the methodology and reporting conventions the report operationalises.
