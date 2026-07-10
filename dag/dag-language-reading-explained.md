<!-- SPDX-License-Identifier: CC-BY-4.0 -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Opus 4.8).

# The language & reading causal DAG: structure, assumptions, and justification

This document explains the causal DAG **as it now stands** after the 2026-07-10 revision — what each part asserts, the evidence for it drawn from the typically-developing (TD), Down-syndrome (DS) and wider intellectual-and-developmental-disability (IDD) reading literature, the honest weaknesses and alternatives, and the case for the choices made. It is the reference companion to the machine-readable graph in [`dag-language-reading.dagitty`](dag-language-reading.dagitty) and its rendered figure; the decision record that produced this structure is [`../notes/202607101100-dag-revision-team-decisions.md`](../notes/202607101100-dag-revision-team-decisions.md), and the two critical reviews that fed it are `../notes/202607091430-dag-critical-review-td-atypical-literature.md` and `../notes/202607091615-vocab-reading-subgraph-critical-review.md` (issue #233). Citations carry DOIs verified against PubMed/publisher pages during drafting; three older references (Stanovich 1986; Metsala & Walley 1998; Share 1995) are flagged _to verify_ before they enter the report, per `METHODS.md`. This is preliminary research; the graph is a working instrument, not a settled theory.

## 1. What this DAG is for, and how to read it

A causal DAG here is an **identification instrument**, not a complete developmental theory. Its job is narrow and specific: to say which observed associations are confounded, what may legitimately be adjusted for, and which effects the study design actually identifies. It sits at the hinge of the project's two-step method — exploratory gradient-boosting models learn _which_ predictors matter, and the DAG governs which of those associations the step-2 Bayesian models may read causally.

Three consequences shape everything below.

- **Roles are assigned per analysis, not fixed in the graph.** The same structure serves the intention-to-treat analysis (exposure `IG`, outcome `WR`), the mechanism analyses (a skill as exposure, `WR` as outcome), the dose-response analyses (`IS` as exposure) and the mediation analyses (`IG` with a skill mediator). Only `GA [latent]` is annotated, because being unobserved is a structural property rather than an analysis-specific role.
- **A missing edge is a strong claim; an extra edge is cheap.** An omitted edge asserts an _absent_ causal path — a real theoretical commitment. An included edge usually only enlarges an adjustment set. So the graph is deliberately generous with edges and sparing with omissions, and the omissions are where the contestable theory concentrates.
- **Most internal edges are untestable in these data and are imported from theory.** Because the latent general-ability node `GA` points into almost every observed variable, nearly every internal conditional independence has a separating set that contains `GA` — so it cannot be checked against the data. The internal structure is therefore theory-led, and its only available audit is contrast with the literature. That is why this document is citation-heavy.

Running through all of it is one population principle: **the causal architecture of early word reading is largely shared between TD and DS children, but the weights differ, and DS carries population-specific inputs — hearing, speech-motor difficulty, verbal short-term memory — that a TD-derived graph never needed to represent.** The 2026-07-10 revision is mostly about putting those DS-specific inputs into the graph and correcting a few TD-inherited weights.

## 2. The graph at a glance

![The revised language & reading DAG (2026-07-10)](dag-language-reading.svg)

Twenty nodes, verified acyclic; four roots (`A`, `GA`, `HS`, `IG`); one primary outcome (`WR`). The figure summarises the edges from the two universal parents `A` and `GA` in a note rather than drawing all 32, for legibility; `HS` is drawn explicitly. The authoritative edge list is the `dag { … }` block in the `.dagitty` file.

| Symbol      | Construct (measure)                                                       | Domain / role                                   |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------- |
| `A`         | Age                                                                       | observed root; maturation + cumulative exposure |
| `GA`        | General ability                                                           | **latent** root; age-residualised stable trait  |
| `HS`        | Hearing status (`hearing_c`: impaired hearing or repeated ear infections) | observed root; DS-specific common cause         |
| `IG`        | Intervention group (randomised)                                           | root; ITT exposure                              |
| `IS`        | Intervention sessions / attendance                                        | dose (a collider — never conditioned on)        |
| `TR` / `TE` | Taught receptive / expressive vocabulary (bespoke b1/b2 word sets)        | direct teaching targets                         |
| `RV` / `EV` | Standardised receptive / expressive vocabulary (rowpvt / eowpvt)          | transfer (generalisation) measures              |
| `RW`        | Word + nonword repetition (erbto) — phonological short-term memory        | capacity                                        |
| `LF`        | Language fundamentals                                                     | language (no path to `WR`)                      |
| `RG` / `EG` | Receptive / expressive grammar (trog / aptgram)                           | language (no path to `WR`)                      |
| `EI`        | Expressive information (aptinfo)                                          | language (no path to `WR`)                      |
| `SP`        | Speech production (deapp_c) — proxy for pervasive speech-motor difficulty | upstream cause                                  |
| `LS`        | Letter–sound knowledge (yarclet)                                          | code route                                      |
| `PA`        | Phonological awareness / blending                                         | code route                                      |
| `NW`        | Nonword reading (decoding)                                                | code route (mediator)                           |
| `PS`        | Phonetic spelling                                                         | reading/spelling outcome (terminal)             |
| `WR`        | Word reading                                                              | **primary outcome**                             |

## 3. Foundational commitments

These four assumptions pervade every analysis the graph serves.

### 3.1 A single latent general ability `GA` causes (almost) everything

`GA` is a latent common cause pointing into every observed skill. It is the graphical encoding of the project's standing position that **only the randomised contrast is causal**: it makes explicit that every skill-to-skill and mediator-to-outcome association is confounded by a shared, stable trait. Crucially, `GA` does _not_ point into `IG` — which is exactly why randomisation survives (it d-separates `IG` from `GA`).

`GA` is deliberately **drawn but never adjusted**. It is latent, so it can never appear in an implementable adjustment set; the subject-level random intercept in the longitudinal models is only a partial, shrunken stand-in, not "control for ability." This is the honest cost: it means the graph _refuses to claim_ that any internal skill-to-skill slope is unconfounded. TD evidence supports the modesty — nonverbal IQ is a weak _unique_ predictor of early word reading (e.g. the weak IQ paths in Burgoyne et al. 2019) — and our own measurement model (mm-001) finds domain factors correlated 0.65–0.80: high, but distinguishable, which is why a single `GA` is a defensible first approximation and a correlated-domain-factor upgrade is a sensible, identification-neutral future refinement rather than an urgent fix.

### 3.2 Age `A` as a universal parent and exposure proxy

`A` points into every observed node. It carries both maturation and cumulative exposure, and it is **not** separated into those channels (only the total age effect is identified). `GA` is defined as age-residualised, so `A → GA` is dropped and age carries the developmental trend directly. Age is treated as a precision covariate in the ITT models rather than an object of inference.

### 3.3 Randomisation of `IG`, and the collider status of dose `IS`

`IG` is a parentless randomised root, so **the total intervention effect `IG → WR` is identified by the empty adjustment set** (ID-1). This is the load-bearing causal claim of the whole project, and — importantly — it is _robust to everything in the internal structure_: no revision to the skill-to-skill edges can disturb it. Dose (`IS`, sessions attended) is a **collider** — a common effect of assignment and of child/family characteristics — so it is never conditioned on, and any dose-response estimate is flagged observational. This collider discipline is more careful than much published mediation work in the area.

### 3.4 One causal coefficient; everything else is an adjusted association

Only τ (the randomised effect) is read causally. Every observational coupling — mechanism slopes, mediator-to-outcome paths, between-child associations — is reported as an **adjusted association**, never "X drives Y" (ID-2; see §11). This convention is what lets the graph carry theory-led internal edges without over-claiming.

## 4. The reading architecture: code route and vocabulary routes

This is the substantive heart of the graph — how the intervention and the child's skills reach word reading.

### 4.1 The code skeleton `LS`/`PA` → decoding → `WR`

Letter–sound knowledge (`LS`) and phoneme awareness (`PA`) feed nonword decoding (`NW`) and word reading (`WR`); `PA` and `LS` also feed each other's downstream products. This is the canonical alphabetic-reading architecture, with the strongest _causal_ (RCT-grade) evidence base in the field: letter–sound knowledge and phoneme awareness are the two best-established causal foundations of word reading in TD samples (Hulme, Bowyer-Crane, Carroll, Duff & Snowling 2012; Bradley & Bryant 1983; the meta-analysis of Melby-Lervåg, Lyster & Hulme 2012), and the simple-view framing (Gough & Tunmer 1986) motivates the decoding-plus-language structure.

**The DS-weighting caution (the honest part).** This is where the DS literature diverges most sharply from the TD canon the skeleton imports. Cossu, Rossini & Marshall (1993) documented DS children reading at a 7-year level while _failing_ phonemic-awareness tasks their reading-age-matched TD controls passed; Hulme et al. (2012, _Developmental Science_) found phoneme awareness predicted reading _growth_ in TD but **not** in DS; Næss (2016) found PA in DS weak relative to controls and _uneven_ across components. Two disclosures follow, and both are recorded rather than papered over: (i) in this population the orthographic/whole-word and language routes are expected to carry relatively **more**, and the phonemic route relatively **less (or later)**, than the TD canon implies; and (ii) our `PA` node is operationalised by **blending alone** — the most reading-proximal and relatively preserved PA component — so the node's label claims more construct than the single task delivers. We keep the code-route edges (PA–reading correlations do exist in DS, PA is trainable, and our own trial moves blending), but the reader should hold the weights lightly. There is a satisfying reconciliation in our own data: the _intervention's_ reading effect runs through letter sounds (what was explicitly taught), while _naturalistic_ reading variance is language-constrained — both routes are real and the graph rightly contains both.

### 4.2 Vocabulary routes to reading — including the new direct edges

Three routes carry vocabulary to reading, and the 2026-07-10 revision changed the balance.

- **`RV → WR` (receptive vocabulary, direct).** Retained. In TD, vocabulary's direct effect on word-level reading is modest and concentrated on exception words (Ricketts, Nation & Bishop 2007); but in DS it is _stronger_ — the key DS longitudinal study (Hulme et al. 2012, _Dev Sci_) found reading more language-constrained than in TD, with vocabulary the better predictor of both level and growth. So `RV → WR` is better justified in DS than a TD-only reading would suggest.
- **`TE → WR` and `EV → WR` (expressive vocabulary, direct) — new in this revision (decision 5).** Previously expressive vocabulary reached `WR` _only_ through the `PA` gateway, which made phoneme awareness a **cut-vertex**: everything expressive vocabulary contributed to reading was funnelled through the single node the DS literature identifies as weakest, so the suite was structurally guaranteed to under-read the vocabulary→reading link regardless of the data. The direct edges retire that bottleneck. The substantive warrant is that DS early reading via RLI / See-and-Learn is heavily **whole-word and paired-associate** — the child must _produce_ the spoken word for a printed target, a role for expressive/naming vocabulary a purely receptive account denies. Gains in expressive vocabulary are therefore expected to help word reading over and above receptive vocabulary and beyond phonology.
- **Taught vs standardised vocabulary as separate nodes, with transfer edges.** `TR`/`TE` (bespoke taught sets) are kept distinct from `RV`/`EV` (norm-referenced), with `TR → RV`, `TR → EV`, `TE → EV`. This converts the trial's "did learning generalise?" question from a design-forced zero into an _estimable transfer effect_ (matching Burgoyne et al. 2012): the taught measures are the proximal product of teaching; the standardised measures capture near-vs-far transfer. Retaining both also matters for identification — `RV` is a confounder for any `EV`-based mechanism, so deleting the standardised nodes would not simplify the graph but blind it (see §12).

**Honesty about the vocabulary routes.** `EV` and `RV` are strongly collinear, so the receptive-vs-expressive asymmetry and the new direct edges are **imposed by theory, not identifiable from these data** — the model cannot discover which route is real; it is told. `EV → WR` is therefore reported as an adjusted association, flagged for possible non-identifiability against the `RV → WR` route.

## 5. Speech production `SP` as an upstream cause

The single largest structural change (decision 2): the 2026-06-23 graph made speech a downstream **sink** of vocabulary (`EV → SP`, `TE → SP`); the revision reverses this to `SP → { TE, EV, LS, PA, NW }`.

**The case.** For these children `SP` (the DEAP picture composite, deapp_c) is a proxy for **pervasive, persistent speech-motor difficulty**, present from the outset, not a downstream reflection of vocabulary size. Three strands support the reversal:

- **TD theory/evidence.** In a large unselected TD sample, speech difficulty at school entry sits _upstream_ of the phonological foundations of reading — Burgoyne, Lervåg, Malone & Hulme (2019) found the speech→reading relationship fully mediated by phoneme awareness, i.e. speech is a cause of the phonological skills, the opposite orientation to `EV → SP`.
- **Same-cohort DS evidence.** Burgoyne, Buckley & Baxter (2021) — _our own children_ — found speech production remarkably **trait-stable** (t1↔t2 PCC r = 0.84 over 21 months, no significant group change) and only weakly tied to vocabulary. That is the empirical profile of a source, not a vocabulary-driven sink. (It also confirms a choice the graph keeps: `SP` has no incoming `IG`/`IS` edge — the intervention does not move it, and τ_SP ≈ 0 in the reporting fit.)
- **Task demands.** The edges `SP → LS`, `SP → PA`, `SP → NW` reflect that every code-route task in this battery **requires producing or sequencing speech sounds**: letter–sound knowledge asks the child to _say_ the sound on cue, phoneme-awareness blending to hold and blend sounds aloud, and nonword reading to articulate novel phoneme strings. Speech-motor capacity is plausibly upstream of all three.

**Honesty and the measurement channel.** The direction is not testable in-sample, and `SP`, `EV` and `PA` are collinear, so the three-way separation is fragile. More subtly, there is a **measurement channel no construct-level edge can fully capture**: the expressive-vocabulary and nonword-reading tasks are _spoken-response_ tasks, so poor articulation depresses the _measured_ score independently of the underlying construct (a `SP → measured-Y` path that exists even if `SP → construct-Y` does not). Our own floor-sitter cut is its face — four children with near-complete letter sounds and some word reading scored **zero** on nonword reading, one of them spelling 53 items phonetically; a child who can _spell_ phonetically but scores zero _reading nonwords aloud_ is pointing straight at the production demand of the task. Finally, a scaling caveat: `deapp_c` in the working data is a lenient trial-era scoring ~20 points above the blind-transcription PCC (mean 43.45%) reported for this cohort in Burgoyne 2021 — `SP`'s _role_ as a motor-difficulty proxy is unaffected, but its scale should not be read as PCC (see the DEAP-scoring memo). The distributional gap between the two populations is itself an argument for reversal: what is a 6.88% clinical tail in TD (PCC ≤ 85%) is essentially the whole DS distribution (~98% below the same cut), so importing the TD _mediating mechanism_ (via phoneme awareness) would be an off-support extrapolation — we adopt the orientation but not the TD mediator.

## 6. Hearing `HS` as a common cause

New in this revision (decision 3): `HS → { TR, RV, TE, EV, SP, RW, PA, LS }`.

**The case.** Hearing is rarely a needed node in a TD reading graph, but conductive hearing loss and recurrent otitis media affect a large fraction of children with DS, and the DS evidence links hearing to speech and language — moderate effects in our own cohort (Burgoyne 2021, d ≈ 0.45–0.60) and a significant association in the larger literature (Laws & Hall 2014, d = 0.87). If hearing is a common cause of vocabulary, speech, phonological memory and the code skills, then omitting it left every one of those associations open to **hearing confounding**. Adding it converts a measured-but-previously-unused variable (`hearing_c` was already in the data) into an **identification asset** — a confounder we can actually adjust for.

**Honesty.** `hearing_c` is a coarse composite (impaired hearing _or_ repeated ear infections), not an audiometric measure, so it under-resolves a fluctuating construct. And `HS` is treated as **exogenous** (no parents) to match the team's specification; age-related change in conductive hearing (glue ear often eases with age) makes `A → HS` defensible, and it is deliberately deferred to the time-lagged workstream rather than forced into the contemporaneous graph.

## 7. Phonological memory `RW`

Widened in this revision (decision 4) from `RW → EV` alone to `RW → { TE, EV, TR, RV, PA, NW, PS }`.

**The case.** Verbal short-term memory is a **signature DS deficit** (Næss, Lyster, Hulme & Melby-Lervåg 2011), which made it an odd node to leave nearly disconnected in a DS-specific graph. The phonological loop is a well-evidenced driver of new-word learning (the Baddeley–Gathercole tradition), and in DS specifically phonological memory predicted _language-comprehension growth_ over five years (Laws & Gunn 2004) — supporting reach to both taught and standardised vocabulary, to blending, and to the nonword/spelling tasks that lean directly on phonological storage. The old single edge understated all of this.

**Honesty.** `RW` (erbto: word + nonword repetition) is itself a spoken-response task, so it shares the production-gating caveat of §5. And `RW` is kept **intervention-free** (no `IG`/`IS` parent) as a clean exogenous-capacity proxy — a modelling choice worth stating explicitly, because the DS reading-as-intervention literature includes claims that reading instruction improves verbal memory (Laws & Gunn 2002), which would contradict the isolation. In a contemporaneous graph the clean-capacity reading is the more conservative one; the reverse channel belongs in the time-lagged workstream.

## 8. Phonetic spelling `PS` as an outcome

Demoted in this revision (decision 6): `PS → WR` is dropped, and `PS` becomes terminal.

**The case.** The `PS` task involves **writing letters**; it is a reading/spelling outcome in its own right, not a cause of word reading. The TD evidence that invented spelling _promotes_ reading (Ouellette & Sénéchal 2008) has no DS support, `PS` is heavily floored in this cohort (≈ 78% / 64% at t1/t2, so there is almost no variance to transmit and the edge was prior-dominated), and the self-teaching direction (`WR → PS`; Share 1995) is at least as defensible. Treating `PS` as an outcome alongside `WR` (and `NW` as a mediator) is the floor-robust, theory-consistent choice. It also removes a collider trap: `PS` was a common effect of `LS` and `PA`, which must never be conditioned on.

## 9. What the graph deliberately excludes

The omissions are strong claims, and each is a considered choice.

- **Grammar and broader language are severed from `WR`.** `RG`, `EG`, `EI`, `LF` have no path to word reading. This is defensible and kept: TD grammar predicts _comprehension_, not word recognition; the outcome here is word-level; and DS grammar is disproportionately impaired (Næss et al. 2011), so it is better cast as an outcome than a decoding cause. The disclosure is that any grammar→`WR` estimate would be structurally forced to null.
- **SES is not a node** but is used as a robustness covariate in a few ITT sensitivity fits. TD SES→language gradients are robust; the split (no structural node, but an optional sensitivity adjustment) is a deliberate, now-recorded choice.
- **RAN (rapid automatized naming) is unmeasured.** It is the second pillar of TD word-reading prediction (Lervåg & Hulme 2009). Nothing to fix in the graph, but it is a named **unmeasured input** to the code route — a potential common cause of `LS`/`PA` and `WR` we cannot adjust for.
- **There are no reverse `WR → language` edges.** Reading instruction as a _route into_ language and verbal memory is the founding hypothesis of the intervention tradition RLI sits in (Laws & Gunn 2002; and print-exposure "Matthew effects" in TD, Stanovich 1986). It cannot be added to the contemporaneous graph without creating a cycle (`EV → PA → WR → EV`), so it is deferred to the time-lagged workstream (§12) rather than denied.

## 10. Domains not yet discussed

For completeness: the **intervention block** (`IG` randomised, `IS` dose) drives the taught targets and the code skills directly (`IG`/`IS → { TR, TE, PA, LS, WR, PS, EI, EG }`), reflecting that the RLI programme explicitly teaches vocabulary, letter sounds, blending and word reading. The **language and grammar** nodes (`LF`, `RG`, `EI`, `EG`) sit as downstream expressive/receptive language outcomes, receiving from vocabulary and grammar but not feeding reading. `RG → EG` encodes that receptive grammar supports expressive grammar.

## 11. Identification: what the DAG buys

- **ID-1 — the ITT effect `IG → WR` is point-identified** with the empty adjustment set, because `IG` is a parentless randomised root. Every structural edit in this revision is internal to the mediation architecture and leaves this untouched: the ITT/joint/DiD suite is unaffected. `HS`, baseline `SP` and `RW` are pre-randomisation child characteristics, so they are now available as optional **precision** covariates that can tighten τ without biasing it.
- **ID-2 — every skill→`WR` and mediator→outcome slope is confounded by latent `GA`** (and, at a single wave, by shared timing) and is **not point-identified**. These are reported as adjusted associations with direction and uncertainty, never as causal effects. What the revision changes is _which observed confounders enter the adjustment sets_: `SP`, `HS` and a wider `RW` are now common causes that must be adjusted in the mechanism, mediation and factor families (the follow-up work tracked in issue #251).

The practical reading rule the project already uses: positive τ means the intervention helps; only τ is causal; observational couplings are adjusted associations.

## 12. Weaknesses, challenges, and open questions

Stated plainly, because the graph's value depends on its limits being visible.

- **The deepest issue: a static acyclic graph for a four-wave panel.** The reciprocal directions the literature actually debates (does `WR` at wave _t_ predict `PA`/`EV` _change_ to _t+1_, or the reverse?) are mostly _resolvable in time_, but a single cross-sectional graph cannot represent them — so reciprocity is collapsed one-way, `WR → language` is unrepresentable, and several directions (`SP`, `PS`, `LS`↔`PA`) are assigned by theory rather than by sequence. A **wave-unrolled companion DAG** would make the reciprocal edges representable without cycles, let each longitudinal model's adjustment set be read off the graph, and convert currently untestable internal edges into cross-lagged hypotheses. This is the single highest-value upgrade available and is scoped as its own workstream (issue #250).
- **The `PA → WR` weight is the most DS-contested commitment** (§4.1). The graph keeps the TD code skeleton while flagging that its phonemic weight is probably too high for DS.
- **`GA` is latent and never adjustable**, so no internal slope is claimed unconfounded; the random intercept is only a partial proxy.
- **Untestable internal structure.** Because `GA` points into almost everything, most internal independencies cannot be checked in-sample; the edges are theory-led.
- **Collinearity makes several separations fragile** — `EV`/`RV`, and `SP`/`EV`/`PA` — so coefficients that partition shared signal (the receptive-vs-expressive asymmetry, the vocabulary→`PA` link) can be unstable.
- **Measurement limits.** Spoken-response tasks are production-gated by `SP` in a way construct edges only partly capture; `PA` is blending only; `PS`, and the floored reading outcomes, transmit little variance; `deapp_c` is not on the PCC scale; `hearing_c` is a coarse composite.
- **`HS` exogeneity** is a simplification (`A → HS` is plausible), deferred to the time-lagged graph.

## 13. Alternatives considered and not adopted

| Alternative                                                                | Why not (this revision)                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Keep `EV → SP` (speech as a downstream sink)                               | Same-cohort trait-stability (r = 0.84) and severity (DS mean PCC below the worst-affected TD child) make speech a source, not a vocabulary-driven sink (§5).                                                                                                                                                                                                                                              |
| Keep `PA` as the sole gateway from expressive vocabulary to reading        | Structurally guarantees under-reading the DS vocabulary→reading link; the new direct `TE`/`EV → WR` edges remove the bottleneck (§4.2).                                                                                                                                                                                                                                                                   |
| Drop the standardised vocabulary nodes (`RV`/`EV`), keep only taught       | Would not simplify but _blind_ the graph: `RV` is a confounder for `EV`-based mechanisms, and the standardised measures are the only near-vs-far **transfer** signal. Taught measures are more proximal, but the fix is role-assignment (taught = proximal mediator/manipulation check; standardised = distal transfer outcome), not deletion — or a latent vocabulary factor if one construct is wanted. |
| Import the TD "speech → phoneme awareness → reading" _mechanism_ wholesale | The TD carrier (phoneme awareness) is precisely the DS-weak pathway, and the TD mediation was estimated off-support for DS severity; we adopt the orientation, quarantine the mechanism (§5).                                                                                                                                                                                                             |
| Replace single `GA` with a bifactor / correlated-domain-factor model now   | Identification-neutral and supported by mm-001 (domain r = 0.65–0.80), but not urgent; a sensible refinement when the measurement model is next revised (§3.1).                                                                                                                                                                                                                                           |
| Add reverse `WR → language` edges to the current graph                     | Closes a cycle; belongs in the wave-unrolled graph, not the contemporaneous one (§9, §12).                                                                                                                                                                                                                                                                                                                |

## 14. References

DOIs verified against PubMed/publisher pages during drafting, except the three marked _to verify_, which must be confirmed before use in the report (per `METHODS.md`).

- Bradley, L., & Bryant, P. E. (1983). Categorising sounds and learning to read — a causal connection. _Nature, 301_, 419–421. https://doi.org/10.1038/301419a0
- Burgoyne, K., Duff, F. J., Clarke, P. J., Buckley, S., Snowling, M. J., & Hulme, C. (2012). Efficacy of a reading and language intervention for children with Down syndrome: a randomised controlled trial. _Journal of Child Psychology and Psychiatry, 53_(10), 1044–1053. https://doi.org/10.1111/j.1469-7610.2012.02557.x
- Burgoyne, K., Lervåg, A., Malone, S., & Hulme, C. (2019). Speech difficulties at school entry are a significant risk factor for later reading difficulties. _Early Childhood Research Quarterly, 49_, 40–48. https://doi.org/10.1016/j.ecresq.2019.06.003
- Burgoyne, K., Buckley, S., & Baxter, R. (2021). Speech production accuracy in children with Down syndrome: relationships with hearing, language, and reading ability and change in speech production accuracy over time. _Journal of Intellectual Disability Research_. https://doi.org/10.1111/jir.12862
- Cossu, G., Rossini, F., & Marshall, J. C. (1993). When reading is acquired but phonemic awareness is not: a study of literacy in Down's syndrome. _Cognition, 46_(2), 129–138. https://doi.org/10.1016/0010-0277(93)90016-O
- Dodd, B., Hua, Z., Crosbie, S., Holm, A., & Ozanne, A. (2002). _Diagnostic Evaluation of Articulation and Phonology (DEAP)._ London: Psychology Corporation. _(test manual, no DOI.)_
- Gough, P. B., & Tunmer, W. E. (1986). Decoding, reading, and reading disability. _Remedial and Special Education, 7_(1), 6–10. https://doi.org/10.1177/074193258600700104
- Hulme, C., Goetz, K., Brigstocke, S., Nash, H. M., Lervåg, A., & Snowling, M. J. (2012). The growth of reading skills in children with Down Syndrome. _Developmental Science, 15_(3), 320–329. https://doi.org/10.1111/j.1467-7687.2011.01129.x
- Hulme, C., Bowyer-Crane, C., Carroll, J. M., Duff, F. J., & Snowling, M. J. (2012). The causal role of phoneme awareness and letter-sound knowledge in learning to read: combining intervention studies with mediation analyses. _Psychological Science, 23_(6), 572–577. https://doi.org/10.1177/0956797611435921
- Laws, G., & Gunn, D. (2002). Relationships between reading, phonological skills and language development in individuals with Down syndrome: a five year follow-up study. _Reading and Writing, 15_, 527–548. https://doi.org/10.1023/A:1016364126817
- Laws, G., & Gunn, D. (2004). Phonological memory as a predictor of language comprehension in Down syndrome: a five-year follow-up study. _Journal of Child Psychology and Psychiatry, 45_(2), 326–337. https://doi.org/10.1111/j.1469-7610.2004.00224.x
- Laws, G., & Hall, A. (2014). Early hearing loss and language abilities in children with Down syndrome. _International Journal of Language & Communication Disorders, 49_(3), 333–342. https://doi.org/10.1111/1460-6984.12077
- Lervåg, A., & Hulme, C. (2009). Rapid automatized naming (RAN) taps a mechanism that places constraints on the development of early reading fluency. _Psychological Science, 20_(8), 1040–1048. https://doi.org/10.1111/j.1467-9280.2009.02405.x
- Melby-Lervåg, M., Lyster, S.-A. H., & Hulme, C. (2012). Phonological skills and their role in learning to read: a meta-analytic review. _Psychological Bulletin, 138_(2), 322–352. https://doi.org/10.1037/a0026744
- Metsala, J. L., & Walley, A. C. (1998). Spoken vocabulary growth and the segmental restructuring of lexical representations. In _Word Recognition in Beginning Literacy_ (pp. 89–120). _(citation/DOI to verify.)_
- Næss, K.-A. B., Lyster, S.-A. H., Hulme, C., & Melby-Lervåg, M. (2011). Language and verbal short-term memory skills in children with Down syndrome: a meta-analytic review. _Research in Developmental Disabilities, 32_(6), 2225–2234. https://doi.org/10.1016/j.ridd.2011.05.014
- Næss, K.-A. B. (2016). Development of phonological awareness in Down syndrome: a meta-analysis and empirical study. _Developmental Psychology, 52_(2), 177–190. https://doi.org/10.1037/a0039840
- Ouellette, G., & Sénéchal, M. (2008). Pathways to literacy: a study of invented spelling and its role in learning to read. _Child Development, 79_(4), 899–913. https://doi.org/10.1111/j.1467-8624.2008.01166.x
- Ricketts, J., Nation, K., & Bishop, D. V. M. (2007). Vocabulary is important for some, but not all reading skills. _Scientific Studies of Reading, 11_(3), 235–257. https://doi.org/10.1080/10888430701344306
- Share, D. L. (1995). Phonological recoding and self-teaching: sine qua non of reading acquisition. _Cognition, 55_(2), 151–218. _(citation/DOI to verify.)_
- Stanovich, K. E. (1986). Matthew effects in reading: some consequences of individual differences in the acquisition of literacy. _Reading Research Quarterly, 21_(4), 360–407. _(citation/DOI to verify.)_

## 15. Related documents and issues

- [`dag-language-reading.dagitty`](dag-language-reading.dagitty) — the authoritative machine-readable graph (paste into dagitty.net); [`dag-language-reading.dot`](dag-language-reading.dot) / [`.svg`](dag-language-reading.svg) — the rendered figure and its source; [`README.md`](README.md) — directory overview.
- [`../notes/202607101100-dag-revision-team-decisions.md`](../notes/202607101100-dag-revision-team-decisions.md) — the decision record for this structure; [`../notes/202606231600-dag-revision-consolidated.md`](../notes/202606231600-dag-revision-consolidated.md) — the superseded 2026-06-23 structure and its deliberation.
- `../notes/202607091430-dag-critical-review-td-atypical-literature.md` and `../notes/202607091615-vocab-reading-subgraph-critical-review.md` — the two critical reviews (issue #233) this exposition draws on.
- Issues #244–#251 — the model-adjustment follow-ups (adjustment sets, PS-as-outcome, figures) and the time-lagged DAG workstream (#250).
