<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Language & reading causal DAG — locked revision and deliberation record

> [!WARNING]
> This note was prepared with an AI tool and may contain mistakes. The causal
> structure encodes deliberate, contestable theoretical commitments (flagged
> throughout as **cautions**). Reading-science citations are named by framework and
> must have their exact references and DOIs verified before they enter the report,
> per `METHODS.md`.

Date: 2026-06-23 — **Status: LOCKED** (supersedes the in-progress drafts in this
note's history).

## Purpose

Records the final causal DAG for the **step-2 Bayesian models** (ITT, joint,
mechanism, mediation, and a prospective dose-response) and the full deliberation
that produced it: a multi-agent structural review followed by a sequence of
targeted design decisions on the latent ability node, age, the intervention and
dose nodes, taught vs standardised vocabulary, the reading/phonics route, phonetic
spelling, and speech production. This is the design artefact those models should be
built against; it changes no code yet.

Machine-readable copy (paste into dagitty.net): [`notes/dag-language-reading.dagitty`](dag-language-reading.dagitty).

## The locked DAG

```
dag {
GA [latent]
IG [exposure]
WR [outcome]
A  -> { IS TR TE PA LS WR PS RV EV LF RG RW EI EG SP NW }
GA -> { IS TR TE PA LS WR PS RV EV LF RG RW EI EG SP NW }
IG -> { IS TR TE PA LS WR PS EI EG }
IS -> { TR TE PA LS WR PS EI EG }
TR -> { TE RV EV LF RG WR }
TE -> { EV EG EI PA SP }
RW -> EV
RV -> { EV LF RG WR }
EV -> { EG EI PA SP }
LS -> { NW PA PS WR }
NW -> WR
PA -> { NW WR PS }
PS -> { WR }
RG -> EG
}
```

### Node definitions

| Symbol | Construct (measure) | Role |
|---|---|---|
| `A` | Age | observed root (maturation + cumulative exposure) |
| `GA` | General ability | **latent** root (age-residualised) |
| `IG` | Intervention group | randomised **exposure** (root) |
| `IS` | Intervention sessions / attendance ("attend") | dose |
| `TR` / `TE` | Taught receptive / expressive vocabulary (LRP74–76) | direct teaching targets |
| `RV` / `EV` | Standardised receptive / expressive vocabulary (rowpvt / eowpvt) | transfer measures |
| `RW` | Word + nonword repetition (erbto) — phonological memory | capacity |
| `LF` | Language fundamentals (CELF basic concepts) | |
| `RG` / `EG` | Receptive / expressive grammar (TROG / APT-grammar) | |
| `EI` | Expressive information (APT-information) | |
| `SP` | Speech production / articulation (DEAP, deapp_c) | outcome (sink) |
| `LS` | Letter–sound knowledge | direct teaching target |
| `PA` | Phonological awareness / blending | direct teaching target |
| `NW` | Nonword reading (decoding) | |
| `PS` | Phonetic spelling | |
| `WR` | Word reading | **outcome** |

### Verified structure (networkx; `output/replication/scratch/dag_v3_check.py`)

- **Acyclic**, 19 nodes. **Roots:** `A`, `GA`, `IG`. **Childless sinks:** `EG`, `EI`, `LF`, `SP`, `WR`.
- **Not ancestors of `WR`** (no causal path to word reading): `EG`, `EI`, `LF`, `RG`, `SP`.
- **Only `RW` is intervention-free** (a structural zero — no directed path from `IG`); every other measured skill is reachable from the intervention.
- **`WR` parents:** `{A, GA, IG, IS, LS, NW, PA, PS, RV, TR}`.
- **ITT identified:** `IG` is a parentless randomised root ⇒ the total effect `IG → WR` is identified by the **empty** adjustment set.

## Deliberation and decisions

The graph began from an in-session draft and went through a multi-agent review
(below) and roughly a dozen design decisions. Each is recorded with its rationale
and the caution it carries.

### 1. General ability `GA` (latent)

- **Decision.** Keep a single latent common cause of (almost) all measures,
  redefined as **age-residualised** stable ability (so `A → GA` is dropped — see §2).
- **Why.** `GA` is the graphical encoding of the project's standing position that
  *only the randomised contrast is causal*: it makes explicit that every skill→skill
  and mediator→outcome association is confounded by a shared stable trait
  (the CLPM / stable-trait argument, `notes/202606201117-longitudinal-modelling-stance-clpm.md`).
  Because `GA` does **not** point into `IG`, the graph also shows *why the ITT
  survives* (randomisation d-separates `IG` from `GA`).
- **Caution.** `GA` is **diagnostic, not a lever** — it is latent, never in an
  implementable adjustment set, and the subject random intercept is only a partial
  (shrunken) stand-in, not "control for ability." It also makes nearly every internal
  conditional independence **untestable** (every separating set contains `GA`), so the
  internal edges are imported from theory, not checkable in these data. A single `g`
  is also the wrong granularity for the observed same-construct clustering
  (`notes/202606231100-gb-selected-features-tables.md`); replacing it with correlated
  domain factors (vocabulary / code / speech / grammar) is identification-neutral but
  a better measurement match — deferred (§ open decisions).

### 2. Age `A`

- **Decision.** Age acts through **two channels**: maturation of ability (`A → GA`)
  *and* a direct **cumulative-exposure / opportunity** channel to every skill
  (`A → {all 15 skill nodes, incl. RW}`). We **drop `A → GA`** and make `GA`
  age-residualised, so age carries the maturation+exposure effect directly and `GA`
  is the stable between-child residual.
- **Why.** Older children have had more time and instruction to learn word meanings,
  to produce words, and to read — independent of general ability. This is *especially*
  forceful here because the likelihood is on **raw bounded counts**, not age-normed
  scores, so the exposure effect is fully present in every measure. It also reconciles
  the graph with the code, which already enters observed age directly (`gamma_A` /
  `f_A`), and it makes age an **observed** confounder of every mechanism — exactly the
  adjustment the age-fix added (`notes/202606172100-mechanism-age-adjustment.md`); this
  DAG would have flagged that bug.
- **Caution.** With `GA` latent, the "via-`g`" vs "direct-exposure" split is **not
  separately identifiable**; only the *total* age effect is. Age is a **proxy** for
  cumulative opportunity/instruction — a measured exposure (schooling/instruction
  months) would be preferable, and the age/wave/schooling collinearity across the four
  waves must be handled explicitly. `RW` (phonological memory) gets a direct age edge
  on the view that capacity grows with practice.

### 3. Intervention group `IG` and sessions/dose `IS`

- **Decision.** `IG` is the randomised exposure. **`IG → IS`** (group drives dose:
  in the randomised window the control arm receives ≈ 0 sessions), **`GA → IS`**
  (abler/better-supported children attend more) and **`A → IS`** (older children
  differ in dose) — so `IS` has parents `{A, GA, IG}`. Direct `IG`/`IS` teaching
  edges go to the taught targets `{LS, PA, WR, TE, TR, EI, EG}` plus `PS`; other
  skills are reached only by transfer.
- **Why.** The waitlist-crossover design means group is near-deterministic of dose in
  the randomised window; within-group dose then varies with ability, age, and
  idiosyncratic reasons. Trimming the direct fan-out to genuine teaching targets makes
  the graph a falsifiable theory of change rather than "the intervention affects
  everything directly," and it matches the observed effect concentration (credible τ
  on `WR`, `LS`; suggestive on `PA`/blending).
- **Caution.** `GA → IS` (plus `IG → IS`) makes `IS` a **collider** on `IG → IS ← GA`.
  Consequences: **never condition on `IS`** (it reopens the latent-`GA` backdoor and
  would bias even the ITT); `IS → skill` couplings are GA-confounded **adjusted
  associations, never dose "effects."** A dose-response on `WR` is therefore
  observational (see §ID-3), not causal.

### 4. Taught (`TE`/`TR`) vs standardised (`EV`/`RV`) vocabulary

- **Decision.** Add the taught-vocabulary nodes as **direct teaching targets**
  (`IG`/`IS → TE`, `TR`), with **transfer edges** to the standardised tests
  (`TE → EV`, `TR → RV`) and broad direct edges onward
  (`TR → {TE, RV, EV, LF, RG, WR}`, `TE → {EV, EG, EI, PA, SP}`).
- **Why.** RLI directly teaches a small curated word set (distinct from the
  standardised ROWPVT/EOWPVT), and the trial found effects **did not transfer** to
  standardised vocabulary (Burgoyne et al., 2012). Modelling taught vocabulary as a
  separate node whose effect *flows onward* lets standardised-vocab τ (and the
  downstream grammar/expressive nulls) be **estimable transfer effects** rather than
  by-design zeros — reconciling the graph with the existing LRP53/54/55 and LRP74–76
  models and with the project stance that a null vocabulary effect is a *finding, not
  a design prediction* (`project_rli_intervention_scope`).
- **Caution.** The `TR`/`TE` fan-outs are **broad**: the direct edges to grammar,
  language, reading and PA partly duplicate the routes carried by `RV`/`EV`. This
  asserts taught vocabulary affects those outcomes *over and above* its (weak) effect
  on standardised vocabulary — coherent, but a strong claim. The leaner alternative
  (route taught vocab through `RV`/`EV` only) was considered and rejected; revisit if
  the direct taught-vocab effects prove unidentifiable.

### 5. Reading / phonics route

- **Decision.** `LS → {NW, PA, PS, WR}`, `PA → {NW, WR, PS}`, `NW → WR`; vocabulary
  feeds reading both directly (`RV → WR`, `TR → WR`) and via the code skills
  (`EV → PA`, etc.). `EV → LS` was **dropped** (letter-sound is print/instruction-
  driven, not vocabulary-driven).
- **Why.** Standard decoding-leads-to-word-reading structure (the simple-view /
  phonics tradition), with vocabulary contributing semantic/lexical support directly
  and through the code route.
- **Caution.** `PA↔NW`, `PA↔WR`, `LS↔NW` are **reciprocal in the literature** but
  collapsed to one direction for an acyclic cross-sectional graph; never read as "PA
  drives reading." Where the panel supports it, prefer a cross-lagged specification.

### 6. Phonetic spelling `PS`

- **Decision.** Keep `PS` on the causal path to reading: `PS → WR`, with `PS` caused
  by `LS`, `PA` and the intervention.
- **Why.** Phonetic encoding can consolidate the grapheme–phoneme knowledge that
  decoding/word-reading reuse (the invented-spelling-promotes-reading view).
- **Caution (significant).** `PS` is **on the floor** in this sample — 78 % floor at
  t1, 64 % at t2, ~36 % movers (`notes/202606171000-measurement-sensitivity-audit.md`)
  — so as a *cause* of `WR` it has little variance to transmit and the `PS → WR`,
  `LS → PS`, `PA → PS` coefficients will be weak / prior-dominated. The **direction is
  contestable**: for early readers the self-teaching view (`WR → PS`, reading builds
  spelling; Share, 1995) is at least as defensible as `PS → WR`. And `PS` remains a
  **collider** (`LS → PS ← PA`, `IG → PS ← GA`) — a must-**not**-condition-on node.
  Removing `PS` from the WR path entirely was analysed (it changes **no** →`WR`
  adjustment set and removes a collider footgun) and remains a live alternative if the
  floor makes `PS → WR` uninformative; deferred.

### 7. Speech production `SP`

- **Decision.** `SP` is a downstream **outcome** caused by vocabulary:
  `EV → SP`, `TE → SP` (plus `A`, `GA`). It is a sink (no children). **For this
  population we adopt `EV → SP`** (vocabulary → speech).
- **Why.** Lexical-restructuring: vocabulary growth drives finer phonological/
  articulatory representations and hence speech. This also re-homes `SP` (it was
  briefly orphaned) so that τ_SP is estimable as a vocab-transfer effect.
- **Caution (recorded at the user's request).** The direction is genuinely
  contestable. (a) Articulation/phonology is often treated as *upstream* of expressive
  vocabulary (`SP → EV`). (b) There is a **measurement dependency**: expressive-vocab
  tasks (EOWPVT, the taught-expressive set) require the child to *say* the word, so
  poor articulation depresses *measured* `EV` — a `SP → EV` channel that runs opposite
  to the chosen arrow, and is material in Down syndrome where speech intelligibility is
  a characteristic constraint. (c) `SP` is a sink, so its direction changes **no** `WR`
  analysis *as drawn*; but if it were flipped to `SP → EV`, `SP` would move onto the
  path to `WR` (via `EV`) and enter the vocab/`WR` adjustment sets. Treat τ_SP and any
  `EV`/`SP` relationship as confounded by this measurement link, not a clean construct
  separation.

### 8. Grammar / expressive language `RG`, `EG`, `EI`, `LF`

- **Decision.** `EG`, `EI` are direct teaching targets (`IG`/`IS →`); `RG` is reached
  via `RV`; `LF` via `RV`. None has a directed path to `WR`.
- **Why.** The expressive language strand is (in part) directly taught; receptive
  grammar and language fundamentals are reached through vocabulary.
- **Caution.** This **hard-codes that grammar/expressive language do not cause word
  reading** — a deliberate decoding-dominant scope choice for a floored early-reading
  sample. Any analysis estimating an effect of `RG`/`EG`/`EI`/`LF` on `WR` is
  structurally guaranteed null. (They remain valid *outcomes* of the intervention.)

## Identification implications

**ID-1 — ITT (`IG → WR`).** Point-identified, minimal adjustment set = ∅ (`IG` a
randomised root; `GA` does not touch `IG`). Baselines and age are precision terms.
**Do not condition on `IS`** (mediator + `GA`-collider).

**ID-2 — Mechanisms / mediator→outcome slopes.** Every skill→`WR` and mediator→outcome
relationship is confounded by latent `GA` and is **not point-identified** — report as
an *adjusted association*, never "X drives Y." Age is now an **observed** confounder
that must appear in every mechanism adjustment set. The subject random intercept is the
partial RI-CLPM repair (removes time-invariant `GA` up to partial-pooling shrinkage),
not full `GA` adjustment.

**ID-3 — Dose-response (`IS → WR`), collapsed over phases.** Estimable but
**observational**: confounded by `GA → IS`/`GA → WR` (and `A`), with only the group
contrast (≈ the ITT) being randomised. Use the all-phases mechanism machinery with `IS`
in the mechanism role, adjust `{IG, A}`, **do not** condition on the reading mediators
(for the total dose effect) and never on `IS` itself; prefer baseline-adjusted post over
raw gains. Run a fixed-effect (full within-child) sensitivity to strip time-invariant
`GA` completely; the **binding limitation is time-varying confounding** (period-specific
engagement/health driving both dose and gain), which the child intercept does not
remove. Report as a confounded within-child association.

## Cautions register (consolidated)

1. `GA` is latent — diagnostic only; internal edges largely untestable; random
   intercept is a partial, not full, stand-in.
2. Age's two channels are not separately identifiable; age is an exposure proxy.
3. `IS` is a collider — never condition on it; `IS → skill` are adjusted associations.
4. `TR`/`TE` fan-outs are broad (direct taught-vocab effects beyond standardised vocab).
5. `PS → WR` rests on a floored measure and a contestable direction; removal is a live
   alternative; `PS` is a collider.
6. `EV → SP` direction is contestable and entangled with a speech-gated measurement of
   `EV` (user-confirmed direction, cautions noted).
7. Grammar/expressive branch is severed from `WR` by design.
8. Reciprocal reading↔PA/decoding edges are collapsed one-way.

## Open / deferred decisions

- **Measurement model.** Replace single `GA` with correlated domain factors / bifactor
  (identification-neutral; better measurement); whether to split bundled nodes `RW`
  (erbto) and `SP` (deapp_c) into sub-scores; whether to **demote `PS` to a weak
  indicator** of a latent code factor rather than a standalone causal node (the
  floor-robust option).
- **`PS → WR` vs removal vs `WR → PS`** — revisit once the floored `PS → WR` slope is
  seen; cross-lagged spec if pursued.
- **`EV → SP` vs `SP → EV`** — re-open only if speech is to be treated as a constraint
  on measured vocabulary rather than a downstream outcome.

## Mapping to the existing models

- ITT / joint (LRP52–55, 60/60a) — unaffected; τ identified by randomisation.
- Mechanism (LRP56–58, 71–73) — adjustment sets must include observed age; slopes are
  adjusted associations (latent-`GA` confounded).
- Mediation (LRP59, 62) — NDE/NIE not identified (latent-`GA` + same-wave); triangulation
  only.
- Taught vocabulary (LRP74–76) — the `TE`/`TR` direct-teaching nodes; standardised-vocab
  τ is a transfer effect.
- Dose-response — new, see ID-3.

## Provenance

Built from an in-session draft via: a 5-dimension multi-agent structural review
(structure / arrows / identification / testable-implications / measurement, each
adversarially verified) which machine-checked acyclicity and the adjustment-set logic
and corrected several d-separation slips; a focused review of whether `IG`/`IS` should
be limited to teaching targets (with an RLI-content mapping grounded in Burgoyne et al.,
2012); and a computed analysis of removing `PS` (adjustment-set diffs verified in
networkx). Structural facts in this note were re-verified with
`output/replication/scratch/dag_v3_check.py`.

## Related notes

- `notes/202606201117-longitudinal-modelling-stance-clpm.md` — stable-trait / CLPM basis for `GA`.
- `notes/202606172100-mechanism-age-adjustment.md` — age-as-confounder fix the exposure edges formalise.
- `notes/202606171000-measurement-sensitivity-audit.md` — the `PS` (and other) floor/ceiling evidence.
- `notes/202606231100-gb-selected-features-tables.md` — same-construct clustering motivating domain factors.
- `notes/202606201500-gb-replication-findings.md` — why mechanism slopes are adjusted associations.

## References (verify exact citations + DOIs before use in the report)

- Burgoyne, K., Duff, F. J., Clarke, P. J., Buckley, S., Snowling, M. J., & Hulme, C.
  (2012). Efficacy of a reading and language intervention for children with Down
  syndrome: a randomized controlled trial. *Journal of Child Psychology and
  Psychiatry, 53*(10), 1044–1053. https://doi.org/10.1111/j.1469-7610.2012.02557.x
- Gough, P. B., & Tunmer, W. E. (1986). Decoding, reading, and reading disability
  (the simple view of reading). *(citation/DOI to verify.)*
- Share, D. L. (1995). Phonological recoding and self-teaching. *Cognition.*
  *(citation/DOI to verify.)*
- Metsala, J. L., & Walley, A. C. (1998). Lexical restructuring and phonological
  awareness. *(citation/DOI to verify.)*
- Ouellette, G., & Sénéchal, M. — invented spelling and learning to read.
  *(citation/DOI to verify.)*

## Reproduction

```bash
"V:\miniconda3\Scripts\conda.exe" run -n dse-language-reading-predictors \
  --no-capture-output python output/replication/scratch/dag_v3_check.py
```
