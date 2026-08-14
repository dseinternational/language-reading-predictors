<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- cspell:ignore Byrne MacDonald Buckley readgrp basread basspel bpvs trog woco basdig bassim basnum basmat rlm dagitty deapp erbto rowpvt eowpvt yarclet aptinfo aptgram -->

> [!NOTE]
> Drafted by a LLM-based AI tool (Claude Code/Fable 5).
>
> Working-graph adoption and lagged companion substantially updated by a LLM-based AI tool (Codex/GPT-5).

> [!IMPORTANT]
> **Status: adopted as the working contemporaneous graph on 2026-08-14.** The lagged companion and the limits on any fitted reciprocal model are recorded in `notes/202608141700-byrne-lagged-dag-decision.md`. This remains a structural assumptions record, not evidence that any arrow is causal.

# A DAG for the Byrne, MacDonald & Buckley (2002) cohort — adopted working graph

Consistent with the authoritative RLI base graph (`dag/dag-language-reading.dagitty`, revised 2026-07-10) but adapted for the measures actually present in the prepared Byrne extract (`data/reading-language-memory/`, `study_id="rlm"`). The graph keeps the RLI causal **skeleton** among the skills that have a Byrne counterpart, drops the RLI nodes with no measure here, collapses the unmeasured decoding route into direct edges, and — the single most consequential change — replaces the **randomised** intervention (`IG`/`IS`) with the **observational** cohort factor `readgrp`.

## Files

- `dag/dag-reading-language-memory.dagitty` — machine-readable graph (paste into dagitty.net). 13 explicit nodes, 42 directed edges plus one bidirected design association; the directed component is acyclic.
- `dag/dag-reading-language-memory.dot` / `.svg` / `.png` — colour-coded left→right rendering (regenerate with `dot -Tsvg dag/dag-reading-language-memory.dot -o dag/dag-reading-language-memory.svg`), styled to match the RLI figure.
- `dag/dag-reading-language-memory-lagged.dagitty` — adopted two-slice annual companion, with rendered `.dot` / `.svg` / `.png` views.

## Construct → measure crosswalk

Node symbols follow the repo convention of study-local Byrne column names (kept separate from the RLI symbol namespace, per `datasets.py`), with the RLI symbol each maps to in brackets.

| RLI node       | Construct                                      | Byrne measure (symbol)                             | Status in the Byrne DAG                                        |
| -------------- | ---------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| `A`            | Age                                            | `age` (months)                                     | kept — universal parent                                        |
| `GA`           | General ability [latent]                       | latent, **now with indicators** `bassim`, `basmat` | kept latent; partially measured (see below)                    |
| `IG`/`IS`      | Intervention group / dose (randomised)         | —                                                  | **replaced by `readgrp`** (observational cohort factor)        |
| `RV`           | Receptive vocabulary                           | `bpvs` (British Picture Vocabulary Scale)          | kept — direct counterpart                                      |
| `RG`           | Receptive grammar                              | `trog` (Test for Reception of Grammar)             | kept — **same instrument** as RLI                              |
| `RW`           | Word/nonword repetition (phon. memory)         | `basdig` (BAS recall of digits)                    | kept — digit-span proxy for verbal STM                         |
| `WR`           | Word reading (primary outcome)                 | `basread` (BAS word reading)                       | kept — direct counterpart, primary outcome                     |
| `PS`           | Phonetic spelling (outcome)                    | `basspel` (BAS spelling)                           | kept — spelling outcome, terminal-ish                          |
| `HS`           | Hearing status                                 | —                                                  | kept **latent**; unmeasured common cause                       |
| `EV`           | Expressive vocabulary                          | —                                                  | **dropped** (no expressive measure)                            |
| `LF`,`EI`,`EG` | Expressive language / grammar / info           | —                                                  | **dropped** (no measures)                                      |
| `SP`           | Speech production                              | —                                                  | **dropped** (no measure); optional latent extension            |
| `LS`,`PA`,`NW` | Letter-sound / blending / nonword (code route) | —                                                  | **dropped as nodes; collapsed** into direct edges to `basread` |
| `TE`,`TR`      | Taught expressive / receptive vocab            | —                                                  | **N/A** — no teaching intervention in this study               |
| —              | Reading comprehension                          | `woco` (WORD reading comprehension)                | **added** — new terminal outcome (Simple View)                 |
| —              | Number / arithmetic                            | `basnum` (BAS number skills)                       | **added** — new academic outcome                               |
| —              | Verbal / non-verbal reasoning                  | `bassim` / `basmat`                                | **added** — GA indicators                                      |
| —              | Processing speed                               | `speed` (wave 3+ only)                             | **documented, not drawn** — too sparse (wide file only)        |

## The one structural change that matters most: `IG`/`IS` → `readgrp`

The RLI DAG's confounding-identification story rests on `IG` being **randomised and parent-less**, so the full-cohort assigned-arm effect `IG → WR` requires the _empty_ confounder adjustment set; the fitted RLI result is nevertheless an **available-case modified ITT estimate** whose causal reading also requires the stated selection assumption. **None of that transfers.** The Byrne study has no intervention: `readgrp` (1 = Down syndrome, 2 = average readers, 3 = reading-matched) is an **observational cohort factor** — a non-manipulable marker of developmental population, not an assigned condition. It is drawn as an exogenous root pointing into every observed skill (population differences pervade every measure), but **no coefficient on `readgrp` is a causal effect**. Every between-group quantity is a descriptive/adjusted association, exactly as `lrp-rlm-hg-001` already labels its group-by-wave growth ("`readgrp` is a **cohort factor**, not a randomised or assigned condition"). This DAG simply makes that status structural.

### `readgrp` carries a selection hazard the RLI graph never had

The **reading-matched** group (3) is, by construction, selected so that its reading level matches the Down-syndrome group's. That is _selection on `basread`_ — a collider. So a Down-syndrome-vs-reading-matched contrast on any reading-associated skill (vocabulary, memory, comprehension) is, whether intended or not, a **conditioned-on-reading** contrast, and can open non-causal paths between `basread` and its correlates. This is not a nuisance to be adjusted away — it _is_ the paper's design, and it is precisely why the reading-matched comparison speaks to "given equal reading, do the groups differ on memory/language?". The base graph does not draw a selection node (matching how RLI keeps design features such as the crossover in prose), but any analysis using group 3 must treat `basread` as a selection variable. An explicit selection-augmented variant is sketched at the end.

## Edge-by-edge: how each Byrne edge descends from RLI

The skill cascade is the RLI structure restricted to measured counterparts, plus two Simple-View reading-comprehension edges that RLI had no node for.

- **`basdig → { bpvs, basread, basspel, basnum, woco }`** — from RLI `RW → {RV, PS}` (giving `basdig→bpvs`, `basdig→basspel`) plus the **collapsed code route**: RLI reaches `WR` from phonological memory only through the unmeasured `LS/PA/NW` chain, so with none of those measured here the direct `basdig → basread` edge stands in for that whole chain. `basdig → basnum` (verbal working memory supports calculation) and `basdig → woco` (verbal WM supports comprehension) are Byrne-specific but well-evidenced in the DS literature.
- **`bpvs → { trog, basread, woco }`** — from RLI `RV → {RG, WR}` (`bpvs→trog`, `bpvs→basread`) plus the Simple-View `bpvs → woco` (vocabulary → reading comprehension).
- **`trog → woco`** — from the language route: grammar/listening comprehension → reading comprehension (Simple View). RLI's `RG → EG` target is unmeasured; RLI has no `RG → WR`, so there is deliberately no `trog → basread`.
- **`basread → woco`** — the Simple View's decoding component of comprehension. This is why `basread` is **no longer a sink** here, unlike RLI where `WR` is terminal.
- **`basspel`, `basnum`** — terminal outcomes, driven by `age`, `GA`, `readgrp`, `basdig`. Consistent with RLI keeping `PS` terminal (no `WR→PS` / `PS→WR`); spelling and word reading are correlated siblings sharing upstream causes rather than one causing the other.
- **`bassim`, `basmat`** — pure **indicators of `GA`** (children of `age`, `GA`, `readgrp`, no outgoing edges); they are proxies to _use_, not causes.

Everything else is the two universal parents (`age`, `GA` → every observed node, as in RLI), latent `HS → {basdig, bpvs, trog, basread, basspel}`, plus `readgrp` → every observed node. Hearing is retained rather than treated as absent because the primary paper describes hearing impairment as common in this population; the extract's failure to measure it creates residual confounding, not structural evidence that it is irrelevant.

## `GA` is no longer purely latent — a genuine analytic gain

RLI's `GA` has no indicators; it can only ever be a latent common cause that biases observational couplings. Byrne measured general ability: `bassim` (verbal reasoning) and `basmat` (non-verbal reasoning, wave 3+) are drawn as GA indicators, so conditioning on them **partially deconfounds `GA`** for the skill-to-skill associations. This is the closest the Byrne data gets to the RLI robustness adjustment that used block-design as a general-ability control (the ITT-017–024 arm). It is partial (two noisy indicators, `basmat` only from wave 3), so it tightens rather than closes the `GA` back-door — report it as an adjusted association, never as "controlled for ability".

## What this identifies — and what it does not

- **No clean causal exposure exists.** Because `readgrp` is observational and every skill edge is an adjusted association (as in every RLI observational coupling), **nothing in this graph is a causal effect**. This is the honest headline. The RLI suite's causal `τ` has no Byrne analogue.
- **What is estimable:** descriptive group-by-wave trajectories per measure (already `lrp-rlm-hg-001`); adjusted skill-to-skill associations with `age`, `GA` (via `bassim`/`basmat`), and `readgrp` in the adjustment set; and — with the lagged companion below — cross-lagged tests of directional precedence.
- **Adjustment sets are read off the graph** the usual way, but none is sufficient using the available measurements because `GA` and `HS` are latent. Example: for a `basdig → basread` association, the back-door set includes `{age, GA, HS, readgrp}`; `bassim`/`basmat` can only proxy `GA`, and no hearing proxy exists. For `bpvs → basread`, add `basdig` as a measured common cause. For any estimand involving the reading-matched group, remember `basread` is a selection variable.

## Remaining latent extension (sensitivity, not the base graph)

Hearing is now part of the working graph. The remaining load-bearing optional extension is:

- **`DEC` [latent] decoding / phonics** (the collapsed `LS/PA/NW` route) → `{basread, basspel, woco}` — makes explicit that `basread`, `basspel` and the decoding arm of `woco` share an unmeasured common cause. Under this variant the direct `basdig→basread` / `bpvs→basread` edges route through `DEC` instead of collapsing it.

`DEC` cannot be conditioned on, so it changes interpretation rather than the point estimate. It should remain a named sensitivity structure until a decoding-route measure is available.

## The reciprocal reading↔language↔memory question needs the lagged companion

The **founding** Byrne question — does learning to read _feed_ language and memory in Down syndrome? (Byrne et al. found _no_ evidence it did) — is inherently longitudinal and **cannot** be expressed in this contemporaneous graph. The adopted companion `dag/dag-reading-language-memory-lagged.dagitty` copies the within-wave cascade, carries each measure forward and adds exactly `basread_t → {bpvs_t1, trog_t1, basdig_t1}`. The paper reports three annual occasions; prepared waves 4–5 are later extensions, and the visual-memory measures in the paper are absent from the repository. The companion therefore licenses a narrow receptive-language/auditory-memory question, not a claim about all language or memory. The graph remains the structural record, but the required recovery study rejected both candidate models; see `notes/202608141812-byrne-lcsm-feasibility.md`.

## Suggested next steps / decisions for the team

1. `readgrp` remains a non-causal cohort marker and `basread` remains a selection variable for group 3 — adopted in the 2026-07-16 decisions.
2. `bassim`/`basmat` remain noisy `GA` indicators; neither licenses the phrase “controlled for ability.”
3. `HS` is promoted to latent in the working graph; `DEC` remains a named sensitivity extension.
4. The lagged companion is green-lit structurally, with three source-anchored reverse edges; the pre-fit recovery study rejected both candidate models, so no fitted model is registered.
5. Phase C is no-go under the current data. Reopening it requires new repeated observations, recovery of missing source measures, or a separately justified narrower question followed by a new pre-fit recovery study. The manual gates for `woco`, `basnum` and `basspel` are orthogonal because none is a reverse target; extract provenance still blocks publication of every RLM fit.
