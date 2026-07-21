# Standard abbreviations for measures and variables

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

The single standard set of symbols for the study's measures and variables, for use in notes and model outputs (issue #374). The authoritative source is the base causal DAG, [`dag/dag-language-reading.dagitty`](../dag/dag-language-reading.dagitty); this note is a readable index of it. Use the **symbol** column everywhere an abbreviation is wanted; name the instrument when the specific test matters.

## Standard symbols (base DAG nodes)

| Symbol | Construct                             | Instrument / measure                                      |
| ------ | ------------------------------------- | --------------------------------------------------------- |
| A      | Age                                   | —                                                         |
| GA     | General ability (latent)              | unobserved                                                |
| HS     | Hearing status                        | `hearing_c` (impaired hearing or repeated ear infections) |
| IG     | Intervention group                    | randomised assignment                                     |
| IS     | Intervention sessions                 | `attend`                                                  |
| WR     | Word reading (primary outcome)        | EWRSWR (YARC Early Word + Single Word Reading)            |
| RV     | Receptive vocabulary                  | ROWPVT                                                    |
| EV     | Expressive vocabulary                 | EOWPVT                                                    |
| LF     | Language fundamentals                 | CELF Preschool-2 basic concepts                           |
| RG     | Receptive grammar                     | TROG-2                                                    |
| RW     | Word repetition / phonological memory | `erbto` (word + nonword repetition)                       |
| EI     | Expressive information                | `aptinfo`                                                 |
| EG     | Expressive grammar                    | `aptgram`                                                 |
| SP     | Speech production                     | `deapp_c` (proxy for persistent speech-motor difficulty)  |
| LS     | Letter-sound knowledge                | YARC-LSK (`yarclet`)                                      |
| NW     | Nonword reading                       | —                                                         |
| PA     | Phonological awareness / blending     | blending task                                             |
| PS     | Phonetic spelling                     | SPPHON                                                    |
| TE     | Taught expressive vocabulary          | bespoke block-1/block-2 taught words                      |
| TR     | Taught receptive vocabulary           | bespoke block-1/block-2 taught words                      |

Study-specific outcomes with no base-DAG node: `UR`/`UE` (not-taught receptive/expressive vocabulary — the not-taught comparison set), and the block-2 variants `TR2`/`TE2`/`UR2`/`UE2`.

## Modelling-code symbols

The Bayesian modelling code (`measures.py`) uses shorter single-letter symbols internally for the eight standardised ITT outcomes plus nonword reading. Human-facing output (report titles, labels, key findings) shows the **DAG symbol**; the map is:

| Code symbol | DAG symbol | Construct                         |
| ----------- | ---------- | --------------------------------- |
| W           | WR         | Word reading                      |
| R           | RV         | Receptive vocabulary              |
| E           | EV         | Expressive vocabulary             |
| L           | LS         | Letter-sound knowledge            |
| P           | PS         | Phonetic spelling                 |
| B           | PA         | Blending / phonological awareness |
| F           | LF         | Language fundamentals (CELF)      |
| T           | RG         | Receptive grammar (TROG)          |
| N           | NW         | Nonword reading                   |

`TR`/`TE` are used unchanged (they already match the DAG). The `DAG_SYMBOL` map in `measures.py` is the code-side source of this mapping.

## Historical (RLM) cohort

The Byrne/RLM historical-growth models use their own dataset variable codes (`basread`, `bpvs`, `trog`, `woco`, `basspel`, …) which are not part of this DAG; keep those codes as-is when referring to that cohort.
