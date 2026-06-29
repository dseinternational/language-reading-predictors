# Measurement-sensitivity audit of the RLI outcome measures

> [!WARNING]
> AI-assisted analysis. Numbers are reproducible from
> `scripts/measurement_audit.py`; interpretation reviewed against the data but
> should be sanity-checked by the study team.

Date: 2026-06-17

## Context

The ITT models (LRP52–LRP55) show a robust word-reading (W) and letter-sound (L)
effect but **no credible vocabulary effect** (receptive R, expressive E). Before
that null is read as "the vocabulary component of RLI does not work", we need to
know whether each outcome even had the **range and reliability to register a
change** in the randomised window (t1 → t2). A floored, ceilinged, or barely
moving measure cannot show a treatment effect regardless of whether one exists.

This audit also selects which phonics-route measures can serve as mediators in the
LRP62 reading-route composite (it needs measures with usable spread in this
window).

## Method

`scripts/measurement_audit.py` (purely descriptive, no MCMC) reads
`data/rli_data_long.csv` (n = 54 children, 4 waves) and, per measure × timepoint,
reports floor/ceiling fractions, fraction of the nominal scale used, a between-child
dispersion proxy, and the fraction of children whose score moves between t1 and t2.
A transparent rule flags a measure **detection-limited** for the t1 → t2 window if
any of: ceiling crowding at t2 (≥ 40 % at max), a narrow t2 range (< 25 % of scale),
a persistent floor at t2 (≥ 40 % at zero), or a heavy t1 floor (≥ 40 %) combined with
few movers (< 40 %). Outputs: `output/measurement_audit/outcome_properties.csv` and
`detectability_verdict.csv`.

## Results (t1 → t2 randomised window)

| Sym   | Measure (max)                 | floor t1 | floor t2 | ceil t2 | scale used t2 | movers | mean Δ | verdict               |
| ----- | ----------------------------- | -------- | -------- | ------- | ------------- | ------ | ------ | --------------------- |
| W     | Word reading (90)             | 40 %     | 15 %     | 0 %     | 62 %          | 77 %   | +3.4   | adequate              |
| R     | Receptive vocab (190)         | 0 %      | 0 %      | 0 %     | 28 %          | 94 %   | +3.1   | adequate              |
| E     | Expressive vocab (170)        | 0 %      | 0 %      | 0 %     | 36 %          | 96 %   | +4.3   | adequate              |
| **L** | **Letter-sound (32)**         | 9 %      | 0 %      | 0 %     | 91 %          | 94 %   | +5.2   | **adequate**          |
| P     | Phonetic spelling (100)       | 78 %     | 64 %     | 0 %     | 92 %          | 36 %   | +5.2   | **detection-limited** |
| **B** | **Blending (10)**             | 4 %      | 2 %      | 6 %     | 100 %         | 78 %   | +0.7   | **adequate**          |
| F     | Basic concepts / CELF (18)    | 2 %      | 4 %      | 6 %     | 100 %         | 89 %   | +0.8   | adequate              |
| T     | Receptive grammar / TROG (32) | 0 %      | 0 %      | 0 %     | 69 %          | 94 %   | +1.3   | adequate              |
| N     | Nonword reading (6)           | 72 %     | 64 %     | 6 %     | 100 %         | 46 %   | +0.3   | **detection-limited** |

One-line verdicts:

- **W** — adequate. Baseline floor (40 % can read no words) clears to 15 % by t2;
  three-quarters of children move. The outcome can register a treatment effect.
- **R, E (vocabulary)** — adequate. **No floor, no ceiling, and 94–96 % of children's
  scores move** (mean +3.1 / +4.3 raw points). They occupy a compressed band of the
  big norm-referenced scales (28 % / 36 % of 190 / 170 items), but that is
  range-compression _with preserved within-band variation and movement_, not a
  floor/ceiling or no-movement problem.
- **L** — adequate, and the best-behaved measure (91 % of scale, 94 % move). Ideal
  composite component.
- **B** — adequate (full scale used, 78 % move). Good composite component.
- **P (phonetic spelling)** — **detection-limited.** 78 % score zero at baseline and
  64 % still score zero at t2; only 36 % move. Phoneme-level spelling is largely
  beyond these children in this window.
- **N (nonword reading)** — **detection-limited.** 72 % floored at baseline, 64 %
  still floored at t2. Most children cannot yet decode nonwords at all.
- **F, T** — adequate.

## Implications

**1. The vocabulary null is most consistent with a _true_ null, not a detectability
artefact.** R and E are not floored, not ceilinged, and show substantial
child-to-child movement across the window; the instruments demonstrably _can_
register change. So "no credible vocabulary treatment effect" should be read as the
RLI vocabulary component not lifting these scores **above maturation**, rather than
as a measure that could not have shown an effect. _Caveat:_ this audit assesses
range / floor / ceiling / movement only — it cannot estimate internal-consistency
**reliability** from single summary scores, so it rules out gross detectability
failure but not a subtler differential-reliability story.

**2. LRP62 route composite = {L, B}; drop P.** Letter-sound (L) and blending (B) both
have usable spread and movement and are sound mediators. Phonetic spelling (P) is
detection-limited (floored) and is **excluded** from the composite. Nonword (N) is
post-only and also floored, reinforcing that it cannot serve as a baseline-conditioned
mediator. This confirms the plan's route block, pruned by data.

## Reproduce

```
python scripts/measurement_audit.py
# -> output/measurement_audit/outcome_properties.csv
# -> output/measurement_audit/detectability_verdict.csv
```
