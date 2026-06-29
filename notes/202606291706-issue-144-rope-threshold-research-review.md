> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Issue #144: research review for ROPE thresholds and meaningful change

Date: 2026-06-29

Related issue: <https://github.com/dseinternational/language-reading-predictors/issues/144>

## Purpose

Issue #144 asks whether the current ITT reporting thresholds for "meaningful
change" are educationally defensible and aligned with prior research. This note is
for team review. It is not the final education sign-off.

The short answer is that prior research does not appear to provide formal
minimally important differences for these exact raw-score measures. It does,
however, give useful anchors:

- the original Burgoyne et al. (2012) RCT reports the same intervention, the same
  20-week randomised comparison, and many of the same measures;
- later Down syndrome reading and vocabulary intervention studies support the
  same broad pattern: directly taught and decoding-related skills can move, while
  broad transfer outcomes are weaker and more variable;
- systematic reviews support caution, because language and communication
  intervention effects are often modest, heterogeneous, and probably overestimated
  in small or lower-rigour studies.

My recommendation is to keep the current domain-paced threshold rule as the
primary reporting convention for now, but to add explicit sensitivity checks for
the two places where the judgement is most fragile:

- word reading (`W`): report the current `delta = 1` item read, plus a sensitivity
  read at `delta = 2` items;
- floored outcomes (`P`, `N`): keep the provisional 10 percentage point
  off-floor risk-difference threshold only as a low-bar primary convention, and
  also show 15 and 20 percentage point sensitivity reads.

## The statistical issue in plain language

The ITT models estimate the randomised intervention effect, `tau`. Positive `tau`
means the immediate-intervention arm did better during the randomised phase.

There are two different questions:

1. Did the intervention probably help at all?
2. Was the benefit large enough to matter educationally?

The first question is a direction question. In the reports this is `pd`, the
posterior probability that the effect is above zero. A value like `pd = 0.99`
means that almost all of the model's plausible effect values are positive.

The second question is a size question. For that we need a threshold, called
`delta`, that says how large a benefit must be before we call it meaningful. The
ROPE, or region of practical equivalence, is the band from `-delta` to `+delta`.
Effects inside that band are treated as close enough to zero for the practical
question at hand.

That means an effect can have strong directional evidence but still be uncertain
in size. Word reading is the motivating example. A positive word-reading effect
is likely, but whether it clears a one-word or two-word meaningful-change bar can
lead to a different practical interpretation.

## Current rule

The current provisional rule is:

> `delta` = half of one period's natural maturation gain, rounded to whole items
> and floored at 1 item.

The "natural maturation gain" is the untreated wait-list arm's t1 to t2 progress
over the first 20-week period. The rule is intentionally educational rather than
purely statistical: it asks whether the intervention added at least a meaningful
fraction of what children would otherwise gain over a period.

That choice has an important consequence. Outcomes that naturally move slowly get
a lower bar. This may be right if "meaningful" should be judged relative to each
domain's pace of progress. It is more lenient than a distribution-based rule, such
as `0.2 * baseline SD`, for some outcomes, especially word reading.

## Data anchors from this study

The table below recomputes the main raw-score anchors from
`data/rli_data_long.csv`, using t1 rows and t1 to t2 gains. Group 1 is the
immediate-intervention group; group 2 is the wait-list control group.

| Outcome | Scale | Current delta | Baseline SD | 0.2 SD | 5% scale | Wait-list gain | Intervention gain | Raw gain gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `L` letter-sound knowledge | 32 | 2 | 8.69 | 1.74 | 1.60 | 3.23 | 6.93 | 3.70 |
| `W` word reading | 79 | 1 | 11.30 | 2.26 | 3.95 | 2.04 | 4.64 | 2.60 |
| `R` receptive vocabulary | 170 | 2 | 13.53 | 2.71 | 8.50 | 3.04 | 3.18 | 0.14 |
| `E` expressive vocabulary | 170 | 2 | 12.78 | 2.56 | 8.50 | 4.31 | 4.36 | 0.05 |
| `TR` taught receptive vocabulary | 24 | 1 | 4.09 | 0.82 | 1.20 | 2.12 | 3.46 | 1.35 |
| `TE` taught expressive vocabulary | 24 | 1 | 3.51 | 0.70 | 1.20 | 1.77 | 3.43 | 1.66 |
| `UR` not-taught receptive vocabulary | 12* | 1 | 2.11 | 0.42 | 0.60 | 0.31 | 1.32 | 1.01 |
| `UE` not-taught expressive vocabulary | 12* | 1 | 2.40 | 0.48 | 0.60 | 0.54 | 0.86 | 0.32 |
| `B` phoneme blending | 10 | 1 | 2.22 | 0.44 | 0.50 | 0.04 | 1.25 | 1.21 |

*The `UR` and `UE` denominators remain unconfirmed in the data dictionary. The
observed maximum is 12, which is consistent with a 12-item not-taught comparison
set, but this should not be treated as final until the original assessment
materials confirm it.

Two things stand out:

- For `L`, `R`, `E`, `TR`, and `TE`, the current delta is close to at least one
  of the independent anchors.
- For `W`, the current one-item delta is clearly lower than the distribution
  anchor (`0.2 SD` is about 2.3 items). This does not make `W = 1` wrong, but it
  does mean the report should show what happens at `W = 2`.

For the floored outcomes, the estimand is not extra raw-score items. It is the
increase in the probability of coming off the floor:

| Outcome | Immediate off floor by t2 | Wait-list off floor by t2 | Raw risk-difference anchor |
|---|---:|---:|---:|
| `P` phonetic spelling | 7/24 = 29% | 2/17 = 12% | +17 percentage points |
| `N` nonword reading | 10/21 = 48% | 2/15 = 13% | +34 percentage points |

This makes the current 10 percentage point threshold plausible as a minimal
off-floor change, but it is a low bar compared with the raw period-1 contrast.
Sensitivity checks at 15 and 20 percentage points would make the conclusion less
dependent on a placeholder.

## What prior research contributes

### Original trial: Burgoyne et al. (2012)

Burgoyne et al. (2012) is the strongest source because it is the original
randomised controlled trial of this intervention. It reports a teaching-assistant
delivered reading and language programme, daily 40-minute sessions, and a
wait-list design with a 20-week randomised phase.

The key pattern is directly relevant to the threshold decision:

- after 20 weeks, children receiving intervention made greater progress in
  single-word reading, letter-sound knowledge, phoneme blending, and taught
  expressive vocabulary;
- effects did not transfer clearly to nonword reading, spelling, standardised
  receptive vocabulary, standardised expressive vocabulary, expressive
  information, or grammar;
- the authors describe single-word reading gains as modest but important in the
  context of Down syndrome and general learning difficulties;
- the paper reports roughly 4.5 words gained per 20 weeks of intervention, versus
  roughly 2 words under typical instruction.

This supports a threshold system that is more permissive for proximal,
directly-taught skills than for distal transfer outcomes. It also supports
keeping word-reading thresholds in raw-item units: even small numbers of
additional readable words can be educationally meaningful for children who begin
with very limited reading.

At the same time, Burgoyne et al. does not settle the exact threshold. It does not
say that one word, two words, or any other raw-score difference is a formal
minimum important difference. The current `W = 1` threshold is therefore a
reasonable domain judgement, not a citation-derived fact.

### Earlier and later Down syndrome reading interventions

The surrounding reading-intervention literature broadly agrees with the
Burgoyne pattern.

Goetz et al. (2008) evaluated training in reading and phoneme awareness. The
Burgoyne discussion notes that this earlier study found similar word-reading
gains, about two words per eight weeks, in a more selected group with emergent
reading skills.

Cologon et al. (2011) studied targeted reading instruction in a small
within-child design. Their abstract reports evidence that phonic reading
instruction generally improved reading skills and phonological awareness.

Lemons and Fuchs (2010) modelled response to reading intervention in children
with Down syndrome. Their study is not an ITT analogue, but it is useful for
expectations: a majority of children showed growth on letter sounds, taught sight
words, and decodable words, while response varied by baseline skill.

Lemons et al. (2012) compared decoding and phonological-awareness interventions
using single-subject designs. The abstract reports improvements in taught
phonetically regular and high-frequency words from the decoding intervention, but
no reliable gains from the phonological-awareness intervention and no
generalisation to oral reading fluency.

Baylis and Snowling (2012) evaluated a 10-week phonological reading programme.
Their abstract reports significant improvement in word reading and alphabet
knowledge, with some children developing decoding strategies, but also
considerable variability in response.

Taken together, these studies do not provide exact deltas, but they do support:

- `L = 2` as a meaningful but not excessive bar for letter-sound knowledge;
- `W = 1` as defensible only if the team wants a domain-paced threshold, with
  `W = 2` needed as a more conservative sensitivity anchor;
- `B = 1` as the minimum credible raw-score movement on a 10-item scale;
- caution for nonword reading and phonetic spelling, because transfer to untaught
  decoding and spelling is often weaker than gains on directly taught words.

### Vocabulary intervention evidence

The taught vocabulary outcomes should be treated separately from standardised
vocabulary. Burgoyne et al. found evidence for directly taught expressive
vocabulary, but not for broader standardised vocabulary.

Naess et al. (2022) provide newer RCT evidence that trained vocabulary in Down
syndrome can be moved by a structured school-delivered programme. The study
reported effects on expressive and receptive vocabulary breadth after a digital
Down Syndrome LanguagePlus intervention. This supports retaining a low but real
raw-score threshold for `TE` and `TR`, because trained vocabulary is a proximal
target.

By contrast, Donolato et al. (2023) reviewed oral-language interventions for
children with neurodevelopmental disorders. The mean post-test effect was modest,
with evidence of publication bias and overestimation. Receptive vocabulary and
omnibus receptive measures had smaller effects than several other language
domains. This supports caution for `R` and `E`: broad standardised vocabulary
should not inherit the more permissive interpretation used for directly taught
items.

Neil and Jones (2016), a systematic review and meta-analysis of communication
interventions for individuals with Down syndrome, similarly concluded that the
evidence base was promising but methodologically limited. That reinforces the
need to report full uncertainty and avoid treating a small point estimate as a
clear educational change.

## Outcome-by-outcome implications

| Outcome | Current threshold | Review implication |
|---|---:|---|
| `L` letter-sound knowledge | 2 items | Retain. It is close to 0.2 SD, close to 5% of the scale, and below the original raw gain gap. It is also a proximal intervention target. |
| `W` word reading | 1 item | Retain only with an explicit `delta = 2` sensitivity read. One item is defensible as half of natural growth, but it is lenient compared with 0.2 SD. |
| `B` phoneme blending | 1 item | Retain. The one-item floor is doing the work, but that is appropriate on a 10-item test where sub-item thresholds cannot be observed. |
| `TE` taught expressive vocabulary | 1 item | Retain. Directly taught vocabulary is a proximal target and the original raw gain gap is larger than one item. |
| `TR` taught receptive vocabulary | 1 item | Retain. The original RCT found weaker receptive than expressive taught-vocabulary evidence, but the raw anchors still support one item as a meaningful trained-word threshold. |
| `UR` not-taught receptive vocabulary | 1 item | Retain provisionally. The denominator must be confirmed. Interpret as a transfer threshold, not as a directly taught-word threshold. |
| `UE` not-taught expressive vocabulary | 1 item | Retain provisionally, but expect sensitivity to be poor because the raw gain gap is small. Denominator confirmation matters. |
| `R` standardised receptive vocabulary | 2 items | Retain. Broad transfer is not strongly supported by the trial or meta-analytic evidence; a threshold below 0.2 SD would be too permissive. |
| `E` standardised expressive vocabulary | 2 items | Retain. Same reasoning as `R`; the observed period-1 raw gap is essentially zero. |
| `P` phonetic spelling off-floor risk difference | 10 percentage points | Keep as provisional and add 15/20 percentage point sensitivity. Burgoyne found little transfer to spelling, and the current models show a floor-limited null. |
| `N` nonword reading off-floor risk difference | 10 percentage points | Keep as provisional and add 15/20 percentage point sensitivity. The raw off-floor contrast is larger than for `P`, but the outcome is coarse and heavily floored. |

## Recommended reporting stance

For the ITT reports, I would separate the decisions like this:

1. Keep `pd = P(effect > 0)` as the direction read only.
2. Keep `P(benefit >= delta)` as the meaningful-benefit read.
3. Lead with posterior medians and credible intervals, not a single evidence
   label.
4. Treat the current deltas as reviewable educational assumptions, not as
   statistical facts.
5. For `W`, show both `delta = 1` and `delta = 2`.
6. For `P` and `N`, show 10, 15, and 20 percentage point thresholds.
7. Mark `UR` and `UE` as denominator-provisional until the assessment materials
   confirm the 12-item denominator.

This would let the team keep the current primary convention while showing readers
where the interpretation changes under stricter, still-plausible thresholds.

## Questions for team review

1. Should "meaningful" be defined relative to each domain's natural pace of
   progress, or should word reading use a stricter distribution anchor?
2. Is a one-word reading benefit over 20 weeks educationally meaningful in this
   population, or should `W = 2` become the primary threshold?
3. For floored outcomes, is a 10 percentage point increase in coming off the
   floor meaningful enough, or should the primary threshold be 15 or 20
   percentage points?
4. Are `UR` and `UE` definitely 12-item tests?
5. Should the report include a small sensitivity table for all outcomes, or only
   for `W`, `P`, and `N`?

## References

- Baylis, P., & Snowling, M. J. (2012). Evaluation of a phonological reading
  programme for children with Down syndrome. *Child Language Teaching and
  Therapy*, 28(1), 39-56. doi:10.1177/0265659011414277
- Burgoyne, K., Duff, F. J., Clarke, P. J., Buckley, S., Snowling, M. J., &
  Hulme, C. (2012). Efficacy of a reading and language intervention for children
  with Down syndrome: A randomized controlled trial. *Journal of Child Psychology
  and Psychiatry*, 53(10), 1044-1053. doi:10.1111/j.1469-7610.2012.02557.x
- Cologon, K., Cupples, L., & Wyver, S. (2011). Effects of targeted reading
  instruction on phonological awareness and phonic decoding in children with Down
  syndrome. *American Journal on Intellectual and Developmental Disabilities*,
  116(2), 111-129. doi:10.1352/1944-7558-116.2.111
- Donolato, E., Toffalini, E., Rogde, K., Nordahl-Hansen, A., Lervag, A.,
  Norbury, C., & Melby-Lervag, M. (2023). Oral language interventions can improve
  language outcomes in children with neurodevelopmental disorders: A systematic
  review and meta-analysis. *Campbell Systematic Reviews*, 19(4), e1368.
  doi:10.1002/cl2.1368
- Gesel, S. A., LeJeune, L. M., & Lemons, C. J. (2019). Teaching phonological
  awareness to preschoolers with Down syndrome: Boosting reading readiness.
  *Young Exceptional Children*, 24(1), 39-51. doi:10.1177/1096250619865953
- Goetz, K., Hulme, C., Brigstocke, S., Carroll, J. M., Nasir, L., & Snowling,
  M. (2008). Training reading and phoneme awareness skills in children with Down
  syndrome. *Reading and Writing*, 21(4), 395-412.
  doi:10.1007/s11145-007-9089-3
- Hulme, C., Goetz, K., Brigstocke, S., Nash, H. M., Lervag, A., & Snowling,
  M. J. (2012). The growth of reading skills in children with Down syndrome.
  *Developmental Science*, 15(3), 320-329.
  doi:10.1111/j.1467-7687.2011.01129.x
- Lemons, C. J., & Fuchs, D. (2010). Modeling response to reading intervention
  in children with Down syndrome: An examination of predictors of differential
  growth. *Reading Research Quarterly*, 45(2), 134-168.
  doi:10.1598/rrq.45.2.1
- Lemons, C. J., & Fuchs, D. (2010). Phonological awareness of children with Down
  syndrome: Its role in learning to read and the effectiveness of related
  interventions. *Research in Developmental Disabilities*, 31(2), 316-330.
  doi:10.1016/j.ridd.2009.11.002
- Lemons, C. J., Mrachko, A. A., Kostewicz, D. E., & Paterra, M. F. (2012).
  Effectiveness of decoding and phonological awareness interventions for children
  with Down syndrome. *Exceptional Children*, 79(1), 67-90.
  doi:10.1177/001440291207900104
- Naess, K.-A. B., Hokstad, S., Engevik, L. I., Lervag, A., & Smith, E. (2022).
  A randomized trial of the digital Down Syndrome LanguagePlus (DSL+) vocabulary
  intervention program. *Remedial and Special Education*, 43(5), 314-327.
  doi:10.1177/07419325211058400
- Neil, N., & Jones, E. A. (2016). Communication intervention for individuals
  with Down syndrome: Systematic review and meta-analysis. *Developmental
  Neurorehabilitation*, 21(1), 1-12. doi:10.1080/17518423.2016.1212947
- Stuebing, K. K., Barth, A. E., Cirino, P. T., Francis, D. J., & Fletcher,
  J. M. (2008). A response to recent reanalyses of the National Reading Panel
  report: Effects of systematic phonics instruction are practically significant.
  *Journal of Educational Psychology*, 100(1), 123-134.
  doi:10.1037/0022-0663.100.1.123
