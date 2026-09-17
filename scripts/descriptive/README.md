> [!NOTE]
> Clarity and currency edits by a LLM-based AI tool (Codex/GPT-6).

# Descriptive plot scripts

Standalone, archivable descriptive figures for the reading-and-language battery.
These are exploratory scratch figures for the responder / non-responder
discussion: per child, how much they gained over the study given their starting
point, plus a developmental-profile trajectory grid across the battery.

**Purely descriptive — no models, no causal language.**

## Standalone by design

Each script defines its own columns, labels and plotting recipe so that a figure can be reviewed and changed on its own. Keep that deliberate separation. The scripts use the shared path resolver to read `data/rli_data_long.csv` and write to the `descriptive/` directory under the selected output root.

The default root is `output/`. Use `--output-dir` or `DSE_LRP_OUTPUT_DIR` to redirect it. Generated figures are not committed.

## Canonical recipe (predictor vs gain, scripts 01–22)

One point per child:

1. Load `data/rli_data_long.csv`.
2. **x** = the predictor measure's level at **time 1** (the fixed baseline).
3. **y** = the outcome measure's **total gain to the last wave**, that is the
   outcome level at the child's last available wave minus the outcome level at
   time 1.
4. Drop children missing either x or y.
5. Plot: faint scatter (alpha about 0.3) plus a quantile-binned **median** (solid)
   and binned **mean** (dashed) across about five bins (skipping bins with fewer
   than four children), plus a dotted zero line.
6. Plain-English axis labels; the title names the measure pair. Saved at 140 dots
   per inch.

### Definition notes

- **"Last wave" = each child's last available wave** for that outcome — the
  largest timepoint with a non-missing outcome value, requiring a later wave than
  baseline. Every child contributes their full observed gain. (If you instead
  want a strict first-to-final-wave gain, restrict the last-wave selection to
  timepoint 4 — a one-line change.)
- The per-period gain columns in the data are **not** used; total gain is derived
  from the plain level columns.

The reference panel is script 06 (baseline letter sounds vs word reading gain),
about 53 children — use it as the visual target for the others.

## Trajectory grid (script 00)

`plot00_progress_over_time.py` draws a 2-by-4 small-multiples grid, one panel per
measure, of level against timepoint (waves 1–4): faint per-child lines plus a bold
mean-per-wave line. It covers the eight measures that have repeated waves. Block
design (a non-verbal ability subtest) is excluded because it is measured at baseline only
and so has no trajectory.

## Measure → column map

Use these columns exactly. Note the letter-sounds trap: **letter sounds is
`yarclet`**, which is a different measure from phonetic spelling (`spphon`).

| Measure                           | Level column |
| --------------------------------- | ------------ |
| Word reading                      | `ewrswr`     |
| Letter-sound knowledge            | `yarclet`    |
| Phonetic spelling                 | `spphon`     |
| Nonword reading                   | `nonword`    |
| Blending (phonological awareness) | `blending`   |
| Expressive vocabulary             | `eowpvt`     |
| Receptive vocabulary              | `rowpvt`     |
| Grammar (TROG)                    | `trog`       |

The trajectory grid (script 00) also uses receptive vocabulary and grammar above.

## Running

Each script runs standalone:

```bash
uv run python scripts/descriptive/plot06_lettersounds_vs_wordreading_gain.py
```

Or run them all (convenience only):

```bash
bash scripts/descriptive/run_all.sh
```

Images are written to `output/descriptive/plotNN_<name>.png`.
