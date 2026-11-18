> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Level-factor prior-sensitivity runner and saved-output audit, 18 November 2026

The level-factor runner passed the selected prior scale twice when the registered model already supplied that argument. Both nuisance sweeps therefore failed before sampling a graded outcome. An explicit dispersion sweep of an off-floor outcome could run despite its Bernoulli model having no dispersion parameter. Both nuisance options also selected the arm-gap CSV as their destination.

The runner now overrides one prior in the registered model. The dispersion sweep (`--axis kappa`) varies the half-normal scale on `1/sqrt(kappa)` over 0.125, 0.25 and 0.5 for the registered graded models. Its default outcomes are W, L and B. The child-intercept sweep (`--axis sigma_child`) varies the half-normal scale on the between-child standard deviation over 0.5, 1.0 and 1.5 and also supports the off-floor outcomes P and N. Both store the scale in the CSV and in each trace. Their respective manifests are `level_dispersion_prior_sensitivity.csv` and `level_child_sd_prior_sensitivity.csv`. Only the treatment-prior sweep supplies evidence for attachment to a primary fit. [Prior calibration and grid rationale](202608231930-level-factors-nuisance-prior-calibration.md)

## Saved reporting outputs

The local output tree contained two level-factor sweep archives, under `output/statistical_models/` and `output/runs/20261001T175406Z-399586f4988d/statistical_models/`. Each contained a combined treatment-prior CSV and five copies attached to individual fits. The 30 distinct cell traces appeared twice each. All 60 files matched their recorded SHA-256 digests. The axis, outcome, treatment-prior scale and recorded primary hashes agreed between each CSV row and its trace. The attached rows matched the combined rows apart from their relative trace paths.

**The dispersion scale is mislabelled in 18 of the 30 cells.** In each archive, the three treatment-prior settings for each of W, L and B record `kappa_sigma=50.0`. The corresponding fits' prior tables give `inv_sqrt_kappa ~ HalfNormal(0.25)`, and the sweep traces contain that free parameter. The value 50 was a hard-coded scale from the earlier prior on concentration. This affects 36 CSV row copies across the two archives. The P and N rows leave dispersion blank, as their likelihood requires.

The incorrect value was written after sampling. Correcting its description requires no refit, and the level-factor release check uses the treatment-prior grid and trace evidence independently of this field. The archived files retain their original contents. For these W, L and B cells, read the dispersion prior as half-normal with scale 0.25 on `1/sqrt(kappa)`.

Every archived cell inspected was a treatment-prior sweep. The local output tree contained no level-factor arm-gap, dispersion or child-intercept sweep files, and the configured public project output prefix returned no entries. The 20 August run note records a 33-cell arm-gap sweep that predates the introduction of the nuisance options. Its underlying CSV was unavailable for inspection, as were any separate private archives. [August arm-gap sweep record](202608201800-level-factors-arm-gap-prior-sweep.md)

## Runner verification

On Windows with `NUMBA_CPU_NAME=sandybridge`, a fresh W development fit and both three-cell sweeps completed with the real sampler in a scratch directory. Each cell saved two chains of 500 draws. All six trace digests and recorded scales matched the two manifests. The short runs failed the convergence gate and supplied no publication evidence.
