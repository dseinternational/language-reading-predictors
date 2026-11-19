> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-6).

# Level-factor prior-sensitivity runner and saved-output audit, 18 November 2026

The level-factor runner passed the selected prior scale twice when the registered model already supplied that argument. Both nuisance sweeps therefore failed before sampling a graded outcome. An explicit dispersion sweep of an off-floor outcome could run despite its Bernoulli model having no dispersion parameter. Both nuisance options also selected the arm-gap CSV as their destination.

The runner now overrides one prior in the registered model. The dispersion sweep (`--axis kappa`) varies the half-normal scale on `1/sqrt(kappa)` over 0.125, 0.25 and 0.5 for the registered graded models. Its default outcomes are W, L and B. The child-intercept sweep (`--axis sigma_child`) varies the half-normal scale on the between-child standard deviation over 0.5, 1.0 and 1.5 and also supports the off-floor outcomes P and N. Both store the scale in the CSV and in each trace. Their respective manifests are `level_dispersion_prior_sensitivity.csv` and `level_child_sd_prior_sensitivity.csv`. Only the treatment-prior sweep supplies evidence for attachment to a primary fit. [Prior calibration and grid rationale](202608231930-level-factors-nuisance-prior-calibration.md)

## Saved reporting outputs

The local output tree contained two level-factor sweep archives, under `output/statistical_models/` and `output/runs/20261001T175406Z-399586f4988d/statistical_models/`. Each contained a combined treatment-prior CSV and five copies attached to individual fits. The 30 distinct cell traces appeared twice each. All 60 files matched their recorded SHA-256 digests. The axis, outcome, treatment-prior scale and recorded primary hashes agreed between each CSV row and its trace. The attached rows matched the combined rows apart from their relative trace paths.

The 18 November audit found a mislabelled dispersion scale in 18 of the 30 cells. In each archive, the three treatment-prior settings for each of W, L and B recorded `kappa_sigma=50.0`. The corresponding fits' prior tables give `inv_sqrt_kappa ~ HalfNormal(0.25)`, and the sweep traces contain that free parameter. After sampling, the runner wrote the hard-coded value 50 from the earlier prior on concentration. The level-factor release check uses the treatment-prior grid and trace evidence independently of this field.

Every archived cell inspected was a treatment-prior sweep. The local output tree contained no level-factor arm-gap, dispersion or child-intercept sweep files, and the configured public project output prefix returned no entries. The 20 August run note records a 33-cell arm-gap sweep that predates the introduction of the nuisance options. Its underlying CSV was unavailable for inspection, as were any separate private archives. [August arm-gap sweep record](202608201800-level-factors-arm-gap-prior-sweep.md)

### Archive correction, 19 November 2026

Both archives now record `kappa_sigma=0.25` for all 18 affected cells. In each archive, the correction covered nine rows in the combined CSV and the matching nine rows in the three CSVs attached to the W, L and B fits. A byte-for-byte comparison of all 12 CSVs with their pre-correction copies verified that only the dispersion field changed in those 36 row copies across eight files. The P and N rows kept their blank dispersion fields.

Before and after the correction, all 60 sweep trace files (30 distinct traces, each stored twice) and all 10 referenced primary traces matched their recorded SHA-256 hashes. No models were refitted for the archive correction.

## Runner verification

On Windows with `NUMBA_CPU_NAME=sandybridge`, a fresh W development fit and both three-cell sweeps completed with the real sampler in a scratch directory. Each cell saved two chains of 500 draws. All six trace digests and recorded scales matched the two manifests. The short runs failed the convergence gate and supplied no publication evidence.
