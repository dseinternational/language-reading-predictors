> [!NOTE]
> Provenance and external-archive handling drafted by a LLM-based AI tool (Codex/GPT-5).

# Data

The checked-in longitudinal RLI files contain the 54 children analysed in the original report after three post-randomisation losses to follow-up. They do not contain the lost children's records. The open-access UK Data Service collection [Reading and language intervention for children with Down syndrome: Experimental data](https://doi.org/10.5255/UKDA-SN-852291) contains screening records for all 57 randomised children and is the source for the word-reading missing-data sensitivity.

The ReShare item leaves its item-level licence field blank, while ReShare's legal terms describe two possible ShareAlike licences for open deposits. The upstream file is therefore not committed or silently relicensed under this repository's CC BY 4.0 data licence. Run `scripts/import_rli_randomised_archive.py --zip /path/to/DSE_Data.zip` or its explicit `--download` mode to install the checksum-pinned CSV under the gitignored `data/generated/` directory. Pass that path to `scripts/fit_statistical_model.py lrp-rli-itt-010 --rli-randomised-archive ...`. The importer and loader verify the upstream checksums; the published flow of 57 randomised children (29 intervention and 28 waiting control) to 54 analysed children (28 and 26); the archive-derived 53 observed t2 word-reading outcomes (28 and 25); and a one-to-one 71-field reconciliation of the 54 included source rows with `rli_data_wide.csv`. The gitignored local raw CSV retains the upstream source identifiers. The loader does not return them, model and emitted data omit them, and no source-to-repository crosswalk is persisted.

## License

The data committed to this directory are licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details. That statement does not relicense locally generated copies of the external UK Data Service archive.
