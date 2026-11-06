# FireProtDB censoring: core analysis code

Core analysis code for **Dynamic-range censoring and reference choice reshape megascale protein stability prediction**.

This compact release contains data preparation, descriptors and sequence grouping, censored-normal and accelerated-failure-time learners, censored stacking, imposed-window experiments, unmasked ESM-2 features and their predictor comparison, double-substitution prediction, record-weighted global-epistasis fitting, external-reference mapping and within-domain reference comparisons. Raw database files, derived measurement tables, predictions, model weights and manuscript files are not redistributed here.

## Data and scope

Obtain `fireprotdb_dump_2025_09_22.zip` from the [official FireProtDB download page](https://loschmidt.chemi.muni.cz/fireprotdb/download/) and observe the [database terms](https://loschmidt.chemi.muni.cz/fireprotdb/terms-of-use/). The source publication is [FireProtDB 2.0](https://doi.org/10.1093/nar/gkaf1211). The analysed archive has SHA-256 `17d392f221016b34fea36c96d13184f79c173f5d0831b6a0a6e7639529d41ba0` and contains `01_fireprotdb_2025-09-20.sql`.

This is a release of the important computational components, not an executable manuscript-production bundle. The manuscript-specific plot exporters and document construction are outside this compact release. The article's Supplementary Information gives the comparison and matching rules. The ESM checkpoint is obtained separately through Hugging Face and retains its own terms.

`config/reference_sequence_candidates.csv` is the frozen candidate identifier configuration used by the cross-assay mapper. It contains database sequence IDs, dataset IDs and overlap statistics, not amino-acid sequences or assay measurements. The original candidate-discovery generator was not retained; the mapper reproduces offset alignment and identity filtering from this supplied candidate list. Its original-source SHA-256 is recorded in the local release manifest.

## Environment

Use a Python environment with the packages in `requirements.txt`:

```bash
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -r requirements.txt
```

The original training environment was not recorded in full. These dependencies identify required libraries; they are not a historical lockfile or a claim of bitwise reproduction. No full retraining was performed when preparing this code release. Some scripts start an analysis when executed or imported; run the numbered entry points intentionally rather than importing them to discover dependencies.

## Main workflow

Run commands from the repository root. Extraction reads PostgreSQL COPY blocks and writes Parquet without executing SQL. It needs substantial disk space beyond the compressed archive.

```bash
python scripts/00_extract_sql_tables.py --zip fireprotdb_dump_2025_09_22.zip --out data_raw
python scripts/01_profile_database.py data_raw results/01_profile
python scripts/02_build_megascale_table.py data_raw data_processed
python scripts/03_build_model_matrix.py
python scripts/06_cross_assay_map.py data_raw
python scripts/05_censored_vs_standard.py
python scripts/07_semisynthetic_censoring.py
python scripts/09_censored_stacking.py
python scripts/16_doubles_censored.py
python scripts/15_global_epistasis.py
```

Optional components:

```bash
python scripts/13_plm_features.py
python scripts/14_plm_recheck.py
python scripts/18_within_domain_validation.py
python scripts/20_dissociation_test.py
python scripts/21_matched_reference.py
python scripts/04_benchmark_21models.py 25000
```

`13_plm_features.py` downloads `facebook/esm2_t30_150M_UR50D` and extracts hidden states and logits from **unmasked wild-type** sequences. It does not calculate masked-marginal scores. `04_benchmark_21models.py` is a budget-limited context benchmark; expensive methods use smaller training samples, as specified in the code. It does not provide a train–test gap audit.

The reference scripts require the mapped measurements from step 06 and the PLM outputs. Step 18 computes variant-count-weighted within-domain correlations. Step 20 pairs eligible domains while retaining each reference's substitution set. Step 21 computes the identical-substitution sensitivity with equal domain weights. Their eligibility sets and estimands must be kept distinct. The principal predictors share eight eligible domains and 300 identical observed substitutions; the channel score covers 299 of those substitutions.

The scripts support `FP_NJ` for selected learners' thread counts. The default analysis settings, seeds and iteration budgets are in the source. Substantial CPU time and memory are required for full runs.

## Measurement and analysis conventions

- MegaScale ΔΔG is mutant minus wild-type unfolding free energy; negative values indicate destabilisation.
- The assay window is on mutant ΔG. It becomes a variant-specific ΔΔG upper bound of `-1 - dG_wt` for left censoring and a lower bound of `5 - dG_wt` for right censoring.
- Singles prioritise and average observed replicate measurements; the modelling population contains 431,989 unique substitutions. The 111 descriptors include measured wild-type free energy and ConSurf. Missing single-site ConSurf grades are filled with 6.
- Double-substitution prediction retains the first eligible record per substitution pair. It uses 218 descriptors and the actual pooled missing-ConSurf fill value of 7, computed before outer splitting. This release preserves that preprocessing; it does not silently replace it with fold-specific preprocessing.
- The global-epistasis analysis retains 156,315 records, representing 151,894 unique double-substitution identities. Each strategy gives equal weight to its included records, and domain bootstrapping carries repeated records together. The observed-only fit omits censored records; the other two strategies include them. A unique-variant weighting sensitivity was not run.
- The 5-mer grouping rule reduces sequence similarity across folds; it is not a guarantee of remote-homology exclusion. An upstream protease-channel reference belongs to the same assay pipeline, whereas thermodynamic and abundance reference endpoints answer different validation questions.
- The imposed windows hide 9.5%, 21.7% and 39.4% of the fully observed parent population. Those are not exact per-training-fold proportions. Three outer holdouts from the five-fold partition were evaluated.

Earlier channel-label training explorations with target-information exposure or interval-loss defects are excluded from this release and from the paper's evidence tables.

## Lightweight verification

```bash
python -m unittest discover -s tests -v
```

The tests compile every released script, compare analytic censored-likelihood derivatives with finite differences, check point and tail contributions against independent distribution calculations, check interval-ordering behaviour and check the softplus link. They do not train a model, download data or rerun the reported experiments.

`source_manifest.json` records source hashes and changes relative to the local analysis source: terminology clarifications, the candidate-configuration path, removal of the invalid legacy channel-training branch from the public reference evaluator, and extraction of the matched-reference calculation from a longer audit script. The retained predictors' scientific calculations and preprocessing were preserved. The public reference evaluator uses an explicit whitelist of valid Drop/Interval representation predictions.
