# Controlled v2 implementation audit

The cohort, ESR==1 label, age/weight filters, missing-value rule, original person-level permutation/splits, 20 epochs, SGD lr0.05/momentum0, nominal batch512, and threshold0.5 are unchanged. Final test outcomes are not loaded/evaluated. Existing reports/pilots were moved intact to V0 before implementation.

## What was verified or changed

- **Feature policy:** primary includes SEX/RAC1P as categorical predictors; secondary reproduces the old exclusion policy. The exclusion-policy training/development matrices were checked element-for-element against the original sparse matrices.
- **Encoding:** reference-fitted age standardization plus one-hot categoricals, NOT_APPLICABLE missingness, unknown categories ignored. Primary has 16 raw inputs/87 encoded columns; secondary has 14/77. The reference partition contains no RAC1P=4 category, so its race block is all zero at prediction time; no records or audit labels were removed.
- **Leakage checks:** ESR, ESP, Y, identifiers, weights, and split labels never enter predictor matrices. Original protected labels remain aligned with saved row IDs. ESR is reloaded only for reference/train/development row IDs, verified against Y, and used for diagnostics, never inputs.
- **Updates:** every full run makes 296 updates per epoch, 5,920 total. DP/control Poisson inclusion probability is 1/296. Opacus normalizes the clipped gradient sum by integer expected batch size511, while ordinary SGD averages by actual batch size (512 except the final smaller batch).
- **Controls:** Poisson-unclipped SGD uses loss sum/511; Poisson-clipped noise-free SGD uses the same Opacus per-example clipping/normalization implementation as DP. They are non-private. Their deterministic sampling streams match within each seed; DP sampling/noise use fresh private randomness.
- **Clipping validation:** a deterministic unit test independently sums clipped per-example gradients, divides by expected batch size8 despite an actual batch of3, and verifies identical parameters after the zero-noise Opacus update. Separate tests check unclipped normalization, undefined rates, leakage, epsilon/noise direction, ESR denominators, and paired aggregation/sample SD.
- **Initialization:** the shared model factory and initialization seed are identical within policy/seed; initialization hashes are checked before computing method-versus-SGD group losses. Different feature policies have different input dimensions, so their weights cannot be exactly identical.
- **Privacy:** full private runs retain C=1, RDP, delta=1/(10*151320), and research RNG settings. No raw training losses/gradients or private RNG state are exported. Raw-data/label checksums are local-only; published fingerprints cover fixed partition IDs, public manifest, and source code.
- **Smoke checks:** all 12 policy/method combinations passed on disjoint public-reference subsets. No additional private-training or test records were used for smoke training/evaluation.

## Household overlap

`household_overlap.csv` reports shared household counts for every partition pair, using SERIALNO/SPORDER only. Overlap exists because the original split was person-level. No split was changed: altering it now would confound the feature-policy/training comparison. These are individual-record privacy claims, not household-level guarantees, and household overlap limits generalization interpretation.

## Outputs

- `feature_policies.json`: exact source and encoded feature lists, split/config identities.
- `audit.json`: checks, changes, and privacy limitations.
- `development_esr_distributions.csv`: ESR1-6 counts separately by sex and all nine race groups, before any new predictions.
- `household_overlap.csv`: overlap counts and split-specific household fractions, without household identifiers.

No fairness intervention, alternative target, final-test evaluation, or publication figure has been introduced.
