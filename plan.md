# Privacy-Fairness Tradeoffs: Controlled Rerun Plan

**Updated November 17, 2026. Progress report: November 18. Final deadline: December 7.**

The controlled-rerun request takes priority over the earlier intervention-first schedule. The previous plan is preserved at [V0/plan.md](V0/plan.md), and the exact request is at [V0/CONTROLLED_RERUN_REQUEST.txt](V0/CONTROLLED_RERUN_REQUEST.txt).

## What changed, and why

| Item | V0 pilots | Controlled v2 |
| --- | --- | --- |
| Results location | Moved to `V0/results/` | New runs only in `results/controlled_v2/` |
| Existing figures/reports/notebooks | Preserved under `V0/` | No edits to the old report or publication figures |
| Feature policy | SEX/RAC1P excluded, 14 raw inputs, 77 encoded columns | Primary `include_sensitive`; secondary `exclude_sensitive` |
| Initialization | One seed: 285 | Five matched seeds: 42, 123, 285, 2026, 2027 |
| Main comparisons | SGD and DP1/4/8, one run each | Same four methods for both feature policies: 40 main runs |
| Diagnostics | No sampling/clipping controls | Two non-private Poisson controls per policy/seed: 20 control runs |
| Evaluation | Pilot accuracy/fairness | Add ROC-AUC/BCE, component rates, paired differences, mean/sample SD, ESR diagnostics |
| Current priority | Interventions and another task next | Validate baseline/DP findings first; interventions and additional tasks wait |

**Unchanged:** California 2018 cohort; ages 17-89; valid PWGTP; Y=(ESR==1); no ESP/ESR predictors; original fixed person-level split assignments; preprocessing fitted on reference only; final test not evaluated. Audit household overlap without changing splits.

## Phase 1: preserve and audit (November 17)

- [x] Create `V0/` and move existing results, figures, reports, and executed notebooks into it; verify moved files by SHA-256.
- [x] Save original configs, code, proposal, plan, logs, data manifest, and `V0/SETTINGS.txt`.
- [x] Review actual filters, feature names, labels, missingness, split alignment, initialization, gradient normalization, and privacy accounting; document findings in `results/controlled_v2/audit/`.
- [x] Commit/push the preservation checkpoint before changing experiment behavior (`c790371`).

## Phase 2: controlled implementation and validation (November 17-18)

- [x] Prepare two feature policies on the existing records/splits. Keep original sensitive labels for auditing (87 primary/77 secondary encoded columns).
- [x] Use the shared MLP input->64->32->1, ReLU, SGD lr0.05/momentum0, 20 epochs, batch512, threshold0.5, fixed final epoch.
- [x] Implement ordinary shuffled SGD, DP-SGD epsilon1/4/8, Poisson SGD without clipping/noise, and Poisson per-example clipped C=1 SGD without noise.
- [x] Match DP/control sampling, number of optimizer updates, expected-batch gradient normalization, and clipping implementation. Label both controls non-private.
- [x] Add deterministic tests comparing manual clipping-only updates with the zero-noise DP optimizer; check leakage, fixed splits, rates/undefined cases, accounting, and paired aggregation.
- [x] Make runs resumable with unique policy/method/seed directories; preserve completed artifacts and never export private sampling/noise RNG state or raw training gradients.
- [x] Run synthetic and public-reference smoke checks (12 smoke configurations passed), then commit/push implementation (`c94624f`; paired-aggregation tests added afterward).

## Phase 3: repeated main experiments (start after validation)

Initialization seeds: **[42, 123, 285, 2026, 2027]**.

- [x] Run 2 feature policies x 5 seeds x (ordinary SGD + DP1 + DP4 + DP8) = **40 main runs**.
- [x] Match model initialization for comparisons within each feature policy/seed. DP sampling/noise use fresh private randomness.
- [x] Log exact achieved epsilon/delta, noise/C, sample rate, updates, normalization, runtime, environment, feature/split fingerprints, and completion state.
- [x] Save development overall/group statistics and local ignored predictions/checkpoints, with fixed final-epoch evaluation.
- [x] Commit/push completed aggregate run artifacts at useful checkpoints (`73b90ff` contains all 40 main runs). Keep test outcomes reserved.

## Phase 4: sampling and clipping controls

- [x] Run 2 feature policies x 5 seeds x 2 controls = **20 additional non-private runs**.
- [x] Compare ordinary SGD versus Poisson-unclipped SGD, Poisson-unclipped versus clipped-noise-free SGD, and clipped-noise-free versus DP-SGD.
- [x] Distinguish sampling/normalization, clipping, and added noise; do not call the diagnostic controls DP.

## Phase 5: aggregate and diagnose

- [x] Produce seed-level and aggregate CSV/JSON tables with mean and sample SD (`ddof=1`). Calculate every fairness gap per run before averaging.
- [x] Report all sex/race/intersection groups, undefined values, denominator support, and a separately labeled race headline comparison restricted to groups with n_development>=500.
- [x] Pair every DP run with its same-policy, same-initialization SGD baseline; report accuracy/group losses and fairness-component differences by seed.
- [x] Compare epsilon1/4/8 differences with training variability and compare sensitive-feature inclusion/exclusion.
- [x] Preserve original ESR1-6; report label-code distributions by sex/race, false-positive composition by ESR, and sex FPR within ESR categories.
- [x] Audit whether female FPR is consistently higher across seeds. Keep the primary ESR==1 target unchanged pending review.
- [x] Write a findings summary, exact reproduction/resume commands, completion counts, and limitations. No fabricated missing runs.
- [x] Prepare a reusable plotting pipeline/specification and recommend at most three main figures; do not render publication figures before review.
- [x] Commit/push the complete findings checkpoint (final controlled-rerun commit).

**Completed evidence:** [findings](results/controlled_v2/FINDINGS.md), [aggregate tables](results/controlled_v2/aggregate/README.md), and [artifact validation](results/controlled_v2/audit/validation.json). All 60 full runs completed; 12 unit tests and 12 public-reference smoke configurations passed. All 30 private accountant bounds and 12 representative checkpoint predictions were verified; the original seed285 exclusion baseline reproduced exactly. No test outcomes, interventions, target changes, final figures, or report edits were introduced.

## Review gate and remaining schedule

Review the controlled results first. Expand important configurations to ten seeds for final analysis if the five-seed variability makes budget effects inconclusive; record the additional seeds before running them.

Review priorities from the completed rerun: (1) the clipping-only controls reproduce most accuracy loss; (2) the sex group with larger accuracy loss changes with feature inclusion/exclusion; (3) race and sex separation can move in opposite directions; (4) epsilon8 is slightly less accurate than epsilon1 in all five inclusion-policy pairs, so retain that observation and investigate before final generalization; (5) ESR2/3/6 diagnostics and the small military cells require careful target interpretation. Keep the target/splits unchanged until a separate comparison is agreed.

After review, schedule EO/LFR and ACSIncome work against the remaining time. They remain project objectives, but are not part of this baseline rerun. Preserve December 1-3 for final test evaluation only after choices are frozen, and December 4-7 for the final report. The existing progress report remains unchanged.

If execution is blocked, finish available implementation/checks and record the precise failed stage, completed runs, and command to resume. Never mark an unexecuted run complete.
