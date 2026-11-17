# Privacy–Fairness Tradeoffs: Execution Plan

**Start:** November 16, 2026

**Progress report:** November 18, 2026

**Final report:** December 7, 2026

Read [proposal.md](proposal.md) for the research questions, exact dataset/features/target, fairness definitions, and privacy assumptions. This file tracks work and deadlines; change research definitions in the proposal before updating implementation or experiment settings.

## Timeline

| Phase | Dates | Deliverable |
| --- | --- | --- |
| 1. Data and baseline | Nov 16 | Data audit, fixed splits, working non-private MLP |
| 2. DP pilots | Nov 17 | Preliminary results at several privacy budgets |
| 3. Progress report | Nov 18 | Submitted report with completed pilot evidence |
| 4. Fairness interventions | Nov 19–24 | EO and LFR with non-private controls |
| 5. Main experiments | Nov 25–30 | Frozen protocol and repeated experiment grid |
| 6. Analysis | Dec 1–3 | Final test results, figures, and findings |
| 7. Final report | Dec 4–7 | Submission package ready Dec 6; submit by Dec 7 |

## Today: November 16

**Goal: understand our records and evaluation rules, then train one ordinary model.**

1. [ ] Review the [research design](proposal.md): filtered employment task, exact label, 14 input columns, both protected attributes, and public-auxiliary assumption.
2. [ ] Create a Python environment and record dependency versions. Add folders for notebooks, source code, configurations, results, figures, and reports; ignore downloaded data and large checkpoints in Git.
3. [x] Create the first data-audit notebook. Download California 2018 ACS 1-Year person data and inspect raw records before missing-value conversion.
4. [x] Verify the filtered task: ages 17–89, valid person weights, 17 source features, and the ESR-based target. Record counts before/after exclusions and exact task/package version.
5. [x] Show raw target-code counts, missingness, label balance, sex/race counts, and race-by-sex counts. Check identifiers for accidental duplicates.
6. [ ] Choose encoding rules and verify group/label support. Save fixed reference/training/development/calibration/test assignments.
7. [x] Put reusable preprocessing and metric functions in Python modules. Check rates against small hand-computed confusion matrices, including undefined-rate cases.
8. [x] Train the non-private MLP and compare it with a majority-class predictor. Save development predictions, group metrics, configuration, and runtime.

**Done today:** a readable data notebook, a documented preparation pipeline, and baseline development results. If time runs short, finish the data audit and split rules first.

**Data preparation completed Nov 16:** 378,817 raw records; 302,640 eligible records; 151,320 training records; 77 encoded inputs after dropping ESP. All notebook code cells executed successfully. Review [the audit notebook](notebooks/01_data_audit.ipynb) before freezing the starting splits/encoding: ESP has been dropped by user decision, and some race-by-sex calibration cells have zero or very few examples. Baseline and metric implementation completed: 20-epoch ordinary SGD pilot, one seed, development evaluation only. Inspect notebook 02 before privacy pilots; the broader design choices remain reviewable.

## November 17: DP-SGD pilots

9. [x] Integrate Opacus and validate the model. Run a short smoke experiment before the pilots.
10. [x] Run epsilon 1, 4, and 8 with one seed and fixed training duration; record achieved budgets and all privacy-accounting settings.
11. [x] Evaluate with the same metric code on development data. Compare each group's accuracy loss against the ordinary model.
12. [x] Save logs, predictions, configurations, and runtimes. Estimate the cost of the main grid.

**Done:** reproducible preliminary private/non-private comparisons. Keep final test outcomes unavailable.

**Completed early on Nov 16:** all three 20-epoch pilots. See [DP training log](dp_training_log.md) and [notebook 03](notebooks/03_dp_sgd.ipynb). One run per budget; test outcomes remain reserved.

## November 18: progress report

13. [ ] Create a population/split summary and preliminary accuracy/fairness table.
14. [ ] Plot accuracy versus epsilon and group accuracy losses; include fairness-gap plots where results are complete.
15. [ ] Write the motivation, connection to Bagdasaryan, exact task and metrics, privacy scope, preliminary findings, and remaining work.
16. [ ] Label results as single-seed development results; identify incomplete experiments. Review citations and submit.

## November 19–24: fairness interventions

**Nov 19–21: equalized-odds post-processing**

17. [ ] Wrap the frozen MLP to expose positive-class scores and fit EO on the calibration set.
18. [ ] Run sex-targeted and race-targeted EO, including ordinary-model controls. Audit both attributes for every output.
19. [x] Record prediction-randomization seeds, calibration support, and variation in development results.

**Nov 22–24: Learning Fair Representations**

20. [ ] Fit LFR on reference data, first for sex and then for the prespecified supported binary race cohort.
21. [ ] Train downstream ordinary/private MLPs on transformed features with original labels. Fit matched controls on identical records.
22. [ ] Check convergence, utility, and runtime; use a small predefined development-only fairness-weight grid.

**Done:** both interventions produce comparable saved outputs, with non-private controls.

## November 25–30: main experiments

23. [ ] On Nov 25, freeze cohorts, splits, preprocessing, metric/group-support rules, tuning choices, epochs, thresholds, seeds, and intervention settings.
24. [ ] Write an experiment manifest. Use the six comparison families and privacy settings specified in [the proposal](proposal.md#5-interventions-and-experimental-controls).
25. [ ] Run at least three training seeds per setting. Reuse raw-model checkpoints for EO and reference-fitted transformations according to the frozen protocol.
26. [ ] Audit missing outputs and failed runs; rerun technical failures consistently. Finish and freeze checkpoints by Nov 30.

**Done:** every run has a configuration, cohort/split identifier, privacy log, predictions, metrics, runtime, and status.

## December 1–3: analysis

27. [ ] Evaluate frozen models on test data. Report seed means/standard deviations and selected paired uncertainty intervals where feasible.
28. [ ] Compare group accuracy losses, both EO components, independence, and binary sufficiency. Check whether smaller gaps accompany improved group performance or prediction collapse.
29. [x] Produce accuracy-versus-epsilon, fairness-versus-epsilon, group-loss, and accuracy–fairness plots. Label targeted attributes, cohorts, non-private references, and variability.
30. [ ] Answer all research questions, including null/mixed findings. Discuss base rates, small groups, observational interpretation, and privacy/data-scope limitations.

## December 4–7: final report

31. [ ] Dec 4: complete the report draft using saved results and figures.
32. [ ] Dec 5: review claims, citations, cohort/metric definitions, privacy logs, and reproducibility instructions.
33. [ ] Dec 6: reproduce representative runs, regenerate figures, proofread, and prepare the submission package.
34. [ ] Dec 7: complete the final submission check and submit before the course deadline.

## Fallback priorities

| Problem | Response |
| --- | --- |
| Pilots incomplete by Nov 18 | Report completed evidence, implementation status, cause, and next experiment |
| Main grid is too expensive | Reduce finite epsilon values to 1, 4, and 8; retain three seeds and both intervention families |
| LFR is slow or unstable | Reduce prototype/tuning grids, cap optimization, and document failures |
| Small race/intersectional groups | Retain descriptive counts, flag uncertainty, and label any supported-cohort comparison |
| Schedule slips | Cut second datasets, combined LFR+EO, and extensive tuning first; preserve analysis and writing time |

A second dataset, additional split seeds, combined LFR+EO, ages-18–64 sensitivity analysis, and clipping-only diagnostics are optional extensions after the core experiment is complete.
