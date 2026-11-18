# Project plan

Updated November 18, 2026. Progress report: November 18. Final deadline: **December 7, 2026**.

## Completed

- [x] California 2018 filtered ACS Employment cohort, missingness audit, ESP removal, fixed five-way splits and reference-only encoding.
- [x] Ordinary SGD/DP-SGD MLP; primary sex/race inclusion and secondary exclusion.
- [x] Five matched seeds at epsilon1/4/8: **40 main runs**.
- [x] Matched non-private Poisson unclipped/clipped controls: **20 runs**.
- [x] Per-group accuracy losses, selection/TPR/FPR/PPV/NPV, four fairness gaps, paired mean/sample SD, ESR and household audits.
- [x] Public-reference smoke checks, numerical unit tests, privacy accounting and representative checkpoint validation.
- [x] Figure/table assets and progress report; imported and committed the team's Overleaf edits.
- [x] Consolidated the record into project_log.md and findings.md; removed V0/duplicate notes while retaining history and all current data/results.

**Current interpretation:** the large performance change is already present in the clipping-only control, whereas epsilon1/4/8 differences at C=1 are small. The next priority is **varying C at fixed epsilon**, followed by cross-task replication. This is a change in experimental emphasis, not proof that noise/epsilon never matter. Group outcomes and all four fairness definitions remain central.

## November 19-24: clipping sensitivity and cross-task robustness

### Priority 1: clipping norm, fixed privacy budget

- [ ] Implement a separate study/output configuration for **C in {0.5,1,2,5}, fixed epsilon1**. Keep the completed controlled_v2 grid/results frozen; do not edit it to reuse different C values under the same paths.
- [ ] Start with the primary include_sensitive policy and the same five initialization seeds, cohort, splits, architecture, learning rate, epochs, sampling and threshold. Reuse the completed C=1 runs as the reference.
- [ ] Calibrate/check the noise multiplier sigma with the RDP accountant for the fixed epsilon/delta/sampling/steps. Noise SD on the clipped sum must be **sigma*C**, then the existing expected-batch normalization. At fixed sampling/steps, the same calibrated sigma normally preserves epsilon while absolute noise changes with C.
- [ ] Add matched **clipped/no-noise Poisson controls** at the new C values, retaining the existing unclipped control. Pair their sampling streams where appropriate; keep private sampling/noise randomness fresh and unpublished. Account for all additional private releases.
- [ ] Compare overall and per-group accuracy losses, selection rate, TPR/FPR, and accuracy parity/independence/separation/binary sufficiency. Retain sample SD and denominator support.
- [ ] Inspect saved development score distributions by true label/group, ROC/AUC and threshold0.5 operating points. Distinguish score shifts from changes in discrimination; keep final test outcomes reserved.
- [ ] If feasible, measure clipping frequency by label/group on permitted public-reference data or via separately budgeted DP training summaries. Do not release raw private gradient norms or unaccounted counts.
- [ ] Prespecify selected ten-seed follow-ups after reviewing the new variability. Do not rerun every old setting or expand epsilon as the default next sweep.

### Priority 2: additional Folktables tasks

- [ ] Add **ACSIncome first**, responding to the instructor's request for additional tasks. Document its target, cohort, missingness and split rules separately in project_log.md.
- [ ] Run matched SGD, clipped/no-noise and DP baselines with five seeds, starting at fixed epsilon1/C=1 and selected C values if feasible. Compare group loss and positive-prediction shifts across tasks.
- [ ] Add a further task only if time permits. Do not mix a different task's labels/cohort into Employment results or imply that ACSIncome is already complete.

## November 24-30: fairness interventions

- [ ] Implement/evaluate LFR preprocessing and EO post-processing, with matched non-private controls.
- [ ] Fit learned interventions only on their designated reference/calibration partitions; document privacy assumptions and any extra privacy cost.
- [ ] Prioritize selected C/privacy settings from the robustness stage. Assess whether interventions reduce disparities and what group/overall accuracy they cost.

## December 1-4: final evaluation

- [ ] Freeze targets, configurations, interventions and reporting choices before evaluating selected models on the held-out test set.
- [ ] Finalize figures and analyze privacy/accuracy/fairness trade-offs, group support and limitations.

## December 5-7: final deliverables

- [ ] Final report, presentation, citations, reproducibility commands and final commit/push by December 7.

No clipping-norm sweep, extra task, intervention, ten-seed expansion or final-test evaluation is claimed complete. Settings/history are in [project_log.md](project_log.md); evidence and interpretations are in [findings.md](findings.md).
