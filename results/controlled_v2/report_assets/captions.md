# Figure captions

## Figure1. Unequal racial impact at epsilon1 (main report)

Paired changes from non-private SGD to DP-SGD at epsilon1 for the five race groups with development n>=500, primary include-sensitive setting. Panels show (a) accuracy loss (SGD minus DP), (b) FPR increase (DP minus SGD), and (c) selection-rate increase (DP minus SGD), all in percentage points. Points and whiskers are means and sample SDs across five initialization-matched runs. Positive values mean lower accuracy or more false/positive predictions under DP. Group labels show fixed development sample sizes. Sample SD represents training variation, not a confidence interval. The smaller race groups are excluded only from this headline display and remain in the full tables. Panels use different horizontal scales, clearly labeled; compare groups within each panel.

## Figure2. Accuracy across sampling, clipping, and DP ablations (main report)

(a) Overall development accuracy for ordinary SGD, Poisson SGD without clipping/noise, Poisson SGD clipped at C=1 without noise, and DP-SGD at epsilon1/4/8. The marked reduction compares the two non-private Poisson controls. (b) Initialization-paired accuracy changes for DP4 minus DP1 and DP8 minus DP1, exposing small budget differences on an explicitly labeled difference scale. Large markers/whiskers show mean +/- sample SD over five seeds; small dots show individual seeds. Panel (a) uses a truncated accuracy axis, not bars; panel (b) uses percentage-point changes, with zero marked. The controls share minibatch streams, while DP uses independently randomized minibatches/noise. Contrasts describe these settings and do not isolate a universal gradient mechanism or establish that noise is irrelevant.

## Figure3. Fairness depends on demographic attribute and criterion (optional supplement)

Four disparity criteria for non-private SGD versus DP1 under the primary include-sensitive configuration, comparing (a) sex groups and (b) the five supported race groups. Accuracy parity and independence are the ranges of group accuracy and selection rates. Separation is the maximum of TPR and FPR ranges. Binary sufficiency is the maximum of PPV and NPV ranges and concerns thresholded predictions. Gaps are computed per run before averaging; whiskers show sample SD across five seeds. Smaller gaps mean closer parity, not necessarily better outcomes within each group. Separation moves in opposite directions across the two demographic attributes. This figure repeats Table1 and is recommended as optional material.
