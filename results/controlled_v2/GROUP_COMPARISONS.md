# DP versus non-private SGD: full group comparisons

[Return to findings](FINDINGS.md). Development set only, fixed threshold 0.5. Levels are percentages; changes are percentage points. Each cell is mean +/- sample SD across five seeds. Changes use same-policy, same-group, same-initialization SGD before averaging. Positive accuracy change means improvement (opposite to the positive-loss convention in FINDINGS). Positive FPR change means more false positives. Equal selection/TPR/FPR/PPV/NPV across groups is the relevant fairness goal; a rate change within one group is not itself a fairness gap.

## Fixed group support

Counts describe people in one development set. Actual-positive and actual-negative counts are the TPR and FPR denominators. Predicted-positive and predicted-negative counts (PPV/NPV denominators) vary by model and seed. The smallest of these four conditional denominators across seeds is shown in each levels table. Counts below30 give weak support. Undefined rates retain missing values.

### SEX

| Group | People | Actual positive | Actual negative | Actual positive (%) |
| --- | --- | --- | --- | --- |
| Male | 14826 | 9124 | 5702 | 61.54 |
| Female | 15438 | 8128 | 7310 | 52.65 |


### RAC1P

| Group | People | Actual positive | Actual negative | Actual positive (%) |
| --- | --- | --- | --- | --- |
| White alone | 18889 | 10710 | 8179 | 56.70 |
| Black or African American alone | 1500 | 735 | 765 | 49.00 |
| American Indian alone | 202 | 90 | 112 | 44.55 |
| Alaska Native alone | 2 | 0 | 2 | 0.00 |
| AIAN specified/unspecified; no other races | 81 | 47 | 34 | 58.02 |
| Asian alone | 5017 | 2950 | 2067 | 58.80 |
| Native Hawaiian / Other Pacific Islander alone | 95 | 58 | 37 | 61.05 |
| Some Other Race alone | 3299 | 1974 | 1325 | 59.84 |
| Two or More Races | 1179 | 688 | 491 | 58.35 |


### RAC1P_SEX

| Group | People | Actual positive | Actual negative | Actual positive (%) |
| --- | --- | --- | --- | --- |
| White alone / Male | 9434 | 5787 | 3647 | 61.34 |
| White alone / Female | 9455 | 4923 | 4532 | 52.07 |
| Black or African American alone / Male | 748 | 346 | 402 | 46.26 |
| Black or African American alone / Female | 752 | 389 | 363 | 51.73 |
| American Indian alone / Male | 101 | 46 | 55 | 45.54 |
| American Indian alone / Female | 101 | 44 | 57 | 43.56 |
| Alaska Native alone / Male | 1 | 0 | 1 | 0.00 |
| Alaska Native alone / Female | 1 | 0 | 1 | 0.00 |
| AIAN specified/unspecified; no other races / Male | 34 | 18 | 16 | 52.94 |
| AIAN specified/unspecified; no other races / Female | 47 | 29 | 18 | 61.70 |
| Asian alone / Male | 2287 | 1474 | 813 | 64.45 |
| Asian alone / Female | 2730 | 1476 | 1254 | 54.07 |
| Native Hawaiian / Other Pacific Islander alone / Male | 51 | 33 | 18 | 64.71 |
| Native Hawaiian / Other Pacific Islander alone / Female | 44 | 25 | 19 | 56.82 |
| Some Other Race alone / Male | 1600 | 1083 | 517 | 67.69 |
| Some Other Race alone / Female | 1699 | 891 | 808 | 52.44 |
| Two or More Races / Male | 570 | 337 | 233 | 59.12 |
| Two or More Races / Female | 609 | 351 | 258 | 57.64 |


## include_sensitive

### SEX

#### sgd: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 80.92 +/- 0.44 | 64.24 +/- 3.94 | 86.69 +/- 3.54 | 28.31 +/- 4.59 | 83.14 +/- 1.73 | 77.41 +/- 3.53 | 4640 |
| Female | 74.06 +/- 0.68 | 56.42 +/- 6.04 | 78.94 +/- 6.34 | 31.37 +/- 5.71 | 73.84 +/- 2.01 | 75.02 +/- 4.13 | 5664 |


#### dp_epsilon_1: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 78.67 +/- 0.59 | 72.25 +/- 0.74 | 91.37 +/- 0.24 | 41.66 +/- 1.69 | 77.83 +/- 0.68 | 80.85 +/- 0.43 | 4006 |
| Female | 72.49 +/- 0.12 | 65.80 +/- 0.71 | 86.36 +/- 0.77 | 42.93 +/- 0.64 | 69.11 +/- 0.14 | 79.02 +/- 0.74 | 5185 |


#### dp_epsilon_1 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| Male | -2.26 +/- 0.85 | +8.01 +/- 3.92 | +4.67 +/- 3.54 | +13.35 +/- 4.72 | -5.30 +/- 1.78 | +3.44 +/- 3.60 |
| Female | -1.57 +/- 0.73 | +9.38 +/- 6.12 | +7.42 +/- 6.48 | +11.56 +/- 5.74 | -4.74 +/- 2.00 | +4.00 +/- 4.27 |


#### dp_epsilon_4: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 78.74 +/- 0.56 | 72.13 +/- 0.72 | 91.33 +/- 0.14 | 41.41 +/- 1.65 | 77.93 +/- 0.67 | 80.84 +/- 0.23 | 4013 |
| Female | 72.49 +/- 0.13 | 65.72 +/- 0.47 | 86.28 +/- 0.55 | 42.85 +/- 0.38 | 69.13 +/- 0.07 | 78.94 +/- 0.56 | 5205 |


#### dp_epsilon_4 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| Male | -2.19 +/- 0.87 | +7.89 +/- 3.85 | +4.63 +/- 3.53 | +13.10 +/- 4.57 | -5.21 +/- 1.72 | +3.43 +/- 3.62 |
| Female | -1.57 +/- 0.70 | +9.30 +/- 6.22 | +7.34 +/- 6.52 | +11.48 +/- 5.90 | -4.72 +/- 2.07 | +3.92 +/- 4.29 |


#### dp_epsilon_8: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 78.56 +/- 0.62 | 72.55 +/- 0.89 | 91.53 +/- 0.24 | 42.19 +/- 1.96 | 77.64 +/- 0.77 | 81.00 +/- 0.17 | 3931 |
| Female | 72.47 +/- 0.11 | 65.98 +/- 0.38 | 86.51 +/- 0.45 | 43.15 +/- 0.31 | 69.03 +/- 0.06 | 79.13 +/- 0.46 | 5205 |


#### dp_epsilon_8 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| Male | -2.36 +/- 0.86 | +8.31 +/- 4.02 | +4.83 +/- 3.58 | +13.88 +/- 4.91 | -5.49 +/- 1.84 | +3.58 +/- 3.64 |
| Female | -1.59 +/- 0.72 | +9.56 +/- 6.21 | +7.57 +/- 6.54 | +11.78 +/- 5.86 | -4.81 +/- 2.04 | +4.11 +/- 4.31 |


### RAC1P

#### sgd: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 77.61 +/- 0.59 | 59.02 +/- 4.89 | 82.31 +/- 4.80 | 28.54 +/- 5.03 | 79.20 +/- 1.97 | 75.87 +/- 3.71 | 6636 |
| Black or African American alone | 79.15 +/- 1.07 | 48.39 +/- 4.67 | 78.10 +/- 5.80 | 19.84 +/- 3.61 | 79.22 +/- 1.83 | 79.46 +/- 3.58 | 645 |
| American Indian alone | 67.43 +/- 0.89 | 47.43 +/- 7.86 | 66.67 +/- 9.13 | 31.96 +/- 6.93 | 62.89 +/- 2.13 | 72.15 +/- 3.44 | 78 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 80.00 +/- 1.03 | 51.36 +/- 5.28 | 77.02 +/- 4.61 | 15.88 +/- 6.44 | 87.35 +/- 4.15 | 72.79 +/- 2.53 | 34 |
| Asian alone | 76.80 +/- 0.67 | 64.01 +/- 5.11 | 84.71 +/- 4.83 | 34.48 +/- 5.53 | 77.92 +/- 1.81 | 75.50 +/- 4.37 | 1531 |
| Native Hawaiian / Other Pacific Islander alone | 83.79 +/- 0.94 | 63.79 +/- 4.80 | 88.97 +/- 3.58 | 24.32 +/- 6.89 | 85.33 +/- 3.12 | 81.77 +/- 3.64 | 29 |
| Some Other Race alone | 77.14 +/- 0.22 | 66.17 +/- 5.17 | 86.19 +/- 4.27 | 36.33 +/- 6.53 | 78.09 +/- 2.28 | 76.04 +/- 3.82 | 938 |
| Two or More Races | 76.61 +/- 1.11 | 64.99 +/- 6.03 | 85.64 +/- 6.10 | 36.05 +/- 5.96 | 77.03 +/- 1.72 | 76.91 +/- 6.18 | 341 |


#### dp_epsilon_1: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 75.98 +/- 0.31 | 66.83 +/- 0.32 | 87.76 +/- 0.35 | 39.44 +/- 0.57 | 74.45 +/- 0.28 | 79.07 +/- 0.49 | 6198 |
| Black or African American alone | 78.04 +/- 0.47 | 54.19 +/- 0.99 | 82.88 +/- 1.43 | 26.61 +/- 0.66 | 74.95 +/- 0.28 | 81.71 +/- 1.12 | 674 |
| American Indian alone | 66.04 +/- 0.75 | 61.49 +/- 1.54 | 80.89 +/- 1.83 | 45.89 +/- 1.62 | 58.62 +/- 0.67 | 77.92 +/- 1.40 | 74 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 77.28 +/- 1.10 | 63.46 +/- 1.66 | 85.11 +/- 1.50 | 33.53 +/- 2.63 | 77.84 +/- 1.28 | 76.38 +/- 1.68 | 29 |
| Asian alone | 74.93 +/- 0.34 | 73.63 +/- 0.52 | 91.29 +/- 0.57 | 48.43 +/- 0.67 | 72.90 +/- 0.25 | 80.59 +/- 0.95 | 1297 |
| Native Hawaiian / Other Pacific Islander alone | 82.53 +/- 0.58 | 73.05 +/- 1.91 | 95.52 +/- 1.97 | 37.84 +/- 1.91 | 79.84 +/- 0.52 | 90.06 +/- 3.84 | 23 |
| Some Other Race alone | 73.13 +/- 0.56 | 79.44 +/- 0.52 | 93.93 +/- 0.38 | 57.86 +/- 1.22 | 70.75 +/- 0.45 | 82.33 +/- 1.01 | 651 |
| Two or More Races | 74.93 +/- 0.52 | 73.99 +/- 0.82 | 91.92 +/- 0.41 | 48.88 +/- 1.56 | 72.49 +/- 0.59 | 81.87 +/- 0.57 | 298 |


#### dp_epsilon_1 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone | -1.63 +/- 0.72 | +7.81 +/- 4.86 | +5.45 +/- 4.80 | +10.90 +/- 4.98 | -4.74 +/- 1.96 | +3.20 +/- 3.73 |
| Black or African American alone | -1.11 +/- 1.19 | +5.80 +/- 4.90 | +4.79 +/- 6.18 | +6.77 +/- 3.69 | -4.27 +/- 1.81 | +2.25 +/- 3.90 |
| American Indian alone | -1.39 +/- 1.18 | +14.06 +/- 9.33 | +14.22 +/- 10.87 | +13.93 +/- 8.22 | -4.27 +/- 2.38 | +5.77 +/- 4.67 |
| Alaska Native alone | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | -2.72 +/- 1.03 | +12.10 +/- 6.01 | +8.09 +/- 5.30 | +17.65 +/- 7.20 | -9.51 +/- 4.31 | +3.59 +/- 3.24 |
| Asian alone | -1.87 +/- 0.86 | +9.62 +/- 5.31 | +6.58 +/- 5.07 | +13.94 +/- 5.72 | -5.02 +/- 1.86 | +5.08 +/- 4.76 |
| Native Hawaiian / Other Pacific Islander alone | -1.26 +/- 0.88 | +9.26 +/- 6.11 | +6.55 +/- 5.23 | +13.51 +/- 7.64 | -5.50 +/- 3.18 | +8.29 +/- 7.09 |
| Some Other Race alone | -4.01 +/- 0.74 | +13.28 +/- 5.30 | +7.74 +/- 4.43 | +21.52 +/- 6.73 | -7.34 +/- 2.33 | +6.29 +/- 4.23 |
| Two or More Races | -1.68 +/- 1.17 | +9.01 +/- 5.97 | +6.28 +/- 6.01 | +12.83 +/- 5.98 | -4.53 +/- 1.74 | +4.96 +/- 6.01 |


#### dp_epsilon_4: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 76.03 +/- 0.29 | 66.73 +/- 0.24 | 87.71 +/- 0.17 | 39.26 +/- 0.57 | 74.53 +/- 0.29 | 79.05 +/- 0.31 | 6238 |
| Black or African American alone | 77.89 +/- 0.43 | 54.25 +/- 0.70 | 82.80 +/- 1.10 | 26.82 +/- 0.45 | 74.79 +/- 0.25 | 81.59 +/- 0.91 | 675 |
| American Indian alone | 65.94 +/- 0.95 | 61.19 +/- 1.24 | 80.44 +/- 1.86 | 45.71 +/- 1.32 | 58.58 +/- 0.77 | 77.58 +/- 1.62 | 76 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 77.53 +/- 1.03 | 63.21 +/- 1.61 | 85.11 +/- 1.50 | 32.94 +/- 2.46 | 78.14 +/- 1.19 | 76.54 +/- 1.66 | 29 |
| Asian alone | 74.96 +/- 0.42 | 73.63 +/- 0.40 | 91.32 +/- 0.42 | 48.39 +/- 0.79 | 72.92 +/- 0.34 | 80.64 +/- 0.80 | 1299 |
| Native Hawaiian / Other Pacific Islander alone | 82.32 +/- 0.88 | 72.42 +/- 1.37 | 94.83 +/- 1.72 | 37.30 +/- 1.21 | 79.94 +/- 0.44 | 88.67 +/- 3.28 | 25 |
| Some Other Race alone | 73.14 +/- 0.46 | 79.25 +/- 0.27 | 93.78 +/- 0.28 | 57.62 +/- 0.84 | 70.80 +/- 0.34 | 82.05 +/- 0.87 | 677 |
| Two or More Races | 75.03 +/- 0.58 | 73.66 +/- 0.73 | 91.72 +/- 0.40 | 48.35 +/- 1.48 | 72.67 +/- 0.59 | 81.65 +/- 0.70 | 304 |


#### dp_epsilon_4 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone | -1.58 +/- 0.72 | +7.70 +/- 4.94 | +5.40 +/- 4.86 | +10.72 +/- 5.11 | -4.67 +/- 2.00 | +3.18 +/- 3.78 |
| Black or African American alone | -1.25 +/- 1.21 | +5.87 +/- 4.77 | +4.71 +/- 6.04 | +6.98 +/- 3.58 | -4.44 +/- 1.78 | +2.13 +/- 3.81 |
| American Indian alone | -1.49 +/- 1.53 | +13.76 +/- 8.93 | +13.78 +/- 10.79 | +13.75 +/- 7.64 | -4.32 +/- 2.24 | +5.43 +/- 4.84 |
| Alaska Native alone | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | -2.47 +/- 0.87 | +11.85 +/- 6.21 | +8.09 +/- 5.30 | +17.06 +/- 7.61 | -9.21 +/- 4.54 | +3.75 +/- 3.08 |
| Asian alone | -1.84 +/- 0.94 | +9.62 +/- 5.24 | +6.61 +/- 5.03 | +13.90 +/- 5.65 | -5.00 +/- 1.84 | +5.14 +/- 4.74 |
| Native Hawaiian / Other Pacific Islander alone | -1.47 +/- 0.94 | +8.63 +/- 5.78 | +5.86 +/- 5.11 | +12.97 +/- 7.00 | -5.39 +/- 2.93 | +6.90 +/- 6.69 |
| Some Other Race alone | -4.01 +/- 0.63 | +13.09 +/- 5.26 | +7.59 +/- 4.47 | +21.28 +/- 6.52 | -7.28 +/- 2.25 | +6.01 +/- 4.38 |
| Two or More Races | -1.58 +/- 1.09 | +8.67 +/- 5.82 | +6.08 +/- 5.81 | +12.30 +/- 5.90 | -4.36 +/- 1.75 | +4.74 +/- 5.68 |


#### dp_epsilon_8: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 75.92 +/- 0.34 | 67.08 +/- 0.46 | 87.92 +/- 0.18 | 39.79 +/- 0.90 | 74.32 +/- 0.41 | 79.20 +/- 0.23 | 6096 |
| Black or African American alone | 77.73 +/- 0.46 | 54.52 +/- 1.01 | 82.91 +/- 1.37 | 27.24 +/- 0.79 | 74.52 +/- 0.38 | 81.61 +/- 1.08 | 666 |
| American Indian alone | 66.14 +/- 1.14 | 61.58 +/- 1.74 | 81.11 +/- 2.22 | 45.89 +/- 1.96 | 58.69 +/- 0.98 | 78.13 +/- 1.89 | 74 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 77.04 +/- 1.41 | 64.20 +/- 0.87 | 85.53 +/- 1.78 | 34.71 +/- 1.32 | 77.30 +/- 0.92 | 76.59 +/- 2.40 | 28 |
| Asian alone | 74.90 +/- 0.37 | 73.90 +/- 0.43 | 91.49 +/- 0.27 | 48.79 +/- 0.89 | 72.80 +/- 0.35 | 80.83 +/- 0.52 | 1289 |
| Native Hawaiian / Other Pacific Islander alone | 82.53 +/- 0.58 | 73.05 +/- 1.91 | 95.52 +/- 1.97 | 37.84 +/- 1.91 | 79.84 +/- 0.52 | 90.06 +/- 3.84 | 23 |
| Some Other Race alone | 73.02 +/- 0.51 | 79.61 +/- 0.52 | 93.98 +/- 0.31 | 58.20 +/- 1.21 | 70.64 +/- 0.43 | 82.33 +/- 0.83 | 643 |
| Two or More Races | 75.00 +/- 0.55 | 74.13 +/- 0.91 | 92.09 +/- 0.49 | 48.96 +/- 1.67 | 72.50 +/- 0.62 | 82.17 +/- 0.69 | 296 |


#### dp_epsilon_8 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone | -1.69 +/- 0.68 | +8.06 +/- 5.00 | +5.62 +/- 4.85 | +11.25 +/- 5.26 | -4.88 +/- 2.08 | +3.33 +/- 3.72 |
| Black or African American alone | -1.41 +/- 1.28 | +6.13 +/- 5.11 | +4.82 +/- 6.49 | +7.40 +/- 3.79 | -4.70 +/- 1.81 | +2.14 +/- 4.14 |
| American Indian alone | -1.29 +/- 1.47 | +14.16 +/- 9.27 | +14.44 +/- 11.19 | +13.93 +/- 7.90 | -4.20 +/- 2.20 | +5.98 +/- 5.12 |
| Alaska Native alone | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | -2.96 +/- 0.68 | +12.84 +/- 5.28 | +8.51 +/- 4.99 | +18.82 +/- 5.73 | -10.05 +/- 3.65 | +3.80 +/- 3.25 |
| Asian alone | -1.90 +/- 0.90 | +9.88 +/- 5.36 | +6.79 +/- 5.10 | +14.30 +/- 5.84 | -5.12 +/- 1.89 | +5.33 +/- 4.80 |
| Native Hawaiian / Other Pacific Islander alone | -1.26 +/- 0.88 | +9.26 +/- 6.11 | +6.55 +/- 5.23 | +13.51 +/- 7.64 | -5.50 +/- 3.18 | +8.29 +/- 7.09 |
| Some Other Race alone | -4.12 +/- 0.70 | +13.45 +/- 5.36 | +7.79 +/- 4.53 | +21.87 +/- 6.71 | -7.45 +/- 2.30 | +6.29 +/- 4.47 |
| Two or More Races | -1.61 +/- 1.33 | +9.14 +/- 5.71 | +6.45 +/- 5.92 | +12.91 +/- 5.50 | -4.53 +/- 1.57 | +5.26 +/- 6.01 |


### RAC1P_SEX

#### sgd: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 80.65 +/- 0.41 | 63.12 +/- 3.99 | 85.68 +/- 3.57 | 27.32 +/- 4.66 | 83.36 +/- 1.81 | 76.47 +/- 3.36 | 3024 |
| White alone / Female | 74.58 +/- 0.78 | 54.94 +/- 5.85 | 78.35 +/- 6.31 | 29.51 +/- 5.36 | 74.42 +/- 1.97 | 75.40 +/- 4.02 | 3612 |
| Black or African American alone / Male | 82.01 +/- 1.32 | 46.02 +/- 4.37 | 80.29 +/- 6.11 | 16.52 +/- 2.90 | 80.83 +/- 1.62 | 83.36 +/- 3.79 | 306 |
| Black or African American alone / Female | 76.30 +/- 0.97 | 50.74 +/- 5.09 | 76.14 +/- 5.74 | 23.53 +/- 4.44 | 77.78 +/- 2.01 | 75.23 +/- 3.40 | 324 |
| American Indian alone / Male | 69.31 +/- 2.32 | 47.72 +/- 9.48 | 68.70 +/- 9.04 | 30.18 +/- 10.17 | 66.39 +/- 5.39 | 73.08 +/- 2.96 | 34 |
| American Indian alone / Female | 65.54 +/- 2.03 | 47.13 +/- 7.78 | 64.55 +/- 10.97 | 33.68 +/- 5.46 | 59.64 +/- 1.18 | 71.30 +/- 4.49 | 35 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 84.71 +/- 1.32 | 48.24 +/- 5.73 | 81.11 +/- 6.33 | 11.25 +/- 5.23 | 89.37 +/- 3.78 | 81.06 +/- 4.36 | 14 |
| AIAN specified/unspecified; no other races / Female | 76.60 +/- 1.50 | 53.62 +/- 5.51 | 74.48 +/- 3.93 | 20.00 +/- 8.43 | 86.09 +/- 4.73 | 66.15 +/- 1.71 | 18 |
| Asian alone / Male | 80.24 +/- 0.49 | 68.40 +/- 3.74 | 87.73 +/- 3.22 | 33.36 +/- 4.70 | 82.73 +/- 1.51 | 75.32 +/- 3.57 | 643 |
| Asian alone / Female | 73.92 +/- 0.82 | 60.34 +/- 6.30 | 81.68 +/- 6.48 | 35.22 +/- 6.12 | 73.35 +/- 1.88 | 75.65 +/- 4.96 | 888 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.53 +/- 2.63 | 67.06 +/- 3.51 | 89.09 +/- 4.60 | 26.67 +/- 2.48 | 85.97 +/- 1.01 | 79.14 +/- 7.04 | 15 |
| Native Hawaiian / Other Pacific Islander alone / Female | 84.09 +/- 4.25 | 60.00 +/- 6.55 | 88.80 +/- 3.35 | 22.11 +/- 12.00 | 84.68 +/- 6.97 | 84.23 +/- 3.10 | 14 |
| Some Other Race alone / Male | 83.81 +/- 0.30 | 74.67 +/- 3.30 | 93.20 +/- 2.57 | 35.86 +/- 4.83 | 84.53 +/- 1.42 | 82.29 +/- 4.38 | 351 |
| Some Other Race alone / Female | 70.87 +/- 0.42 | 58.15 +/- 6.99 | 77.67 +/- 6.40 | 36.63 +/- 7.66 | 70.30 +/- 2.77 | 72.46 +/- 3.43 | 587 |
| Two or More Races / Male | 80.21 +/- 0.53 | 64.46 +/- 5.27 | 87.77 +/- 4.88 | 30.73 +/- 5.86 | 80.65 +/- 2.16 | 80.27 +/- 5.12 | 172 |
| Two or More Races / Female | 73.23 +/- 1.77 | 65.48 +/- 6.98 | 83.59 +/- 7.56 | 40.85 +/- 6.21 | 73.68 +/- 1.27 | 73.75 +/- 7.19 | 169 |


#### dp_epsilon_1: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 78.62 +/- 0.53 | 70.42 +/- 0.74 | 89.97 +/- 0.23 | 39.39 +/- 1.63 | 78.38 +/- 0.67 | 79.21 +/- 0.30 | 2720 |
| White alone / Female | 73.35 +/- 0.12 | 63.26 +/- 0.71 | 85.16 +/- 0.76 | 39.47 +/- 0.67 | 70.09 +/- 0.19 | 78.97 +/- 0.66 | 3417 |
| Black or African American alone / Male | 80.03 +/- 0.64 | 54.52 +/- 0.52 | 87.34 +/- 1.07 | 26.27 +/- 0.57 | 74.10 +/- 0.52 | 87.13 +/- 0.97 | 338 |
| Black or African American alone / Female | 76.06 +/- 0.45 | 53.86 +/- 1.52 | 78.92 +/- 1.77 | 27.00 +/- 1.34 | 75.81 +/- 0.57 | 76.40 +/- 1.20 | 336 |
| American Indian alone / Male | 64.75 +/- 0.89 | 59.01 +/- 0.89 | 76.09 +/- 0.00 | 44.73 +/- 1.63 | 58.74 +/- 0.87 | 73.42 +/- 0.58 | 40 |
| American Indian alone / Female | 67.33 +/- 0.99 | 63.96 +/- 2.77 | 85.91 +/- 3.73 | 47.02 +/- 2.29 | 58.52 +/- 0.75 | 83.16 +/- 3.39 | 33 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 82.94 +/- 1.32 | 58.24 +/- 1.32 | 88.89 +/- 0.00 | 23.75 +/- 2.80 | 80.84 +/- 1.88 | 85.90 +/- 0.43 | 14 |
| AIAN specified/unspecified; no other races / Female | 73.19 +/- 1.17 | 67.23 +/- 3.23 | 82.76 +/- 2.44 | 42.22 +/- 4.97 | 76.01 +/- 1.87 | 67.62 +/- 2.13 | 14 |
| Asian alone / Male | 78.45 +/- 0.54 | 78.51 +/- 0.56 | 94.19 +/- 0.28 | 50.09 +/- 1.46 | 77.32 +/- 0.51 | 82.58 +/- 0.79 | 478 |
| Asian alone / Female | 71.97 +/- 0.23 | 69.54 +/- 0.87 | 88.39 +/- 0.92 | 47.35 +/- 0.86 | 68.72 +/- 0.21 | 79.41 +/- 1.03 | 813 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.92 +/- 0.88 | 76.08 +/- 2.91 | 96.36 +/- 2.54 | 38.89 +/- 3.93 | 81.99 +/- 1.19 | 90.73 +/- 6.13 | 10 |
| Native Hawaiian / Other Pacific Islander alone / Female | 80.91 +/- 1.24 | 69.55 +/- 1.24 | 94.40 +/- 2.19 | 36.84 +/- 0.00 | 77.12 +/- 0.41 | 89.67 +/- 3.61 | 13 |
| Some Other Race alone / Male | 79.67 +/- 0.99 | 82.44 +/- 1.14 | 95.88 +/- 0.34 | 54.27 +/- 3.22 | 78.74 +/- 0.99 | 84.11 +/- 1.19 | 264 |
| Some Other Race alone / Female | 66.97 +/- 0.24 | 76.62 +/- 0.78 | 91.56 +/- 0.78 | 60.15 +/- 0.87 | 62.67 +/- 0.21 | 81.09 +/- 1.16 | 383 |
| Two or More Races / Male | 77.33 +/- 0.97 | 75.05 +/- 1.91 | 94.30 +/- 0.99 | 47.21 +/- 3.42 | 74.31 +/- 1.24 | 86.56 +/- 1.47 | 133 |
| Two or More Races / Female | 72.68 +/- 0.19 | 73.00 +/- 0.38 | 89.63 +/- 0.38 | 50.39 +/- 0.47 | 70.76 +/- 0.16 | 77.86 +/- 0.55 | 162 |


#### dp_epsilon_1 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | -2.03 +/- 0.74 | +7.30 +/- 4.00 | +4.30 +/- 3.54 | +12.06 +/- 4.89 | -4.97 +/- 1.90 | +2.74 +/- 3.34 |
| White alone / Female | -1.23 +/- 0.86 | +8.32 +/- 5.76 | +6.81 +/- 6.30 | +9.96 +/- 5.18 | -4.33 +/- 1.88 | +3.58 +/- 4.05 |
| Black or African American alone / Male | -1.98 +/- 1.68 | +8.50 +/- 4.60 | +7.05 +/- 6.61 | +9.75 +/- 3.01 | -6.72 +/- 1.70 | +3.78 +/- 4.26 |
| Black or African American alone / Female | -0.24 +/- 1.01 | +3.11 +/- 5.25 | +2.78 +/- 5.97 | +3.47 +/- 4.51 | -1.97 +/- 2.00 | +1.17 +/- 3.57 |
| American Indian alone / Male | -4.55 +/- 2.85 | +11.29 +/- 9.61 | +7.39 +/- 9.04 | +14.55 +/- 10.60 | -7.66 +/- 5.64 | +0.34 +/- 3.10 |
| American Indian alone / Female | +1.78 +/- 2.75 | +16.83 +/- 10.07 | +21.36 +/- 14.07 | +13.33 +/- 7.30 | -1.12 +/- 1.74 | +11.86 +/- 7.54 |
| Alaska Native alone / Male | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| Alaska Native alone / Female | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races / Male | -1.76 +/- 1.61 | +10.00 +/- 5.34 | +7.78 +/- 6.33 | +12.50 +/- 4.42 | -8.53 +/- 3.08 | +4.84 +/- 4.49 |
| AIAN specified/unspecified; no other races / Female | -3.40 +/- 1.90 | +13.62 +/- 7.61 | +8.28 +/- 5.23 | +22.22 +/- 11.79 | -10.08 +/- 5.81 | +1.47 +/- 2.49 |
| Asian alone / Male | -1.78 +/- 0.92 | +10.11 +/- 3.69 | +6.46 +/- 3.37 | +16.73 +/- 4.47 | -5.41 +/- 1.42 | +7.26 +/- 4.10 |
| Asian alone / Female | -1.95 +/- 0.84 | +9.20 +/- 6.67 | +6.71 +/- 6.79 | +12.14 +/- 6.57 | -4.62 +/- 2.03 | +3.76 +/- 5.23 |
| Native Hawaiian / Other Pacific Islander alone / Male | +0.39 +/- 3.22 | +9.02 +/- 5.30 | +7.27 +/- 6.28 | +12.22 +/- 4.65 | -3.98 +/- 1.48 | +11.59 +/- 11.60 |
| Native Hawaiian / Other Pacific Islander alone / Female | -3.18 +/- 4.13 | +9.55 +/- 7.08 | +5.60 +/- 4.56 | +14.74 +/- 12.00 | -7.56 +/- 6.83 | +5.44 +/- 5.10 |
| Some Other Race alone / Male | -4.14 +/- 1.25 | +7.76 +/- 3.03 | +2.68 +/- 2.45 | +18.41 +/- 5.03 | -5.80 +/- 1.52 | +1.82 +/- 4.39 |
| Some Other Race alone / Female | -3.90 +/- 0.56 | +18.47 +/- 7.50 | +13.89 +/- 6.92 | +23.51 +/- 8.18 | -7.64 +/- 2.86 | +8.63 +/- 4.20 |
| Two or More Races / Male | -2.88 +/- 1.23 | +10.60 +/- 5.11 | +6.53 +/- 4.59 | +16.48 +/- 6.21 | -6.34 +/- 2.29 | +6.29 +/- 4.74 |
| Two or More Races / Female | -0.56 +/- 1.82 | +7.52 +/- 6.87 | +6.04 +/- 7.50 | +9.53 +/- 6.06 | -2.92 +/- 1.24 | +4.12 +/- 7.16 |


#### dp_epsilon_4: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 78.71 +/- 0.47 | 70.30 +/- 0.72 | 89.95 +/- 0.21 | 39.12 +/- 1.54 | 78.49 +/- 0.63 | 79.23 +/- 0.11 | 2725 |
| White alone / Female | 73.36 +/- 0.12 | 63.17 +/- 0.40 | 85.08 +/- 0.49 | 39.36 +/- 0.30 | 70.13 +/- 0.05 | 78.91 +/- 0.46 | 3442 |
| Black or African American alone / Male | 79.76 +/- 0.69 | 54.57 +/- 0.35 | 87.11 +/- 0.88 | 26.57 +/- 0.67 | 73.84 +/- 0.62 | 86.88 +/- 0.85 | 336 |
| Black or African American alone / Female | 76.04 +/- 0.26 | 53.94 +/- 1.22 | 78.97 +/- 1.39 | 27.11 +/- 1.06 | 75.75 +/- 0.41 | 76.41 +/- 0.93 | 336 |
| American Indian alone / Male | 65.15 +/- 0.83 | 58.61 +/- 0.83 | 76.09 +/- 0.00 | 44.00 +/- 1.52 | 59.13 +/- 0.84 | 73.68 +/- 0.52 | 41 |
| American Indian alone / Female | 66.73 +/- 1.13 | 63.76 +/- 2.68 | 85.00 +/- 3.80 | 47.37 +/- 2.15 | 58.07 +/- 0.81 | 82.15 +/- 3.51 | 33 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 83.53 +/- 1.61 | 57.65 +/- 1.61 | 88.89 +/- 0.00 | 22.50 +/- 3.42 | 81.68 +/- 2.31 | 86.10 +/- 0.52 | 14 |
| AIAN specified/unspecified; no other races / Female | 73.19 +/- 1.17 | 67.23 +/- 3.23 | 82.76 +/- 2.44 | 42.22 +/- 4.97 | 76.01 +/- 1.87 | 67.62 +/- 2.13 | 14 |
| Asian alone / Male | 78.51 +/- 0.65 | 78.54 +/- 0.70 | 94.26 +/- 0.14 | 50.04 +/- 1.88 | 77.36 +/- 0.65 | 82.75 +/- 0.57 | 469 |
| Asian alone / Female | 71.98 +/- 0.26 | 69.52 +/- 0.68 | 88.37 +/- 0.76 | 47.32 +/- 0.67 | 68.73 +/- 0.19 | 79.39 +/- 0.91 | 817 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.53 +/- 1.75 | 74.90 +/- 2.15 | 95.15 +/- 2.71 | 37.78 +/- 2.48 | 82.20 +/- 0.98 | 87.86 +/- 5.80 | 12 |
| Native Hawaiian / Other Pacific Islander alone / Female | 80.91 +/- 1.24 | 69.55 +/- 1.24 | 94.40 +/- 2.19 | 36.84 +/- 0.00 | 77.12 +/- 0.41 | 89.67 +/- 3.61 | 13 |
| Some Other Race alone / Male | 79.79 +/- 0.93 | 82.17 +/- 1.00 | 95.77 +/- 0.12 | 53.69 +/- 2.99 | 78.90 +/- 0.93 | 83.91 +/- 0.76 | 271 |
| Some Other Race alone / Female | 66.87 +/- 0.21 | 76.50 +/- 0.67 | 91.36 +/- 0.64 | 60.12 +/- 0.77 | 62.63 +/- 0.19 | 80.73 +/- 0.95 | 380 |
| Two or More Races / Male | 77.47 +/- 1.10 | 74.63 +/- 1.78 | 94.07 +/- 0.89 | 46.52 +/- 3.39 | 74.54 +/- 1.28 | 86.21 +/- 1.34 | 137 |
| Two or More Races / Female | 72.74 +/- 0.12 | 72.74 +/- 0.42 | 89.46 +/- 0.40 | 50.00 +/- 0.47 | 70.88 +/- 0.12 | 77.72 +/- 0.53 | 162 |


#### dp_epsilon_4 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | -1.94 +/- 0.73 | +7.18 +/- 3.99 | +4.27 +/- 3.56 | +11.80 +/- 4.79 | -4.87 +/- 1.86 | +2.76 +/- 3.40 |
| White alone / Female | -1.22 +/- 0.81 | +8.23 +/- 5.94 | +6.73 +/- 6.41 | +9.85 +/- 5.44 | -4.29 +/- 2.00 | +3.51 +/- 4.10 |
| Black or African American alone / Male | -2.25 +/- 1.76 | +8.56 +/- 4.39 | +6.82 +/- 6.47 | +10.05 +/- 2.77 | -6.99 +/- 1.58 | +3.52 +/- 4.17 |
| Black or African American alone / Female | -0.27 +/- 0.94 | +3.19 +/- 5.22 | +2.83 +/- 5.86 | +3.58 +/- 4.58 | -2.03 +/- 2.07 | +1.18 +/- 3.46 |
| American Indian alone / Male | -4.16 +/- 2.56 | +10.89 +/- 9.10 | +7.39 +/- 9.04 | +13.82 +/- 9.67 | -7.26 +/- 5.14 | +0.60 +/- 3.34 |
| American Indian alone / Female | +1.19 +/- 2.84 | +16.63 +/- 9.81 | +20.45 +/- 13.73 | +13.68 +/- 7.17 | -1.57 +/- 1.91 | +10.86 +/- 7.37 |
| Alaska Native alone / Male | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| Alaska Native alone / Female | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races / Male | -1.18 +/- 1.61 | +9.41 +/- 5.66 | +7.78 +/- 6.33 | +11.25 +/- 5.23 | -7.69 +/- 3.69 | +5.03 +/- 4.43 |
| AIAN specified/unspecified; no other races / Female | -3.40 +/- 1.90 | +13.62 +/- 7.61 | +8.28 +/- 5.23 | +22.22 +/- 11.79 | -10.08 +/- 5.81 | +1.47 +/- 2.49 |
| Asian alone / Male | -1.72 +/- 1.04 | +10.14 +/- 3.60 | +6.53 +/- 3.30 | +16.68 +/- 4.45 | -5.37 +/- 1.43 | +7.44 +/- 3.98 |
| Asian alone / Female | -1.94 +/- 0.90 | +9.18 +/- 6.64 | +6.69 +/- 6.79 | +12.11 +/- 6.52 | -4.61 +/- 2.01 | +3.74 +/- 5.27 |
| Native Hawaiian / Other Pacific Islander alone / Male | +0.00 +/- 3.92 | +7.84 +/- 4.80 | +6.06 +/- 6.43 | +11.11 +/- 3.93 | -3.77 +/- 1.62 | +8.72 +/- 11.15 |
| Native Hawaiian / Other Pacific Islander alone / Female | -3.18 +/- 4.13 | +9.55 +/- 7.08 | +5.60 +/- 4.56 | +14.74 +/- 12.00 | -7.56 +/- 6.83 | +5.44 +/- 5.10 |
| Some Other Race alone / Male | -4.03 +/- 1.21 | +7.50 +/- 2.96 | +2.57 +/- 2.54 | +17.83 +/- 4.55 | -5.64 +/- 1.36 | +1.62 +/- 4.72 |
| Some Other Race alone / Female | -3.99 +/- 0.57 | +18.35 +/- 7.48 | +13.69 +/- 6.87 | +23.49 +/- 8.19 | -7.68 +/- 2.87 | +8.27 +/- 4.15 |
| Two or More Races / Male | -2.74 +/- 1.28 | +10.18 +/- 4.88 | +6.29 +/- 4.26 | +15.79 +/- 6.18 | -6.11 +/- 2.36 | +5.94 +/- 4.06 |
| Two or More Races / Female | -0.49 +/- 1.77 | +7.26 +/- 6.82 | +5.87 +/- 7.44 | +9.15 +/- 5.99 | -2.80 +/- 1.20 | +3.97 +/- 7.05 |


#### dp_epsilon_8: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 78.52 +/- 0.58 | 70.74 +/- 0.96 | 90.16 +/- 0.32 | 39.93 +/- 1.98 | 78.19 +/- 0.79 | 79.36 +/- 0.14 | 2662 |
| White alone / Female | 73.33 +/- 0.14 | 63.43 +/- 0.32 | 85.30 +/- 0.42 | 39.67 +/- 0.25 | 70.02 +/- 0.07 | 79.07 +/- 0.41 | 3428 |
| Black or African American alone / Male | 79.52 +/- 0.78 | 54.92 +/- 0.72 | 87.23 +/- 1.16 | 27.11 +/- 0.96 | 73.47 +/- 0.75 | 86.90 +/- 1.06 | 332 |
| Black or African American alone / Female | 75.96 +/- 0.39 | 54.12 +/- 1.48 | 79.07 +/- 1.63 | 27.38 +/- 1.41 | 75.59 +/- 0.63 | 76.43 +/- 1.08 | 332 |
| American Indian alone / Male | 65.15 +/- 1.63 | 59.01 +/- 1.13 | 76.52 +/- 0.97 | 44.36 +/- 2.44 | 59.08 +/- 1.53 | 73.89 +/- 1.46 | 40 |
| American Indian alone / Female | 67.13 +/- 0.83 | 64.16 +/- 2.75 | 85.91 +/- 3.73 | 47.37 +/- 2.15 | 58.33 +/- 0.56 | 83.07 +/- 3.37 | 33 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 82.94 +/- 1.32 | 58.24 +/- 1.32 | 88.89 +/- 0.00 | 23.75 +/- 2.80 | 80.84 +/- 1.88 | 85.90 +/- 0.43 | 14 |
| AIAN specified/unspecified; no other races / Female | 72.77 +/- 1.78 | 68.51 +/- 1.78 | 83.45 +/- 2.89 | 44.44 +/- 0.00 | 75.14 +/- 0.65 | 67.74 +/- 3.77 | 14 |
| Asian alone / Male | 78.38 +/- 0.72 | 78.78 +/- 0.91 | 94.34 +/- 0.21 | 50.55 +/- 2.28 | 77.19 +/- 0.76 | 82.81 +/- 0.50 | 453 |
| Asian alone / Female | 71.98 +/- 0.19 | 69.81 +/- 0.46 | 88.64 +/- 0.45 | 47.64 +/- 0.56 | 68.65 +/- 0.19 | 79.67 +/- 0.53 | 810 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.92 +/- 0.88 | 76.08 +/- 2.91 | 96.36 +/- 2.54 | 38.89 +/- 3.93 | 81.99 +/- 1.19 | 90.73 +/- 6.13 | 10 |
| Native Hawaiian / Other Pacific Islander alone / Female | 80.91 +/- 1.24 | 69.55 +/- 1.24 | 94.40 +/- 2.19 | 36.84 +/- 0.00 | 77.12 +/- 0.41 | 89.67 +/- 3.61 | 13 |
| Some Other Race alone / Male | 79.53 +/- 1.00 | 82.69 +/- 1.14 | 95.96 +/- 0.15 | 54.89 +/- 3.30 | 78.56 +/- 1.01 | 84.16 +/- 0.70 | 258 |
| Some Other Race alone / Female | 66.90 +/- 0.19 | 76.72 +/- 0.75 | 91.58 +/- 0.62 | 60.32 +/- 0.93 | 62.61 +/- 0.22 | 81.06 +/- 0.83 | 380 |
| Two or More Races / Male | 77.47 +/- 0.84 | 75.26 +/- 1.92 | 94.60 +/- 1.10 | 47.30 +/- 3.26 | 74.33 +/- 1.15 | 87.18 +/- 1.66 | 132 |
| Two or More Races / Female | 72.68 +/- 0.38 | 73.07 +/- 0.39 | 89.69 +/- 0.55 | 50.47 +/- 0.51 | 70.74 +/- 0.24 | 77.93 +/- 0.94 | 160 |


#### dp_epsilon_8 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | -2.13 +/- 0.73 | +7.62 +/- 4.15 | +4.48 +/- 3.57 | +12.61 +/- 5.22 | -5.17 +/- 2.03 | +2.89 +/- 3.31 |
| White alone / Female | -1.25 +/- 0.83 | +8.49 +/- 5.88 | +6.95 +/- 6.38 | +10.15 +/- 5.37 | -4.40 +/- 1.97 | +3.68 +/- 4.08 |
| Black or African American alone / Male | -2.49 +/- 1.81 | +8.90 +/- 4.84 | +6.94 +/- 6.98 | +10.60 +/- 3.17 | -7.36 +/- 1.74 | +3.54 +/- 4.57 |
| Black or African American alone / Female | -0.35 +/- 1.17 | +3.38 +/- 5.44 | +2.93 +/- 6.25 | +3.86 +/- 4.66 | -2.19 +/- 2.05 | +1.21 +/- 3.77 |
| American Indian alone / Male | -4.16 +/- 2.56 | +11.29 +/- 9.51 | +7.83 +/- 9.91 | +14.18 +/- 9.74 | -7.31 +/- 4.95 | +0.81 +/- 3.98 |
| American Indian alone / Female | +1.58 +/- 2.77 | +17.03 +/- 10.06 | +21.36 +/- 14.16 | +13.68 +/- 7.17 | -1.30 +/- 1.68 | +11.77 +/- 7.61 |
| Alaska Native alone / Male | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| Alaska Native alone / Female | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races / Male | -1.76 +/- 1.61 | +10.00 +/- 5.34 | +7.78 +/- 6.33 | +12.50 +/- 4.42 | -8.53 +/- 3.08 | +4.84 +/- 4.49 |
| AIAN specified/unspecified; no other races / Female | -3.83 +/- 0.95 | +14.89 +/- 6.38 | +8.97 +/- 5.23 | +24.44 +/- 8.43 | -10.95 +/- 4.42 | +1.59 +/- 3.51 |
| Asian alone / Male | -1.85 +/- 1.08 | +10.37 +/- 3.85 | +6.61 +/- 3.34 | +17.20 +/- 5.13 | -5.54 +/- 1.65 | +7.49 +/- 3.96 |
| Asian alone / Female | -1.94 +/- 0.85 | +9.47 +/- 6.67 | +6.96 +/- 6.88 | +12.42 +/- 6.44 | -4.69 +/- 1.93 | +4.02 +/- 5.40 |
| Native Hawaiian / Other Pacific Islander alone / Male | +0.39 +/- 3.22 | +9.02 +/- 5.30 | +7.27 +/- 6.28 | +12.22 +/- 4.65 | -3.98 +/- 1.48 | +11.59 +/- 11.60 |
| Native Hawaiian / Other Pacific Islander alone / Female | -3.18 +/- 4.13 | +9.55 +/- 7.08 | +5.60 +/- 4.56 | +14.74 +/- 12.00 | -7.56 +/- 6.83 | +5.44 +/- 5.10 |
| Some Other Race alone / Male | -4.29 +/- 1.25 | +8.01 +/- 3.03 | +2.75 +/- 2.62 | +19.03 +/- 4.63 | -5.97 +/- 1.36 | +1.87 +/- 4.96 |
| Some Other Race alone / Female | -3.97 +/- 0.59 | +18.56 +/- 7.60 | +13.92 +/- 6.90 | +23.69 +/- 8.39 | -7.70 +/- 2.94 | +8.60 +/- 4.08 |
| Two or More Races / Male | -2.74 +/- 1.17 | +10.81 +/- 4.78 | +6.82 +/- 4.51 | +16.57 +/- 5.48 | -6.32 +/- 2.00 | +6.91 +/- 4.75 |
| Two or More Races / Female | -0.56 +/- 1.90 | +7.59 +/- 6.69 | +6.10 +/- 7.41 | +9.61 +/- 5.75 | -2.93 +/- 1.14 | +4.19 +/- 7.11 |


## exclude_sensitive

### SEX

#### sgd: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 80.48 +/- 0.29 | 63.69 +/- 2.33 | 85.89 +/- 2.12 | 28.17 +/- 2.66 | 83.02 +/- 0.96 | 76.19 +/- 2.16 | 4844 |
| Female | 73.63 +/- 0.14 | 64.40 +/- 2.23 | 86.12 +/- 2.02 | 40.26 +/- 2.47 | 70.42 +/- 0.77 | 79.56 +/- 1.79 | 4945 |


#### dp_epsilon_1: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 79.44 +/- 0.13 | 68.65 +/- 0.48 | 89.07 +/- 0.31 | 35.97 +/- 0.76 | 79.85 +/- 0.29 | 78.55 +/- 0.30 | 4574 |
| Female | 72.18 +/- 0.08 | 69.47 +/- 0.43 | 89.55 +/- 0.35 | 47.15 +/- 0.52 | 67.87 +/- 0.16 | 81.98 +/- 0.35 | 4661 |


#### dp_epsilon_1 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| Male | -1.04 +/- 0.25 | +4.96 +/- 2.46 | +3.18 +/- 2.18 | +7.80 +/- 2.91 | -3.17 +/- 1.06 | +2.36 +/- 2.18 |
| Female | -1.45 +/- 0.15 | +5.07 +/- 2.19 | +3.44 +/- 1.97 | +6.89 +/- 2.44 | -2.55 +/- 0.77 | +2.43 +/- 1.73 |


#### dp_epsilon_4: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 79.45 +/- 0.08 | 68.68 +/- 0.31 | 89.11 +/- 0.21 | 35.99 +/- 0.48 | 79.84 +/- 0.18 | 78.60 +/- 0.22 | 4588 |
| Female | 72.16 +/- 0.07 | 69.48 +/- 0.26 | 89.54 +/- 0.21 | 47.17 +/- 0.34 | 67.85 +/- 0.11 | 81.96 +/- 0.21 | 4666 |


#### dp_epsilon_4 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| Male | -1.03 +/- 0.23 | +4.99 +/- 2.54 | +3.22 +/- 2.24 | +7.82 +/- 3.03 | -3.17 +/- 1.11 | +2.40 +/- 2.25 |
| Female | -1.47 +/- 0.20 | +5.08 +/- 2.27 | +3.43 +/- 2.00 | +6.91 +/- 2.58 | -2.57 +/- 0.83 | +2.41 +/- 1.74 |


#### dp_epsilon_8: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Male | 79.47 +/- 0.11 | 68.57 +/- 0.51 | 89.03 +/- 0.35 | 35.83 +/- 0.77 | 79.91 +/- 0.29 | 78.53 +/- 0.34 | 4597 |
| Female | 72.19 +/- 0.10 | 69.39 +/- 0.42 | 89.48 +/- 0.34 | 47.05 +/- 0.52 | 67.90 +/- 0.17 | 81.91 +/- 0.34 | 4677 |


#### dp_epsilon_8 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| Male | -1.01 +/- 0.21 | +4.88 +/- 2.50 | +3.14 +/- 2.19 | +7.65 +/- 2.99 | -3.11 +/- 1.09 | +2.33 +/- 2.19 |
| Female | -1.44 +/- 0.16 | +4.98 +/- 2.25 | +3.36 +/- 2.00 | +6.79 +/- 2.54 | -2.52 +/- 0.81 | +2.36 +/- 1.74 |


### RAC1P

#### sgd: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 77.25 +/- 0.10 | 61.78 +/- 2.30 | 84.42 +/- 2.10 | 32.14 +/- 2.57 | 77.50 +/- 0.93 | 76.97 +/- 1.82 | 6513 |
| Black or African American alone | 79.53 +/- 0.26 | 53.68 +/- 1.88 | 83.89 +/- 1.81 | 24.65 +/- 1.97 | 76.61 +/- 1.03 | 83.00 +/- 1.26 | 649 |
| American Indian alone | 69.90 +/- 1.01 | 59.21 +/- 2.20 | 82.67 +/- 1.86 | 40.36 +/- 2.71 | 62.24 +/- 1.20 | 81.10 +/- 1.23 | 75 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 80.49 +/- 0.55 | 58.77 +/- 1.10 | 83.83 +/- 1.17 | 24.12 +/- 1.32 | 82.78 +/- 0.66 | 77.26 +/- 1.11 | 32 |
| Asian alone | 76.39 +/- 0.25 | 70.26 +/- 2.04 | 89.67 +/- 1.66 | 42.55 +/- 2.61 | 75.06 +/- 0.80 | 79.68 +/- 2.02 | 1330 |
| Native Hawaiian / Other Pacific Islander alone | 82.95 +/- 2.02 | 68.42 +/- 2.23 | 92.07 +/- 0.94 | 31.35 +/- 5.27 | 82.21 +/- 2.45 | 84.65 +/- 1.74 | 28 |
| Some Other Race alone | 75.28 +/- 0.43 | 70.88 +/- 2.99 | 88.57 +/- 2.85 | 44.51 +/- 3.20 | 74.80 +/- 0.75 | 76.81 +/- 3.59 | 831 |
| Two or More Races | 77.29 +/- 0.56 | 69.13 +/- 2.29 | 89.77 +/- 2.41 | 40.20 +/- 2.14 | 75.79 +/- 0.49 | 80.85 +/- 3.20 | 329 |


#### dp_epsilon_1: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 76.04 +/- 0.11 | 66.65 +/- 0.43 | 87.65 +/- 0.32 | 39.15 +/- 0.58 | 74.57 +/- 0.22 | 79.00 +/- 0.29 | 6229 |
| Black or African American alone | 78.51 +/- 0.20 | 57.83 +/- 0.58 | 87.07 +/- 0.57 | 29.73 +/- 0.66 | 73.79 +/- 0.33 | 84.99 +/- 0.47 | 619 |
| American Indian alone | 66.73 +/- 0.65 | 63.37 +/- 1.05 | 83.78 +/- 0.61 | 46.96 +/- 1.49 | 58.91 +/- 0.65 | 80.27 +/- 0.38 | 72 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 79.75 +/- 1.10 | 65.93 +/- 1.66 | 89.36 +/- 2.13 | 33.53 +/- 1.61 | 78.66 +/- 0.71 | 81.98 +/- 2.82 | 26 |
| Asian alone | 75.08 +/- 0.07 | 75.14 +/- 0.50 | 92.71 +/- 0.37 | 50.07 +/- 0.68 | 72.55 +/- 0.19 | 82.77 +/- 0.53 | 1217 |
| Native Hawaiian / Other Pacific Islander alone | 82.53 +/- 0.94 | 71.37 +/- 0.47 | 94.14 +/- 0.94 | 35.68 +/- 1.21 | 80.53 +/- 0.64 | 87.51 +/- 1.89 | 27 |
| Some Other Race alone | 73.92 +/- 0.27 | 77.56 +/- 0.49 | 93.02 +/- 0.23 | 54.54 +/- 0.93 | 71.76 +/- 0.31 | 81.39 +/- 0.29 | 727 |
| Two or More Races | 75.76 +/- 0.22 | 73.67 +/- 0.50 | 92.35 +/- 0.38 | 47.49 +/- 0.75 | 73.15 +/- 0.26 | 83.06 +/- 0.56 | 304 |


#### dp_epsilon_1 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone | -1.20 +/- 0.05 | +4.87 +/- 2.32 | +3.23 +/- 2.07 | +7.01 +/- 2.65 | -2.93 +/- 0.98 | +2.03 +/- 1.75 |
| Black or African American alone | -1.03 +/- 0.42 | +4.15 +/- 2.08 | +3.18 +/- 2.01 | +5.07 +/- 2.22 | -2.82 +/- 1.17 | +1.99 +/- 1.44 |
| American Indian alone | -3.17 +/- 1.47 | +4.16 +/- 2.20 | +1.11 +/- 1.76 | +6.61 +/- 3.07 | -3.32 +/- 1.54 | -0.83 +/- 1.39 |
| Alaska Native alone | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | -0.74 +/- 1.41 | +7.16 +/- 2.03 | +5.53 +/- 2.85 | +9.41 +/- 1.32 | -4.12 +/- 0.56 | +4.72 +/- 3.56 |
| Asian alone | -1.31 +/- 0.25 | +4.89 +/- 2.17 | +3.04 +/- 1.78 | +7.52 +/- 2.76 | -2.52 +/- 0.83 | +3.09 +/- 2.20 |
| Native Hawaiian / Other Pacific Islander alone | -0.42 +/- 2.18 | +2.95 +/- 2.51 | +2.07 +/- 1.44 | +4.32 +/- 5.60 | -1.68 +/- 2.57 | +2.86 +/- 2.53 |
| Some Other Race alone | -1.36 +/- 0.55 | +6.69 +/- 2.96 | +4.45 +/- 2.82 | +10.02 +/- 3.24 | -3.04 +/- 0.79 | +4.57 +/- 3.56 |
| Two or More Races | -1.53 +/- 0.51 | +4.55 +/- 2.41 | +2.59 +/- 2.43 | +7.29 +/- 2.42 | -2.64 +/- 0.62 | +2.21 +/- 3.17 |


#### dp_epsilon_4: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 76.03 +/- 0.06 | 66.63 +/- 0.29 | 87.62 +/- 0.24 | 39.15 +/- 0.37 | 74.56 +/- 0.13 | 78.96 +/- 0.24 | 6252 |
| Black or African American alone | 78.68 +/- 0.15 | 57.84 +/- 0.45 | 87.27 +/- 0.34 | 29.57 +/- 0.57 | 73.93 +/- 0.30 | 85.20 +/- 0.25 | 623 |
| American Indian alone | 66.93 +/- 0.65 | 63.17 +/- 1.03 | 83.78 +/- 0.61 | 46.61 +/- 1.47 | 59.10 +/- 0.65 | 80.38 +/- 0.41 | 72 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 80.74 +/- 1.10 | 66.42 +/- 1.35 | 90.64 +/- 1.90 | 32.94 +/- 1.32 | 79.18 +/- 0.65 | 83.90 +/- 2.58 | 26 |
| Asian alone | 75.08 +/- 0.09 | 75.33 +/- 0.29 | 92.86 +/- 0.19 | 50.30 +/- 0.44 | 72.49 +/- 0.14 | 82.99 +/- 0.26 | 1216 |
| Native Hawaiian / Other Pacific Islander alone | 82.53 +/- 0.94 | 71.37 +/- 0.47 | 94.14 +/- 0.94 | 35.68 +/- 1.21 | 80.53 +/- 0.64 | 87.51 +/- 1.89 | 27 |
| Some Other Race alone | 73.88 +/- 0.20 | 77.61 +/- 0.22 | 93.02 +/- 0.11 | 54.64 +/- 0.50 | 71.72 +/- 0.18 | 81.35 +/- 0.26 | 726 |
| Two or More Races | 75.83 +/- 0.33 | 73.54 +/- 0.19 | 92.30 +/- 0.34 | 47.25 +/- 0.43 | 73.24 +/- 0.22 | 83.01 +/- 0.69 | 309 |


#### dp_epsilon_4 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone | -1.22 +/- 0.08 | +4.85 +/- 2.40 | +3.20 +/- 2.13 | +7.01 +/- 2.75 | -2.94 +/- 1.01 | +1.99 +/- 1.82 |
| Black or African American alone | -0.85 +/- 0.34 | +4.16 +/- 2.08 | +3.37 +/- 1.99 | +4.92 +/- 2.20 | -2.68 +/- 1.14 | +2.20 +/- 1.42 |
| American Indian alone | -2.97 +/- 1.53 | +3.96 +/- 2.35 | +1.11 +/- 1.76 | +6.25 +/- 3.28 | -3.14 +/- 1.62 | -0.73 +/- 1.33 |
| Alaska Native alone | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | +0.25 +/- 1.03 | +7.65 +/- 2.37 | +6.81 +/- 2.77 | +8.82 +/- 2.08 | -3.60 +/- 0.75 | +6.63 +/- 3.19 |
| Asian alone | -1.32 +/- 0.24 | +5.07 +/- 2.26 | +3.19 +/- 1.83 | +7.75 +/- 2.90 | -2.58 +/- 0.88 | +3.31 +/- 2.26 |
| Native Hawaiian / Other Pacific Islander alone | -0.42 +/- 2.18 | +2.95 +/- 2.51 | +2.07 +/- 1.44 | +4.32 +/- 5.60 | -1.68 +/- 2.57 | +2.86 +/- 2.53 |
| Some Other Race alone | -1.41 +/- 0.28 | +6.73 +/- 3.13 | +4.45 +/- 2.82 | +10.13 +/- 3.60 | -3.08 +/- 0.90 | +4.54 +/- 3.40 |
| Two or More Races | -1.46 +/- 0.40 | +4.41 +/- 2.34 | +2.53 +/- 2.22 | +7.05 +/- 2.55 | -2.55 +/- 0.72 | +2.17 +/- 2.77 |


#### dp_epsilon_8: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone | 76.05 +/- 0.12 | 66.51 +/- 0.46 | 87.54 +/- 0.33 | 38.99 +/- 0.64 | 74.62 +/- 0.25 | 78.90 +/- 0.27 | 6249 |
| Black or African American alone | 78.57 +/- 0.26 | 57.81 +/- 0.54 | 87.13 +/- 0.51 | 29.65 +/- 0.66 | 73.85 +/- 0.37 | 85.05 +/- 0.43 | 624 |
| American Indian alone | 67.03 +/- 0.56 | 63.07 +/- 1.03 | 83.78 +/- 0.61 | 46.43 +/- 1.41 | 59.19 +/- 0.59 | 80.43 +/- 0.31 | 72 |
| Alaska Native alone | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races | 80.25 +/- 0.87 | 65.93 +/- 1.41 | 89.79 +/- 1.78 | 32.94 +/- 1.32 | 79.03 +/- 0.55 | 82.68 +/- 2.34 | 26 |
| Asian alone | 75.07 +/- 0.07 | 75.27 +/- 0.58 | 92.81 +/- 0.47 | 50.24 +/- 0.73 | 72.50 +/- 0.19 | 82.91 +/- 0.72 | 1214 |
| Native Hawaiian / Other Pacific Islander alone | 82.11 +/- 0.74 | 70.95 +/- 0.58 | 93.45 +/- 0.77 | 35.68 +/- 1.21 | 80.42 +/- 0.58 | 86.24 +/- 1.50 | 27 |
| Some Other Race alone | 73.99 +/- 0.21 | 77.47 +/- 0.41 | 93.00 +/- 0.17 | 54.32 +/- 0.77 | 71.84 +/- 0.25 | 81.41 +/- 0.14 | 734 |
| Two or More Races | 75.88 +/- 0.31 | 73.52 +/- 0.31 | 92.33 +/- 0.28 | 47.17 +/- 0.64 | 73.28 +/- 0.27 | 83.09 +/- 0.53 | 307 |


#### dp_epsilon_8 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone | -1.20 +/- 0.04 | +4.73 +/- 2.43 | +3.12 +/- 2.14 | +6.84 +/- 2.82 | -2.88 +/- 1.04 | +1.93 +/- 1.80 |
| Black or African American alone | -0.96 +/- 0.46 | +4.13 +/- 1.98 | +3.24 +/- 1.92 | +4.99 +/- 2.13 | -2.76 +/- 1.13 | +2.05 +/- 1.39 |
| American Indian alone | -2.87 +/- 1.23 | +3.86 +/- 1.99 | +1.11 +/- 1.76 | +6.07 +/- 2.63 | -3.05 +/- 1.31 | -0.67 +/- 1.37 |
| Alaska Native alone | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races | -0.25 +/- 0.55 | +7.16 +/- 1.83 | +5.96 +/- 1.78 | +8.82 +/- 2.08 | -3.75 +/- 0.88 | +5.41 +/- 2.07 |
| Asian alone | -1.32 +/- 0.25 | +5.01 +/- 2.15 | +3.14 +/- 1.78 | +7.68 +/- 2.71 | -2.56 +/- 0.81 | +3.23 +/- 2.23 |
| Native Hawaiian / Other Pacific Islander alone | -0.84 +/- 2.40 | +2.53 +/- 2.05 | +1.38 +/- 0.77 | +4.32 +/- 5.60 | -1.80 +/- 2.66 | +1.59 +/- 2.03 |
| Some Other Race alone | -1.29 +/- 0.49 | +6.59 +/- 2.97 | +4.43 +/- 2.81 | +9.81 +/- 3.26 | -2.97 +/- 0.79 | +4.60 +/- 3.52 |
| Two or More Races | -1.41 +/- 0.56 | +4.39 +/- 2.41 | +2.56 +/- 2.39 | +6.97 +/- 2.55 | -2.51 +/- 0.71 | +2.24 +/- 3.10 |


### RAC1P_SEX

#### sgd: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 80.20 +/- 0.35 | 61.48 +/- 2.37 | 83.97 +/- 2.22 | 25.80 +/- 2.63 | 83.81 +/- 0.99 | 74.58 +/- 2.07 | 3270 |
| White alone / Female | 74.30 +/- 0.21 | 62.08 +/- 2.22 | 84.94 +/- 1.96 | 37.25 +/- 2.52 | 71.26 +/- 0.89 | 79.39 +/- 1.56 | 3243 |
| Black or African American alone / Male | 82.17 +/- 0.45 | 51.68 +/- 1.67 | 86.59 +/- 1.54 | 21.64 +/- 1.85 | 77.53 +/- 1.19 | 87.19 +/- 1.05 | 343 |
| Black or African American alone / Female | 76.91 +/- 0.20 | 55.66 +/- 2.12 | 81.49 +/- 2.14 | 27.99 +/- 2.12 | 75.76 +/- 0.88 | 78.46 +/- 1.55 | 306 |
| American Indian alone / Male | 69.90 +/- 1.66 | 54.65 +/- 3.00 | 76.96 +/- 2.48 | 36.00 +/- 3.94 | 64.21 +/- 2.07 | 76.89 +/- 1.46 | 41 |
| American Indian alone / Female | 69.90 +/- 0.89 | 63.76 +/- 1.66 | 88.64 +/- 1.61 | 44.56 +/- 2.00 | 60.57 +/- 0.87 | 86.38 +/- 1.45 | 34 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 82.35 +/- 2.08 | 52.94 +/- 2.08 | 83.33 +/- 3.93 | 18.75 +/- 0.00 | 83.31 +/- 0.66 | 81.38 +/- 3.61 | 15 |
| AIAN specified/unspecified; no other races / Female | 79.15 +/- 1.78 | 62.98 +/- 1.17 | 84.14 +/- 1.89 | 28.89 +/- 2.48 | 82.44 +/- 1.39 | 73.59 +/- 2.71 | 17 |
| Asian alone / Male | 79.97 +/- 0.27 | 71.93 +/- 2.11 | 90.26 +/- 1.55 | 38.70 +/- 3.14 | 80.90 +/- 0.99 | 77.75 +/- 1.99 | 569 |
| Asian alone / Female | 73.40 +/- 0.25 | 68.86 +/- 2.00 | 89.08 +/- 1.79 | 45.06 +/- 2.27 | 69.96 +/- 0.65 | 81.14 +/- 2.04 | 761 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.92 +/- 2.15 | 70.59 +/- 1.96 | 92.12 +/- 1.66 | 31.11 +/- 4.97 | 84.48 +/- 2.11 | 82.69 +/- 3.39 | 14 |
| Native Hawaiian / Other Pacific Islander alone / Female | 81.82 +/- 2.78 | 65.91 +/- 2.78 | 92.00 +/- 0.00 | 31.58 +/- 6.45 | 79.43 +/- 3.48 | 86.60 +/- 1.03 | 14 |
| Some Other Race alone / Male | 82.69 +/- 0.81 | 70.28 +/- 2.97 | 89.12 +/- 2.79 | 30.79 +/- 3.35 | 85.87 +/- 0.94 | 75.55 +/- 3.98 | 415 |
| Some Other Race alone / Female | 68.31 +/- 0.26 | 71.44 +/- 3.01 | 87.90 +/- 2.94 | 53.29 +/- 3.11 | 64.54 +/- 0.60 | 78.05 +/- 3.28 | 416 |
| Two or More Races / Male | 80.28 +/- 0.60 | 66.21 +/- 2.40 | 89.32 +/- 2.53 | 32.79 +/- 2.24 | 79.77 +/- 0.65 | 81.49 +/- 3.16 | 176 |
| Two or More Races / Female | 74.48 +/- 0.53 | 71.86 +/- 2.19 | 90.20 +/- 2.30 | 46.90 +/- 2.07 | 72.36 +/- 0.40 | 80.13 +/- 3.26 | 153 |


#### dp_epsilon_1: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 79.30 +/- 0.13 | 66.37 +/- 0.45 | 87.23 +/- 0.32 | 33.28 +/- 0.69 | 80.62 +/- 0.27 | 76.70 +/- 0.30 | 3130 |
| White alone / Female | 72.80 +/- 0.10 | 66.93 +/- 0.41 | 88.15 +/- 0.34 | 43.88 +/- 0.51 | 68.58 +/- 0.18 | 81.34 +/- 0.30 | 3094 |
| Black or African American alone / Male | 80.61 +/- 0.42 | 55.21 +/- 0.63 | 88.73 +/- 0.54 | 26.37 +/- 0.88 | 74.34 +/- 0.59 | 88.36 +/- 0.45 | 327 |
| Black or African American alone / Female | 76.41 +/- 0.15 | 60.43 +/- 0.58 | 85.60 +/- 0.66 | 33.44 +/- 0.54 | 73.28 +/- 0.19 | 81.19 +/- 0.58 | 292 |
| American Indian alone / Male | 65.94 +/- 0.89 | 58.22 +/- 1.08 | 76.52 +/- 0.97 | 42.91 +/- 1.63 | 59.87 +/- 0.90 | 74.41 +/- 0.85 | 41 |
| American Indian alone / Female | 67.52 +/- 0.83 | 68.51 +/- 1.29 | 91.36 +/- 1.02 | 50.88 +/- 1.75 | 58.10 +/- 0.73 | 88.07 +/- 1.13 | 30 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 84.12 +/- 1.61 | 57.06 +/- 1.61 | 88.89 +/- 0.00 | 21.25 +/- 3.42 | 82.53 +/- 2.31 | 86.29 +/- 0.52 | 14 |
| AIAN specified/unspecified; no other races / Female | 76.60 +/- 2.13 | 72.34 +/- 2.13 | 89.66 +/- 3.45 | 44.44 +/- 0.00 | 76.45 +/- 0.69 | 77.29 +/- 5.96 | 12 |
| Asian alone / Male | 78.74 +/- 0.18 | 76.51 +/- 0.54 | 92.86 +/- 0.34 | 46.86 +/- 0.96 | 78.23 +/- 0.30 | 80.42 +/- 0.51 | 520 |
| Asian alone / Female | 72.02 +/- 0.14 | 74.00 +/- 0.51 | 92.56 +/- 0.42 | 52.15 +/- 0.64 | 67.63 +/- 0.18 | 84.54 +/- 0.58 | 697 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.53 +/- 1.07 | 71.76 +/- 1.07 | 92.73 +/- 1.66 | 33.33 +/- 0.00 | 83.60 +/- 0.25 | 83.43 +/- 3.13 | 14 |
| Native Hawaiian / Other Pacific Islander alone / Female | 81.36 +/- 1.02 | 70.91 +/- 1.02 | 96.00 +/- 0.00 | 37.89 +/- 2.35 | 76.94 +/- 1.08 | 92.18 +/- 0.29 | 12 |
| Some Other Race alone / Male | 81.71 +/- 0.42 | 76.95 +/- 0.63 | 93.33 +/- 0.20 | 42.63 +/- 1.60 | 82.10 +/- 0.53 | 80.42 +/- 0.30 | 358 |
| Some Other Race alone / Female | 66.58 +/- 0.17 | 78.14 +/- 0.39 | 92.64 +/- 0.32 | 62.15 +/- 0.53 | 62.17 +/- 0.15 | 82.34 +/- 0.51 | 364 |
| Two or More Races / Male | 78.35 +/- 0.27 | 71.58 +/- 0.64 | 92.23 +/- 0.49 | 41.72 +/- 0.98 | 76.18 +/- 0.36 | 83.84 +/- 0.68 | 158 |
| Two or More Races / Female | 73.33 +/- 0.22 | 75.63 +/- 0.46 | 92.48 +/- 0.32 | 52.71 +/- 0.73 | 70.47 +/- 0.24 | 82.22 +/- 0.50 | 145 |


#### dp_epsilon_1 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | -0.90 +/- 0.30 | +4.89 +/- 2.46 | +3.25 +/- 2.24 | +7.48 +/- 2.82 | -3.19 +/- 1.08 | +2.12 +/- 2.05 |
| White alone / Female | -1.51 +/- 0.24 | +4.85 +/- 2.18 | +3.21 +/- 1.87 | +6.63 +/- 2.51 | -2.69 +/- 0.91 | +1.95 +/- 1.44 |
| Black or African American alone / Male | -1.55 +/- 0.77 | +3.53 +/- 1.98 | +2.14 +/- 1.83 | +4.73 +/- 2.31 | -3.19 +/- 1.51 | +1.17 +/- 1.32 |
| Black or African American alone / Female | -0.51 +/- 0.26 | +4.76 +/- 2.25 | +4.11 +/- 2.33 | +5.45 +/- 2.18 | -2.47 +/- 0.88 | +2.73 +/- 1.74 |
| American Indian alone / Male | -3.96 +/- 2.21 | +3.56 +/- 3.54 | -0.43 +/- 3.22 | +6.91 +/- 4.71 | -4.33 +/- 2.55 | -2.48 +/- 2.21 |
| American Indian alone / Female | -2.38 +/- 1.50 | +4.75 +/- 1.29 | +2.73 +/- 1.02 | +6.32 +/- 2.35 | -2.47 +/- 1.30 | +1.69 +/- 1.24 |
| Alaska Native alone / Male | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| Alaska Native alone / Female | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races / Male | +1.76 +/- 3.35 | +4.12 +/- 1.61 | +5.56 +/- 3.93 | +2.50 +/- 3.42 | -0.79 +/- 2.77 | +4.91 +/- 3.97 |
| AIAN specified/unspecified; no other races / Female | -2.55 +/- 2.77 | +9.36 +/- 2.43 | +5.52 +/- 3.93 | +15.56 +/- 2.48 | -5.98 +/- 1.56 | +3.69 +/- 6.49 |
| Asian alone / Male | -1.22 +/- 0.30 | +4.58 +/- 2.36 | +2.61 +/- 1.70 | +8.17 +/- 3.59 | -2.67 +/- 1.11 | +2.68 +/- 2.18 |
| Asian alone / Female | -1.38 +/- 0.33 | +5.14 +/- 2.01 | +3.48 +/- 1.87 | +7.10 +/- 2.24 | -2.33 +/- 0.63 | +3.40 +/- 2.21 |
| Native Hawaiian / Other Pacific Islander alone / Male | -0.39 +/- 2.15 | +1.18 +/- 2.63 | +0.61 +/- 2.54 | +2.22 +/- 4.97 | -0.88 +/- 2.02 | +0.74 +/- 4.57 |
| Native Hawaiian / Other Pacific Islander alone / Female | -0.45 +/- 2.96 | +5.00 +/- 2.96 | +4.00 +/- 0.00 | +6.32 +/- 6.86 | -2.49 +/- 3.62 | +5.58 +/- 1.08 |
| Some Other Race alone / Male | -0.98 +/- 0.90 | +6.68 +/- 3.01 | +4.21 +/- 2.77 | +11.84 +/- 3.67 | -3.77 +/- 1.07 | +4.87 +/- 3.97 |
| Some Other Race alone / Female | -1.73 +/- 0.28 | +6.70 +/- 2.93 | +4.74 +/- 2.89 | +8.86 +/- 3.00 | -2.37 +/- 0.57 | +4.29 +/- 3.19 |
| Two or More Races / Male | -1.93 +/- 0.45 | +5.37 +/- 2.73 | +2.91 +/- 2.62 | +8.93 +/- 2.90 | -3.60 +/- 0.92 | +2.35 +/- 3.19 |
| Two or More Races / Female | -1.15 +/- 0.58 | +3.78 +/- 2.12 | +2.28 +/- 2.24 | +5.81 +/- 2.03 | -1.88 +/- 0.43 | +2.08 +/- 3.17 |


#### dp_epsilon_4: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 79.29 +/- 0.07 | 66.36 +/- 0.32 | 87.22 +/- 0.27 | 33.28 +/- 0.43 | 80.62 +/- 0.16 | 76.69 +/- 0.28 | 3142 |
| White alone / Female | 72.77 +/- 0.08 | 66.89 +/- 0.30 | 88.08 +/- 0.24 | 43.87 +/- 0.37 | 68.56 +/- 0.13 | 81.26 +/- 0.22 | 3105 |
| Black or African American alone / Male | 80.78 +/- 0.32 | 55.27 +/- 0.65 | 88.96 +/- 0.38 | 26.27 +/- 0.89 | 74.46 +/- 0.57 | 88.59 +/- 0.23 | 329 |
| Black or African American alone / Female | 76.60 +/- 0.21 | 60.40 +/- 0.34 | 85.76 +/- 0.39 | 33.22 +/- 0.42 | 73.45 +/- 0.22 | 81.40 +/- 0.38 | 294 |
| American Indian alone / Male | 66.14 +/- 0.83 | 58.02 +/- 1.13 | 76.52 +/- 0.97 | 42.55 +/- 1.63 | 60.08 +/- 0.86 | 74.53 +/- 0.78 | 41 |
| American Indian alone / Female | 67.72 +/- 0.89 | 68.32 +/- 0.99 | 91.36 +/- 1.02 | 50.53 +/- 1.47 | 58.27 +/- 0.72 | 88.14 +/- 1.26 | 31 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 85.88 +/- 2.46 | 57.65 +/- 1.61 | 91.11 +/- 3.04 | 20.00 +/- 2.80 | 83.68 +/- 2.10 | 88.95 +/- 3.59 | 14 |
| AIAN specified/unspecified; no other races / Female | 77.02 +/- 1.78 | 72.77 +/- 1.78 | 90.34 +/- 2.89 | 44.44 +/- 0.00 | 76.60 +/- 0.58 | 78.39 +/- 5.04 | 12 |
| Asian alone / Male | 78.79 +/- 0.11 | 76.69 +/- 0.30 | 93.04 +/- 0.23 | 47.04 +/- 0.48 | 78.20 +/- 0.14 | 80.76 +/- 0.40 | 525 |
| Asian alone / Female | 71.96 +/- 0.13 | 74.19 +/- 0.30 | 92.68 +/- 0.17 | 52.42 +/- 0.46 | 67.54 +/- 0.15 | 84.67 +/- 0.17 | 691 |
| Native Hawaiian / Other Pacific Islander alone / Male | 83.53 +/- 1.07 | 71.76 +/- 1.07 | 92.73 +/- 1.66 | 33.33 +/- 0.00 | 83.60 +/- 0.25 | 83.43 +/- 3.13 | 14 |
| Native Hawaiian / Other Pacific Islander alone / Female | 81.36 +/- 1.02 | 70.91 +/- 1.02 | 96.00 +/- 0.00 | 37.89 +/- 2.35 | 76.94 +/- 1.08 | 92.18 +/- 0.29 | 12 |
| Some Other Race alone / Male | 81.64 +/- 0.35 | 77.00 +/- 0.36 | 93.31 +/- 0.05 | 42.82 +/- 1.10 | 82.03 +/- 0.37 | 80.32 +/- 0.30 | 358 |
| Some Other Race alone / Female | 66.57 +/- 0.12 | 78.18 +/- 0.19 | 92.66 +/- 0.22 | 62.20 +/- 0.24 | 62.16 +/- 0.09 | 82.36 +/- 0.40 | 368 |
| Two or More Races / Male | 78.46 +/- 0.31 | 71.54 +/- 0.47 | 92.28 +/- 0.36 | 41.55 +/- 0.83 | 76.26 +/- 0.34 | 83.97 +/- 0.57 | 159 |
| Two or More Races / Female | 73.37 +/- 0.37 | 75.40 +/- 0.18 | 92.31 +/- 0.45 | 52.40 +/- 0.32 | 70.56 +/- 0.22 | 81.98 +/- 0.94 | 148 |


#### dp_epsilon_4 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | -0.90 +/- 0.29 | +4.88 +/- 2.56 | +3.24 +/- 2.32 | +7.48 +/- 2.94 | -3.19 +/- 1.12 | +2.11 +/- 2.14 |
| White alone / Female | -1.54 +/- 0.27 | +4.81 +/- 2.24 | +3.14 +/- 1.91 | +6.62 +/- 2.60 | -2.70 +/- 0.94 | +1.87 +/- 1.48 |
| Black or African American alone / Male | -1.39 +/- 0.69 | +3.58 +/- 2.05 | +2.37 +/- 1.70 | +4.63 +/- 2.44 | -3.06 +/- 1.58 | +1.40 +/- 1.14 |
| Black or African American alone / Female | -0.32 +/- 0.32 | +4.73 +/- 2.12 | +4.27 +/- 2.29 | +5.23 +/- 1.96 | -2.31 +/- 0.76 | +2.94 +/- 1.75 |
| American Indian alone / Male | -3.76 +/- 2.15 | +3.37 +/- 3.47 | -0.43 +/- 3.22 | +6.55 +/- 4.56 | -4.13 +/- 2.47 | -2.36 +/- 2.20 |
| American Indian alone / Female | -2.18 +/- 1.63 | +4.55 +/- 1.50 | +2.73 +/- 1.02 | +5.96 +/- 2.66 | -2.31 +/- 1.43 | +1.76 +/- 1.25 |
| Alaska Native alone / Male | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| Alaska Native alone / Female | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races / Male | +3.53 +/- 3.83 | +4.71 +/- 3.35 | +7.78 +/- 6.33 | +1.25 +/- 2.80 | +0.37 +/- 2.29 | +7.57 +/- 6.43 |
| AIAN specified/unspecified; no other races / Female | -2.13 +/- 1.50 | +9.79 +/- 2.43 | +6.21 +/- 2.89 | +15.56 +/- 2.48 | -5.84 +/- 1.00 | +4.79 +/- 4.43 |
| Asian alone / Male | -1.17 +/- 0.17 | +4.76 +/- 2.35 | +2.78 +/- 1.77 | +8.34 +/- 3.42 | -2.70 +/- 1.04 | +3.01 +/- 2.37 |
| Asian alone / Female | -1.44 +/- 0.32 | +5.33 +/- 2.19 | +3.60 +/- 1.90 | +7.37 +/- 2.57 | -2.42 +/- 0.75 | +3.53 +/- 2.16 |
| Native Hawaiian / Other Pacific Islander alone / Male | -0.39 +/- 2.15 | +1.18 +/- 2.63 | +0.61 +/- 2.54 | +2.22 +/- 4.97 | -0.88 +/- 2.02 | +0.74 +/- 4.57 |
| Native Hawaiian / Other Pacific Islander alone / Female | -0.45 +/- 2.96 | +5.00 +/- 2.96 | +4.00 +/- 0.00 | +6.32 +/- 6.86 | -2.49 +/- 3.62 | +5.58 +/- 1.08 |
| Some Other Race alone / Male | -1.05 +/- 0.62 | +6.73 +/- 3.25 | +4.19 +/- 2.79 | +12.03 +/- 4.26 | -3.84 +/- 1.26 | +4.77 +/- 3.79 |
| Some Other Race alone / Female | -1.74 +/- 0.20 | +6.73 +/- 3.02 | +4.76 +/- 2.86 | +8.91 +/- 3.21 | -2.38 +/- 0.64 | +4.31 +/- 3.06 |
| Two or More Races / Male | -1.82 +/- 0.38 | +5.33 +/- 2.67 | +2.97 +/- 2.51 | +8.76 +/- 2.92 | -3.51 +/- 0.95 | +2.49 +/- 2.98 |
| Two or More Races / Female | -1.12 +/- 0.46 | +3.55 +/- 2.05 | +2.11 +/- 1.96 | +5.50 +/- 2.27 | -1.80 +/- 0.58 | +1.85 +/- 2.58 |


#### dp_epsilon_8: levels (%)

| Group | Accuracy | Selection | TPR | FPR | PPV | NPV | Min conditional n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | 79.32 +/- 0.12 | 66.23 +/- 0.50 | 87.13 +/- 0.35 | 33.07 +/- 0.76 | 80.70 +/- 0.30 | 76.62 +/- 0.31 | 3144 |
| White alone / Female | 72.79 +/- 0.14 | 66.80 +/- 0.42 | 88.02 +/- 0.31 | 43.75 +/- 0.56 | 68.61 +/- 0.21 | 81.21 +/- 0.27 | 3105 |
| Black or African American alone / Male | 80.70 +/- 0.36 | 55.24 +/- 0.79 | 88.84 +/- 0.70 | 26.32 +/- 0.97 | 74.40 +/- 0.60 | 88.48 +/- 0.54 | 328 |
| Black or African American alone / Female | 76.46 +/- 0.16 | 60.37 +/- 0.41 | 85.60 +/- 0.45 | 33.33 +/- 0.44 | 73.35 +/- 0.19 | 81.21 +/- 0.40 | 295 |
| American Indian alone / Male | 66.34 +/- 0.70 | 57.82 +/- 1.13 | 76.52 +/- 0.97 | 42.18 +/- 1.52 | 60.28 +/- 0.75 | 74.65 +/- 0.68 | 41 |
| American Indian alone / Female | 67.72 +/- 0.54 | 68.32 +/- 1.21 | 91.36 +/- 1.02 | 50.53 +/- 1.47 | 58.27 +/- 0.52 | 88.15 +/- 1.05 | 30 |
| Alaska Native alone / Male | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| Alaska Native alone / Female | 100.00 +/- 0.00 | 0.00 +/- 0.00 | undefined | 0.00 +/- 0.00 | undefined | 100.00 +/- 0.00 | 0 |
| AIAN specified/unspecified; no other races / Male | 85.29 +/- 2.08 | 57.06 +/- 1.61 | 90.00 +/- 2.48 | 20.00 +/- 2.80 | 83.53 +/- 2.00 | 87.71 +/- 2.90 | 14 |
| AIAN specified/unspecified; no other races / Female | 76.60 +/- 2.13 | 72.34 +/- 2.13 | 89.66 +/- 3.45 | 44.44 +/- 0.00 | 76.45 +/- 0.69 | 77.29 +/- 5.96 | 12 |
| Asian alone / Male | 78.73 +/- 0.12 | 76.66 +/- 0.60 | 92.97 +/- 0.44 | 47.08 +/- 0.91 | 78.17 +/- 0.26 | 80.61 +/- 0.72 | 520 |
| Asian alone / Female | 72.01 +/- 0.07 | 74.10 +/- 0.58 | 92.64 +/- 0.53 | 52.28 +/- 0.65 | 67.59 +/- 0.16 | 84.65 +/- 0.74 | 694 |
| Native Hawaiian / Other Pacific Islander alone / Male | 82.75 +/- 0.88 | 70.98 +/- 0.88 | 91.52 +/- 1.36 | 33.33 +/- 0.00 | 83.42 +/- 0.20 | 81.14 +/- 2.56 | 14 |
| Native Hawaiian / Other Pacific Islander alone / Female | 81.36 +/- 1.02 | 70.91 +/- 1.02 | 96.00 +/- 0.00 | 37.89 +/- 2.35 | 76.94 +/- 1.08 | 92.18 +/- 0.29 | 12 |
| Some Other Race alone / Male | 81.79 +/- 0.34 | 76.88 +/- 0.51 | 93.33 +/- 0.15 | 42.40 +/- 1.30 | 82.18 +/- 0.43 | 80.49 +/- 0.20 | 361 |
| Some Other Race alone / Female | 66.65 +/- 0.20 | 78.02 +/- 0.36 | 92.59 +/- 0.22 | 61.96 +/- 0.56 | 62.24 +/- 0.18 | 82.33 +/- 0.32 | 369 |
| Two or More Races / Male | 78.49 +/- 0.38 | 71.44 +/- 0.50 | 92.23 +/- 0.33 | 41.37 +/- 0.99 | 76.33 +/- 0.41 | 83.91 +/- 0.53 | 159 |
| Two or More Races / Female | 73.43 +/- 0.27 | 75.47 +/- 0.25 | 92.42 +/- 0.32 | 52.40 +/- 0.42 | 70.58 +/- 0.19 | 82.20 +/- 0.65 | 148 |


#### dp_epsilon_8 minus matched SGD (pp)

| Group | Accuracy change | Selection change | TPR change | FPR change | PPV change | NPV change |
| --- | --- | --- | --- | --- | --- | --- |
| White alone / Male | -0.88 +/- 0.27 | +4.75 +/- 2.56 | +3.16 +/- 2.30 | +7.28 +/- 2.99 | -3.11 +/- 1.15 | +2.04 +/- 2.09 |
| White alone / Female | -1.51 +/- 0.28 | +4.71 +/- 2.30 | +3.08 +/- 1.95 | +6.50 +/- 2.68 | -2.66 +/- 0.97 | +1.82 +/- 1.50 |
| Black or African American alone / Male | -1.47 +/- 0.68 | +3.56 +/- 1.94 | +2.25 +/- 1.77 | +4.68 +/- 2.23 | -3.12 +/- 1.43 | +1.29 +/- 1.25 |
| Black or African American alone / Female | -0.45 +/- 0.33 | +4.71 +/- 2.05 | +4.11 +/- 2.10 | +5.34 +/- 2.05 | -2.41 +/- 0.87 | +2.75 +/- 1.55 |
| American Indian alone / Male | -3.56 +/- 1.80 | +3.17 +/- 3.39 | -0.43 +/- 3.22 | +6.18 +/- 4.19 | -3.92 +/- 2.16 | -2.24 +/- 2.04 |
| American Indian alone / Female | -2.18 +/- 1.08 | +4.55 +/- 0.89 | +2.73 +/- 1.02 | +5.96 +/- 1.57 | -2.31 +/- 0.94 | +1.77 +/- 1.13 |
| Alaska Native alone / Male | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| Alaska Native alone / Female | +0.00 +/- 0.00 | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 | undefined | +0.00 +/- 0.00 |
| AIAN specified/unspecified; no other races / Male | +2.94 +/- 2.08 | +4.12 +/- 1.61 | +6.67 +/- 2.48 | +1.25 +/- 2.80 | +0.21 +/- 2.03 | +6.34 +/- 2.24 |
| AIAN specified/unspecified; no other races / Female | -2.55 +/- 1.78 | +9.36 +/- 2.43 | +5.52 +/- 3.08 | +15.56 +/- 2.48 | -5.98 +/- 1.12 | +3.69 +/- 5.10 |
| Asian alone / Male | -1.23 +/- 0.26 | +4.73 +/- 2.29 | +2.71 +/- 1.74 | +8.39 +/- 3.34 | -2.73 +/- 1.02 | +2.86 +/- 2.35 |
| Asian alone / Female | -1.39 +/- 0.27 | +5.25 +/- 2.03 | +3.56 +/- 1.83 | +7.22 +/- 2.30 | -2.36 +/- 0.65 | +3.51 +/- 2.13 |
| Native Hawaiian / Other Pacific Islander alone / Male | -1.18 +/- 2.24 | +0.39 +/- 1.64 | -0.61 +/- 1.36 | +2.22 +/- 4.97 | -1.06 +/- 2.15 | -1.55 +/- 3.07 |
| Native Hawaiian / Other Pacific Islander alone / Female | -0.45 +/- 2.96 | +5.00 +/- 2.96 | +4.00 +/- 0.00 | +6.32 +/- 6.86 | -2.49 +/- 3.62 | +5.58 +/- 1.08 |
| Some Other Race alone / Male | -0.90 +/- 0.76 | +6.60 +/- 3.10 | +4.21 +/- 2.76 | +11.61 +/- 3.90 | -3.69 +/- 1.14 | +4.94 +/- 3.84 |
| Some Other Race alone / Female | -1.66 +/- 0.27 | +6.58 +/- 2.87 | +4.69 +/- 2.88 | +8.66 +/- 2.87 | -2.31 +/- 0.52 | +4.27 +/- 3.23 |
| Two or More Races / Male | -1.79 +/- 0.55 | +5.23 +/- 2.56 | +2.91 +/- 2.42 | +8.58 +/- 2.86 | -3.45 +/- 0.96 | +2.42 +/- 2.88 |
| Two or More Races / Female | -1.05 +/- 0.60 | +3.61 +/- 2.29 | +2.22 +/- 2.36 | +5.50 +/- 2.30 | -1.77 +/- 0.52 | +2.07 +/- 3.34 |


## Source tables and reproduction

Levels: `aggregate/group_seed_metrics.csv` and `aggregate/group_mean_std.csv`. Paired changes: `aggregate/group_rate_changes_seed.csv` and `aggregate/group_rate_changes_mean_std.csv`. Accuracy losses: `aggregate/group_accuracy_losses_mean_std.csv`. All raw rate/change CSV values are fractions except explicitly named `_pp` columns. Mean fairness gaps in FINDINGS are computed per seed, not from rounded table entries. Regenerate with `scripts/run_controlled.py --stage summarize`; no retraining or test evaluation is needed.
