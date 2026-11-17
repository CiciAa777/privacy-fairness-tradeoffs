## Overall (percent; loss in percentage points)

| method | epsilon | accuracy | balanced_accuracy | accuracy_loss_pp |
| --- | --- | --- | --- | --- |
| SGD | undefined | 77.08 | 75.73 | 0.0 |
| DP-SGD | 0.999 | 75.82 | 73.67 | 1.26 |
| DP-SGD | 3.999 | 75.81 | 73.69 | 1.27 |
| DP-SGD | 8.0 | 75.78 | 73.63 | 1.3 |

## Fairness gaps (percentage points; undefined = a required group rate is undefined)

| method | target_epsilon | SEX_accuracy_gap | SEX_independence | SEX_separation | SEX_sufficiency | RAC1P_accuracy_gap | RAC1P_independence | RAC1P_separation | RAC1P_sufficiency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SGD | undefined | 6.67 | 0.64 | 12.01 | 12.58 | 31.19 | 69.46 | undefined | undefined |
| DP-SGD | 1.0 | 7.38 | 0.78 | 11.38 | 12.12 | 32.18 | 76.93 | undefined | undefined |
| DP-SGD | 4.0 | 7.33 | 0.91 | 11.51 | 12.16 | 32.18 | 77.17 | undefined | undefined |
| DP-SGD | 8.0 | 7.43 | 0.87 | 11.54 | 12.19 | 32.18 | 77.02 | undefined | undefined |

## Group accuracy loss (pp) by target epsilon

| attribute | group | n | 1.0 | 4.0 | 8.0 | name | flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RAC1P | 1 | 18889 | 1.24 | 1.18 | 1.23 | White alone |  |
| RAC1P | 2 | 1500 | 1.07 | 1.27 | 1.4 | Black alone |  |
| RAC1P | 3 | 202 | 0.99 | 0.99 | 0.99 | American Indian alone |  |
| RAC1P | 4 | 2 | 0.0 | 0.0 | 0.0 | Alaska Native alone | n<100: unstable |
| RAC1P | 5 | 81 | 1.23 | 1.23 | 1.23 | Other AIAN category | n<100: unstable |
| RAC1P | 6 | 5017 | 1.63 | 1.69 | 1.65 | Asian alone |  |
| RAC1P | 7 | 95 | 3.16 | 3.16 | 3.16 | NHPI alone | n<100: unstable |
| RAC1P | 8 | 3299 | 0.67 | 0.97 | 0.88 | Some other race alone |  |
| RAC1P | 9 | 1179 | 1.7 | 1.61 | 1.78 | Two or more races |  |
| SEX | 1 | 14826 | 0.9 | 0.93 | 0.91 | Male |  |
| SEX | 2 | 15438 | 1.61 | 1.59 | 1.67 | Female |  |

## Privacy settings

| target_epsilon | epsilon | delta | noise_multiplier | sample_rate | steps | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.99916 | 1e-06 | 1.411133 | 0.003378 | 5920 | 48.188988 |
| 4.0 | 3.999411 | 1e-06 | 0.740509 | 0.003378 | 5920 | 78.737063 |
| 8.0 | 7.999651 | 1e-06 | 0.597458 | 0.003378 | 5920 | 67.610846 |
