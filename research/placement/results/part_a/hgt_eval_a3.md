
### spike_ecoli  (635 positives, 60 core, 4313 negatives; high tier: recall 0.559 with 7 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.979 | 0.763 | 0.405 | 0.564 | 0.701 |  |  |
| rules | 0.979 | 0.772 | 0.583 | 0.654 | 0.654 |  |  |
| gbm_hits | 0.983 | 0.776 | 0.323 | 0.561 | 0.672 | -0.008 … +0.019 | -0.368 … -0.100 |
| rule_ai_near | 0.983 | 0.730 | 0.143 | 0.342 | 0.474 | -0.071 … -0.015 | -0.590 … -0.313 |
| rule_hidx_near | 0.983 | 0.707 | 0.128 | 0.211 | 0.348 | -0.094 … -0.036 | -0.540 … -0.341 |
| rule_lca_adapt | 0.980 | 0.679 | 0.000 | 0.000 | 0.000 | -0.125 … -0.063 | -0.654 … -0.476 |
| place_disc | 0.982 | 0.675 | 0.087 | 0.209 | 0.405 | -0.132 … -0.069 | -0.576 … -0.398 |
| place_node | 0.983 | 0.788 | 0.394 | 0.562 | 0.672 | -0.005 … +0.033 | -0.269 … -0.055 |
| place_p1 | 0.982 | 0.707 | 0.290 | 0.372 | 0.391 | -0.091 … -0.043 | -0.369 … -0.175 |
| place_p2 | 0.982 | 0.693 | 0.246 | 0.296 | 0.356 | -0.108 … -0.055 | -0.431 … -0.249 |
| place_top | 0.982 | 0.703 | 0.028 | 0.331 | 0.521 | -0.108 … -0.036 | -0.628 … -0.438 |
| place_mean5 | 0.982 | 0.779 | 0.356 | 0.688 | 0.754 | -0.019 … +0.027 | -0.388 … -0.105 |
| core genes allowed through | rule tier recall | ai | rules | gbm_hits | rule_ai_near | rule_hidx_near | rule_lca_adapt | place_disc | place_node | place_p1 | place_p2 | place_top | place_mean5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 of 60 (high) | 0.559 | 0.761 | 0.761 | 0.794 | 0.802 | 0.802 | 0.775 | 0.688 | 0.789 | 0.704 | 0.682 | 0.794 | 0.773 |
| 0 of 60 (high or medium) | 0.654 | 0.761 | 0.761 | 0.794 | 0.802 | 0.802 | 0.775 | 0.688 | 0.789 | 0.704 | 0.682 | 0.794 | 0.773 |
| 0 of 60 (any call) | 0.761 | 0.761 | 0.761 | 0.794 | 0.802 | 0.802 | 0.775 | 0.688 | 0.789 | 0.704 | 0.682 | 0.794 | 0.773 |
- recall at high-tier FP by divergence, rules: 0.0: 0.80, 0.3: 0.71, 0.5: 0.55, 0.7: 0.27
- recall at high-tier FP by divergence, gbm_hits: 0.0: 0.60, 0.3: 0.35, 0.5: 0.26, 0.7: 0.08
- recall at high-tier FP by divergence, place_disc: 0.0: 0.17, 0.3: 0.09, 0.5: 0.09, 0.7: 0.00
- recall at high-tier FP by divergence, place_node: 0.0: 0.66, 0.3: 0.46, 0.5: 0.33, 0.7: 0.12
- recall at high-tier FP by divergence, rule_ai_near: 0.0: 0.37, 0.3: 0.16, 0.5: 0.04, 0.7: 0.00
- genes with an exact token set (plus genes without hits): 593 positives, 4106 negatives; AP vs all / recall at high-tier FP: ai 0.744 / 0.371, rules 0.755 / 0.555, gbm_hits 0.759 / 0.307, rule_ai_near 0.705 / 0.118, place_disc 0.652 / 0.088, place_node 0.769 / 0.354

### spike_entamoeba  (635 positives, 163 core, 7919 negatives; high tier: recall 0.583 with 287 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.960 | 0.550 | 0.598 | 0.728 | 0.302 |  |  |
| rules | 0.958 | 0.546 | 0.589 | 0.685 | 0.301 |  |  |
| gbm_hits | 0.959 | 0.706 | 0.726 | 0.762 | 0.580 | +0.126 … +0.189 | +0.070 … +0.185 |
| rule_ai_near | 0.898 | 0.335 | 0.323 | 0.443 | 0.148 | -0.234 … -0.177 | -0.332 … -0.213 |
| rule_hidx_near | 0.831 | 0.412 | 0.350 | 0.441 | 0.213 | -0.169 … -0.095 | -0.310 … -0.183 |
| rule_lca_adapt | 0.959 | 0.466 | 0.392 | 0.392 | 0.000 | -0.119 … -0.034 | -0.276 … -0.131 |
| place_disc | 0.962 | 0.721 | 0.773 | 0.792 | 0.710 | +0.141 … +0.212 | +0.119 … +0.235 |
| place_node | 0.960 | 0.718 | 0.726 | 0.769 | 0.603 | +0.141 … +0.202 | +0.076 … +0.187 |
| place_p1 | 0.962 | 0.701 | 0.694 | 0.764 | 0.573 | +0.123 … +0.186 | +0.044 … +0.155 |
| place_p2 | 0.961 | 0.753 | 0.775 | 0.794 | 0.685 | +0.172 … +0.241 | +0.120 … +0.239 |
| place_top | 0.962 | 0.730 | 0.781 | 0.792 | 0.707 | +0.152 … +0.221 | +0.124 … +0.247 |
| place_mean5 | 0.962 | 0.722 | 0.776 | 0.795 | 0.687 | +0.141 … +0.214 | +0.119 … +0.240 |
| core genes allowed through | rule tier recall | ai | rules | gbm_hits | rule_ai_near | rule_hidx_near | rule_lca_adapt | place_disc | place_node | place_p1 | place_p2 | place_top | place_mean5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 of 163 (high) | 0.583 | 0.757 | 0.685 | 0.595 | 0.222 | 0.098 | 0.806 | 0.795 | 0.775 | 0.794 | 0.770 | 0.795 | 0.806 |
| 11 of 163 (high or medium) | 0.685 | 0.802 | 0.794 | 0.750 | 0.317 | 0.145 | 0.806 | 0.808 | 0.798 | 0.811 | 0.802 | 0.808 | 0.813 |
| 21 of 163 (any call) | 0.811 | 0.811 | 0.811 | 0.776 | 0.439 | 0.165 | 0.806 | 0.813 | 0.806 | 0.814 | 0.808 | 0.814 | 0.813 |
- recall at high-tier FP by divergence, rules: 0.0: 0.83, 0.3: 0.74, 0.5: 0.55, 0.7: 0.23
- recall at high-tier FP by divergence, gbm_hits: 0.0: 0.88, 0.3: 0.85, 0.5: 0.74, 0.7: 0.43
- recall at high-tier FP by divergence, place_disc: 0.0: 0.94, 0.3: 0.90, 0.5: 0.79, 0.7: 0.46
- recall at high-tier FP by divergence, place_node: 0.0: 0.93, 0.3: 0.89, 0.5: 0.70, 0.7: 0.39
- recall at high-tier FP by divergence, rule_ai_near: 0.0: 0.67, 0.3: 0.40, 0.5: 0.19, 0.7: 0.04
- genes with an exact token set (plus genes without hits): 597 positives, 7879 negatives; AP vs all / recall at high-tier FP: ai 0.518 / 0.576, rules 0.515 / 0.566, gbm_hits 0.685 / 0.710, rule_ai_near 0.300 / 0.296, place_disc 0.704 / 0.760, place_node 0.694 / 0.707

### transplant_legionella  (240 positives, 48 core, 3036 negatives; high tier: recall 0.721 with 22 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.983 | 0.864 | 0.750 | 0.867 | 0.808 |  |  |
| rules | 0.983 | 0.859 | 0.787 | 0.858 | 0.817 |  |  |
| gbm_hits | 0.988 | 0.886 | 0.754 | 0.887 | 0.825 | +0.005 … +0.056 | -0.071 … +0.136 |
| rule_ai_near | 0.990 | 0.544 | 0.000 | 0.408 | 0.350 | -0.371 … -0.258 | -0.842 … -0.314 |
| rule_hidx_near | 0.990 | 0.668 | 0.400 | 0.492 | 0.438 | -0.234 … -0.134 | -0.461 … -0.194 |
| rule_lca_adapt | 0.986 | 0.594 | 0.167 | 0.346 | 0.167 | -0.313 … -0.212 | -0.679 … -0.435 |
| place_disc | 0.989 | 0.907 | 0.842 | 0.875 | 0.854 | +0.021 … +0.082 | +0.002 … +0.221 |
| place_node | 0.990 | 0.916 | 0.863 | 0.896 | 0.863 | +0.030 … +0.092 | +0.029 … +0.233 |
| place_p1 | 0.989 | 0.900 | 0.825 | 0.858 | 0.833 | +0.015 … +0.076 | -0.019 … +0.196 |
| place_p2 | 0.989 | 0.895 | 0.804 | 0.846 | 0.842 | +0.008 … +0.070 | -0.042 … +0.196 |
| place_top | 0.990 | 0.920 | 0.863 | 0.900 | 0.887 | +0.035 … +0.094 | +0.025 … +0.237 |
| place_mean5 | 0.988 | 0.909 | 0.867 | 0.883 | 0.871 | +0.022 … +0.082 | +0.033 … +0.238 |
| core genes allowed through | rule tier recall | ai | rules | gbm_hits | rule_ai_near | rule_hidx_near | rule_lca_adapt | place_disc | place_node | place_p1 | place_p2 | place_top | place_mean5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 of 48 (high) | 0.721 | 0.900 | 0.900 | 0.929 | 0.938 | 0.938 | 0.917 | 0.908 | 0.933 | 0.854 | 0.846 | 0.938 | 0.917 |
| 0 of 48 (high or medium) | 0.854 | 0.900 | 0.900 | 0.929 | 0.938 | 0.938 | 0.917 | 0.908 | 0.933 | 0.854 | 0.846 | 0.938 | 0.917 |
| 0 of 48 (any call) | 0.900 | 0.900 | 0.900 | 0.929 | 0.938 | 0.938 | 0.917 | 0.908 | 0.933 | 0.854 | 0.846 | 0.938 | 0.917 |
- recall at high-tier FP by divergence, rules: 0.0: 0.72, 0.5: 0.83, 1.0: 0.81
- recall at high-tier FP by divergence, gbm_hits: 0.0: 0.72, 0.5: 0.78, 1.0: 0.76
- recall at high-tier FP by divergence, place_disc: 0.0: 0.81, 0.5: 0.94, 1.0: 0.79
- recall at high-tier FP by divergence, place_node: 0.0: 0.81, 0.5: 0.94, 1.0: 0.84
- recall at high-tier FP by divergence, rule_ai_near: 0.0: 0.00, 0.5: 0.00, 1.0: 0.00
- genes with an exact token set (plus genes without hits): 201 positives, 2915 negatives; AP vs all / recall at high-tier FP: ai 0.830 / 0.711, rules 0.823 / 0.751, gbm_hits 0.858 / 0.716, rule_ai_near 0.469 / 0.000, place_disc 0.886 / 0.821, place_node 0.895 / 0.836
