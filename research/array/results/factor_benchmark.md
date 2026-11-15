
### spike_ecoli  (635 positives, 60 core, 4313 negatives; high tier: recall 0.559 with 7 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.979 | 0.763 | 0.405 | 0.564 | 0.701 |  |  |
| rules | 0.979 | 0.772 | 0.583 | 0.654 | 0.654 |  |  |
| ai_all | 0.983 | 0.788 | 0.405 | 0.564 | 0.701 | +0.005 … +0.028 | -0.258 … -0.025 |
| factor.own | 0.957 | 0.227 | 0.050 | 0.083 | 0.098 | -0.576 … -0.511 | -0.611 … -0.422 |
| factor.home | 0.954 | 0.290 | 0.054 | 0.080 | 0.110 | -0.517 … -0.444 | -0.608 … -0.419 |
| place.node | 0.983 | 0.801 | 0.546 | 0.652 | 0.737 | +0.010 … +0.047 | -0.154 … +0.058 |
| core genes allowed through | rule tier recall | ai | rules | ai_all | factor.own | factor.home | place.node |
|---|---|---|---|---|---|---|---|
| 0 of 60 (high) | 0.559 | 0.761 | 0.761 | 0.792 | 0.279 | 0.411 | 0.784 |
| 0 of 60 (high or medium) | 0.654 | 0.761 | 0.761 | 0.792 | 0.279 | 0.411 | 0.784 |
| 0 of 60 (any call) | 0.761 | 0.761 | 0.761 | 0.792 | 0.279 | 0.411 | 0.784 |
- recall at high-tier FP by divergence, ai: 0.0: 0.73, 0.3: 0.52, 0.5: 0.28, 0.7: 0.08
- recall at high-tier FP by divergence, rules: 0.0: 0.80, 0.3: 0.71, 0.5: 0.55, 0.7: 0.27
- recall at high-tier FP by divergence, ai_all: 0.0: 0.73, 0.3: 0.52, 0.5: 0.28, 0.7: 0.08
- recall at high-tier FP by divergence, factor.own: 0.0: 0.06, 0.3: 0.08, 0.5: 0.05, 0.7: 0.01
- recall at high-tier FP by divergence, factor.home: 0.0: 0.06, 0.3: 0.07, 0.5: 0.06, 0.7: 0.02
- recall at high-tier FP by divergence, place.node: 0.0: 0.77, 0.3: 0.64, 0.5: 0.50, 0.7: 0.27

### spike_entamoeba  (635 positives, 163 core, 7919 negatives; high tier: recall 0.583 with 287 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.960 | 0.550 | 0.598 | 0.728 | 0.302 |  |  |
| rules | 0.958 | 0.546 | 0.589 | 0.685 | 0.301 |  |  |
| ai_all | 0.963 | 0.553 | 0.598 | 0.728 | 0.302 | -0.004 … +0.020 | -0.036 … +0.047 |
| factor.own | 0.724 | 0.101 | 0.069 | 0.088 | 0.030 | -0.479 … -0.408 | -0.588 … -0.463 |
| factor.home | 0.769 | 0.134 | 0.077 | 0.156 | 0.028 | -0.445 … -0.376 | -0.577 … -0.450 |
| place.node | 0.959 | 0.718 | 0.751 | 0.783 | 0.569 | +0.142 … +0.202 | +0.097 … +0.206 |
| core genes allowed through | rule tier recall | ai | rules | ai_all | factor.own | factor.home | place.node |
|---|---|---|---|---|---|---|---|
| 1 of 163 (high) | 0.583 | 0.757 | 0.685 | 0.757 | 0.005 | 0.014 | 0.764 |
| 11 of 163 (high or medium) | 0.685 | 0.802 | 0.794 | 0.802 | 0.035 | 0.041 | 0.800 |
| 21 of 163 (any call) | 0.811 | 0.811 | 0.811 | 0.811 | 0.061 | 0.091 | 0.813 |
- recall at high-tier FP by divergence, ai: 0.0: 0.89, 0.3: 0.80, 0.5: 0.53, 0.7: 0.18
- recall at high-tier FP by divergence, rules: 0.0: 0.83, 0.3: 0.74, 0.5: 0.55, 0.7: 0.23
- recall at high-tier FP by divergence, ai_all: 0.0: 0.89, 0.3: 0.80, 0.5: 0.53, 0.7: 0.18
- recall at high-tier FP by divergence, factor.own: 0.0: 0.11, 0.3: 0.09, 0.5: 0.05, 0.7: 0.02
- recall at high-tier FP by divergence, factor.home: 0.0: 0.10, 0.3: 0.13, 0.5: 0.06, 0.7: 0.02
- recall at high-tier FP by divergence, place.node: 0.0: 0.94, 0.3: 0.89, 0.5: 0.74, 0.7: 0.44

### transplant_legionella  (240 positives, 48 core, 3036 negatives; high tier: recall 0.721 with 22 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.983 | 0.864 | 0.750 | 0.867 | 0.808 |  |  |
| rules | 0.983 | 0.859 | 0.787 | 0.858 | 0.817 |  |  |
| ai_all | 0.988 | 0.873 | 0.750 | 0.867 | 0.808 | +0.004 … +0.025 | -0.075 … +0.092 |
| factor.own | 0.953 | 0.161 | 0.058 | 0.087 | 0.067 | -0.743 … -0.640 | -0.794 … -0.567 |
| factor.home | 0.967 | 0.286 | 0.108 | 0.163 | 0.129 | -0.627 … -0.508 | -0.742 … -0.510 |
| place.node | 0.990 | 0.917 | 0.871 | 0.900 | 0.875 | +0.032 … +0.091 | +0.029 … +0.240 |
| core genes allowed through | rule tier recall | ai | rules | ai_all | factor.own | factor.home | place.node |
|---|---|---|---|---|---|---|---|
| 0 of 48 (high) | 0.721 | 0.900 | 0.900 | 0.921 | 0.392 | 0.625 | 0.938 |
| 0 of 48 (high or medium) | 0.854 | 0.900 | 0.900 | 0.921 | 0.392 | 0.625 | 0.938 |
| 0 of 48 (any call) | 0.900 | 0.900 | 0.900 | 0.921 | 0.392 | 0.625 | 0.938 |
- recall at high-tier FP by divergence, ai: 0.0: 0.69, 0.5: 0.81, 1.0: 0.75
- recall at high-tier FP by divergence, rules: 0.0: 0.72, 0.5: 0.83, 1.0: 0.81
- recall at high-tier FP by divergence, ai_all: 0.0: 0.69, 0.5: 0.81, 1.0: 0.75
- recall at high-tier FP by divergence, factor.own: 0.0: 0.01, 0.5: 0.04, 1.0: 0.11
- recall at high-tier FP by divergence, factor.home: 0.0: 0.12, 0.5: 0.09, 1.0: 0.11
- recall at high-tier FP by divergence, place.node: 0.0: 0.80, 0.5: 0.94, 1.0: 0.88
