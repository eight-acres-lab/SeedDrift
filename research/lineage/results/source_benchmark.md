
### spike_ecoli  (635 positives, 60 core, 4313 negatives; high tier: recall 0.559 with 7 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.979 | 0.763 | 0.405 | 0.564 | 0.701 |  |  |
| rules | 0.979 | 0.772 | 0.583 | 0.654 | 0.654 |  |  |
| ai_all | 0.983 | 0.788 | 0.405 | 0.564 | 0.701 | +0.005 … +0.028 | -0.258 … -0.025 |
| lineage.plm | 0.993 | 0.762 | 0.055 | 0.142 | 0.302 | -0.047 … +0.039 | -0.602 … -0.405 |
| lineage.raw_plm | 0.994 | 0.871 | 0.386 | 0.513 | 0.602 | +0.065 … +0.138 | -0.313 … -0.049 |
| lineage.composition | 0.970 | 0.379 | 0.025 | 0.049 | 0.069 | -0.438 … -0.346 | -0.632 … -0.446 |
| lineage.raw_composition | 0.948 | 0.343 | 0.016 | 0.038 | 0.079 | -0.474 … -0.380 | -0.640 … -0.456 |
| place.node | 0.983 | 0.801 | 0.546 | 0.652 | 0.737 | +0.010 … +0.047 | -0.154 … +0.058 |
| core genes allowed through | rule tier recall | ai | rules | ai_all | lineage.plm | lineage.raw_plm | lineage.composition | lineage.raw_composition | place.node |
|---|---|---|---|---|---|---|---|---|---|
| 0 of 60 (high) | 0.559 | 0.761 | 0.761 | 0.792 | 0.287 | 0.587 | 0.324 | 0.017 | 0.784 |
| 0 of 60 (high or medium) | 0.654 | 0.761 | 0.761 | 0.792 | 0.287 | 0.587 | 0.324 | 0.017 | 0.784 |
| 0 of 60 (any call) | 0.761 | 0.761 | 0.761 | 0.792 | 0.287 | 0.587 | 0.324 | 0.017 | 0.784 |
- recall at high-tier FP by divergence, ai: 0.0: 0.73, 0.3: 0.52, 0.5: 0.28, 0.7: 0.08
- recall at high-tier FP by divergence, rules: 0.0: 0.80, 0.3: 0.71, 0.5: 0.55, 0.7: 0.27
- recall at high-tier FP by divergence, ai_all: 0.0: 0.73, 0.3: 0.52, 0.5: 0.28, 0.7: 0.08
- recall at high-tier FP by divergence, lineage.plm: 0.0: 0.07, 0.3: 0.06, 0.5: 0.07, 0.7: 0.02
- recall at high-tier FP by divergence, lineage.raw_plm: 0.0: 0.14, 0.3: 0.30, 0.5: 0.49, 0.7: 0.62
- recall at high-tier FP by divergence, lineage.composition: 0.0: 0.07, 0.3: 0.03, 0.5: 0.00, 0.7: 0.00
- recall at high-tier FP by divergence, lineage.raw_composition: 0.0: 0.02, 0.3: 0.03, 0.5: 0.01, 0.7: 0.01
- recall at high-tier FP by divergence, place.node: 0.0: 0.77, 0.3: 0.64, 0.5: 0.50, 0.7: 0.27

### spike_entamoeba  (635 positives, 163 core, 7919 negatives; high tier: recall 0.583 with 287 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.960 | 0.550 | 0.598 | 0.728 | 0.302 |  |  |
| rules | 0.958 | 0.546 | 0.589 | 0.685 | 0.301 |  |  |
| ai_all | 0.963 | 0.553 | 0.598 | 0.728 | 0.302 | -0.004 … +0.020 | -0.036 … +0.047 |
| lineage.plm | 0.942 | 0.703 | 0.696 | 0.805 | 0.438 | +0.112 … +0.203 | +0.031 … +0.176 |
| lineage.raw_plm | 0.751 | 0.331 | 0.230 | 0.293 | 0.128 | -0.254 … -0.172 | -0.430 … -0.297 |
| lineage.composition | 0.961 | 0.793 | 0.805 | 0.852 | 0.592 | +0.203 … +0.291 | +0.139 … +0.278 |
| lineage.raw_composition | 0.907 | 0.706 | 0.676 | 0.746 | 0.485 | +0.104 … +0.207 | +0.009 … +0.157 |
| place.node | 0.959 | 0.718 | 0.751 | 0.783 | 0.569 | +0.142 … +0.202 | +0.097 … +0.206 |
| core genes allowed through | rule tier recall | ai | rules | ai_all | lineage.plm | lineage.raw_plm | lineage.composition | lineage.raw_composition | place.node |
|---|---|---|---|---|---|---|---|---|---|
| 1 of 163 (high) | 0.583 | 0.757 | 0.685 | 0.757 | 0.104 | 0.072 | 0.121 | 0.186 | 0.764 |
| 11 of 163 (high or medium) | 0.685 | 0.802 | 0.794 | 0.802 | 0.471 | 0.115 | 0.691 | 0.331 | 0.800 |
| 21 of 163 (any call) | 0.811 | 0.811 | 0.811 | 0.811 | 0.597 | 0.135 | 0.784 | 0.466 | 0.813 |
- recall at high-tier FP by divergence, ai: 0.0: 0.89, 0.3: 0.80, 0.5: 0.53, 0.7: 0.18
- recall at high-tier FP by divergence, rules: 0.0: 0.83, 0.3: 0.74, 0.5: 0.55, 0.7: 0.23
- recall at high-tier FP by divergence, ai_all: 0.0: 0.89, 0.3: 0.80, 0.5: 0.53, 0.7: 0.18
- recall at high-tier FP by divergence, lineage.plm: 0.0: 0.81, 0.3: 0.74, 0.5: 0.66, 0.7: 0.58
- recall at high-tier FP by divergence, lineage.raw_plm: 0.0: 0.54, 0.3: 0.28, 0.5: 0.06, 0.7: 0.04
- recall at high-tier FP by divergence, lineage.composition: 0.0: 0.72, 0.3: 0.79, 0.5: 0.84, 0.7: 0.87
- recall at high-tier FP by divergence, lineage.raw_composition: 0.0: 0.60, 0.3: 0.69, 0.5: 0.72, 0.7: 0.70
- recall at high-tier FP by divergence, place.node: 0.0: 0.94, 0.3: 0.89, 0.5: 0.74, 0.7: 0.44

### transplant_legionella  (240 positives, 48 core, 3036 negatives; high tier: recall 0.721 with 22 false positives)
| score | AP vs core | AP vs all | recall at high-tier FP | recall at high+medium FP | recall at 1% FP | AP vs all minus rules (95%) | recall at high-tier FP minus rules (95%) |
|---|---|---|---|---|---|---|---|
| ai | 0.983 | 0.864 | 0.750 | 0.867 | 0.808 |  |  |
| rules | 0.983 | 0.859 | 0.787 | 0.858 | 0.817 |  |  |
| ai_all | 0.988 | 0.873 | 0.750 | 0.867 | 0.808 | +0.004 … +0.025 | -0.075 … +0.092 |
| lineage.plm | 0.968 | 0.660 | 0.412 | 0.562 | 0.471 | -0.260 … -0.130 | -0.461 … -0.183 |
| lineage.raw_plm | 0.929 | 0.581 | 0.338 | 0.508 | 0.354 | -0.339 … -0.204 | -0.552 … -0.273 |
| lineage.composition | 0.931 | 0.435 | 0.233 | 0.317 | 0.254 | -0.496 … -0.342 | -0.629 … -0.368 |
| lineage.raw_composition | 0.828 | 0.335 | 0.142 | 0.212 | 0.179 | -0.589 … -0.448 | -0.719 … -0.464 |
| place.node | 0.990 | 0.917 | 0.871 | 0.900 | 0.875 | +0.032 … +0.091 | +0.029 … +0.240 |
| core genes allowed through | rule tier recall | ai | rules | ai_all | lineage.plm | lineage.raw_plm | lineage.composition | lineage.raw_composition | place.node |
|---|---|---|---|---|---|---|---|---|---|
| 0 of 48 (high) | 0.721 | 0.900 | 0.900 | 0.921 | 0.525 | 0.279 | 0.229 | 0.083 | 0.938 |
| 0 of 48 (high or medium) | 0.854 | 0.900 | 0.900 | 0.921 | 0.525 | 0.279 | 0.229 | 0.083 | 0.938 |
| 0 of 48 (any call) | 0.900 | 0.900 | 0.900 | 0.921 | 0.525 | 0.279 | 0.229 | 0.083 | 0.938 |
- recall at high-tier FP by divergence, ai: 0.0: 0.69, 0.5: 0.81, 1.0: 0.75
- recall at high-tier FP by divergence, rules: 0.0: 0.72, 0.5: 0.83, 1.0: 0.81
- recall at high-tier FP by divergence, ai_all: 0.0: 0.69, 0.5: 0.81, 1.0: 0.75
- recall at high-tier FP by divergence, lineage.plm: 0.0: 0.35, 0.5: 0.44, 1.0: 0.44
- recall at high-tier FP by divergence, lineage.raw_plm: 0.0: 0.39, 0.5: 0.27, 1.0: 0.35
- recall at high-tier FP by divergence, lineage.composition: 0.0: 0.24, 0.5: 0.22, 1.0: 0.24
- recall at high-tier FP by divergence, lineage.raw_composition: 0.0: 0.15, 0.5: 0.16, 1.0: 0.12
- recall at high-tier FP by divergence, place.node: 0.0: 0.80, 0.5: 0.94, 1.0: 0.88
