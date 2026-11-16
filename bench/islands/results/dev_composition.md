Chromosomes: 24, of which 18 have at least 50 island and 50 backbone genes and enter the summary.

| score | AUROC mean | AUROC median | recall at 5% backbone FP | recall at 1% backbone FP | chromosomes |
|---|---|---|---|---|---|
| comp.gc3 | 0.683 | 0.710 | 0.290 | 0.133 | 18 |
| comp.codon | 0.875 | 0.879 | 0.548 | 0.316 | 18 |
| comp.kmer | 0.870 | 0.889 | 0.533 | 0.296 | 18 |
| comp.gc3_nb | 0.839 | 0.854 | 0.444 | 0.191 | 18 |
| comp.codon_nb | 0.974 | 0.977 | 0.884 | 0.701 | 18 |
| comp.kmer_nb | 0.966 | 0.979 | 0.848 | 0.667 | 18 |
| comp.short | 0.605 | 0.606 | 0.116 | 0.006 | 18 |

| contrast | AUROC difference (95%) | recall at 5% FP difference (95%) | recall at 1% FP difference (95%) | chromosomes won |
|---|---|---|---|---|
| comp.kmer_nb-comp.kmer | +0.096 (+0.083 to +0.110) | +0.314 (+0.288 to +0.341) | +0.371 (+0.278 to +0.451) | 18 of 18 |
| comp.kmer_nb-comp.gc3_nb | +0.127 (+0.105 to +0.152) | +0.404 (+0.342 to +0.466) | +0.476 (+0.389 to +0.556) | 18 of 18 |
| comp.kmer-comp.gc3 | +0.188 (+0.167 to +0.209) | +0.244 (+0.206 to +0.281) | +0.163 (+0.126 to +0.202) | 18 of 18 |
| comp.kmer-comp.codon | -0.005 (-0.021 to +0.009) | -0.015 (-0.045 to +0.011) | -0.020 (-0.046 to +0.007) | 13 of 18 |

Per chromosome (AUROC):

| chromosome | organism | island genes | backbone genes | island genes with mobile-element words | comp.gc3 | comp.codon | comp.kmer | comp.gc3_nb | comp.codon_nb | comp.kmer_nb | comp.short |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NC_008253.1 | Escherichia coli 536 | 206 | 518 | 0.08 | 0.715 | 0.877 | 0.884 | 0.867 | 0.970 | 0.966 | 0.536 |
| NC_008563.1 | Escherichia coli APEC O1 | 304 | 527 | 0.27 | 0.724 | 0.835 | 0.840 | 0.861 | 0.956 | 0.956 | 0.632 |
| NC_004431.1 | Escherichia coli CFT073 | 433 | 470 | 0.16 | 0.721 | 0.860 | 0.890 | 0.904 | 0.977 | 0.988 | 0.592 |
| NC_009801.1 | Escherichia coli E24377A | 226 | 469 | 0.22 | 0.712 | 0.897 | 0.913 | 0.903 | 0.992 | 0.991 | 0.634 |
| NC_009800.1 | Escherichia coli HS | 36 | 532 | 0.19 | 0.592 | 0.768 | 0.793 | 0.733 | 0.836 | 0.901 | 0.457 |
| NC_002695.1 | Escherichia coli O157:H7 str. Sakai | 718 | 465 | 0.22 | 0.712 | 0.838 | 0.845 | 0.892 | 0.951 | 0.957 | 0.690 |
| NC_010498.1 | Escherichia coli SECEC SMS-3-5 | 102 | 526 | 0.17 | 0.744 | 0.914 | 0.930 | 0.899 | 0.995 | 0.996 | 0.498 |
| NC_007946.1 | Escherichia coli UTI89 | 474 | 511 | 0.21 | 0.708 | 0.843 | 0.854 | 0.852 | 0.955 | 0.959 | 0.569 |
| NC_010473.1 | Escherichia coli str. K-12 substr. DH10B | 44 | 480 | 0.29 | 0.717 | 0.937 | 0.951 | 0.939 | 0.979 | 0.977 | 0.696 |
| NC_006905.1 | Salmonella enterica subsp. enterica serovar C | 117 | 493 | 0.30 | 0.688 | 0.882 | 0.874 | 0.857 | 0.952 | 0.957 | 0.682 |
| NC_006511.1 | Salmonella enterica subsp. enterica serovar P | 48 | 471 | 0.31 | 0.739 | 0.848 | 0.841 | 0.905 | 0.955 | 0.933 | 0.609 |
| NC_010102.1 | Salmonella enterica subsp. enterica serovar P | 129 | 475 | 0.19 | 0.735 | 0.929 | 0.942 | 0.907 | 0.989 | 0.985 | 0.544 |
| NC_004631.1 | Salmonella enterica subsp. enterica serovar T | 212 | 487 | 0.32 | 0.732 | 0.876 | 0.895 | 0.848 | 0.974 | 0.983 | 0.613 |
| NC_003198.1 | Salmonella enterica subsp. enterica serovar T | 214 | 491 | 0.34 | 0.715 | 0.882 | 0.898 | 0.845 | 0.977 | 0.984 | 0.636 |
| NC_007613.1 | Shigella boydii Sb227 | 49 | 377 | 0.51 | 0.585 | 0.846 | 0.847 | 0.864 | 0.992 | 0.996 | 0.582 |
| NC_007606.1 | Shigella dysenteriae Sd197 | 71 | 375 | 0.42 | 0.621 | 0.912 | 0.905 | 0.815 | 0.988 | 0.981 | 0.635 |
| NC_004741.1 | Shigella flexneri 2a str. 2457T | 98 | 407 | 0.45 | 0.607 | 0.886 | 0.892 | 0.775 | 0.984 | 0.992 | 0.600 |
| NC_008258.1 | Shigella flexneri 5 str. 8401 | 68 | 497 | 0.38 | 0.691 | 0.910 | 0.912 | 0.871 | 0.978 | 0.977 | 0.685 |
| NC_007384.1 | Shigella sonnei Ss046 | 128 | 469 | 0.46 | 0.679 | 0.883 | 0.888 | 0.828 | 0.986 | 0.988 | 0.628 |
| NC_010159.1 | Yersinia pestis Angola | 48 | 655 | 0.29 | 0.736 | 0.925 | 0.904 | 0.924 | 0.975 | 0.963 | 0.561 |
| NC_004088.1 | Yersinia pestis KIM | 41 | 769 | 0.20 | 0.642 | 0.844 | 0.834 | 0.822 | 0.990 | 0.985 | 0.561 |
| NC_009708.1 | Yersinia pseudotuberculosis IP 31758 | 216 | 121 | 0.22 | 0.561 | 0.855 | 0.804 | 0.661 | 0.977 | 0.944 | 0.581 |
| NC_006155.1 | Yersinia pseudotuberculosis IP 32953 | 198 | 212 | 0.19 | 0.611 | 0.839 | 0.749 | 0.778 | 0.979 | 0.924 | 0.571 |
| NC_010465.1 | Yersinia pseudotuberculosis YPIII | 246 | 118 | 0.20 | 0.610 | 0.830 | 0.748 | 0.733 | 0.948 | 0.859 | 0.562 |
