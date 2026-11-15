
## unseen_class: 71 genomes
| test | raw_plm | plm | raw_composition | composition |
|---|---|---|---|---|
| verify, others from all | 0.884 | 0.894 | 0.807 | 0.827 |
| verify, others from other domain | 0.968 | 0.985 | 0.869 | 0.917 |
| verify, others from same domain | 0.882 | 0.904 | 0.785 | 0.787 |
| verify, others from same phylum | 0.670 | 0.837 | 0.574 | 0.723 |
| verify, others from same class | 0.518 | 0.404 | 0.629 | 0.497 |
| verify, others from same order | 0.544 | 0.453 | 0.599 | 0.518 |
| verify, others from same family | 0.581 | 0.661 | 0.433 | 0.581 |
| swap, ortholog from other domain (n=1765) | 0.844 / 0.92 | 0.940 / 0.92 | 0.728 / 0.86 | 0.839 / 0.82 |
| swap, ortholog from same domain (n=3369) | 0.770 / 0.83 | 0.835 / 0.83 | 0.694 / 0.79 | 0.745 / 0.77 |
| swap, ortholog from same phylum (n=5066) | 0.585 / 0.65 | 0.664 / 0.67 | 0.542 / 0.64 | 0.634 / 0.64 |
| swap, ortholog from same class (n=6541) | 0.498 / 0.48 | 0.419 / 0.50 | 0.595 / 0.49 | 0.489 / 0.49 |
| swap, ortholog from same order (n=4851) | 0.475 / 0.51 | 0.412 / 0.51 | 0.565 / 0.50 | 0.497 / 0.48 |
| swap, ortholog from same family (n=1452) | 0.548 / 0.59 | 0.673 / 0.61 | 0.468 / 0.62 | 0.593 / 0.61 |
| swap, homolog of rank 1 (n=6362) | 0.527 / 0.54 | 0.537 / 0.56 | 0.537 / 0.54 | 0.544 / 0.53 |
| swap, homolog of rank 4 (n=6061) | 0.560 / 0.59 | 0.568 / 0.60 | 0.568 / 0.59 | 0.580 / 0.58 |
| swap, homolog of rank 20 (n=5561) | 0.615 / 0.66 | 0.622 / 0.67 | 0.628 / 0.65 | 0.623 / 0.64 |
| swap, homolog of rank 100 (n=5060) | 0.639 / 0.70 | 0.640 / 0.70 | 0.658 / 0.67 | 0.639 / 0.65 |
verify: AUROC. swap: AUROC / share of pairs in which the native protein is closer to its genome than the homolog.

## bench_lineage: 47 genomes
| test | raw_plm | plm | raw_composition | composition |
|---|---|---|---|---|
| verify, others from all | 0.662 | 0.700 | 0.669 | 0.683 |
| verify, others from other domain | 0.966 | 0.977 | 0.840 | 0.931 |
| verify, others from same phylum | 0.408 | 0.341 | 0.260 | 0.297 |
| verify, others from same order | 0.623 | 0.663 | 0.650 | 0.654 |
| verify, others from same family | 0.614 | 0.658 | 0.634 | 0.636 |
| swap, ortholog from other domain (n=269) | 0.773 / 0.73 | 0.796 / 0.71 | 0.714 / 0.69 | 0.707 / 0.70 |
| swap, ortholog from same domain (n=1470) | 0.769 / 0.87 | 0.901 / 0.90 | 0.680 / 0.81 | 0.837 / 0.85 |
| swap, ortholog from same phylum (n=1367) | 0.580 / 0.70 | 0.776 / 0.82 | 0.521 / 0.55 | 0.759 / 0.77 |
| swap, ortholog from same class (n=2627) | 0.566 / 0.71 | 0.661 / 0.72 | 0.637 / 0.73 | 0.663 / 0.68 |
| swap, ortholog from same order (n=4549) | 0.580 / 0.67 | 0.640 / 0.67 | 0.630 / 0.69 | 0.646 / 0.66 |
| swap, ortholog from same family (n=6221) | 0.510 / 0.54 | 0.533 / 0.54 | 0.556 / 0.56 | 0.557 / 0.54 |
| swap, homolog of rank 1 (n=4398) | 0.539 / 0.58 | 0.566 / 0.58 | 0.564 / 0.60 | 0.580 / 0.57 |
| swap, homolog of rank 4 (n=4273) | 0.566 / 0.61 | 0.614 / 0.62 | 0.598 / 0.63 | 0.624 / 0.62 |
| swap, homolog of rank 20 (n=4093) | 0.577 / 0.66 | 0.652 / 0.67 | 0.604 / 0.64 | 0.656 / 0.66 |
| swap, homolog of rank 100 (n=3739) | 0.610 / 0.77 | 0.743 / 0.80 | 0.641 / 0.72 | 0.724 / 0.73 |
verify: AUROC. swap: AUROC / share of pairs in which the native protein is closer to its genome than the homolog.
