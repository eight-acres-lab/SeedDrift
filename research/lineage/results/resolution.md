pairs 229316 (test 45655), queries 64554
learned comparator on held-out queries: correlation with identity 0.855, mean absolute error 7.8 points

| which of two homologs is closer: agreement with the alignment, by gap in identity (points) | 0-2 | 2-5 | 5-10 | 10-20 | 20-101 |
|---|---|---|---|---|---|
| cosine, layer 6 | 0.597 | 0.670 | 0.758 | 0.870 | 0.958 |
| cosine, layer 20 | 0.580 | 0.643 | 0.729 | 0.815 | 0.911 |
| cosine, layer 33 | 0.580 | 0.629 | 0.723 | 0.838 | 0.945 |
| learned comparator (identity) | 0.575 | 0.635 | 0.728 | 0.849 | 0.954 |
| learned comparator (bits) | 0.563 | 0.635 | 0.733 | 0.851 | 0.957 |
| pairs | 9174 | 12045 | 13046 | 14579 | 13445 |

| which of two homologs is closer: agreement with the alignment, by gap in bits per residue | 0-0.05 | 0.05-0.1 | 0.1-0.2 | 0.2-0.4 | 0.4-9 |
|---|---|---|---|---|---|
| cosine, layer 6 | 0.635 | 0.682 | 0.766 | 0.869 | 0.962 |
| cosine, layer 20 | 0.610 | 0.657 | 0.733 | 0.823 | 0.920 |
| cosine, layer 33 | 0.617 | 0.651 | 0.726 | 0.833 | 0.947 |
| learned comparator (identity) | 0.586 | 0.635 | 0.719 | 0.829 | 0.949 |
| learned comparator (bits) | 0.589 | 0.636 | 0.727 | 0.840 | 0.955 |
| pairs | 9933 | 9546 | 12194 | 13844 | 17108 |
