# Event registry report (rules version 0.2; no scores)

10 records (claim_set 3, event 7). 5 of them take part in grouping and form 5 groups (overlapping exclusion radii, a shared paper or a shared assembly); disputed, overturned and grade-D records and claim sets stay ungrouped (RULES.md section 7).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV022 (Podospora) | C 1 | - |
| G02 | EV023 (Penicillium) | A 1 | - |
| G03 | EV024 (Osmerus) | A 1 | - |
| G04 | EV026 (Enterobacterales) | A 1 | - |
| G05 | EV028 (Helicobacter) | B 1 | - |

Ungrouped: EV021 (Saccharomycetaceae; D; event); EV025 (Mimiviridae; C; claim_set); EV027 (Campylobacter; A; claim_set); EV029 (Schistosoma; overturned; event); EV030 (Homo; overturned; claim_set).

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 4 | 3 |
| B | 1 | 1 |
| C | 2 | 1 |
| D | 1 | 0 |
| disputed | 0 | 0 |
| overturned | 2 | 0 |

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: overturned 2, replicated 3, standing 5. Discovery route: ? 3, N 3, P 2, S 2. Truth class: contamination 1, native 1, transfer 8. Category: protein_coding 10.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok | A1 B1 |  |  |  |  |  |  |  |  |  |  | A1 |
| prok>euk |  |  | overturned1 |  |  |  | D1 |  |  |  |  |  |
| euk>prok |  |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  | A1 | overturned1 |  | A1 C1 |  |  |  |  |  |
| virus>host |  |  |  |  |  |  |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  |  |  | C1 |  |
| virus>virus |  |  |  |  |  |  |  |  |  |  |  |  |

Complex scenarios of charter section 1 (scenario tags), discovery routes and whether the full text was read, by label:

| | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near | 3 | 1 |  |  |  |  |
| replacement | 1 | 1 |  |  |  |  |
| ancient |  |  |  |  |  |  |
| donor_unsampled | 1 |  |  |  |  |  |
| partial |  |  |  |  |  |  |
| serial | 2 |  | 1 |  |  |  |
| high_foreign | 1 |  |  |  |  |  |
| chimeric_virus |  |  |  |  |  |  |
| contamination_vs_integration |  |  |  |  |  | 1 |
| multicopy |  |  |  |  |  |  |
| intron_gain |  |  |  |  |  |  |
| large_insert | 1 |  | 1 |  |  |  |
| route ? |  |  | 1 | 1 |  | 1 |
| route N | 2 | 1 |  |  |  |  |
| route P | 1 |  | 1 |  |  |  |
| route S | 1 |  |  |  |  | 1 |
| full text n |  | 1 | 1 | 1 |  | 1 |
| full text partial | 4 |  | 1 |  |  | 1 |

## 3. Recheckable evidence

5 of 10 records (50%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 3 of the 5 groups. By label: A 4 of 4, B 0 of 1, C 0 of 2, D 0 of 1, overturned 1 of 2.

## 4. Exclusion and exposure

Provisional list covers the radius: n 4, partial 1, y 5. Genomes inside the radius in the medium corpus: ? 6, n 4. Overlap with sealed genera or the sealed test part: ? 5, n 5. Exposure: ? 9, L5 1.
