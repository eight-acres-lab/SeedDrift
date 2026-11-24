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

Group numbers follow row order and change when records are added or split: compare groups by their members, not by number.

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

Concentration in groups: largest group G02 holds 1 of the 4 A; largest group G05 holds 1 of the 1 B; largest group G01 holds 1 of the 2 C.

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

### Concentration of sources (RULES.md section 8a; events only, claim sets apart)

10 source clusters (records joined by a shared DOI in papers). A cell is flagged when its largest source cluster or its largest exclusion lineage holds more than 30% of its events.

Charter section 3 also reports any cell with fewer than 30 events or 5 lineages case by case only, so a small cell is flagged by its size alone; the flag matters where a source fills a cell that is otherwise large.

| cell | events | sources | lineages | largest source (share) | largest lineage (share) | over 30% | claim sets |
|---|---|---|---|---|---|---|---|
| euk>euk|fungi | 2 | 2 | 2 | S02 (10.1016/j.cub.2010.12.020) (50%) | Podospora (50%) | **yes** | 0 |
| euk>euk|invertebrates | 1 | 1 | 1 | S09 (10.1016/j.molbiopara.2006.07.009) (100%) | Schistosoma (100%) | **yes** | 0 |
| euk>euk|other_vertebrates | 1 | 1 | 1 | S04 (10.1186/1471-2148-12-190) (100%) | Osmerus (100%) | **yes** | 0 |
| prok>euk|fungi | 1 | 1 | 1 | S01 (10.1007/s00438-004-0995-7) (100%) | Saccharomycetaceae (100%) | **yes** | 0 |
| prok>prok|bacteria | 1 | 1 | 1 | S08 (10.1007/s002849900447) (100%) | Helicobacter (100%) | **yes** | 1 |
| prok>prok|plasmid | 1 | 1 | 1 | S06 (10.1016/s1473-3099(15)00424-7) (100%) | Enterobacterales (100%) | **yes** | 0 |

Cells with claim sets only: host>virus|giant_virus (1), prok>euk|mammals (1).

The same cells counted once per source cluster (a cluster counts once for each label it holds in the cell):

| cell | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| euk>euk|fungi | 1 |  | 1 |  |  |  |
| euk>euk|invertebrates |  |  |  |  |  | 1 |
| euk>euk|other_vertebrates | 1 |  |  |  |  |  |
| prok>euk|fungi |  |  |  | 1 |  |  |
| prok>prok|bacteria |  | 1 |  |  |  |  |
| prok>prok|plasmid | 1 |  |  |  |  |  |

## 3. Recheckable evidence

5 of 10 records (50%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 3 of the 5 groups. By label: A 4 of 4, B 0 of 1, C 0 of 2, D 0 of 1, overturned 1 of 2.

## 4. Exclusion and exposure

Provisional list covers the radius: n 4, partial 1, y 5. Genomes inside the radius in the medium corpus: n 8, y 2. Overlap with sealed genera or the sealed test part: n 10. Exposure: ? 9, L5 1.
