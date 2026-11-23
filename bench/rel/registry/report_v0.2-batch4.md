# Event registry report (rules version 0.2; no scores)

10 records (claim_set 3, event 7). 5 of them take part in grouping and form 4 groups (overlapping exclusion radii, a shared paper or a shared assembly); disputed, overturned and grade-D records and claim sets stay ungrouped (RULES.md section 7).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV037 (Giardia); EV038 (Giardia) | C 2 | EV037-EV038: radius Giardia / Giardia; paper 10.1128/ec.1.2.181-190.2002 |
| G02 | EV039 (Braconidae) | C 1 | - |
| G03 | EV042 (Acyrthosiphon) | A 1 | - |
| G04 | EV045 (Wolbachia) | C 1 | - |

Ungrouped: EV036 (Trichomonas; D; claim_set); EV040 (Chordopoxvirinae; C; claim_set); EV041 (Nicotiana; D; event); EV043 (Halobacteria; C; claim_set); EV044 (Prosthecobacter; D; event).

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 1 | 1 |
| B | 0 | 0 |
| C | 6 | 3 |
| D | 3 | 0 |
| disputed | 0 | 0 |
| overturned | 0 | 0 |

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: narrowed 1, replicated 1, standing 8. Discovery route: ? 7, P 2, S 1. Truth class: transfer 10. Category: endogenous_viral_element 1, protein_coding 9.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok | C1 | C1 |  |  |  |  |  |  |  |  |  |  |
| prok>euk |  |  |  |  | A1 |  |  | C2 D1 |  |  |  |  |
| euk>prok | D1 |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  |  |  |  |  |  |  |  |  |  |
| virus>host |  |  |  |  | C1 | D1 |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  |  |  | C1 |  |
| virus>virus |  |  |  |  |  |  |  |  |  |  |  |  |

Complex scenarios of charter section 1 (scenario tags), discovery routes and whether the full text was read, by label:

| | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near |  |  | 1 |  |  |  |
| replacement |  |  | 1 |  |  |  |
| ancient |  |  | 2 |  |  |  |
| donor_unsampled |  |  |  | 1 |  |  |
| partial | 1 |  |  |  |  |  |
| serial |  |  | 1 |  |  |  |
| high_foreign |  |  | 1 |  |  |  |
| chimeric_virus |  |  |  |  |  |  |
| contamination_vs_integration |  |  |  |  |  |  |
| multicopy |  |  |  | 2 |  |  |
| intron_gain | 1 |  |  |  |  |  |
| large_insert |  |  | 1 |  |  |  |
| route ? |  |  | 4 | 3 |  |  |
| route P |  |  | 2 |  |  |  |
| route S | 1 |  |  |  |  |  |
| full text n |  |  | 6 | 3 |  |  |
| full text y | 1 |  |  |  |  |  |

## 3. Recheckable evidence

0 of 10 records (0%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 0 of the 4 groups. By label: A 0 of 1, C 0 of 6, D 0 of 3.

## 4. Exclusion and exposure

Provisional list covers the radius: n 3, y 7. Genomes inside the radius in the medium corpus: ? 5, n 5. Overlap with sealed genera or the sealed test part: ? 5, n 5. Exposure: ? 10.
