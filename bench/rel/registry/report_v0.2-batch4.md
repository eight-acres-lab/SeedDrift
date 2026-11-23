# Event registry report (rules version 0.2; no scores)

10 records (claim_set 2, event 8). 6 of them take part in grouping and form 5 groups (overlapping exclusion radii, a shared paper or a shared assembly); disputed, overturned and grade-D records and claim sets stay ungrouped (RULES.md section 7).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV037 (Giardia); EV038 (Giardia) | C 2 | EV037-EV038: radius Giardia / Giardia; paper 10.1128/ec.1.2.181-190.2002 |
| G02 | EV039 (Braconidae) | C 1 | - |
| G03 | EV040 (Molluscipoxvirus) | B 1 | - |
| G04 | EV042 (Acyrthosiphon) | A 1 | - |
| G05 | EV045 (Wolbachia) | C 1 | - |

Group numbers follow row order and change when records are added or split: compare groups by their members, not by number.

Ungrouped: EV036 (Trichomonas; D; claim_set); EV041 (Nicotiana; D; event); EV043 (Halobacteria; C; claim_set); EV044 (Prosthecobacter; D; event).

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 1 | 1 |
| B | 1 | 1 |
| C | 5 | 3 |
| D | 3 | 0 |
| disputed | 0 | 0 |
| overturned | 0 | 0 |

Concentration in groups: largest group G04 holds 1 of the 1 A; largest group G03 holds 1 of the 1 B; largest group G01 holds 2 of the 5 C.

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: narrowed 1, replicated 1, standing 8. Discovery route: ? 6, P 2, S 2. Truth class: transfer 10. Category: endogenous_viral_element 1, protein_coding 9.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok | C1 | C1 |  |  |  |  |  |  |  |  |  |  |
| prok>euk |  |  |  |  | A1 |  |  | C2 D1 |  |  |  |  |
| euk>prok | D1 |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  |  |  |  |  |  |  |  |  |  |
| virus>host |  |  |  |  | C1 | D1 |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  |  |  | B1 |  |
| virus>virus |  |  |  |  |  |  |  |  |  |  |  |  |

Complex scenarios of charter section 1 (scenario tags), discovery routes and whether the full text was read, by label:

| | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near |  |  | 1 |  |  |  |
| replacement |  |  | 1 |  |  |  |
| ancient |  |  | 2 |  |  |  |
| donor_unsampled |  |  |  | 1 |  |  |
| partial | 1 |  |  |  |  |  |
| serial |  |  |  |  |  |  |
| high_foreign |  |  | 1 |  |  |  |
| chimeric_virus |  |  |  |  |  |  |
| contamination_vs_integration |  |  |  |  |  |  |
| multicopy |  |  |  | 2 |  |  |
| intron_gain | 1 |  |  |  |  |  |
| large_insert |  |  | 1 |  |  |  |
| route ? |  |  | 3 | 3 |  |  |
| route P |  |  | 2 |  |  |  |
| route S | 1 | 1 |  |  |  |  |
| full text n |  |  | 5 | 3 |  |  |
| full text y | 1 | 1 |  |  |  |  |

### Concentration of sources (RULES.md section 8a; events only, claim sets apart)

9 source clusters (records joined by a shared DOI in papers). A cell is flagged when its largest source cluster or its largest exclusion lineage holds more than 30% of its events.

Charter section 3 also reports any cell with fewer than 30 events or 5 lineages case by case only, so a small cell is flagged by its size alone; the flag matters where a source fills a cell that is otherwise large.

| cell | events | sources | lineages | largest source (share) | largest lineage (share) | over 30% | claim sets |
|---|---|---|---|---|---|---|---|
| euk>prok|bacteria | 1 | 1 | 1 | S08 (10.1073/pnas.0502859102) (100%) | Prosthecobacter (100%) | **yes** | 0 |
| host>virus|giant_virus | 1 | 1 | 1 | S04 (10.1186/1471-2148-8-67) (100%) | Molluscipoxvirus (100%) | **yes** | 0 |
| prok>euk|invertebrates | 1 | 1 | 1 | S06 (10.1371/journal.pgen.1000827) (100%) | Acyrthosiphon (100%) | **yes** | 0 |
| prok>euk|protists | 2 | 1 | 1 | S02 (10.1128/ec.1.2.181-190.2002) (100%) | Giardia (100%) | **yes** | 1 |
| prok>prok|bacteria | 1 | 1 | 1 | S09 (10.1128/jb.187.15.5406-5418.2005) (100%) | Wolbachia (100%) | **yes** | 0 |
| virus>host|invertebrates | 1 | 1 | 1 | S03 (10.1126/science.1166788) (100%) | Braconidae (100%) | **yes** | 0 |
| virus>host|plants_algae | 1 | 1 | 1 | S05 (10.1073/pnas.93.2.759) (100%) | Nicotiana (100%) | **yes** | 0 |

Cells with claim sets only: prok>prok|archaea (1).

The same cells counted once per source cluster (a cluster counts once for each label it holds in the cell):

| cell | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| euk>prok|bacteria |  |  |  | 1 |  |  |
| host>virus|giant_virus |  | 1 |  |  |  |  |
| prok>euk|invertebrates | 1 |  |  |  |  |  |
| prok>euk|protists |  |  | 1 |  |  |  |
| prok>prok|bacteria |  |  | 1 |  |  |  |
| virus>host|invertebrates |  |  | 1 |  |  |  |
| virus>host|plants_algae |  |  |  | 1 |  |  |

## 3. Recheckable evidence

0 of 10 records (0%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 0 of the 5 groups. By label: A 0 of 1, B 0 of 1, C 0 of 5, D 0 of 3.

## 4. Exclusion and exposure

Provisional list covers the radius: n 3, y 7. Genomes inside the radius in the medium corpus: n 9, y 1. Overlap with sealed genera or the sealed test part: n 10. Exposure: ? 10.
