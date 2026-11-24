# Event registry report (rules version 0.2; no scores)

5 records (event 5). 5 of them take part in grouping and form 1 groups (overlapping exclusion radii, a shared paper or a shared assembly); disputed, overturned and grade-D records and claim sets stay ungrouped (RULES.md section 7).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV031 (Kyanoviridae); EV032 (Kyanoviridae); EV033 (Sechaudvirinae); EV034 (Kyanoviridae); EV035 (Kyanoviridae) | A 4, C 1 | EV031-EV032: radius Kyanoviridae / Kyanoviridae; paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV031-EV033: paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV031-EV034: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV031-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV032-EV033: paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV032-EV034: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV032-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV033-EV034: paper 10.1371/journal.pbio.0040234; EV033-EV035: paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV034-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234 |

Group numbers follow row order and change when records are added or split: compare groups by their members, not by number.

Ungrouped: none.

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 4 | 1 |
| B | 0 | 0 |
| C | 1 | 1 |
| D | 0 | 0 |
| disputed | 0 | 0 |
| overturned | 0 | 0 |

Concentration in groups: largest group G01 holds 4 of the 4 A; largest group G01 holds 1 of the 1 C.

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: replicated 1, standing 4. Discovery route: N 5. Truth class: transfer 5. Category: protein_coding 5.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok |  |  |  |  |  |  |  |  |  |  |  |  |
| prok>euk |  |  |  |  |  |  |  |  |  |  |  |  |
| euk>prok |  |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  |  |  |  |  |  |  |  |  |  |
| virus>host |  |  |  |  |  |  |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  | A4 C1 |  |  |  |
| virus>virus |  |  |  |  |  |  |  |  |  |  |  |  |

Complex scenarios of charter section 1 (scenario tags), discovery routes and whether the full text was read, by label:

| | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near |  |  |  |  |  |  |
| replacement |  |  |  |  |  |  |
| ancient |  |  |  |  |  |  |
| donor_unsampled |  |  |  |  |  |  |
| partial |  |  |  |  |  |  |
| serial | 4 |  | 1 |  |  |  |
| high_foreign |  |  |  |  |  |  |
| chimeric_virus |  |  |  |  |  |  |
| contamination_vs_integration |  |  |  |  |  |  |
| multicopy |  |  |  |  |  |  |
| intron_gain |  |  |  |  |  |  |
| large_insert |  |  |  |  |  |  |
| route N | 4 |  | 1 |  |  |  |
| full text partial | 4 |  | 1 |  |  |  |

### Concentration of sources (RULES.md section 8a; events only, claim sets apart)

1 source clusters (records joined by a shared DOI in papers). A cell is flagged when its largest source cluster or its largest exclusion lineage holds more than 30% of its events.

Charter section 3 also reports any cell with fewer than 30 events or 5 lineages case by case only, so a small cell is flagged by its size alone; the flag matters where a source fills a cell that is otherwise large.

| cell | events | sources | lineages | largest source (share) | largest lineage (share) | over 30% | claim sets |
|---|---|---|---|---|---|---|---|
| host>virus|phage | 5 | 1 | 2 | S01 (10.1073/pnas.0401526101) (100%) | Kyanoviridae (80%) | **yes** | 0 |

The same cells counted once per source cluster (a cluster counts once for each label it holds in the cell):

| cell | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| host>virus|phage | 1 |  | 1 |  |  |  |

## 3. Recheckable evidence

5 of 5 records (100%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 1 of the 1 groups. By label: A 4 of 4, C 1 of 1.

## 4. Exclusion and exposure

Provisional list covers the radius: n 5. Genomes inside the radius in the medium corpus: n 4, y 1. Overlap with sealed genera or the sealed test part: n 5. Exposure: ? 5.
