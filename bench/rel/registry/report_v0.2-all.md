# Event registry report (rules version 0.2; no scores)

35 records (claim_set 9, event 26). 18 of them take part in grouping and form 12 groups (overlapping exclusion radii, a shared paper or a shared assembly); disputed, overturned and grade-D records and claim sets stay ungrouped (RULES.md section 7).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV001 (Drosophila) | A 1 | - |
| G02 | EV002 (Acyrthosiphon); EV003 (Aphidomorpha) | A 1, C 1 | EV002-EV003: radius Acyrthosiphon / Aphidomorpha |
| G03 | EV004 (Bemisia) | C 1 | - |
| G04 | EV008 (Legionella) | C 1 | - |
| G05 | EV011 (Kyanoviridae); EV031 (Kyanoviridae); EV032 (Kyanoviridae); EV033 (Sechaudvirinae); EV034 (Kyanoviridae); EV035 (Kyanoviridae) | A 5, C 1 | EV011-EV031: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV011-EV032: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV011-EV033: paper 10.1371/journal.pbio.0040234; EV011-EV034: radius Kyanoviridae / Kyanoviridae; paper 10.1038/424741a, 10.1073/pnas.0401478101, 10.1371/journal.pbio.0040234; EV011-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV031-EV032: radius Kyanoviridae / Kyanoviridae; paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV031-EV033: paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV031-EV034: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV031-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV032-EV033: paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV032-EV034: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234; EV032-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV033-EV034: paper 10.1371/journal.pbio.0040234; EV033-EV035: paper 10.1073/pnas.0401526101, 10.1371/journal.pbio.0040234; EV034-EV035: radius Kyanoviridae / Kyanoviridae; paper 10.1371/journal.pbio.0040234 |
| G06 | EV012 (environmental samples) | C 1 | - |
| G07 | EV014 (Methanosarcina) | C 1 | - |
| G08 | EV022 (Podospora) | C 1 | - |
| G09 | EV023 (Penicillium) | A 1 | - |
| G10 | EV024 (Osmerus) | A 1 | - |
| G11 | EV026 (Enterobacterales) | A 1 | - |
| G12 | EV028 (Helicobacter) | B 1 | - |

Ungrouped: EV005 (Triticeae; D; event); EV006 (Alloteropsis; A; claim_set); EV007 (Ipomoea; D; event); EV009 (Hominoidea; D; event); EV010 (Muroidea; D; event); EV013 (Thermotoga; D; claim_set); EV015 (Neisseria; D; event); EV016 (Bdelloidea; A; claim_set); EV017 (Hypsibius; overturned; claim_set); EV018 (Vertebrata; overturned; claim_set); EV019 (Bacillariophyta; overturned; claim_set); EV020 (Elysia; overturned; event); EV021 (Saccharomycetaceae; D; event); EV025 (Mimiviridae; C; claim_set); EV027 (Campylobacter; A; claim_set); EV029 (Schistosoma; overturned; event); EV030 (Homo; overturned; claim_set).

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 13 | 6 |
| B | 1 | 1 |
| C | 8 | 7 |
| D | 7 | 0 |
| disputed | 0 | 0 |
| overturned | 6 | 0 |

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: overturned 6, replicated 10, standing 19. Discovery route: ? 6, N 13, P 5, S 11. Truth class: contamination 3, native 2, transfer 29, unresolved 1. Category: organelle_derived 1, protein_coding 34.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok | A1 B1 D2 | C1 |  |  |  |  |  |  |  |  |  | A1 |
| prok>euk |  |  | overturned2 |  | A3 overturned1 | D1 | D1 |  |  |  |  |  |
| euk>prok | C1 |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  | A1 | C2 overturned2 | A1 D1 | A1 C1 | overturned1 |  |  |  |  |
| virus>host |  |  | D2 |  |  |  |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  | A5 C1 |  | C1 |  |
| virus>virus |  |  |  |  |  |  |  |  |  | C1 |  |  |

Complex scenarios of charter section 1 (scenario tags), discovery routes and whether the full text was read, by label:

| | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near | 4 | 1 |  | 1 |  |  |
| replacement | 1 | 1 |  | 1 |  |  |
| ancient | 1 |  | 3 | 2 |  | 1 |
| donor_unsampled | 1 |  | 1 |  |  |  |
| partial |  |  |  |  |  |  |
| serial | 8 |  | 2 |  |  |  |
| high_foreign | 2 |  |  | 1 |  | 1 |
| chimeric_virus |  |  | 1 |  |  |  |
| contamination_vs_integration | 1 |  |  |  |  | 3 |
| multicopy | 2 |  | 1 |  |  |  |
| intron_gain | 1 |  |  |  |  |  |
| large_insert | 3 |  | 1 |  |  |  |
| route ? |  |  | 4 | 1 |  | 1 |
| route N | 7 | 1 | 1 | 4 |  |  |
| route P | 1 |  | 2 |  |  | 2 |
| route S | 5 |  | 1 | 2 |  | 3 |
| full text n |  | 1 | 3 | 6 |  | 2 |
| full text partial | 12 |  | 4 | 1 |  | 4 |
| full text y | 1 |  | 1 |  |  |  |

Grades that would change if `donor_structure` were accepted as evidence of foreign origin (RULES.md section 11): EV007 D -> A; EV009 D -> A.

## 3. Recheckable evidence

21 of 35 records (60%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 8 of the 12 groups. By label: A 12 of 13, B 0 of 1, C 4 of 8, D 2 of 7, overturned 3 of 6.

## 4. Exclusion and exposure

Provisional list covers the radius: n 12, partial 6, y 17. Genomes inside the radius in the medium corpus: ? 7, n 22, y 6. Overlap with sealed genera or the sealed test part: ? 6, n 29. Exposure: ? 33, L4 1, L5 1.
