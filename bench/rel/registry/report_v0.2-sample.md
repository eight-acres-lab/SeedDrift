# Event registry report (rules version 0.2; no scores)

20 records (claim_set 7, event 13). 7 of them take part in grouping and form 6 groups (overlapping exclusion radii, a shared paper or a shared assembly); disputed, overturned and grade-D records and claim sets stay ungrouped (RULES.md section 7).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV001 (Drosophila) | A 1 | - |
| G02 | EV002 (Acyrthosiphon); EV003 (Aphidomorpha) | A 1, C 1 | EV002-EV003: radius Acyrthosiphon / Aphidomorpha |
| G03 | EV004 (Bemisia) | C 1 | - |
| G04 | EV008 (Legionella) | C 1 | - |
| G05 | EV012 (environmental samples) | C 1 | - |
| G06 | EV014 (Methanosarcina) | C 1 | - |

Ungrouped: EV005 (Triticeae; D; event); EV006 (Alloteropsis; A; claim_set); EV007 (Ipomoea; D; event); EV009 (Hominoidea; D; event); EV010 (Muroidea; D; event); EV011 (Caudoviricetes; A; claim_set); EV013 (Thermotoga; D; claim_set); EV015 (Neisseria; D; event); EV016 (Bdelloidea; A; claim_set); EV017 (Hypsibius; overturned; claim_set); EV018 (Vertebrata; overturned; claim_set); EV019 (Bacillariophyta; overturned; claim_set); EV020 (Elysia; overturned; event).

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 5 | 2 |
| B | 0 | 0 |
| C | 5 | 5 |
| D | 6 | 0 |
| disputed | 0 | 0 |
| overturned | 4 | 0 |

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: overturned 4, replicated 6, standing 10. Discovery route: ? 3, N 5, P 3, S 9. Truth class: contamination 2, native 1, transfer 16, unresolved 1. Category: organelle_derived 1, protein_coding 19.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok | D2 | C1 |  |  |  |  |  |  |  |  |  |  |
| prok>euk |  |  | overturned1 |  | A3 overturned1 | D1 |  |  |  |  |  |  |
| euk>prok | C1 |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  |  | C2 overturned1 | A1 D1 |  | overturned1 |  |  |  |  |
| virus>host |  |  | D2 |  |  |  |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  | A1 |  |  |  |
| virus>virus |  |  |  |  |  |  |  |  |  | C1 |  |  |

Complex scenarios of charter section 1 (scenario tags), discovery routes and whether the full text was read, by label:

| | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near | 1 |  |  | 1 |  |  |
| replacement |  |  |  | 1 |  |  |
| ancient | 1 |  | 3 | 2 |  | 1 |
| donor_unsampled |  |  | 1 |  |  |  |
| partial |  |  |  |  |  |  |
| serial | 2 |  |  |  |  |  |
| high_foreign | 1 |  |  | 1 |  | 1 |
| chimeric_virus |  |  | 1 |  |  |  |
| contamination_vs_integration | 1 |  |  |  |  | 2 |
| multicopy | 2 |  | 1 |  |  |  |
| intron_gain | 1 |  |  |  |  |  |
| large_insert | 2 |  |  |  |  |  |
| route ? |  |  | 3 |  |  |  |
| route N | 1 |  |  | 4 |  |  |
| route P |  |  | 1 |  |  | 2 |
| route S | 4 |  | 1 | 2 |  | 2 |
| full text n |  |  | 2 | 5 |  | 1 |
| full text partial | 4 |  | 2 | 1 |  | 3 |
| full text y | 1 |  | 1 |  |  |  |

Grades that would change if `donor_structure` were accepted as evidence of foreign origin (RULES.md section 11): EV007 D -> A; EV009 D -> A.

## 3. Recheckable evidence

11 of 20 records (55%) name evidence a fixed check can re-examine (read archives, a second assembly, population data); they lie in 4 of the 6 groups. By label: A 4 of 5, C 3 of 5, D 2 of 6, overturned 2 of 4.

## 4. Exclusion and exposure

Provisional list covers the radius: n 3, partial 5, y 12. Genomes inside the radius in the medium corpus: ? 1, n 14, y 5. Overlap with sealed genera or the sealed test part: ? 1, n 19. Exposure: ? 19, L4 1.

## 5. Double registration (rules 0.1, before regrading)

8 records registered twice; 37 of 48 fields agree (77%). The first registration is taken as compiled, before any reconciliation.

| field | agree |
|---|---|
| grade_transfer | 6 of 8 |
| grade_donor | 7 of 8 |
| status | 7 of 8 |
| discovery | 3 of 8 |
| exclusion_taxon | 6 of 8 |
| truth_class | 8 of 8 |

| record | field | first | second | note |
|---|---|---|---|---|
| EV001 | discovery | ? | S | second: the abstract says host genomes were searched for Wolbachia-derived sequences; first had not read the route |
| EV003 | grade_transfer | A | C | first misread the 2010 abstract (it reports no genetic mapping); the draft is corrected to C |
| EV003 | status | standing | replicated | second found an independent phylogeny in 23 aphid genomes (10.1016/j.ibmb.2025.104433); added to the draft |
| EV003 | discovery | ? | S | second assumed a homology screen; neither read the route |
| EV003 | exclusion_taxon | Aphidomorpha (infraorder) | Aphidoidea (superfamily) | adelgids share the transfer and sit outside Aphidoidea, so the lowest common taxon is Aphidomorpha (to be checked against NCBI on the server) |
| EV007 | discovery | N | S | found in siRNA assemblies by similarity to T-DNA genes: a mixed route |
| EV009 | grade_donor | A | - | retroviral origin: the donor's own structure (first) or no donor lineage claimed (second) |
| EV012 | grade_transfer | A | C | virus_chimera rule: the Rep-capsid junction in one PCR-confirmed circle counts as position evidence (first), or a breakpoint analysis is needed (second) |
| EV012 | discovery | S | N | a best hit came first, the chimera from assembly: a mixed route |
| EV012 | exclusion_taxon | cruciviruses (no rank) | BSL RDHV (no rank) | no ICTV taxon either way; the group (first) or the single virus (second) |
| EV014 | discovery | ? | N | a pathway-targeted phylogeny: neither route as defined |
