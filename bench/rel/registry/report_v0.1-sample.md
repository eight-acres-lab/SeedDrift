# Event registry report (rules version 0.1; no scores)

20 records in 17 groups (groups: exclusion radii that overlap, a shared paper or a shared assembly).

## 1. Association graph by exclusion lineage

| group | records | labels | links inside the group |
|---|---|---|---|
| G01 | EV001 (Drosophila) | A 1 | - |
| G02 | EV002 (Acyrthosiphon); EV003 (Aphidomorpha) | A 1, C 1 | EV002-EV003: radius Acyrthosiphon / Aphidomorpha |
| G03 | EV004 (Bemisia) | C 1 | - |
| G04 | EV005 (Triticeae) | D 1 | - |
| G05 | EV006 (Alloteropsis) | A 1 | - |
| G06 | EV007 (Ipomoea) | A 1 | - |
| G07 | EV008 (Legionella) | C 1 | - |
| G08 | EV009 (Hominoidea); EV010 (Muroidea); EV018 (Vertebrata) | A 2, overturned 1 | EV009-EV018: radius Hominoidea / Vertebrata; EV010-EV018: radius Muroidea / Vertebrata |
| G09 | EV011 (Caudoviricetes) | A 1 | - |
| G10 | EV012 (cruciviruses) | A 1 | - |
| G11 | EV013 (Thermotoga) | D 1 | - |
| G12 | EV014 (Methanosarcina) | C 1 | - |
| G13 | EV015 (Neisseria) | D 1 | - |
| G14 | EV016 (Bdelloidea) | A 1 | - |
| G15 | EV017 (Hypsibius) | overturned 1 | - |
| G16 | EV019 (Bacillariophyta) | overturned 1 | - |
| G17 | EV020 (Elysia) | overturned 1 | - |

## 2. Counts

| label | records | groups |
|---|---|---|
| A | 9 | 8 |
| B | 0 | 0 |
| C | 4 | 4 |
| D | 3 | 3 |
| disputed | 0 | 0 |
| overturned | 4 | 4 |

Label: the transfer grade, except that disputed and overturned records count under their status. Statuses: overturned 4, replicated 7, standing 9. Discovery route: ? 6, N 6, S 8. Truth class: contamination 2, native 1, transfer 16, unresolved 1. Category: organelle_derived 1, protein_coding 19.

Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty cell has no record yet.

| direction | bacteria | archaea | mammals | other_vertebrates | invertebrates | plants_algae | fungi | protists | phage | eukaryotic_virus | giant_virus | plasmid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prok>prok | D2 | C1 |  |  |  |  |  |  |  |  |  |  |
| prok>euk |  |  | overturned1 |  | A3 overturned1 | A1 |  |  |  |  |  |  |
| euk>prok | C1 |  |  |  |  |  |  |  |  |  |  |  |
| euk>euk |  |  |  |  | C2 overturned1 | A1 D1 |  | overturned1 |  |  |  |  |
| virus>host |  |  | A2 |  |  |  |  |  |  |  |  |  |
| host>virus |  |  |  |  |  |  |  |  | A1 |  |  |  |
| virus>virus |  |  |  |  |  |  |  |  |  | A1 |  |  |

Complex scenarios of charter section 1 (scenario tags) and discovery routes, by label:

| tag | A | B | C | D | disputed | overturned |
|---|---|---|---|---|---|---|
| near | 1 |  |  | 1 |  |  |
| replacement |  |  |  | 1 |  |  |
| ancient | 3 |  | 3 |  |  | 1 |
| donor_unsampled |  |  | 1 |  |  |  |
| partial |  |  |  |  |  |  |
| serial | 2 |  |  |  |  |  |
| high_foreign | 1 |  |  | 1 |  | 1 |
| chimeric_virus | 1 |  |  |  |  |  |
| contamination_vs_integration | 1 |  |  |  |  | 2 |
| multicopy | 2 |  | 1 |  |  |  |
| intron_gain | 1 |  |  |  |  |  |
| large_insert | 2 |  |  |  |  |  |
| route ? | 1 |  | 4 |  |  | 1 |
| route N | 3 |  |  | 2 |  | 1 |
| route S | 5 |  |  | 1 |  | 2 |

## 3. Recheckable evidence

11 of 20 records (55%) in 11 of 17 groups name evidence a fixed check can re-examine (read archives, a second assembly, population data). By label: A 7 of 9, C 2 of 4, D 0 of 3, overturned 2 of 4.

## 4. Exclusion and exposure

Provisional list covers the radius: n 3, partial 5, y 12. Genomes inside the radius in the medium corpus: ? 7, n 12, y 1. Overlap with sealed genera or the sealed test part: ? 7, n 13. Exposure: ? 19, L4 1.

## 5. Double registration

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
