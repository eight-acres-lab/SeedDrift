# Benchmark v2, family A re-cut: the rules (development chromosomes first)

The rules below are fixed before the run that first applies them; every threshold is in
`configs/benchmark_v2/family_a.yaml`. A threshold changes only by a dated amendment below, on label-against-label
evidence; no score ever informs a label rule. The frozen rules are later applied blind to new units. Decided with the
user on 2026-11-20 ("Yes, with defaults"; the defaults of the design: GTDB R232 taxonomy, three graded outgroup
genera with two genomes each, presence by DIAMOND blastx on genome DNA, the 30 aa length floor, one matched arm,
units by the merge rule).

## Part 1: taxonomy, panels and outgroups (fixed before the first family_a run)

**Chromosomes.** The development part of `bench/hard/results/v2_reference_set.tsv`, intersected with
`bench/islands/results/dev_genomes.tsv` (24). Nothing is computed for a sealed or candidate chromosome or genus; the
code never opens `bench/hard/results/sealed.tsv` or `v2_sealed_candidates.tsv`.

**Taxonomy.** GTDB release 232 (`bac120_taxonomy_r232.tsv.gz`, `bac120_r232.tree.gz`, each checked against the
release's MD5SUM.txt) for species, genera and the tree; the NCBI taxonomy (the CI taxdump) for the exclusion checks.
An assembly maps to GTDB by the digits of its GCA/GCF accession. A development chromosome takes the GTDB species of its
assembly (the record's DBLINK), else the GTDB species most of the assemblies of its NCBI species fall into.

**Panels.** For each development chromosome, from the ENA assemblies (complete genome or chromosome, full
representation) of the NCBI genera of its unit, mapped to GTDB, without library assemblies and island-set taxa:
- same: up to 5 assemblies of its GTDB species;
- other: one assembly per other GTDB species of its GTDB genus, up to 5; at least 3 are required.
Order: ascending SHA-256 of "<assembly>|panel-v2" (the old panels' salt), species ordered by their first assembly.

**Outgroups.** From the GTDB tree: the most recent common ancestor of the unit genus's representatives, then each node
towards the root; the sister clades at successive nodes are the candidate clades, nearest first. A genus is eligible
when it has at least 2 ENA complete-genome assemblies mapped to it, its genomes' NCBI order is the focal order, and it
is neither an island-set genus nor a genus or family of `EXCLUDE.tsv`, of a sealed_v1 or candidate chromosome, and its
assemblies are not library assemblies. From each of the first 3 clades that hold an eligible genus: its genera in
descending number of GTDB species representatives in the clade (ties by SHA-256 of the name), the first eligible one;
from it 2 genomes of different GTDB species where possible, in ascending SHA-256 of "<assembly>|outgroup-v1", with the
next 3 in that order as reserves. A genome with an N-run of 100 or more is replaced by the next reserve. The patristic
distance of each clade is recorded.

## Part 2: presence, labels, units and evaluation

(Written before the first run that computes labels.)
