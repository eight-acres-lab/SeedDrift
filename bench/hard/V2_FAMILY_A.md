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

**Amendment of 2026-11-21 (before any label; evidence: taxonomy and genome lengths only).** The first family_a run
(20261120T152733, commit af06821) failed G4_famA_taxa. GTDB R232 splits *E. coli* into three genera: Escherichia
(*E. coli* proper with the other Escherichia species), ECMA0423 (one species: *Shigella* and *E. coli* HS, E24377A)
and G047199095 (one species: the K-12 lineage; its representative sits next to Kosakonia in the tree). So 6
development chromosomes had no other species in their GTDB genus. Separately, the third outgroup clade of Yersinia
holds only reduced endosymbiont genomes, and Buchnera (0.4 to 0.6 Mb against Yersinia's 4.7 Mb) was chosen; a genome
that lacks most genes makes outgroup absence uninformative. The rules become (the first decided by the user on
2026-11-21: "Nearest larger clade", then "Place at E. coli's genus"):
- a chromosome whose GTDB genus offers fewer than 3 other species is walked. Anchor: the pool row of its GTDB
  species' representative gives an NCBI species; when more than half of the pool's GTDB-mapped assemblies of that
  NCBI species sit in one other GTDB genus G', the walk starts at the MRCA of G', else at the MRCA of its own genus.
  (The representatives of ECMA0423 and G047199095 are NCBI *E. coli* genomes, and most of the pool's *E. coli* sits in
  Escherichia; a walk from G047199095's own place would give K-12 Kosakonia genomes as other species.) From there,
  towards the root, the first node under which at least 3 GTDB species other than its own (never one of a GTDB genus
  named after a forbidden genus) have pool assemblies; the other species come from those, in the same order and
  quota. Same stays its own GTDB species. The unit's outgroups are the sister clades above that node. A chromosome
  whose genus meets the part 1 rule keeps it unchanged;
- a genus is eligible as an outgroup only when the median length of its eligible assemblies is at least half the
  median of the unit's pool; at least 2 outgroup genera are required (3 are still sought).

## Part 2: presence and labels (fixed before the first run that computes labels, 2026-11-20)

The labels come from the genomes' own evidence only. The published islands (the store's `labels/`, from SupTable1
and SupTable2) never enter a class; they are written next to it as an agreement stratum. The literature islands
(SupTable3) set only the edge margin N (below).

**Genes.** A development chromosome's genes are its record's CDS, named and kept the way the island stage names them
(`seeddrift.inputs.from_genbank`, `seeddrift.pipeline.clean_proteins`). A gene fails QC when its location is partial,
its length is not a multiple of 3 or it has a stop before its last codon (table 11), its coding sequence has a
non-ACGT base, or it is joined over the origin of a circular record. A record must hold one replicon.
A gene is multi-copy when another gene of the same chromosome matches it at 40 % identity over 80 % of both (DIAMOND
blastp).

**Presence.** Per unit, the development proteins of its chromosomes form one database. Every panel and outgroup
genome's DNA (in 200 kb windows overlapping by 30 kb) is searched against it with DIAMOND blastx (sensitive,
frameshift 15, genetic code 11, e-value 1e-5, identity at least 30 %, target coverage at least 30 % per hit, every
HSP reported). The hits of one protein on one replicon and strand merge into a locus when within 3 x the gene length +
3 kb; of them, the hits that do not overlap a better one on the protein by more than 10 % give the locus its covered
share of the protein, its length-weighted identity and its score. Loci of different proteins of the chromosome
overlapping by half of both, on either strand, are one place, owned by the highest score. Calls of a gene in a genome:
- ortho: an owned locus at 40 % identity and 70 % coverage;
- hom: any locus at 40 % identity and 50 % coverage;
- og: an owned locus at 30 % identity and 50 % coverage;
- pos: ortho, with an ortho locus of a neighbour (one of the next two genes in genome order) within 10 kb of the
  gene's best ortho locus on each side (the shorter way round a circular replicon).

**Shares.** Over the other-species genomes of the chromosome (at least 3, else every gene that passes QC is no_panel): f_ortho,
f_hom and f_pos, the share of genomes with that call. f_same, the ortho share in same-species genomes, is reported
only.

**Runs.** In genome order (wrapping on a circular record), with one-gene gaps bridged, a run is at least 3 genes and
4 kb long and starts and ends with a member gene:
- S1: genes without an ortho call in any other-species genome;
- S2: genes with f_ortho of at most 0.5.
The boundary band is the genes of an S2 run that holds an S1 run, outside the S1 run. U is the union of S1 and S2
runs.

**Polarity.** A gene is outgroup-absent when it has no og call in both genomes of the nearest outgroup genus that has
genomes, and in both genomes of at least one further genus; at least 2 genera are needed.

**Site.** For each S1 run, its flanks are the nearest 3 genes on each side that are single-copy, pass QC and are outside
U. In every other-species and outgroup genome that lacks the run (no ortho call, og for outgroups, in all its genes),
the innermost flank on each side with a locus there (ortho, og for outgroups) gives the site:
- empty: both flanks on one replicon with consistent strands, at most 5 kb apart;
- replaced: 5 to 100 kb apart;
- broken (different replicons or more than 100 kb), inverted, or flank_lost.
A run passes when at least 1 genome shows it empty and at least 2 show it empty or replaced.

**Edge margin N.** The 0.9 quantile of the distances between the literature islands' edges (SupTable3, the
development chromosomes only) and the nearest S1-run edge of the same side within 20 kb, rounded up to whole kb and
clipped to 1 to 10 kb; 5 kb when fewer than 10 edges match.

**Classes** (first that applies):
1. qc_fail; no_panel;
2. in an S1 run: bridged (a gap gene, called ortho somewhere), edge_in (within N of the run's nearer edge),
   no_polarity (not outgroup-absent), no_flank (its run does not pass), else **insertion**; distances to a run's
   edges are from the gene's ends, and a run spans from its smallest start to its largest end;
3. boundary (the band); unresolved (in U);
4. accessory (no ortho call in any other species, outside every run);
5. single-copy with f_pos of at least 0.8: **core** when not within 2 genes of a U-run gene and more than N away
   from every U run, else edge;
6. unresolved.

Insertion genes are the positives and core genes the negatives; every other class is excluded from the metrics and
reported. Outputs: `data/famA/labels/<chromosome>.tsv` (the class, the columns above, the published label),
`segments/<chromosome>.json` (every S1 run with its site states) and `index.json` (N, class counts and the
class-by-published-label table per chromosome).

## Part 3: units and evaluation

(Written before the first run that evaluates scores on these labels.)
