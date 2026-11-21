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
- broken (different replicons or more than 100 kb apart), inverted (closer, with strands inconsistent with the
  focal ones), or flank_lost.
A run passes when at least 1 genome shows it empty and at least 2 show it empty or replaced.

**Edge margin N.** The median (the 0.9 quantile until the amendment below) of the distances between the literature
islands' edges (SupTable3, the development chromosomes only) and the nearest S1-run edge of the same side within
20 kb, rounded up to whole kb and clipped to 1 to 10 kb; 5 kb when fewer than 10 edges match.

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

**Amendment of 2026-11-21, after the first labels run (20261120T164342, commit 1c99f93; evidence: labels and
literature edges only, no score; decided by the user: "改成中位数 3 kb").** The 0.9 quantile reached the 10 kb cap.
It rested on 80 matched edges of the 3 development chromosomes with literature islands, and with the 20 kb matching
window it measured mostly the coarseness of the literature coordinates (matched distances: median 2.65 kb, upper
quartile 6.1 kb, 0.9 quantile 12.4 kb). A 10 kb margin removes every gene of an S1 segment shorter than 20 kb (the
median segment is 8.1 kb), so all 187 insertion genes came from 23 segments of 20 kb or more: the length bias this
benchmark is meant to remove. N is now the median of the matched distances (here 3 kb). Also: a site whose flanks lie
more than 100 kb apart is broken whatever the strands (it was counted inverted when the strands disagreed); no
segment's pass changes by this.

## Part 3: units and evaluation (fixed before the first run that evaluates scores on these labels, 2026-11-21)

The evaluation is its own pipeline (family_a_eval). It reads the labels of a passed family_a run and never changes
them; the units are computed from the labels alone, before any score is read.

**Inputs.**
- Labels: a family_a run whose G4_famA_labels passed.
- Scores and covariates of record:
  - the per-gene columns of a passed audit run: cmp_codon, cmp_gc3, cmp_hits, cmp_short, cmp_kmer and tv_* (scores),
    md_cat_* (categories);
  - the model columns of the named island runs (scores/<variant>/scores_<acc>.tsv), each named `<label>:<column>`;
  - the first island run's hits (hits/<acc>.npz, after the V1 view) and reference library, for the covariates and lk_og.
- Every chromosome's gene list equals the island run's genes.tsv and the audit's columns, in order.

**Element clusters and units.** The segments are the S1 runs that hold at least one insertion gene. Their QC-pass,
single-copy genes are searched all against all (DIAMOND blastp, sensitive). Two segments of different chromosomes are
linked when at least max(3, half the genes of the smaller S1 run) have reciprocal hits at 80 % identity or more over
80 % or more of both proteins; a gene with such hits in 3 or more segments of one chromosome (IS-like) links nothing.
Element clusters are the connected components of the segments. Units (the merge rule): the GTDB genera of the
family A units of Part 1, joined wherever an element cluster holds segments of two of them; a unit is a connected
component. Every cross-genus link is reported with its element cluster.

**Chromosomes.** A chromosome is evaluated when it has at least 10 insertion genes and 100 core genes.

**Covariates.** len, the log protein length; hom, log(1 + the gene's hit rows in the island run, after the view);
ident, the best identity of those rows (0 without one); has_hit, at least one row. Within a chromosome each of len,
hom and ident becomes a quantile over the arm's positives and core genes.

**Arms** (per chromosome; every other gene is excluded):

| arm | positives | negatives |
|---|---|---|
| main | insertion genes with a match | their 1:1 matched core genes |
| core | insertion | all core genes, unmatched |
| noise | insertion | accessory genes (a sensitivity view only) |
| v1 | the old island set's island genes (labels_panel) | its native genes, on the audit's chromosomes and units |

Matching: the positives in ascending SHA-256 of "<chromosome>|<gene>|match-v1" each take the unused core gene with
the same has_hit, every quantile difference at most 0.1 and the smallest sum of differences (ties by the same key of
the core gene); a positive without a partner is dropped. The kept share is reported per chromosome and per category
(a gene's first md_cat flag in the order mobile, ribosomal, transport, regulator, hypothetical, other), and as a
ladder: len; len + hom; len + hom + ident (the main arm); then adding CAI (tv_cai); then the
neighbourhood (tv_win_gc, tv_win_dev); then the category as an exact field. Only the main arm is gated.

**Metrics.** For every arm and score: the AUROC per chromosome (ties count one half, a missing score ranks lowest),
the mean over the chromosomes of a unit, the mean over units, with its 95 % interval from the protocol's paired
cluster bootstrap (1000 draws; positive clusters are runs of positives in genome order, negative clusters blocks of
10; chromosomes resampled within units; on the main arm the pairs are not kept together, so the interval is that of
two independent samples). Contrasts: every model column against cmp_codon, cmp_kmer, cmp_hits and
tv_win_dev, on the main and the core arm. Strata on the core arm: insertion genes in a published island, and the rest.

**Controls.**
- Balance: on the main arm, the standardised mean difference between positives and their partners of each of the
  quantiles of len, hom and ident is at most 0.1 in every evaluated unit, and every evaluated unit has matched pairs
  (gated).
- Own scores: on the main arm, the 95 % interval of each matched covariate's own AUROC contains 0.5, from at least
  90 % finite draws (gated).
- Shuffle: with the labels permuted within each chromosome (200 draws), every score's mean AUROC over all draws and
  chromosomes on the core arm is 0.5 ± 0.01 (gated).
- Reproduction: every audit column's AUROC on the v1 arm equals the audit report's to 1e-9 (gated).
- lk_og (the leak column; the count and the identity themselves until the amendment below): per gene, minus the
  number of the chromosome's outgroup genera (the NCBI genera named by its GTDB outgroup genera, suffix removed) with
  a visible hit row (lk_og), and minus the best identity of those rows (lk_og_id), so that, like every score, higher
  is more island-like: an insertion gene is absent from the outgroups by its label, and a score that reads their hits
  can see that. Its AUROC on the main arm is reported; above 0.6, the hit-reading scores (every score but cmp_codon,
  cmp_gc3, cmp_kmer, cmp_short and tv_*) are marked "label evidence visible" in every arm. lk_og_id is reported and
  marks nothing.
- Inputs: the first island run's hits were searched under a view, and every island run used this run's taxdump.

**Outputs.** data/famA/units.json; data/famA/eval/report.json, arms.tsv, balance.tsv and ladder.tsv.

**Amendment of 2026-11-21, after the first evaluation run (20261121T111443, commit 97d05ac; evidence: that run's
report, scores included; decided by the user: "只翻转方向").** lk_og was the count itself (higher = more outgroup
genera with hits), so a leak moved its AUROC below 0.5 and the flag above 0.6 could never mark it. Only the direction
changes: both columns are negated; the threshold (0.6), the column the flag reads (lk_og alone) and everything else
stay. The outcome was known before the change: on the first run's main arm lk_og becomes 0.577 (0.423 before), below
the flag, so nothing is marked; lk_og_id becomes 0.620 (0.380 before) and marks nothing by the rule.
