# Hard benchmark: protocol of the first freeze

Written 2026-11-16, before any score of a sealed chromosome was joined to a label. This file is the operative
protocol of the first freeze (F1) and of the first unsealing (U1): real island genes on bacterial chromosomes against
natives certified by hidden genomes of the same genus. The full design, with all later stages and the constructed
hard scenarios, is in [DESIGN.md](DESIGN.md); where the two differ for the first freeze, this file rules. Where this
file is silent on a first-freeze rule, DESIGN.md as patched and corrected applies.

## 0. What has been done and seen before this protocol

- The first pre-registration ([bench/islands/README.md](../islands/README.md), commit `620f18c`) fixed the island
  set, the selection of 18 chromosomes (one per genus) and the gene label rule. Its dated amendment supersedes parts
  of it (section 11).
- Deep search of the 18 selected chromosomes: done. Their published labels: counted
  (`bench/islands/results/label_counts.tsv`). No score of any method has been joined to a label of any of them.
- Looked at freely: the 24 development chromosomes of the same set (Escherichia, Shigella, Salmonella, Yersinia),
  with every score that exists. All tuning happens there.
- Scope of every verdict: bacterial chromosomes, one genome at a time, with the genome's own genus hidden from every
  method. Nothing is claimed for eukaryote, archaeal or viral recipients before their components are run.
- This text was revised after three reviews of it (fidelity to DESIGN.md; whether every rule can be computed; the
  code against the text), still before any score of a sealed chromosome was joined to a label.
- Known at the time of this revision, all from development chromosomes or from label files: the development gates
  of section 2.4 pass in all three units (median share of backbone genes in the positional core 0.94 to 0.97, median
  conflict share 0.00); panel labels exist for the 24 development chromosomes and, as label files only, for 3 sealed
  genera; the scores of every method of section 5 have been evaluated on the 4 searched development chromosomes with
  the gated labels (3 units; every registered contrast reads "unresolved" there, and two trivial scores lead).

## 1. The question and the classes

On a real chromosome, which input separates recently acquired genes from genes kept in place since the genus
diverged, at a strict threshold, gene by gene and block by block: DNA composition, protein similarity, a taxonomy
rule, or a fusion?

- **Unit.** One gene: an annotated non-pseudo coding sequence whose protein has at least 30 residues (the pipeline's
  cleaned protein set).
- **Positive, "island gene".** An island-labelled gene, also called a published island gene (section 2.1; a gene with
  the label "both" is not one), with a positional orthologue in at most 50% of the panel genomes of other species of
  its genus.
- **Negative, "native" (positional core).** Not a published island gene; single copy in the chromosome; a positional
  orthologue in at least 80% of the panel genomes of other species; not within 5 genes of a published island gene.
  For this class a gene with the label "both" (section 2.1) counts as a published island gene.
- **Everything else is unlabelled** and enters no rate: published island genes that the other species keep in place
  ("conflict"), positional-core genes next to an island ("edge"), multi-copy genes, genes present elsewhere in the
  other species, genes with a share between 0.5 and 0.8, genes with the label "both".
- **Names of the two label sets.** Gated labels = the island and native classes above. Published labels = the labels
  of section 2.1 as they are.
- **What the classes are not.** The positive class is "accessory relative to the other species, inside a block the
  published comparison could not align": gain is not told apart from loss in the others. The negative class is "kept
  in place since the genus diverged": it contains old transfers and gains at the root of the genus, which a detector
  of transfers of any age is right to flag. Analysis S-f (section 8) brackets that effect; it is not removed.

## 2. Labels

All label code reads genome records only: no score, no reference library, no sequence composition.

**2.1 Published regions** (`bench/islands/labels.py`). A gene is island-labelled when at least half of its coding
region lies in one published island region, backbone-labelled when at least half lies in one published backbone
region. A gene with at least half in a published island region and at least half in a published backbone region (the
two region sets come from different published runs) has the label "both": it is neither island-labelled nor
backbone-labelled in any class, count, share or comparison of this file, except that it counts as island-labelled for
the native class, the 5-gene margin and the native blocks (it is never a native and never inside a native block).

**2.2 Genus panel** (`bench/hard/panel.py`), per chromosome. INSDC assemblies from ENA of level "complete genome"
or "chromosome", full representation, with the submitters' annotation. Left out: assemblies behind a reference
proteome of the search library (by assembly accession); assemblies whose taxon is that of any chromosome of the
island set; assemblies with fewer than 300 translated coding sequences or with under 80% of them translated. Chosen
in ascending order of SHA-256("<assembly accession.version>|panel-v2"): up to 5 of the same species, up to 5 of other
species of the genus (one per species), 4 of other genera of the family (one per genus; from the order if the family
has fewer than 4). The candidate list always comes from ENA; the records of the chosen assemblies come from ENA or,
when its record service does not answer, from the NCBI mirror of the same INSDC records (`--source ncbi`). A failed
download stops the build of that panel instead of letting the next assembly in hash order take the place. ENA
replaces RefSeq because it is the mirror that is fast from the server; the genomes are the
same, the annotation is the submitters'.

**2.3 Panel classes** (`bench/hard/panel_labels.py`).

- Hit of gene g in panel genome P: the best alignment of g in P with at least 40% identity over at least 70% of g
  (DIAMOND `--sensitive`, E <= 1e-5). Best = the highest bit score; among equal bit scores, the first line of the
  search output.
- Positional orthologue: g has a hit h in P, and at least one of the 2 genes before g and at least one of the 2 genes
  after g has its own hit in P within 5 genes of h on the same replicon (counted around the origin as well where the
  panel record says that the replicon is circular).
- f = share of the other-species panel genomes with a positional orthologue; defined with at least 3 such genomes.
- Single copy: no other gene of the chromosome aligns with at least 40% identity over at least 80% of both genes.
- Classes as in section 1, with f <= 0.5 for island genes and f >= 0.8 for natives.
- Block units. Island region: a published island region with at least 3 island-class genes that are at least 50% of
  its island-labelled genes; only those genes carry the region. Native block: 11 consecutive genes of one contig, none
  island-labelled or within 5 genes of an island-labelled gene (the label "both" counts as island-labelled here), at
  least 9 of them native; only its native genes enter its score; tiled from the first gene of each contig, 11 genes
  forward after a block, 1 gene forward after a window that fails.
- Gene order and topology. Every sealed and development chromosome is one record with one contig; the evaluator
  stops if a gene table holds more than one. Gene order = the genes of section 1 sorted by start, then end;
  pseudogenes and coding sequences outside the gene set hold no position. A chromosome is closed when its GenBank
  LOCUS line says "circular"; the topology of each of the 41 chromosomes (17 sealed, 24 development) is written into
  the freeze. On a closed chromosome the window W11 (section 5), the 5-gene margin around island-labelled genes, the
  neighbours of the positional-orthologue rule and the label-shift null (section 6) wrap around the origin; the
  tiling of native blocks does not.
- Flags (strata only): `absent` = island gene without any alignment (40% identity over 50% of the gene) in any other
  species; `pool` = island gene with a 70% / 70% alignment outside its position in another species; `deep` = native
  with a hit in at least 2 sibling-genus genomes; `fast` = native in the lowest tenth, within the chromosome, of mean
  identity to its positional orthologues.

**2.4 Checks of the labels against each other** (no score involved).

- Development gates, on every development unit whose chromosomes have at least 3 other-species panel genomes. Once
  the panels were built this was all three units (Escherichia with Shigella, 14 chromosomes; Salmonella, 5; Yersinia,
  5: each chromosome has 5 other-species genomes). Both gates use all chromosomes of those units, not only the
  searched ones; a chromosome with fewer than 3 other-species panel genomes enters no median. Gate 1: per development chromosome, the share of its
  single-copy published backbone genes with f >= 0.8; the median over the chromosomes of each unit is at least 80%,
  in every such unit. Gate 2: per development chromosome, the conflict share c = island-labelled genes with f >= 0.8 /
  island-labelled genes; in each unit, the median c over its chromosomes whose backbone-labelled genes outnumber
  their island-labelled genes is at most 5%, and the median c over its other chromosomes is at most 20%; a group
  without a chromosome passes. Gates 1 and 2 must both pass in every such unit. Gate 3 (printed, never triggers the
  fallback): at most 5% of the genes of the literature-curated islands of Sakai and CFT073 (`SupTable3.txt` of the
  island-set download, `Lit_ID` column) have f >= 0.8; it is not computed if the table lacks them. If gate 1 or 2
  fails after at most half a day of fixes on development, the positional-core class is withdrawn and the fallback of
  section 3 applies.
- Per sealed genus, before unsealing: under 70% of the single-copy backbone genes with f >= 0.8 = "panel
  unreliable"; a genus without any single-copy backbone-labelled gene is also "panel unreliable"; c above 20% =
  "label implausible". Either removes the genus from the primary population.

## 3. Genomes, eligibility, split

- **Development (open).** The 24 Enterobacterales chromosomes of the island set, in 3 units: Escherichia with
  Shigella (14 chromosomes), Salmonella (5), Yersinia (5). DNA scores and panel labels on all 24; protein methods on
  the 4 with a deep search (NC_002695.1, NC_003198.1, NC_007384.1, NC_006155.1). A unit has a positional-core class
  when its chromosomes have at least 3 other-species panel genomes (all three units, once the panels were built). A unit's
  value is the mean over its chromosomes of the per-chromosome metric; a development chromosome enters a unit mean
  with at least 100 natives and 20 island genes.
- **Sealed.** 17 genera, one chromosome each: the 18 selected genera without Brucella. Replicon rule: only the
  largest replicon of an assembly can be a test chromosome. The island set holds two Brucella records and both are
  chromosome II (NC_010167.1, NC_009504.1), so no largest replicon exists for the genus and Brucella leaves the
  sealed set. NC_010167.1 stays listed as sealed in `bench/hard/results/sealed.tsv`; it is a "secondary replicon":
  descriptive, in no rule, scored in stage 1b (the work after U1, see DESIGN.md). The unit of replication is the
  genus (by the genus taxon of the server's taxonomy); genera of one family are never split between Test A and
  Reserve B: Rhizobium with Sinorhizobium, Bradyrhizobium with Rhodopseudomonas.
- **Eligibility**, from label files only. The counts are computed for all 17 genera before the split.
  - P1: at least 3 other-species panel genomes; not "panel unreliable"; not "label implausible"; at least 500
    natives; at least 40 island genes.
  - P2: the same panel conditions; at least 5 island regions; at least 50 native blocks. Ladder, decided from label
    files only, after the split and before any score is joined. Step 1: a native block needs 9 natives among its 11
    genes. If fewer than 8 Test A genera are P2-eligible under step 1, step 2 holds for all genera: a native block
    needs 7 natives among its 11 genes. The step in force is written into the freeze; under step 2 every P2 figure
    carries "relaxed blocks". If step 2 also gives fewer than 8 genera, P2 is reported without verdicts under step 1
    and every "better than" verdict of P1 reads "at gene level only". The number of native blocks under both steps is
    printed for the 24 development chromosomes and, per sealed genus, beside its published island regions, island
    regions and natives.
  - S-pub (published labels as they are): at least 300 backbone and 50 island genes; at block level at least 20
    published backbone regions and 5 published island regions, counted as units: a published region is a unit when
    at least 3 genes carry its label.
- **Split**, fixed now. Family units are ordered inside two strata by ascending SHA-256("<alphabetically first genus
  of the unit>|reserve-B-v2"). High stratum: Mycobacterium, Acinetobacter, Corynebacterium, Rhizobium+Sinorhizobium,
  Pseudomonas, Burkholderia, Bacillus, Bradyrhizobium+Rhodopseudomonas. Low stratum: Streptococcus, Staphylococcus,
  Shewanella, Haemophilus, Prochlorococcus, Vibrio, Rickettsia. The two lists are final as printed, without
  Brucella. The criterion that formed them (high = a genus with at least 300 published backbone genes on the selected
  record) was applied to the counts taken before the label fix; the lists are not recomputed after it. With n_P1 =
  the number of P1-eligible genera:
  - n_P1 >= 14: Reserve B = the first 2 single-genus units of the high list and the first 3 units of the low list,
    skipping units that are not P1-eligible. Test A = the other genera (12). Reserve B is scored by nothing until the
    first DNA-reading model is frozen.
  - 9 <= n_P1 <= 13: no Reserve B; all 17 genera are Test A.
  - n_P1 < 9, or development gate 1 or 2 fails: fallback PUB. All 17 genera are Test A; P1 and P2 are reported on
    the published labels without verdicts. Gene level: island against backbone genes in the genera with at least 300
    backbone and 50 island genes; mean recall over 0-5% as the headline, because only 3-14 backbone genes sit at the
    1% point. Block level: published regions in the genera with at least 20 published backbone regions and 5
    published island regions, one backbone region allowed. Per-genus values, paired differences, t-intervals and
    sign-flip p are printed under the mark "published labels, not gated"; no "better than" statement is made and no
    concordance condition is evaluated. Where the positional-core class stands, the gated figures of the eligible
    genera are printed beside, descriptive.
- **Floors.** P1 needs 9 eligible Test A genera, P2 needs 8. Below its floor an endpoint is reported without verdicts.

## 4. What a method may see

A view, also called a regime, fixes which reference proteomes of the library every method may see. One reference
proteome of the library is a row. There are two regimes, V1 and V2.

- **V1 (ranked).** Hidden from every method: (a) every library proteome of the genome's genus; (b) every proteome r
  that is as close as the closest own-genus proteome, that is, with s(r) at least the s of that proteome; (c) every
  r with s(r) >= 90. s(r) = the median, over the genes with a hit covering at least 70% of the gene in r, of the best
  such identity; defined when at least 30% of the genes have such a hit (`bench/hard/views.py`). In clause (b)
  own-genus proteomes with s(r) >= 99 are not used as the reference, because the genome's own strain may or may not
  be in the library; if no own-genus proteome with a defined s(r) under 99 exists, clause (b) is inactive and only
  (a) and (c) act. A genus name alone would leave Shigella visible to an Escherichia genome. The hidden set of every
  chromosome is written into the freeze, and the three counts (rows hidden by (a), added by (b), added by (c)) are
  printed per genus.
- **V2 (required concordance, not an endpoint).** V1 plus every proteome of the family: the same genes as if the
  lineage were poorly sampled. A genus where V2 hides fewer than 3 proteomes more than V1 is "V2 not evaluable"; the
  other genera are V2-evaluable. Under V2 every score that reads hits is computed from the V2 view and none is
  copied from V1; the scores that read no hit (`gc3`, `codon`, `kmer`, `short`, `random`) are the same under both
  regimes.
- **Exactness.** A view filters the stored hit list. Genes whose 1,000-hit list is full are searched once more to
  4,000 hits. A list is full when the stored file holds at least 1,000 lines for the gene, counted before any taxon
  filter. If a stored list is missing for a searched development chromosome, all of its proteins go into the same
  run. A view keeps the first 1,000 visible hits and is flagged exact when 1,000 visible hits remain or the stored
  list holds fewer than 4,000. The share of exact views is printed per class and regime.
- **Input contract.** Scoring scripts read the hit table, the genome record and the label-free gene table
  `genes/<accession>.tsv` (gene_id, contig, start, end, strand) that `bench/islands/labels.py` writes. They never
  open a label file, and `views.py` is never given one. No product text reaches a method or a printout of a sealed
  genome, the rule-tier tables included.

## 5. Scores

Higher = more foreign. Every method declares its receptive field: per-gene (the score depends on the gene's own
sequence or hits and on genome-wide statistics) or context (it depends on neighbouring genes). Per-gene methods:
`gc3`, `codon`, `kmer`, `wide`, `wide_rule_ai`, `ai_all`, `ai_narrow`, `ai_gene`, `fuse`, `fuse_triv` and every
trivial score. Context methods: every `x_nb` (below).

| name | input | what it is |
|---|---|---|
| `gc3`, `codon`, `kmer` | DNA | `bench/islands/composition.py`: deviation of a gene's coding sequence from the chromosome |
| `wide` | proteins | the taxonomy-free table model of `research/array/wide.py`, checkpoint of seed 0; genes are scored in 3 random groupings of 64 genes (seed 7 of `score_genomes`) and the score is the mean of the logit over the groupings; per-gene |
| `wide_rule_ai` | proteins | the fixed rule on wide's own rows: best similarity outside the genome's nearest rows minus best inside |
| `ai_all` | proteins + taxonomy | Alien Index on the hit table, home group = phylum |
| `ai_narrow` | proteins + taxonomy | the same, home group = the narrowest of family, order, class with at least 10 visible genera besides the own, else the phylum |
| `ai_gene` | proteins + taxonomy | the same, home rank chosen per gene: the narrowest of family, order, class, phylum in which the gene has a visible hit covering at least 40% of it; if the gene has no visible covering hit inside its phylum, the home group is the phylum (empty home side, E = 1), as for `ai_all` |
| `fuse` | DNA + proteins | fusion of `codon` and `wide` (below) |
| `fuse_triv` | DNA + proteins | the same fusion with `absent_near` in place of `wide` |

**Alien Index.** Every Alien Index = ln(best home E-value) minus ln(best outside E-value) over the visible hits that
cover at least 40% of the gene. E-values are floored at 1e-200; a side without such a hit counts as E = 1. A gene
without any such hit is unscored.

"Linear rules" = {`ai_all`, `ai_narrow`, `ai_gene`, `wide_rule_ai`}, the table scores of `bench/hard/views.py`. The
pipeline's own Alien Index and its confidence tiers (tag `rules-R0`) are printed beside as counts and enter no test;
if the Spearman correlation of the table `ai_all` with the pipeline's (median over the development chromosomes with
a pipeline run, table computed under the pipeline's own hidden set) is below 0.9, every verdict on the linear rules
reads "table rules, not the pipeline".

**Route of `wide`.** Its columns are built per regime from the visible hits that `views.py` writes for that regime
(`--hits-out`), so `wide` and the table scores read the same first 1,000 visible hits of a gene. Its genome profile
comes from the unfiltered table. The hidden set of the view (`--hidden-out`) is passed to it.

**Trivial scores** are methods under test, in both windows: `short` (minus log length), `nohit` (no visible hit),
`n_rows` (minus the number of visible proteomes with a hit covering at least 40% of the gene), `absent_near` (1 minus
the best bit score per residue, relative to 2 bits per residue, clipped to 0-1, among the 12 visible proteomes with
the highest s(r); ties at the 12th place go to the lower taxid; if fewer than 12 visible proteomes have a defined
s(r), all of them are used; 1 without a hit there), `best_id` (minus the best visible identity), `sim_anom` (distance
of the best visible bit score per residue from the genome's median), `copies` (the gene's visible hits divided by the
number of distinct visible proteomes among them), `viral` (share of viral proteomes among the 10 best visible hits),
`random` (the first 8 bytes of the SHA-256 digest of the UTF-8 string "<gene_id of the gene table>|random-v1", read
as a big-endian unsigned integer). A gene without any visible hit has `nohit` = 1, `absent_near` = 1 and `n_rows` =
0, its highest value; `best_id`, `sim_anom`, `copies` and `viral` do not score it. Only the scores named in this
section are methods: the columns `full`, `exact` and `hidx` that `views.py` also writes are not.

**Quantile.** q_x = (rank - 0.5) / n of score x among the n genes of the chromosome that x scores, ties at the average
rank. A gene is scored by x when x gives the gene its own value, so that q_x exists; an empty cell or an absent row of
a score file means unscored; an absent column is a missing score file (section 9).

**Fusion.** Candidates, in this order: (1) max(q_codon, q_wide); (2), (3), (4) the weighted mean of the normal scores
of the two quantiles with weights codon:wide 1:1, 3:1, 1:3. A candidate is computed on the genes both components
score and expressed as a quantile among them; a gene scored by one component takes that component's q; a gene scored
by neither is unscored. The development value of a candidate = the mean, over the development units with a
positional-core class, of the unit value (mean over the unit's chromosomes that have a deep search) of W1 R@1% under
V1 with the gated labels. The candidate with the highest development value is fixed at the freeze; ties go to the
first in the list. Under fallback PUB the same rule is applied on the 4 searched development chromosomes (3 units)
with the published island genes and the single-copy published backbone genes, the backbone genes taking the place of
the natives; the candidate then carries "published labels, not gated". `fuse_triv` uses the same candidate.

**Windows**, one definition for every score. W1 = q. W11 = the mean of q over the scored genes among the gene and its
5 neighbours on each side (same contig; circular only on a closed chromosome); an unscored gene takes the mean of its
scored neighbours, which does not make it scored; with fewer than 3 scored genes in its window a gene has no W11
value. `x_nb` = W11 of x; `fuse_nb` = W11 of the quantile of `fuse`. The evaluator computes every `x_nb` itself from
x; the `*_nb` columns written by `bench/islands/composition.py` are a different quantity and are never read. In W1 a
gene a method cannot score ranks last, is never called and stays in every denominator; coverage is printed per method
and class. This holds in every label set and in every analysis; the only exception is the denominator of the two
analyses that section 8 takes over the genes both methods of a pair score (the hard-native strata and the printed-only
"scorable genes only"), where an unscored gene is left out and is still never called. The unscored genes form one
group below every scored gene, and the expectation rule for ties (section 6) is not applied inside that group. In the
gene-level figures at W11 a gene without a W11 value is treated in the same way. Sensitivity S-w: unscored genes enter
the window at the lowest rank (`metrics.rank_window`).

## 6. Metrics

- **Gene level.** R@1% = share of a chromosome's island genes above the score exceeded by 1% of its natives. With n
  natives, R@1% is recall at a native call rate of exactly 0.01 (`metrics.recall_never` at 0.01). Sort the scored
  genes from high to low and take the group of equal scores that holds the (floor(0.01 x n) + 1)-th highest native.
  Let a = the number of natives ranked above that group and k = the number of natives inside it: the island genes
  ranked above the group count in full, the island genes inside the group count with the weight (0.01 x n - a) / k,
  and no island gene below the group counts (ties at their expectation). Without ties this is the share of island
  genes ranked above the (floor(0.01 x n) + 1)-th highest native, so floor(0.01 x n) natives lie above the threshold.
  Unscored genes are never called (section 5): if fewer than floor(0.01 x n) + 1 natives are scored, every scored
  island gene is counted and no unscored one. A method that has a score file but scores no gene of a chromosome has
  R@1% = 0 and LR@2% = 0 there, and the genus stays in every paired difference. With fewer than 100 natives no native
  is allowed; such a value is printed with the mark "under 100 natives" and enters no mean. Printed beside: mean
  recall over native call rates 0-5%; R@0.3% where a chromosome has at least 1,000 natives; recall with every island
  region weighing 1; AUROC (descriptive; unscored genes tied below the scored ones, ties counted one half); the call
  split = the number of genes above the threshold by class (island, native, conflict, unresolved genes; "unresolved
  genes" = every gene that is not island, native or conflict, edge genes included).
- **Block level.** Region score = median W11 value over the region's island genes; block score = median W11 value over
  the block's natives. The median runs over those genes that have a W11 value; the median of an even number of values
  is the mean of the two middle values. A region or block is scored when at least 3 of its island genes (natives) are
  scored by the method itself (the gene has its own q, section 5; holding a W11 value is not enough) and at least one
  of its island genes (natives) has a W11 value; otherwise it is unscored: an island region then counts as missed, a
  native block as not called. LR@2%: allowed = max(1, floor(0.02 x native blocks of the chromosome)), where the count
  holds scored and unscored blocks; a region is found when its score is strictly above the (allowed + 1)-th highest
  block score. That score is the value at that position in the list of scored blocks sorted from high to low,
  duplicates kept; a region whose score equals it is not found. If fewer than allowed + 1 native blocks are scored,
  every scored region is found and every unscored region is missed. Printed beside: the same with floor(0.01 x blocks)
  and with 0 allowed; region AUROC (unscored regions and blocks tied below the scored ones, ties counted one half);
  the number of island regions and of native blocks that are unscored for the method; for each P2 contrast, LR@2% over
  the regions that both methods score (descriptive); the label-shift null (all labels moved together by one random
  offset along the chromosome; 1,000 offsets drawn from 1 to n - 1 with numpy `default_rng(1)`, n = the number of
  genes of the chromosome), at region level and for R@1% at W11.
- **Transport** (descriptive). Raw score = the value a method writes, before any quantile. Transported threshold of
  a method, per regime = the raw score exceeded by 1% of the natives pooled over the development chromosomes with a
  positional-core class for which the method has scores (`metrics.threshold_at` at 1% on the W1 raw scores). It is
  computed for raw scores only, not for `fuse`, `fuse_triv` or any W11 score. It is applied unchanged to Test A, a
  gene being called when its raw score is strictly above it: realised native call rate per genus, recall, calls by
  class. Under fallback PUB the thresholds come from the single-copy published backbone genes, pooled over the
  development chromosomes that have at least 300 of them and scores of the method, and carry "published labels, not
  gated".
- **Rule tiers** (pipeline `rules-R0`, as deployed): native call rate and island recall of the high tier, high or
  medium, any tier, as counts with exact binomial intervals, on the chromosomes whose pipeline run has finished
  before F1 (at most 6 sealed).
- **Audit and strata** (descriptive; these are the rows M1-M7 and M15 of the scenario matrix of DESIGN.md that are
  opened at U1; M1 is the real-island result of P1 and P2 itself). M15, confident calls of unknown truth: per method
  and Test A chromosome, among its top 1% and top 3% of genes, the shares of island genes, of positional-core genes
  (natives and edge genes) and of all other genes, median over genera, with the trivial and random scores beside; it
  enters no rule. M2, poorly sampled lineage: native call rate under V2 at the V1 threshold, split into deep and
  other natives (only the deep part counts as false). M3, genes without a visible hit: share of scored genes per
  method and class; AUROC inside the stratum of genes without a visible hit. M4, short genes: R@1% and coverage by
  protein length (under 100, 100-200, over 200 residues). M5, within-genome families (a family = a connected
  component of the alignments of the single-copy rule of section 2.3): recall with one unit per family and the share
  of each method's calls on unlabelled genes that fall in families of 5 or more. M6, highly expressed natives, and
  M7, `fast` natives: the call rates of the hard-native rule (section 8) and also the AUROC of the stratum against
  the other natives. Each method's top-1% natives are listed with their panel flags.

## 7. Primary endpoints

- **P1, gene level.** Per-genus R@1%, window W1, regime V1, per-gene methods, on the P1-eligible genera of Test A.
  Contrasts, Holm over 3: `codon` minus `wide`; `fuse` minus `codon`; `wide` minus the linear rules (wide must beat
  each of the four; the set enters the family as one p-value, section 8).
- **P2, block level.** Per-genus LR@2%, window W11, regime V1, on the P2-eligible genera of Test A. Contrasts, Holm
  over 2: `codon_nb` minus `wide_nb`; `fuse_nb` minus `codon_nb`.

The Holm families have fixed sizes, 3 for P1 and 2 for P2: a registered contrast that cannot be formed enters its
family with p = 1.

Nothing else is primary at the first unsealing. The endpoints on constructed cases (P3 gene-specific loss, P4 single
spliced genes, P5 spliced prophages) are registered in [DESIGN.md](DESIGN.md), are built in stage 1b and are opened
once, at the second unsealing (U2).

## 8. When a difference counts

- **Inference.** Unit = genus. Differences are paired within genus. Point estimate of a contrast = the mean of the
  per-genus differences. Interval: two-sided 95% t-interval on the per-genus differences (n - 1 degrees of freedom).
  "Equivalent" and "not worse" statements, and only these, use the interval obtained by inverting the sign-flip
  test: the set of shifts d0 for which the exact two-sided sign-flip p of the per-genus differences minus d0 is at
  least 0.05. Test: exact sign-flip permutation over genera. Statistic = the mean of the per-genus differences; all
  2^n sign assignments are enumerated; p = the share of assignments whose absolute mean is at least the observed
  absolute mean (two-sided, the observed assignment included, tolerance 1e-12). Holm inside each endpoint, with the
  family sizes of section 7. The smallest attainable p for the realised number of genera, 2 / 2^n, is printed beside
  every result. Every comparison with a limit of this protocol is made on unrounded values.
- **Set contrasts.** A contrast against a set (the four linear rules) is an intersection-union test: every member
  must be beaten. Point estimate = the smallest mean difference over the set; interval limit = the smallest lower
  limit. Each member gets the two-sided sign-flip p, replaced by 1 when its mean difference is not above 0; the
  set's p is the largest member p and enters the Holm family as one p-value; members are not adjusted among
  themselves. The trivial scores are not such a set: they are judged member by member as in condition 7, with no
  margin and outside every Holm family; their largest sign-flip p is printed, not required.
- **Direction.** A simple contrast is read in both directions: with A and B exchanged the same seven conditions
  apply, condition 7 to the winner. A set contrast has one direction only: "A better than the set", "equivalent" or
  "unresolved"; no verdict "member better than A" is issued, and the per-member table is printed. In conditions 4 to
  6 a set contrast uses the same rule inside each analysis: the sign of the smallest mean difference, the smallest
  lower limit.
- **"A better than B"** needs all of:
  1. the interval excludes 0;
  2. the point estimate is at least 0.05;
  3. the Holm-adjusted p is below 0.05;
  4. the same sign in S-pub, S-a and S-b (below), each over the genera it uses (bullet "Genera an analysis uses"
     below). No analysis is waived: with fewer than 5 genera the sign is still taken and the claim carries "<analysis>
     on n genera"; if an analysis has no genus, the condition fails for every contrast of that endpoint and no "better
     than" verdict is issued there (bullet "Verdict of a contrast" below);
  5. if A or B reads similarity: the same sign and an interval excluding 0 on that side under V2, on the eligible
     genera that are V2-evaluable; with fewer than 7 such genera the same sign only, and the claim carries "V2 not
     tested"; with no V2-evaluable genus, no "better than" verdict is issued when the winner A reads similarity, and
     when only the loser B reads similarity the claim carries "V2 not evaluable". "V1 and V2 disagree" = this
     condition fails: the V2 estimate and interval are printed beside the V1 ones, V1 is marked "comparators
     possibly visible", and no verdict is taken from V2;
  6. if A and B belong to different input classes (DNA; similarity with or without taxonomy; both): an interval
     excluding 0 on the same side also under S-f, on the genera with at least 300 deep natives (P1) or with at least
     20 native blocks whose natives are all deep (P2); with fewer than 7 such genera the same sign only, and "S-f
     not tested"; with no genus that meets the S-f size rule, no "better than" verdict is issued when the winner A
     reads DNA only, and otherwise the claim carries "S-f not evaluable";
  7. A beats every registered trivial score of its window: for every member the mean paired difference is above 0
     and its interval excludes 0; otherwise the verdict reads "not shown to detect origin beyond a trivial proxy".
     For `fuse`, `fuse_nb` and any method that reads both DNA and similarity, the set also contains `fuse_triv` at the
     same window; failing it reads "fusion not shown to add to composition plus a trivial similarity proxy".
- "The same sign" = the mean paired difference in the analysis lies strictly on the same side of 0 as the primary
  point estimate; a mean of exactly 0 fails the condition.
- **Verdict of a contrast.** If conditions 1 to 3 do not all hold: "equivalent" or "unresolved". If they hold and one
  of 4 to 6 fails: "unresolved", naming the failed condition. Where condition 4, 5 or 6 issues no "better than"
  verdict for lack of genera, the condition counts as failed: the contrast reads "equivalent" or "unresolved" if
  conditions 1 to 3 do not all hold, and "unresolved: <analysis> not evaluable" (S-pub, S-a, S-b, V2 or S-f) if they
  hold. If 1 to 6 hold and 7 fails: the sentence of condition 7.
- "Equivalent at this resolution" = the inverted sign-flip interval lies inside (-0.05, +0.05); for a set contrast,
  the inverted sign-flip interval of every member lies inside it. Everything else = "unresolved". With 9-13 genera,
  true differences under about 0.08-0.10 will mostly read unresolved; this is stated with every table.
- **Registered analyses.** S-pub: the published labels as they are (island against backbone genes; at block level
  published regions with one backbone region allowed); its negative rate is called "core-backbone call rate". For
  S-pub a published region is a unit when at least 3 genes carry its label; its score is the median W11 value over
  those genes, with the same rule for a scored region (section 6); allowed = 1 whatever the number of backbone
  regions. S-a: positives = island genes flagged `absent` (the part similarity can decide); at block level an island
  region is kept when at least half of its island genes are flagged `absent`, and a kept region is scored from all its
  island genes. S-b: without the genera flagged "label visible" (below) and without the genera whose published island
  genes outnumber the backbone genes. S-f: thresholds from deep natives only; at block level from the native blocks
  whose natives are all deep, with allowed = max(1, floor(0.02 x those blocks)) and the threshold taken among those
  blocks only. S-w: section 5.
- **Genera an analysis uses**, all fixed without joining a score of a method under test to a label (from label files;
  S-b also from the restatement flags, V2 from the hidden sets), and all inside the endpoint's eligible Test A genera
  except for S-pub. S-pub: every Test A genus that meets the S-pub counts of section 3 (gene counts for P1, region
  counts for P2), whether or not it is P1- or P2-eligible. S-a: at least 20 island genes flagged `absent` (P1); at
  least 3 kept regions (P2). S-b: the genera left after its two exclusions. S-f: at least 300 deep natives (P1); at
  least 20 native blocks whose natives are all deep (P2). V2: the V2-evaluable genera.
- **Printed only**: recall with one unit per within-genome family (section 6, M5); R@1% on the genes that both
  methods of a pair score ("scorable genes only"); the contrasts with orders as units.
- **Single-row restatement diagnostic.** Per genus and per row r visible under V1, the indicator is 1 for a gene
  without a hit in r of at least 80% identity that covers at least 40% of the gene, and 0 for a gene with one
  (higher = more foreign, as for every score). AUROC with published island genes as positives and published
  backbone genes as negatives, ties counted one half; if the maximum over r is above 0.85 the genus is flagged
  "label visible". It is computed for every Test A genus that has at least one island-labelled and one
  backbone-labelled gene. Timing: on development before the freeze; on Test A only by the frozen evaluator at U1; it
  is never run on a sealed chromosome before F1.
- **How the two endpoints combine.** Pairs across the endpoints: `codon` with `codon_nb`, `wide` with `wide_nb`,
  `fuse` with `fuse_nb`. An unqualified "A better than B" needs both P1 and P2. "Opposite signs" = both endpoints are
  at or above their floors and the two point estimates have opposite signs, whether or not the second endpoint gives a
  "better than" verdict; then no winner is named. A "better than" verdict on one endpoint and none on the other,
  without opposite signs, reads "at gene level only" or "at block level only". The contrast of `wide` with the linear
  rules has no block-level form and can only read "at gene level only". If one endpoint is below its floor, it is
  printed without verdicts, its point estimates take no part in the "opposite signs" rule, and every "better than"
  verdict of the other endpoint reads "at gene level only" or "at block level only". Error across the two endpoints is
  not controlled, and the report says so.
- **Hard natives** (strata, no label change): `fast`; highly expressed genes (natives matched by
  `research/placement/markers.py --strict`); natives under 100 residues; natives without any visible hit under V1
  (`nohit` = 1), for the pairs in which both methods score them (so in no pair with a similarity-only method; they are
  also reported apart). The pairs are the simple contrasts of P1. For a pair, the threshold is the score exceeded by
  1% of the natives outside the stratum that both methods score, and each call rate is taken over the stratum's genes
  that both score (an unscored gene is an abstention, printed as coverage, never a non-call). A genus with fewer than
  30 such genes is left out of that stratum; a stratum enters with at least 30 such genes in at least 6 genera. "A not
  worse on hard natives than B" = the upper limit of the inverted sign-flip interval, over genera, of [call rate of A
  minus call rate of B] is at most 5 points in every stratum that entered. "A better for the user than B" = "A better
  than B" and "A not worse on hard natives than B". This statement is made only if, on development (mean over the
  units with a positional-core class), at least one stratum reaches a call rate of 3% or more for `codon` or `kmer`;
  otherwise the report says "no hard native measured for DNA inputs". The lists of highly expressed genes (`markers.py
  --strict` on the protein FASTA, no label read) are computed for the 24 development and the Test A chromosomes before
  F1 and are part of F1; if pyhmmer or Pfam-A.hmm is missing, the stratum is not measured and the 3% trigger uses the
  other strata.
- Every statement that compares DNA with similarity carries: "island anchor only" and "own genus hidden" (the
  evidence that defines the labels, the other species of the genus, is withheld from similarity methods while
  composition keeps all of its evidence).
- **Exposure of the learned model.** Per genus: whether its library proteome, family or class was among the
  training genomes of `wide`; results are printed with and without exposed genera.

## 9. Sealing

- **What is sealed** is the join of scores with labels. Label files, panels, eligibility tables, the split, searches
  and scores of sealed chromosomes may exist before the freeze; none is an outcome. Only the accessions listed as
  development (`bench/islands/results/dev_genomes.tsv`) are open. Every other accession, listed in
  `bench/hard/results/sealed.tsv` or not, needs `--unseal` with a tag whose first letter is the letter of its tier (A
  for Test A, as in `A1` and `A1r`; B for Reserve B; the split writes the tier of each sealed accession into that
  table; NC_010167.1 keeps the tier "sealed", as does every accession absent from the table, and no tag opens them).
  The same check stands in every script that can join scores with labels of these chromosomes,
  `bench/islands/evaluate.py` included.
- **Freeze F1** is a tagged commit that holds: this protocol; the amended README of `bench/islands`; SHA-256 of the
  library FASTA and DIAMOND database, the taxonomy dump and the `wide` checkpoint; the tag `rules-R0`; all label and
  panel files, hidden sets and eligibility tables; the topology of the 41 chromosomes; the split, the block step in
  force and the lists of eligible Test A genera of P1 and of P2; the lists of highly expressed genes; the fusion
  candidate; the transported thresholds; the replication table for rows M1-M7 and M15 (one line per method and row:
  pass or fail expected, written on development); the SHA-256 of every score file and filtered hit table of the Test A
  chromosomes under V1 and V2, taken after a coverage check of the scores that reads no label; the evaluator.
- **Unsealing U1.** What U1 contains is fixed at F1. If F1 is not complete, U1 moves as a whole and no component is
  unsealed alone. No genus is added after an unsealing. The evaluator is run once with `--unseal A1`. Before it reads
  a label it checks the hashes of the score files and filtered hit tables against F1, and it stops if a registered
  method (the members of the P1 and P2 contrasts, the four linear rules, the trivial scores and `fuse_triv`; for
  `fuse` and `fuse_triv` the score files of their components) lacks a score file, or its column in that file, for a
  Test A chromosome that is eligible for P1, P2 or S-pub: under V1 for every registered method, under V2 for those
  that read hits (a score that reads no hit has one file for both regimes, section 4). It writes the log line (date,
  commit, hashes of labels, scores and evaluator) to `bench/hard/results/unseal.log` before any output, and it refuses
  an empty commit and a repeated tag. It writes only the registered tables. Anything computed afterwards that is not
  registered here is labelled exploratory. After U1, Test A is "exposed at design level"; a later model family may be
  evaluated on it at most 3 times, each time logged with its running count.
- **Registered tables of U1**: eligibility, split and the lists of eligible Test A genera of P1 and of P2; per genus
  and method R@1% at W1 and W11 and LR@2% with the figures "printed beside" of section 6, coverage and exact-view
  shares; the contrasts of P1 and P2 with mean, t-interval, inverted sign-flip interval, sign-flip p, Holm p, smallest
  attainable p and verdict; the same contrasts under S-pub, S-a, S-b, S-f, V2 and S-w with their genera; every primary
  method against every trivial score and `fuse_triv`; call split; transport; rule tiers; the audit and strata of
  section 6; hard-native strata; restatement flags; exposure; the label-shift null; the printed-only analyses.
- **A run that aborts.** If the run aborts before all tables are written, nothing it wrote is read. The fault is
  fixed in a commit that changes no rule of this protocol, both commits and the diff are logged, and the run is
  repeated once; because a repeated tag is refused, the repeat runs under the tag `A1r`.
- **Reserve B**, if it exists: its hit tables, labels, scores and pipeline outputs live in a directory that the
  evaluator refuses to read without `--unseal B`. Pipeline runs of sealed chromosomes are queued in ascending
  SHA-256 of the accession; the outputs of chromosomes that the split sends to Reserve B are moved unread to that
  directory. It is opened once, when the first DNA-reading model is frozen; all methods, old and new, are then
  scored on it together.
- **Training exclusions**: [EXCLUDE.tsv](EXCLUDE.tsv), a dated append-only list (the 18 selected genera, Brucella
  included, and their families, every panel genome, later every donor and viral genus the benchmark uses). No future
  model may train on them, and training code may not import the test-only generators of `bench/hard`.

## 10. Order of work

1. This protocol, the README amendment, checksums, exclusions. Replicon lengths (Brucella leaves the sealed set,
   section 3); label fix.
2. Panels for the 17 sealed and the 24 development chromosomes; panel classes; development gates. The label code is
   tagged after the gates have passed.
3. One library run at 4,000 hits: the genes with full lists (section 4).
4. Views (first 1,000 visible hits, exact flag), `ai_gene`, `absent_near`, `random`; the restatement diagnostic, on
   development only; `wide` under an explicit hidden set, with its columns built per regime from the visible hits.
5. Evaluator: per-chromosome metrics, scored-only windows, fusion candidates, the block rule, t-intervals, sign-flip
   and its inverted interval, Holm, set contrasts, the label-shift null, the registered analyses, `--unseal` with
   its log.
6. Development numbers with published and gated labels; fusion candidate, transported thresholds and replication
   table fixed; lists of highly expressed genes of the development chromosomes.
7. Labels only for the 17 sealed genera, produced by the label code tagged after the development gates have passed
   (`panel.py` and `panel_labels.py` do not change after that tag): eligibility, n_P1, split, block step of P2.
   Reserve B material moved unread to the locked directory; Test A chromosomes scored without label files; lists of
   highly expressed genes of the Test A chromosomes.
8. Coverage check of the Test A scores (no label read), the lists of eligible Test A genera of P1 and of P2, hashes;
   freeze F1; unsealing U1; the 3 largest deviations from the replication table are investigated before the report;
   report (`docs/hard-results.md`) with the coverage table of the scenario matrix.

The U1 report prints:

- beside every interval the realised n, the realised s (the standard deviation of the per-genus differences) and the
  minimum detectable difference (the half-width of the t-interval, t(n - 1) x s / sqrt(n)), with the statement that
  "unresolved" is the expected reading for true differences under about 0.08-0.10, not evidence of equality;
- the standard deviation of `wide` over groupings beside every `wide` interval, with the note "no seed variance";
- beside the fusion candidate and the transport table, that the fusion candidate and the transported thresholds of the
  methods that read hits rest on the 4 searched development chromosomes in 3 units with a
  positional-core class, while a method scored on more development chromosomes (the DNA scores, on all 24) has its
  threshold pooled over all of them that have a positional-core class (up to 19);
- under every call-split table, that its counts can differ from R@1% times the class size;
- per genus every eligibility count and the reason for each exclusion, the genera lost with their genome size and GC
  content, c, the agreement tables (backbone genes with f >= 0.8, island genes with f >= 0.8), panel roles and
  annotation source;
- the evaluator commit, and that earlier development baselines (pooled per genus, old window) are not comparable
  with P1 and P2;
- in the limits: that the generators of the constructed scenarios of DESIGN.md (its components HV, SP and FR) are
  written by the same engineer after the real-label results of Test A are known; that SP and FR are not yet frozen;
  that no model can yet pass or fail criteria (e) and (f) of the bar that DESIGN.md sets for a learned DNA-reading
  model;
- in the coverage table: rows M1 and M3 as "control not yet measured" (their control is the view V0 of DESIGN.md,
  which hides the species only and is not in the first freeze) and secondary replicons as "not measured";
- the training-data manifest; a glossary; the measured time of one `wide` pass and the revised date of U2.

Under fallback PUB every table header carries "published labels, not gated".

Dropped first if time runs short: rule tiers of sealed chromosomes; two of the three groupings of `wide` under V2.
Never dropped: P1 and P2 under V1 with gated labels, the V2 concordance, S-pub, S-a, S-b, S-f, the trivial and
fusion scores, the label gates, eligibility and split, the freeze and the logged unsealing.

## 11. What this supersedes in the first pre-registration

The dated amendment in [bench/islands/README.md](../islands/README.md) lists the changes: the three outcomes
registered there are withdrawn before any run (they compared methods against the published backbone genes only,
with thresholds set by a handful of genes, and needed a second pipeline run per chromosome with a home group that
is undefined once the family is hidden); negatives are the positional-core natives; the own genus is hidden by
measured closeness; one window rule for every score; largest replicon only, so Brucella leaves the sealed set (both
of its records in the island set are chromosome II) and 17 genera remain; genus as the unit; the README rule that a
genome with fewer than 50 positive or negative genes enters no summary is replaced by the eligibility rules of
section 3.

## 12. Known limits

- Blocks only: the island set starts at regions of about 4 kb. Single-gene transfers, old transfers and donors are
  not in the positives; they are the business of the constructed scenarios of stage 1b.
- Positives are the island extreme, negatives the positional-core extreme; most genes stay unlabelled.
- Both real-label endpoints rest on the panel code; its safeguards are label-against-label gates and the required
  agreement with the published labels (S-pub).
- The other species of a genus lie at very different distances in different genera; the median identity to the panel
  is printed per genus, not corrected. Species are NCBI species, which is uneven.
- Pathogen- and Proteobacteria-heavy; 4 of the 17 genera lie in one order.
- One engineer builds labels, runs methods and writes the evaluator; the seal is procedural (hashes, a list of open
  accessions, a locked directory, one logged run).

## 13. Glossary

- **Gated labels**: the island and native classes of section 1. **Published labels**: the labels of section 2.1 as
  they are.
- **Label "both"**: a gene with at least half of its length in a published island region and at least half in a
  published backbone region (section 2.1).
- **Regime**: a view, V1 or V2 (section 4). **Row**: one reference proteome of the library.
- **Covering hit**: a library hit (a hit in a reference proteome of the search library, section 4) that covers at
  least 40% of the gene (Alien Index, `n_rows`, restatement diagnostic). Among the library hits only s(r) asks for
  70%; the panel hit of section 2.3 (40% identity over 70% of the gene, in a panel genome) is a different object.
- **Full list**: a stored hit list of at least 1,000 lines, counted before any taxon filter (section 4).
- **Scored**: a gene is scored by a method when the method gives the gene its own value (q exists); a W11 value taken
  from neighbours does not count. A region or block is scored when at least 3 of its island genes (natives) are scored
  and at least one of its island genes (natives) has a W11 value (section 6).
- **Raw score**: the value a method writes, before any quantile (section 6, transport).
- **Per-gene method, context method**: the two receptive fields; section 5 lists the methods of each.
- **Registered method**: a member of a P1 or P2 contrast, one of the four linear rules, a trivial score, or
  `fuse_triv`.
- **Units with a positional-core class**: development units whose chromosomes have at least 3 other-species panel
  genomes (all three development units, once the panels were built).
- **Island region**: the block unit of section 2.3 (P2 eligibility, LR@2%). **Published island region, published
  backbone region**: a region of the published tables; in S-pub it is a unit when at least 3 genes carry its label.
- **Unresolved genes** (call split): every gene that is not island, native or conflict, edge genes included. This is
  not the verdict "unresolved" of section 8, which is the reading of a contrast that is neither "better" nor
  "equivalent".
- **Closed chromosome**: a chromosome whose GenBank LOCUS line says "circular" (section 2.3).
- **Point estimate, the same sign, opposite signs, inverted sign-flip interval**: section 8.
- **Minimum detectable difference**: the half-width of the t-interval, t(n - 1) x s / sqrt(n), with s = the standard
  deviation of the per-genus differences.
- **Fallback PUB**: the report on published labels without verdicts (section 3).
- **Secondary replicon**: a replicon that is not the largest of its assembly; descriptive, in no rule, scored in
  stage 1b (NC_010167.1).
- **Stage 1b, U2**: the work after U1 that builds the constructed scenarios of DESIGN.md; it ends with the second
  freeze and the second unsealing, U2.
- **Rows M1-M7 and M15**: the rows of the scenario matrix of DESIGN.md that are opened at U1 (section 6).
  **Replication table**: one line per method and row with the outcome expected from development (pass or fail),
  committed with F1.
