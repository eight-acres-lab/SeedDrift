# Hard benchmark: protocol of the first freeze

Written 2026-11-16, before any score of a sealed chromosome was joined to a label. This file is the operative
protocol of the first freeze (F1) and of the first unsealing (U1): real island genes on bacterial chromosomes against
natives certified by hidden genomes of the same genus. The full design, with all later stages and the constructed
hard scenarios, is in [DESIGN.md](DESIGN.md); where the two differ for the first freeze, this file rules.

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

## 1. The question and the classes

On a real chromosome, which input separates recently acquired genes from genes kept in place since the genus
diverged, at a strict threshold, gene by gene and block by block: DNA composition, protein similarity, a taxonomy
rule, or a fusion?

- **Unit.** One gene: an annotated non-pseudo coding sequence whose protein has at least 30 residues (the pipeline's
  cleaned protein set).
- **Positive, "island gene".** A gene with at least 50% of its length in a published island region AND with a
  positional orthologue in at most 50% of the panel genomes of other species of its genus.
- **Negative, "native" (positional core).** Not a published island gene; single copy in the chromosome; a positional
  orthologue in at least 80% of the panel genomes of other species; not within 5 genes of a published island gene.
- **Everything else is unlabelled** and enters no rate: published island genes that the other species keep in place
  ("conflict"), positional-core genes next to an island ("edge"), multi-copy genes, genes present elsewhere in the
  other species, genes with a share between 0.5 and 0.8.
- **What the classes are not.** The positive class is "accessory relative to the other species, inside a block the
  published comparison could not align": gain is not told apart from loss in the others. The negative class is "kept
  in place since the genus diverged": it contains old transfers and gains at the root of the genus, which a detector
  of transfers of any age is right to flag. Analysis S-f (section 8) brackets that effect; it is not removed.

## 2. Labels

All label code reads genome records only: no score, no reference library, no sequence composition.

**2.1 Published regions** (`bench/islands/labels.py`). A gene is island-labelled when at least half of its coding
region lies in one island region, backbone-labelled when at least half lies in one backbone region; a gene with at
least half in both (the two region sets come from different published runs) is unlabelled.

**2.2 Genus panel** (`bench/hard/panel.py`), per chromosome. INSDC assemblies from ENA of level "complete genome"
or "chromosome", full representation, with the submitters' annotation. Left out: assemblies behind a reference
proteome of the search library (by assembly accession); assemblies whose taxon is that of any chromosome of the
island set; assemblies with fewer than 300 translated coding sequences or with under 80% of them translated. Chosen
in ascending order of SHA-256("<assembly accession>|panel-v2"): up to 5 of the same species, up to 5 of other species
of the genus (one per species), 4 of other genera of the family (one per genus; from the order if the family has
fewer than 4). ENA replaces RefSeq because it is the mirror that is fast from the server; the genomes are the same,
the annotation is the submitters'.

**2.3 Panel classes** (`bench/hard/panel_labels.py`).

- Hit of gene g in panel genome P: the best alignment of g in P with at least 40% identity over at least 70% of g
  (DIAMOND `--sensitive`, E <= 1e-5).
- Positional orthologue: g has a hit h in P, and at least one of the 2 genes before g and at least one of the 2 genes
  after g has its own hit in P within 5 genes of h on the same replicon.
- f = share of the other-species panel genomes with a positional orthologue; defined with at least 3 such genomes.
- Single copy: no other gene of the chromosome aligns with at least 40% identity over at least 80% of both genes.
- Classes as in section 1, with f <= 0.5 for island genes and f >= 0.8 for natives.
- Block units. Island region: a published island region with at least 3 island-class genes that are at least 50% of
  its island-labelled genes; only those genes carry the region. Native block: 11 consecutive genes of one contig, none
  island-labelled or within 5 genes of an island-labelled gene, at least 9 of them native; only its native genes
  enter its score; tiled from the first gene of each contig, 11 genes forward after a block, 1 gene forward after a
  window that fails.
- Flags (strata only): `absent` = island gene without any alignment (40% identity over 50% of the gene) in any other
  species; `pool` = island gene with a 70% / 70% alignment outside its position in another species; `deep` = native
  with a hit in at least 2 sibling-genus genomes; `fast` = native in the lowest tenth, within the chromosome, of mean
  identity to its positional orthologues.

**2.4 Checks of the labels against each other** (no score involved).

- Development gates, on the units with at least 3 other-species genomes (Escherichia with Shigella; Yersinia). Gate
  1: at least 80% of the single-copy published backbone genes have f >= 0.8 (median over the chromosomes of a unit).
  Gate 2: the conflict share c = island-labelled genes with f >= 0.8 / island-labelled genes; its median over the
  chromosomes of a unit is at most 5% where backbone genes outnumber island genes and at most 20% otherwise. If a
  gate fails after at most half a day of fixes on development, the positional-core class is withdrawn and the
  fallback of section 3 applies.
- Per sealed genus, before unsealing: under 70% of the single-copy backbone genes with f >= 0.8 = "panel
  unreliable"; c above 20% = "label implausible". Either removes the genus from the primary population.

## 3. Genomes, eligibility, split

- **Development (open).** The 24 Enterobacterales chromosomes of the island set, in 3 units: Escherichia with
  Shigella (14 chromosomes), Salmonella (5), Yersinia (5). DNA scores and panel labels on all 24; protein methods on
  the 4 with a deep search (NC_002695.1, NC_003198.1, NC_007384.1, NC_006155.1). A unit's value is the mean over its
  chromosomes.
- **Sealed.** The 18 selected genera, one chromosome each. Replicon rule: only the largest replicon of an assembly
  can be a test chromosome. Brucella suis chromosome II (NC_010167.1) is therefore replaced by the next Brucella
  chromosome of the README ranking that is a largest replicon. The unit of replication is the genus (by the genus
  taxon of the server's taxonomy); genera of one family are never split between Test A and Reserve B: Rhizobium with
  Sinorhizobium, Bradyrhizobium with Rhodopseudomonas.
- **Eligibility**, from label files only, computed for all 18 genera before the split.
  - P1: at least 3 other-species panel genomes; not "panel unreliable"; not "label implausible"; at least 500
    natives; at least 40 island genes.
  - P2: the same panel conditions; at least 5 island regions; at least 50 native blocks. Ladder, decided from label
    files only: if fewer than 8 Test A genera are P2-eligible, a native block needs 7 natives among its 11 genes
    instead of 9, for all genera; the step in force is written into the freeze.
  - S-pub (published labels as they are): at least 300 backbone and 50 island genes; at block level at least 20
    backbone and 5 island regions.
- **Split**, fixed now. Family units are ordered inside two strata by ascending SHA-256("<alphabetically first genus
  of the unit>|reserve-B-v2"). High stratum (a genus with at least 300 published backbone genes on the selected
  record): Mycobacterium, Acinetobacter, Corynebacterium, Rhizobium+Sinorhizobium, Pseudomonas, Burkholderia,
  Bacillus, Bradyrhizobium+Rhodopseudomonas. Low stratum: Streptococcus, Staphylococcus, Brucella, Shewanella,
  Haemophilus, Prochlorococcus, Vibrio, Rickettsia. With n_P1 = the number of P1-eligible genera:
  - n_P1 >= 14: Reserve B = the first 2 single-genus units of the high list and the first 3 units of the low list,
    skipping units that are not P1-eligible. Test A = the other 13 genera. Reserve B is scored by nothing until the
    first DNA-reading model is frozen.
  - 9 <= n_P1 <= 13: no Reserve B; all 18 genera are Test A.
  - n_P1 < 9, or a development gate fails: fallback PUB. All 18 genera are Test A; P1 and P2 are reported on the
    published labels without verdicts (island against backbone genes in the genera with at least 300 backbone and 50
    island genes; mean recall over 0-5% as the headline, because only 3-14 backbone genes sit at the 1% point).
- **Floors.** P1 needs 9 eligible Test A genera, P2 needs 8. Below its floor an endpoint is reported without verdicts.

## 4. What a method may see

- **V1 (ranked).** Hidden from every method: every library proteome of the genome's genus; every proteome r that is
  as close as the closest own-genus proteome; every r with s(r) >= 90. s(r) = the median, over the genes with a hit
  covering at least 70% of the gene in r, of the best such identity; defined when at least 30% of the genes have such
  a hit (`bench/hard/views.py`). A genus name alone would leave Shigella visible to an Escherichia genome. The hidden
  set of every chromosome is written into the freeze.
- **V2 (required concordance, not an endpoint).** V1 plus every proteome of the family: the same genes as if the
  lineage were poorly sampled. A genus where V2 hides fewer than 3 proteomes more than V1 is "V2 not evaluable".
- **Exactness.** A view filters the stored hit list. Genes whose 1,000-hit list is full are searched once more to
  4,000 hits; a view keeps the first 1,000 visible hits and is flagged exact when 1,000 visible hits remain or the
  stored list holds fewer than 4,000. The share of exact views is printed per class and regime.
- **Input contract.** Scoring scripts read the hit table, the genome record and a gene table (gene, contig, start,
  end, strand). They never open a label file. No product text reaches a method or a printout of a sealed genome.

## 5. Scores

Higher = more foreign. Every method declares its receptive field: per-gene (the score depends on the gene's own
sequence or hits and on genome-wide statistics) or context (it depends on neighbouring genes).

| name | input | what it is |
|---|---|---|
| `gc3`, `codon`, `kmer` | DNA | `bench/islands/composition.py`: deviation of a gene's coding sequence from the chromosome |
| `wide` | proteins | the taxonomy-free table model of `research/array/wide.py`, checkpoint of seed 0, mean over 3 random groupings of 64 genes; per-gene |
| `wide_rule_ai` | proteins | the fixed rule on wide's own rows: best similarity outside the genome's nearest rows minus best inside |
| `ai_all` | proteins + taxonomy | Alien Index on the hit table, home group = phylum |
| `ai_narrow` | proteins + taxonomy | the same, home group = the narrowest of family, order, class with at least 10 visible genera besides the own, else the phylum |
| `ai_gene` | proteins + taxonomy | the same, home rank chosen per gene: the narrowest of family, order, class, phylum in which the gene has a visible hit covering at least 40% of it |
| `fuse` | DNA + proteins | fusion of `codon` and `wide` (below) |
| `fuse_triv` | DNA + proteins | the same fusion with `absent_near` in place of `wide` |

"Linear rules" = {`ai_all`, `ai_narrow`, `ai_gene`, `wide_rule_ai`}, the table scores of `bench/hard/views.py`. The
pipeline's own Alien Index and its confidence tiers (tag `rules-R0`) are printed beside as counts and enter no test;
if the Spearman correlation of the table `ai_all` with the pipeline's (median over the development chromosomes with
a pipeline run, table computed under the pipeline's own hidden set) is below 0.9, every verdict on the linear rules
reads "table rules, not the pipeline".

**Trivial scores** are methods under test, in both windows: `short` (minus log length), `nohit` (no visible hit),
`n_rows` (minus the number of visible proteomes with a covering hit), `absent_near` (1 minus the best bit score per
residue, relative to 2 bits per residue, clipped to 0-1, among the 12 visible proteomes with the highest s(r); 1
without a hit there), `best_id` (minus the best visible identity), `sim_anom` (distance of the best visible bit score
per residue from the genome's median), `copies` (stored hits per visible proteome with a hit), `viral` (share of
viral proteomes among the 10 best visible hits), `random` (the first 8 bytes of SHA-256("<gene id>|random-v1")).

**Quantile.** q_x = (rank - 0.5) / n of score x among the n genes of the chromosome that x scores, ties at the
average rank.

**Fusion.** Candidates, in this order: (1) max(q_codon, q_wide); (2), (3), (4) the weighted mean of the normal scores
of the two quantiles with weights codon:wide 1:1, 3:1, 1:3. A candidate is computed on the genes both components
score and expressed as a quantile among them; a gene scored by one component takes that component's q; a gene scored
by neither is unscored. The candidate with the highest development value of the P1 metric (V1, gated labels, units
with a positional-core class) is fixed at the freeze; ties go to the first in the list. `fuse_triv` uses the same
candidate.

**Windows**, one definition for every score. W1 = q. W11 = the mean of q over the scored genes among the gene and
its 5 neighbours on each side (same contig; circular only on a closed chromosome); an unscored gene takes the mean of
its scored neighbours; a window with fewer than 3 scored genes is unscored. `x_nb` = W11 of x; `fuse_nb` = W11 of the
quantile of `fuse`. In W1 a gene a method cannot score ranks last, is never called and stays in every denominator;
coverage is printed per method and class. Sensitivity S-w: unscored genes enter the window at the lowest rank
(`metrics.rank_window`).

## 6. Metrics

- **Gene level.** R@1% = share of a chromosome's island genes above the score exceeded by 1% of its natives, ties
  at their expectation (`metrics.recall_at`). Printed beside: mean recall over native call rates 0-5%; R@0.3% where
  a chromosome has at least 1,000 natives; recall with every island region weighing 1; AUROC (descriptive); the
  number of genes above the threshold by class (island / native / conflict / unresolved).
- **Block level.** Region score = median W11 value over the region's island genes; block score = median W11 value
  over the block's natives; a unit with fewer than 3 scored genes among those is unscored (an island region then
  counts as missed, a native block as not called). LR@2%: allowed = max(1, floor(0.02 x native blocks of the
  chromosome)); a region is found when its score is strictly above the (allowed + 1)-th highest block score. Printed
  beside: the same with floor(0.01 x blocks) and with 0 allowed; region AUROC; the label-shift null (all labels moved
  together by one random offset along the chromosome, 1,000 offsets), at region level and for R@1% at W11.
- **Transport** (descriptive). The raw score exceeded by 1% of the natives pooled over the development chromosomes,
  per method and regime, applied unchanged to Test A: realised native call rate per genus, recall, calls by class.
- **Rule tiers** (pipeline `rules-R0`, as deployed): native call rate and island recall of the high tier, high or
  medium, any tier, as counts with exact binomial intervals, on the chromosomes whose pipeline run has finished.

## 7. Primary endpoints

- **P1, gene level.** Per-genus R@1%, window W1, regime V1, per-gene methods, on the P1-eligible genera of Test A.
  Contrasts, Holm over 3: `codon` minus `wide`; `fuse` minus `codon`; `wide` minus the linear rules (wide must beat
  each of the four).
- **P2, block level.** Per-genus LR@2%, window W11, regime V1, on the P2-eligible genera of Test A. Contrasts, Holm
  over 2: `codon_nb` minus `wide_nb`; `fuse_nb` minus `codon_nb`.

Nothing else is primary at the first unsealing. The endpoints on constructed cases (P3 gene-specific loss, P4 single
spliced genes, P5 spliced prophages) are registered in [DESIGN.md](DESIGN.md), are built in stage 1b and are opened
once, at the second unsealing.

## 8. When a difference counts

- **Inference.** Unit = genus. Differences are paired within genus. Interval: two-sided 95% t-interval on the
  per-genus differences. Test: exact sign-flip permutation over genera; Holm inside each endpoint; the smallest
  attainable p for the realised number of genera is printed beside every result. A contrast against a set (the four
  linear rules; all trivial scores) is an intersection-union test: point estimate = the smallest mean difference over
  the set, p = the largest p, interval limit = the smallest lower limit; every member must be beaten.
- **"A better than B"** needs all of:
  1. the interval excludes 0;
  2. the point estimate is at least 0.05;
  3. the Holm-adjusted p is below 0.05;
  4. the same sign in S-pub, S-a and S-b (below), each over the genera it can use. No analysis is waived: with fewer
     than 5 genera the sign is still taken and the claim carries "<analysis> on n genera"; if an analysis has no
     genus, no verdict is issued on that endpoint;
  5. if A or B reads similarity: the same sign and an interval excluding 0 under V2, on the eligible genera that are
     V2-evaluable; with fewer than 7 such genera the same sign only, and the claim carries "V2 not tested"; where V1
     and V2 disagree the V2 result is reported and V1 is marked "comparators possibly visible";
  6. if A and B belong to different input classes (DNA; similarity with or without taxonomy; both): an interval
     excluding 0 also under S-f, on the genera with at least 300 deep natives; with fewer than 7 such genera the same
     sign only, and "S-f not tested";
  7. A beats every registered trivial score of its window: for every member the mean paired difference is above 0
     and its interval excludes 0; otherwise the verdict reads "not shown to detect origin beyond a trivial proxy".
     For `fuse`, `fuse_nb` and any method that reads both DNA and similarity, the set also contains `fuse_triv` at the
     same window; failing it reads "fusion not shown to add to composition plus a trivial similarity proxy".
- "Equivalent at this resolution" = the interval lies inside (-0.05, +0.05). Everything else = "unresolved". With
  9-13 genera, differences under about 0.08 will mostly read unresolved; this is stated with every table.
- **Registered analyses.** S-pub: the published labels as they are (island against backbone genes; at block level
  published regions with one backbone region allowed); its negative rate is called "core-backbone call rate". S-a:
  positives = island genes flagged `absent` (the part similarity can decide; at block level regions whose island genes
  are at least half `absent`). S-b: without the genera flagged "label visible" (below) and without the genera whose
  published island genes outnumber the backbone genes. S-f: thresholds from deep natives only (at block level from
  the native blocks whose natives are all deep). S-w: section 5. Printed only: one unit per within-genome family;
  scorable genes only; units = orders.
- **Single-row restatement diagnostic.** Per genus, the AUROC, published island against backbone genes, of "has a
  covering hit at 80% identity or more in row r", maximised over the rows visible under V1; above 0.85 the genus is
  flagged "label visible".
- **How the two endpoints combine.** An unqualified "A better than B" needs both P1 and P2; one alone is reported as
  "at gene level only" or "at block level only"; opposite signs mean no winner. Error across the two endpoints is not
  controlled, and the report says so.
- **Hard natives** (strata, no label change): `fast`; highly expressed genes (natives matched by
  `research/placement/markers.py --strict`); natives under 100 residues. Each stratum's call rate is taken at the
  threshold that passes 1% of the natives outside the stratum, on the genes that both methods of a pair score (an
  unscored gene is an abstention, printed as coverage, never a non-call). A stratum enters with at least 30 such
  genes in at least 6 genera. "A not worse on hard natives than B" = the upper limit of [A minus B] is at most 5
  points in every stratum that entered. Natives without any hit are reported apart: no similarity method scores
  them.
- Every statement that compares DNA with similarity carries: "island anchor only" and "own genus hidden" (the
  evidence that defines the labels, the other species of the genus, is withheld from similarity methods while
  composition keeps all of its evidence).
- **Exposure of the learned model.** Per genus: whether its library proteome, family or class was among the
  training genomes of `wide`; results are printed with and without exposed genera.

## 9. Sealing

- **What is sealed** is the join of scores with labels. Label files, panels, eligibility tables, the split, searches
  and scores of sealed chromosomes may exist before the freeze; none is an outcome.
- **Freeze F1** is a tagged commit that holds: this protocol; the amended README of `bench/islands`; SHA-256 of the
  library FASTA and DIAMOND database, the taxonomy dump and the `wide` checkpoint; the tag `rules-R0`; all label and
  panel files, hidden sets and eligibility tables; the split and the unit lists of P1 and P2; the fusion candidate;
  the transported thresholds; the evaluator.
- **Unsealing U1**: the evaluator is run once with `--unseal A1`, writes only the registered tables and appends date,
  commit and hashes to `bench/hard/results/unseal.log`. Anything computed afterwards that is not registered here is
  labelled exploratory. After U1, Test A is "exposed at design level"; a later model family may be evaluated on it at
  most 3 times, each time logged.
- **Reserve B**, if it exists: its labels and scores live in a directory that the evaluator refuses to read without
  `--unseal B`. It is opened once, when the first DNA-reading model is frozen; all methods, old and new, are then
  scored on it together.
- **Training exclusions**: [EXCLUDE.tsv](EXCLUDE.tsv), a dated append-only list (the 18 genera and their families,
  every panel genome, later every donor and viral genus the benchmark uses). No future model may train on them, and
  training code may not import the test-only generators of `bench/hard`.

## 10. Order of work

1. This protocol, the README amendment, checksums, exclusions. Replicon lengths; the Brucella replacement; label fix.
2. Panels for the 18 sealed and the 24 development chromosomes; panel classes; development gates.
3. One library run at 4,000 hits: the Brucella replacement and the genes with full lists.
4. Views (first 1,000 visible hits, exact flag), `ai_gene`, `absent_near`, `random`, restatement diagnostic;
   `wide` under an explicit hidden set.
5. Evaluator: per-chromosome metrics, scored-only windows, fusion candidates, the block rule, t-intervals, sign-flip,
   Holm, set contrasts, the label-shift null, the registered analyses, `--unseal` with its log.
6. Development numbers with published and gated labels; fusion candidate and transported thresholds fixed.
7. Labels only for the 18 sealed genera: eligibility, n_P1, split. Sealed chromosomes scored without label files.
8. Freeze F1, unsealing U1, report (`docs/hard-results.md`) with the coverage table of the scenario matrix.

Dropped first if time runs short: rule tiers of sealed chromosomes; two of the three groupings of `wide` under V2.
Never dropped: P1 and P2 under V1 with gated labels, the V2 concordance, S-pub, S-a, S-b, S-f, the trivial and
fusion scores, the label gates, eligibility and split, the freeze and the logged unsealing.

## 11. What this supersedes in the first pre-registration

The dated amendment in [bench/islands/README.md](../islands/README.md) lists the changes: the three outcomes
registered there are withdrawn before any run (they compared methods against the published backbone genes only,
with thresholds set by a handful of genes, and needed a second pipeline run per chromosome with a home group that
is undefined once the family is hidden); negatives are the positional-core natives; the own genus is hidden by
measured closeness; one window rule for every score; largest replicon only; genus as the unit.

## 12. Known limits

- Blocks only: the island set starts at regions of about 4 kb. Single-gene transfers, old transfers and donors are
  not in the positives; they are the business of the constructed scenarios of stage 1b.
- Positives are the island extreme, negatives the positional-core extreme; most genes stay unlabelled.
- Both real-label endpoints rest on the panel code; its safeguards are label-against-label gates and the required
  agreement with the published labels (S-pub).
- The other species of a genus lie at very different distances in different genera; the median identity to the panel
  is printed per genus, not corrected. Species are NCBI species, which is uneven.
- Pathogen- and Proteobacteria-heavy; 5 of the 18 genera lie in one order.
- One engineer builds labels, runs methods and writes the evaluator; the seal is procedural (hashes, a locked
  directory, one logged run).
