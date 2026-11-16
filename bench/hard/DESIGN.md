# Hard benchmark: the full design

This is the design as it stood on 2026-11-16 after six independent proposals, three critiques, a synthesis, three verifications, a revision and a second

verification. It is kept verbatim as the registered plan of all stages. Read it together with the two lists at the end:

the patches of the second verification and the corrections of its two judges. Where they differ, a patch rules over the design text and a correction rules

over a patch. For the first freeze the operative text is [PROTOCOL.md](PROTOCOL.md).

Staging after the patches: the first freeze (F1, unsealing U1) holds the components RI and GP with the primary endpoints P1 and P2. The components HV, VR, XK,

SP and FR, with the primary endpoints P3, P4 and P5, are stage 1b: they are built on development genomes, frozen once (F2) and opened once (U2). Wherever the

design text below says "stage 1b" for the components PX and VM, read "stage 1c". Day numbers in the design text are superseded by the build order of the patches.

No part of this text was written after a score of a sealed chromosome was joined to a label.

## Estimands

UNIT. One gene: an annotated non-pseudo CDS whose protein has at least 30 residues (the pipeline's cleaned protein set, as bench/islands/labels.py already does; shorter CDS are counted, printed and enter no denominator), keyed by contig, start, end, strand; a residue block for chimeric genes; of one physical assembly.

CLASSES. N native: the locus has descended vertically in this genome's lineage since depth tau. T transferred: the DNA entered the lineage from another lineage (cellular, phage or plasmid pool) after tau AND is physically joined to native DNA. C contaminant: sequence that is not part of the organism (own contig, no link to native DNA). M mobile replicon: a plasmid or phage present as its own molecule. U undecidable: the constructed truth stands but no admitted single-genome input can recover it; always reported, never used to rank.

PRIMARY ESTIMAND E_recent, written as the labels actually measure it. Reference = the other species of the genus in a panel of congener genomes that no method sees (component GP). tau = divergence of the test genome from those species; it differs between genera and is printed per genus (median protein identity to the panel).

- Positive ('gated island gene'): a gene that (a) lies with at least 50% of its length in a published island region (a block of at least 4 kb in the merged published set, without alignment to the comparison genomes of at least one of three published runs) and (b) has a positional orthologue in at most 50% of the other-species panel genomes.

- Negative ('PCn native'): not island-labelled, single copy in the genome, positional orthologue in at least 80% of the other-species panel genomes, not within 5 genes of an island-labelled gene.

- Both classes are read against the same panel, so they share one reference depth. Every other gene is unlabelled and enters no rate.

- What this is NOT: the positive class is 'accessory relative to congeners, inside a block the published comparison could not align'; gain is not polarised against loss before stage 3 (no empty-site test). The negative class is 'kept in place since the genus diverged'; it contains ancient transfers and gains at the root of the genus.

- The published labels as they are (island region against backbone region = at least 8 kb aligned in ALL comparison genomes of the 2008 run, one of them distant) are the registered sensitivity analysis S-pub. Their negative rate is called 'core-backbone call rate', never 'false-positive rate over natives'.

SECONDARY ESTIMAND E_deep (regression check only): ancestry outside the home lineage at phylum or domain rank, at any age; positives are far-donor constructs.

VIRAL RECIPIENTS. Stage 1: E_recent in viral form, 'not gained since the viral genus diverged' (viral genus-core natives of component VR; defined for every method class). Stage 1b: E_cell, cellular (host-derived) against viral ancestry of a gene in a viral genome.

GENOMES WITHOUT A MAJORITY (mosaic phages, 50:50 chimeras). 'Horizontal versus vertical' is defined only relative to a majority co-ancestry class. If one class holds at least 80% of the genes, the minority blocks are T. Otherwise the binary question is ill-posed and the estimand is the co-origin relation over gene pairs, scored as a partition and always relative to the intact parents, because natural phage genomes are already mosaic.

COORDINATES, not classes: donor class and distance; the radius hidden around the donor (an unsampled donor lineage and an older transfer are the same thing to similarity); block size; composition of the inserted gene under the recipient's model.

HOW LABEL ERRORS DEPEND ON THE METHODS' INPUTS (direction, then countermeasure).

(1) Toward DNA and neighbourhood methods: island positives include gains from the own genus pool (invisible to similarity once the genus is hidden, visible to composition) and are contiguous blocks. Countermeasure: S-a (positives absent from every other-species panel genome), window parity, block-permutation null.

(2) Toward similarity methods: the published comparison genomes were chosen by a proteome-distance window with no genus condition, so a comparator can sit in the library under another genus name; one visible row then restates the label, and an ancestral region lost in the comparators (labelled island) is 'found' for that reason. Countermeasure: V1 hides by measured distance as well as by genus; concordance under V2 (family hidden) is required; a single-row restatement diagnostic flags genera; gate (b) removes island genes that congeners conserve in place.

(3) Against any-age detectors: PCn natives that are genus-root or family-root gains are N under E_recent although an any-age detector is right to flag them. Countermeasure: S-f (thresholds from 'deep' PCn natives only) brackets the effect; it is not removed.

(4) Deployment condition: every verdict holds for 'own genus hidden'. The evidence that defines E_recent (congeners) is withheld from similarity methods while composition keeps all of its evidence. A method designed to use congeners cannot be ranked by this protocol. This sentence is printed next to every DNA-versus-similarity figure.

MATCH TO EACH METHOD. rules, ai_all: E_deep at phylum rank (kingdom or supergroup for eukaryotes, per home_group() in seeddrift/single.py); blind by design to in-phylum donors and to exchange inside one viral home group; their recall under E_recent is a comparison number, their native call rates are valid. ai_narrow: the same at the narrowest sampled rank of the genome. ai_gene: the same with the home rank chosen per gene (narrowest rank in which this gene still has a visible hit). place.node: lineage disagreement at variable rank, any age; inherits the labels of the references. wide: 'similarity pattern differs from that of the surrounding genes', taxonomy-free, age-agnostic, assumes a majority; undefined under 14 genes and for mosaics. gc3, codon, kmer: E_recent only, while the donor's composition differs and has not ameliorated; window versions assume blocks. A future DNA model: E_recent plus linkage (can separate T from C); blind to composition-typical single genes unless it also reads similarity.

USER-FACING QUANTITY. The share of true T among the calls at a frozen operating point. It is not identified on a real genome and is never reported as a point. What is printed, per method and genus, is the number of genes called at the P1 threshold split into gated island / PCn native (1% of that class by construction) / label conflict / unresolved. The 'precision standardised to a prevalence of 5%' of the first version is deleted.

SCOPE. Every stage-1 verdict and every verdict of the bar for the next model concerns bacterial chromosomes and tailed phages only. Nothing is claimed for eukaryote recipients before component EU is reported (stage 2c) or for archaeal recipients before stage 3e.

## Components

### RI. Real recent block transfers: the published island set gated by a hidden congener panel, near references hidden by distance, all input classes, window parity

**Purpose.**

On real chromosomes, which input (DNA composition, protein similarity, a taxonomy rule, or their fusion) separates recently acquired island genes from positional-core native genes at a strict threshold, gene by gene and block by block, once every reference as close as the own genus is hidden?

**Labels.**

BASE LABELS (bench/islands/labels.py, one fix: a gene lying at least 50% in an island region AND at least 50% in a backbone region is unlabelled and counted per chromosome; the two region sets come from different published runs).

PRIMARY CLASSES (label-only, built in component GP before any score is joined). Gene track: positive = gated island gene; negative = PCn native (definitions under ESTIMANDS). Block track: positive unit = gated island region (published island region with at least 3 labelled genes of which at least 50% are gated island genes); negative unit = native block (11 consecutive genes, none island-labelled or within 5 genes of an island-labelled gene, at least 9 of them PCn, tiled without overlap). Island-labelled genes with a positional orthologue in at least 80% of the other species are 'label conflicts': unlabelled, and their share c is printed per genus.

AMENDMENTS. Commit 620f18c holds the registered README. First bench/hard and the modified bench/islands/evaluate.py are committed as they are. Then ONE dated commit supersedes the README before any score is joined to a label:

(1) Replicon rule: only the largest replicon of an assembly (lengths from the assembly report) can be a test chromosome; every other replicon is the stratum 'secondary replicon' (scored, descriptive). The substring rule of the first version is removed. Brucella suis ATCC 23445 chromosome II (NC_010167.1, 419 of 1,231 genes labelled island) is replaced under README rule 3 by the next Brucella chromosome that is a largest replicon.

(2) One chromosome per genus. The 'up to 2 further chromosomes per genus' of the first version is REMOVED: PCn natives give every genus hundreds of negatives from one chromosome, and pooled strains repeated the same orthologues and needed a pooling rule.

(3) Units: the genus by current genus taxid of the server's taxonomy dump (not the first word of the 2019 organism string). Genera of one family form one split unit. Escherichia and Shigella are one development unit.

(4) README text corrected (lines 11, 16, 18, 74): comparison genomes were chosen by a proteome-distance window, not by genus; the merged island set starts at 4 kb; positive and negative meanings as under ESTIMANDS. The three registered outcomes are WITHDRAWN before any run: they need all 18 chromosomes (Reserve B is locked), the pipeline's ai_family (degenerate when the family is hidden, and a second pipeline run per chromosome) and AUROC or 5% recall against saturated backbone negatives. Outcome 1 survives as a descriptive figure on Test A ('core-backbone call rate of the high tier').

(5) Visibility. V0 = same species hidden (descriptive only). V1 (ranked) = hidden set H1 = (a) every library proteome of the same genus taxid; (b) every library proteome r with s(r) >= s of the closest same-genus proteome; (c) every r with s(r) >= 90, where s(r) = median, over the genes with a hit covering at least 70% of the gene in r, of the best such identity (defined when at least 30% of the genes have such a hit; from the stored deep table). On development, Escherichia and Shigella are hidden together. H1 is written into the freeze per genus with the number of rows added by (b) and (c). V2 = H1 plus the whole family (required concordance, see decision rules); a genus where V2 hides fewer than 3 proteomes more than V1 is 'V2 not evaluable'. V3 (order hidden) is REMOVED. Exactness: one deeper search (top 4,000 hits, nothing excluded) of every gene whose 1,000-hit list is full; a view filters the stored list and keeps the first 1,000 visible hits; flag 'exact' when at least 1,000 visible hits remain or the stored list holds fewer than 4,000. The taxonomy-enabled DIAMOND database and all per-genome exclusion runs are REMOVED.

(6) Gene universe as under ESTIMANDS; identifiers renumbered by position; no product text reaches any method or any printout of a sealed genome before its unsealing. In W1 a gene a method cannot score ranks last, is never called and stays in every denominator; coverage is printed per method and class.

(7) Windows, one definition for every score. q = within-chromosome quantile of the score among the genes the method scores (label-free; also the input of every fusion). W1 = q. Wk (k = 5, 11, 21 genes) = mean of q over the SCORED genes of the window centred on the gene (same contig, circular for a closed chromosome); an unscored gene takes the mean of its scored neighbours; a window with fewer than 3 scored genes is unscored; a region or block with fewer than 3 scored genes is unscored (an island region then counts as missed, a native block as not called). Registered sensitivity S-w: unscored genes enter with q = 0 (the first version's rule). Development prints both.

(8) Receptive field. Every method declares 'per-gene' (the score depends on the gene's own sequence or hits and on genome-wide statistics) or 'context' (it depends on neighbouring genes or on position). Per-gene methods are compared in P1. Context methods (every Wk score now, any neighbourhood-reading model later) are compared at gene level in the context track P1c: same classes and metric as P1, against EVERY registered context baseline (codon, kmer, fuse at W5, W11, W21: 9 scores), with the block-permutation null and the edge-versus-interior analysis as part of the result. P1 against unsmoothed fuse is valid for per-gene methods only.

METHODS BY DECLARED INPUT. Proteins: wide, wide_rule_ai, profile misfit. Proteins plus taxonomy: rule tiers, ai_all (table) and the pipeline's own ai_all where a pipeline run exists, ai_narrow, ai_gene, place.node. DNA: gc3, codon, kmer. Both: fuse, fuse_triv (absent_near in place of wide). 'Linear rules' = the set {ai_all, ai_narrow, ai_gene, wide_rule_ai}. fuse = one of 4 candidates (maximum of the two quantiles; mean of normal scores with weights codon:wide 1:1, 3:1, 1:3), the one with the highest development R@1% is fixed at freeze F1. wide = the checkpoints of the third-version 'all rows' model that exist at F1 (seed 0 exists; at most 3 seeds), 3 random 64-gene groupings per seed (at most 9 passes), score = mean over seeds and groupings. No checkpoint is trained for the benchmark. If the SD over seeds and groupings of the per-unit R@1% on development exceeds 0.02 at this cap, that variance is added to the intervals.

STRATA (they change no label): protein length (<100, 100-200, >200 aa); no hit under the regime; within-genome family size (mmseqs easy-cluster, 40% identity, 80% coverage: 1, 2-4, 5 or more); masked fraction (tantan, at least 20% against under 5%; if tantan cannot be installed the stratum is dropped); HEG = PCn natives matched by research/placement/markers.py (Pfam families Ribosomal_, RNA_pol_, EFG_, GTP_EFTU; E <= 1e-10); FAST, deep, genus-pool, absent-from-genus flags from GP; edge = non-island genes within 5 genes of an island-labelled gene; count-flagged 'distant-comparator' genera (island genes outnumber backbone genes: 7 of 18 today); secondary replicons; on development the literature-curated islands (Sakai 24, CT18 18, CFT073 13) and the 9 cryptic prophages of K-12.

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage).

M1 Mobile elements and islands in cellular genomes, real | T = gated island gene, N = PCn native (E_recent) | published regions gated by the panel, under V1 | V0; S-pub; block-permutation null; trivial scores | rules and ai_all (in-phylum donors); every similarity method on island genes without a hit and on gains from the genus pool; composition where islands are not compositionally distinct | R@1% (P1), LR@2% (P2) | stage 1, unsealing U1.

M2 Genome from a poorly sampled lineage | labels unchanged | the same genes under V2 | the same genes under V1 | ai_narrow widens its home rank; wide if it reads missing near rows as foreignness; DNA scores do not change (printed as such) | realised PCn call rate under V2 at the V1 threshold, split into deep and genus-restricted PCn natives (only the deep part is counted as false); R@1% with the V2 threshold | U1.

M3 Huge gene pool, no close homologue (orphans) | both classes exist: gated island genes without a hit (T) and PCn natives without a hit (N); 'no hit' is evidence for neither | real genes | the same genes under V0 | every similarity method ties them at the bottom; a method that ranks no-hit genes high turns native orphans into false calls | scorable share per method and class; AUROC inside the no-hit stratum; every headline metric includes these genes | U1.

M4 Short genes, natural | labels unchanged | genes under 100 aa of both classes | genes over 200 aa of the same class | similarity methods (against about 1.5e10 library residues a 50-aa protein cannot reach E <= 1e-5 below roughly 55-60% identity; proteins under 30 aa are dropped); single-gene composition (variance grows as 1/length) | R@1% and coverage by length stratum | U1 (matched version M24, stage 1b).

M5 Family expansion on real genomes | labels unchanged (PCn natives are single-copy by definition, so multi-copy natives are unlabelled) | calls counted per within-genome family, each family weighing 1 | gene-level counts | best-hit rules on multi-copy families | family-collapsed recall and the share of each method's unlabelled calls that fall in families of 5 or more (S-c) | U1.

M6 Hard natives for DNA methods: highly expressed genes | N | HEG stratum | other PCn natives | codon and kmer (expression-linked codon bias); similarity methods pass | call rate of the stratum at the threshold that passes 1% of the PCn natives OUTSIDE the stratum; AUROC of the stratum against the other PCn natives | U1.

M7 Fast-evolving natives | N (a FAST gene replaced in place by homologous recombination would be T: noted, direction against a correct method) | FAST stratum | other PCn natives | wide, profile misfit, sim_anom; taxonomy rules mostly pass | as M6 (threshold from PCn natives outside the stratum, so the rate is not capped) | U1.

**Genomes.**

Development (open): the 24 Enterobacterales chromosomes of the island set in 3 units (Escherichia+Shigella 14, Salmonella 5, Yersinia 5); DNA scores on all 24; protein methods on 12 (3 per genus by the README ranking: 4 searches queued, 8 more = about 36,000 proteins); E. coli K-12 MG1655 with the DH10B-mapped labels (464 backbone, 26 island genes).

Sealed: 18 genera = 16 family units (two-genus units: Rhizobium+Sinorhizobium; Bradyrhizobium+Rhodopseudomonas; if the server's taxonomy groups differently the units are rebuilt by the same rule before any score exists).

Measured today (island / backbone genes of the selected record): Acinetobacter 130/907, Bacillus 61/1443, Bradyrhizobium 911/334, Brucella chr II 419/158 (to be replaced), Burkholderia 374/663, Corynebacterium 118/1199, Haemophilus 105/17, Mycobacterium 282/637, Prochlorococcus 55/211, Pseudomonas 252/440, Rhizobium 165/578, Rhodopseudomonas 308/83, Rickettsia 18/15, Shewanella 265/110, Sinorhizobium 92/579, Staphylococcus 52/251, Streptococcus 76/187, Vibrio 169/135.

ELIGIBILITY, from label files only, computed on day 2 for all 18 genera and written into freeze F1.

- P1: panel with at least 3 other-species genomes; at least 70% of the single-copy published backbone genes are positional core (else 'panel unreliable'); conflict share c at most 20% (else 'label implausible'); at least 500 PCn natives; at least 40 gated island genes.

- P2: the panel and c conditions; at least 50 native blocks; at least 5 gated island regions.

- Constructed endpoints P3-P5: at least 500 PCn natives (island labels are not needed).

- S-pub (published labels as they are): at least 300 backbone and 50 island genes, or 20 backbone and 5 island regions. Certain today: 9 genera for both (Bacillus, Corynebacterium, Acinetobacter, Burkholderia, Mycobacterium, Sinorhizobium, Rhizobium, Pseudomonas, Bradyrhizobium). Rickettsia (18 island genes) can never be P1-eligible, so T, the number of P1-eligible genera, is between 0 and 17 and is not known before day 2.

SPLIT, fixed now. Strata from today's counts: 'high' = a family unit with a genus of at least 300 backbone genes, 'low' = the rest. Order inside a stratum = ascending SHA-256 (hex) of the UTF-8 string '<alphabetically first genus of the unit>|reserve-B-v2'. Computed today: high = Mycobacterium, Acinetobacter, Corynebacterium, Rhizobium+Sinorhizobium, Pseudomonas, Burkholderia, Bacillus, Bradyrhizobium+Rhodopseudomonas; low = Streptococcus, Staphylococcus, Brucella, Shewanella, Haemophilus, Prochlorococcus, Vibrio, Rickettsia.

- If T >= 14: Reserve B = the first 2 single-genus units of the high list and the first 3 units of the low list, skipping units that are not P1-eligible (with all eligible: Mycobacterium, Acinetobacter, Streptococcus, Staphylococcus, Brucella). Test A = the other 13 genera; it then holds T - 5 >= 9 P1-eligible genera. Non-eligible genera of Test A are scored and shown, in no mean.

- If 9 <= T <= 13: there is NO Reserve B for island labels; all 18 genera are Test A.

- If T < 9, or if the PC class fails its development gates (GP): fallback F1 = published labels are the primary classes, all 18 genera are Test A (9 evaluable for P1 and P2), verdicts carry the mark 'published labels, not gated' and need the same sign without Bradyrhizobium (the one count-flagged genus among the 9); no Reserve B.

- Floors: P1 needs 9 eligible Test A genera, P2 needs 8. Below the floor the endpoint is reported without verdicts.

Under the default split the published-label genera are 7 in Test A (Bacillus, Corynebacterium, Burkholderia, Sinorhizobium, Rhizobium, Pseudomonas, Bradyrhizobium) and 2 in Reserve B.

The genera lost to eligibility are printed with genome size and GC content: that list is the stated limit of the evaluated population (small AT-rich genomes are the likely losses).

**Controls.**

1. V0 is printed and never ranked.

2. Overlap and restatement. Inventory by taxid and assembly accession of library proteomes that are the same assembly, species or genus as a test chromosome. Single-row restatement diagnostic: per genus the AUROC, published island against published backbone genes, of 'has a covering hit at 80% identity or more in row r', maximised over the rows visible under V1; above 0.85 the genus is flagged 'label visible'. Computed on development now and on Test A by the frozen evaluator at unsealing.

3. Trivial scores are methods under test in every window: length; no-hit indicator; n_rows; absent_near (1 minus the best bit score per residue relative to the self score among the 12 nearest visible references); best_id; sim_anom; copies; viral_share; random. A method must exceed EVERY registered trivial score of its window on Test A (intersection-union; nothing is selected on development, the development ranking is printed as information).

4. Fixed untrained taxonomy-free comparators in every rule-versus-model contrast: wide_rule_ai and the profile-misfit statistic of research/array/ceiling.py (adapter for whole genomes).

5. Block structure: block-permutation null for every Wk and region figure (all labels shifted together, 1,000 shifts); the gain from smoothing a weak score (length: 0.61 at gene level on development) printed beside every method; edge analysis (first and last two genes of an island against its interior).

6. Registered analyses. Required for a claim (each addresses a named bias): S-pub (published labels as they are); S-a (positives = island genes absent from every other-species panel genome: the part similarity can decide and the handle on loss-mislabelled islands); S-b (without genera flagged 'label visible' and without the count-flagged distant-comparator genera). Required for cross-class contrasts: S-f (thresholds from deep PCn natives only; direction: S-f favours similarity methods, all PCn natives favour recent-only detectors). Printed only: S-c one unit per within-genome family; S-d scorable genes only; S-e units = orders; S-w unscored genes ranked last in windows.

7. Share of full 1,000-hit lists and of 'exact' views per class and regime. Table ai_all against the pipeline's ai_all: median Spearman of at least 0.9 on development is required before F1; otherwise the table scores are renamed ai_table, flagged 'not the pipeline's index', and the pipeline's own ai_all is the rule comparator in P1 wherever a pipeline run exists. A bit-score Alien Index is printed beside the E-value one.

8. Exposure of the learned models: per genus whether its library proteome, family or class was among the 732 training genomes; results with and without exposed genera. Known directions: the development order was held out of training and the sealed genera were not (so a transport difference between wide and codon is not a property of the methods); wide's training labelled real islands and prophages of training genomes as native (a bias against wide). The retraining control is stage 2d.

9. Call split: beside every R@1%, the number of genes above the threshold by class (gated island / PCn native / label conflict / unresolved), and each method's top-1% PCn natives with their panel flags.

10. Variance of wide over seeds and groupings measured on development.

**Metrics.**

Gene track (P1, P1c). R@1% = share of a genus's gated island genes above the score exceeded by 1% of its PCn natives (same chromosome, same regime, ties at expectation). Co-reported: mean recall over PCn call rates 0-5% (less threshold-noisy); R@0.3% where a genus has at least 1,000 PCn natives; recall with every island region weighing 1; AUROC (descriptive).

Block track (P2). Region score = median W11 value of the region's genes. LR@2% = share of gated island regions above the score exceeded by 2% of the chromosome's native blocks (at least 1 block of at least 50). Printed beside: the same at 1% and with no native block allowed; the realised region-level rate; region AUROC; the block-permutation null.

Transport: the raw score exceeded by 1% of the PCn natives pooled over the development chromosomes with a PC class, per method and regime, applied unchanged to Test A: realised PCn call rate per genus (median, maximum), recall, genes called by class. No precision is computed.

Rule tiers (pipeline tag rules-R0, as deployed with the genus hidden; flagged where H1 hides more rows): PCn call rate and recall for high, high+medium, all, as counts with exact binomial intervals; recall of the continuous scores at each tier's call count is descriptive and never a contrast.

Intervals and tests as under decision rules. No operating point below 0.3% is reported.

**Cost.**

Stage 1, days 0-3. Searches: none for the 18 selected chromosomes (running); Brucella replacement (2,000-3,300 proteins); 8 development chromosomes (about 36,000 proteins, 1 h); the top-4,000 re-search of full lists for 12 development and 18 sealed chromosomes (roughly half of about 110,000 genes; 2-4 h, rate measured on day 0). Rule pipeline: 4 development and the Test A chromosomes, one at a time at night, in hash order; if the day-0 measurement exceeds 1 h per chromosome only 4 development and 6 Test A chromosomes are run. Code: 563 lines exist in bench/hard (views 166, metrics 140, evaluate 257); about 350 new lines (H1 by distance, top-4,000 views, ai_gene, scored-only windows, context track, t-intervals with sign-flip tests and intersection-union contrasts, adapters: job file from a taxid without a pipeline run, explicit hidden-row mask for wide, profile misfit for whole genomes). Downloads: 1 GenBank record.

**Limits.**

Blocks only; positives are the island extreme and negatives the positional-core extreme (most genes stay unlabelled); gain is not polarised against loss; no donor; no single genes; no old transfers; pathogen- and Proteobacteria-heavy; 5 of the 18 genera lie in one order (Hyphomicrobiales) and at most about 12 orders are represented. Both real-label endpoints now rest on new panel code (GP), whose only safeguards are label-versus-label gates and the required concordance with the published labels. Any-age detectors are still charged for genus-root gains among PCn natives; S-f brackets this. All figures hold for 'own genus hidden'. With 9-12 genera, differences under about 0.07-0.08 are not resolved. Rule tiers are the deployed pipeline on unedited genomes only. TRAINING NEED: the 18 genera and their families stay out of training queries; whole replicons as DNA with coordinates; queries without any hit must keep a target (14.5% of the E2 queries are discarded today).

### HV. Hit-table views: certified natives made hard for similarity methods (gene-specific loss, hidden paralogy, a false reference cell)

**Purpose.**

Does a similarity method turn a certified native gene into a call when the evidence about that one gene is degraded in a way that leaves its ancestry unchanged, and can it still tell such a gene from a near-donor transfer that suffered the same degradation?

**Labels.**

Base genes are PCn natives of the edit chromosome (native by positional conservation in hidden congeners). Each edit only removes, weakens or relabels evidence in the stored hit table, so N stays exact under E_recent. For an any-age detector N is exact only in the deep-vertical stratum DV defined below. Edits are written in the existing hit-table format; ai_all, ai_narrow, ai_gene, place.node, wide, wide_rule_ai and the trivial scores read them unchanged. Lists are refilled from the top-4,000 search by one procedure for natives and positives.

LOSS CLADE L of a recipient = the narrowest of family, order, class that holds at least 5 visible library proteomes outside H1; recorded per genus with its size and the median identity of its proteomes to the recipient (a family is 1 to 104 proteomes, so the rank alone says little).

ELIGIBLE BASE GENE: its best visible hit under V1 lies in L and it has hits in at least 4 proteomes of L (otherwise the edit deletes nothing that matters). Up to 300 per chromosome by ascending SHA-256('<accession>|<gene id>'); a unit with fewer than 100 eligible genes is descriptive.

DV (deep-vertical stratum, fixed from the unedited table): eligible base genes with a covering hit (at least 40% of the gene) in at least half of the visible sister clades of L one rank up (at least 2 sister clades, else one rank higher). There N also holds beyond L, so the edit is a pure test of the mechanism. The other base genes are 'N under E_recent only' and are reported apart.

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage).

M8 Long-term differential loss in relatives (patchy distribution) | N | graded LOSS: for each base gene the hits in a random 25, 50, 75 or 100% of the proteomes of L are removed (subset drawn per gene, seed SHA-256('<gene id>|<grade>')); every other gene stays under V1; the genome profile comes from the unedited table | C0 = the same gene unedited (paired); for grade 100% with L = family also the same gene when the whole genome is under V2 | ai_narrow and ai_gene as the home group empties (computed and printed, not assumed); ai_all changes when the best in-phylum hit was in L (computed); place.node partly; wide passes only if it compares the remaining cells with the genome profile; DNA methods do not change | paired inflation at the unedited threshold, grades 50% and 75% pooled (P3a); grade 100% is printed as 'leaning undecidable' (losing every member of a clade of 5 or more proteomes is more parsimoniously a gain at the root of the genus) and enters no verdict | stage 1, unsealing U1.

Matched positives for M8: the reciprocal-best-hit orthologues, in a donor genome, of the SAME base genes. Donor = a complete genome of a species absent from the library in a sister clade of L one rank up (another family of the order when L is the family; this is the MID donor of component SP). Each orthologue receives the same deletion as its partner (same subset of L's proteomes, same refill). Visibility = the recipient's H1 plus the donor's genus and its distance set, applied to natives and positives alike. Cell D1 = this. Pairing by orthologue removes differences of breadth, length, rate and family between the classes.

M9 Family expansion and hidden paralogy | N (orthologue lost in relatives, out-paralogue kept) | PSWAP: for base genes with at least 2 stored hits in at least 3 proteomes of L (raw hit files: the table columns keep one hit per proteome), the best hit in each such proteome is deleted, so the best remaining cell there is the paralogue; far rows untouched | uniform swap (best hit deleted in every proteome with at least 2 hits, near and far); the unedited gene | best-hit rules whose home rank is at or inside L; place.node; wide if lowered near cells read as foreignness | inflation relative to the uniform swap | U1 (first HV item to drop).

M10 Mislabelled or contaminated reference proteome | N; the far hit is a property of the reference | MISLABEL: the gene's real hit in the hidden same-genus proteome (near-identical, real alignment numbers) is assigned to one library proteome R of another phylum in which the gene has no hit (R per gene = smallest SHA-256('<gene id>|<proteome id>') among such proteomes) | the same cell assigned to a visible proteome of another genus of the home family that has no hit for this gene | ai_all, rule logic and place.node trust R's label; wide should discount a one-row anomaly but has never seen one | inflation at the unedited threshold; descriptive: share of each method's calls on real genomes that rest on a single row at 90% identity or more | U1; real contaminated entries: stage 3 (M38).

M11 Ambiguity of the topology, near donors | T by construction, U when no visible row tells donor from recipient | D2: the positives of D1 with the donor's own clade at the rank of L hidden as well, against natives at grade 100% | the cell is its own control | every method must sit near 0.5 | AUROC, expected with an interval overlapping 0.45-0.55 | U1.

**Genomes.**

Edit chromosomes: the 4 development chromosomes with a deep search (3 units) and every Test A genus with at least 500 PCn natives and a loss clade (island labels are not needed, so genera that are not P1-eligible can still be units). Reserve B's edits are generated with the same frozen code when B is opened. Matched positives: at most 300 orthologues per recipient (about 6,300 proteins in all, searched to 4,000 hits); one donor genome per recipient, shared with component SP where L is the family, otherwise one extra genome from another order of the class. The unit list of P3 is computed from the library table and the label files and written into freeze F1.

**Controls.**

Every edit has a paired control on the same gene. wide's genome profile always comes from the unedited table of the regime, and edited genes are scored in batches so that no 64-gene table holds more than 2 of them (new argument of the scoring adapter). Positives and natives share deletion, refill procedure and hidden set. Every trivial score is scored in BOTH orientations on every HV cell and a method is compared with all of them (the first version registered only the orientation that is anti-predictive here). Rates are also printed conditional on the gene keeping at least one visible hit, so that abstention is not counted as robustness. Results are binned by the measured identity to the nearest visible proteome (at least 70%, 50-70%, under 50%). The figures of ai_all, ai_narrow and ai_gene are computed, never printed as 'by construction'. Rule tiers are not available on edited tables in stage 1 (the pipeline runs its own searches); the rule family is represented by the table scores until the stage-2 pipeline hook, and the tree step of the pipeline is exercised on no hard cell before then.

**Metrics.**

All at ONE absolute threshold per method: the score exceeded by 1% of the PCn natives of the unedited chromosome under V1.

Inflation (P3a) = call rate of the edited genes minus call rate of the same genes unedited, paired per unit, grades 50% and 75% pooled; the ratio edited/unedited is printed beside it; reported for all eligible base genes, for DV, and conditional on a visible hit. For grade 100% with L = family the inflation is decomposed into [whole-genome V2 rate minus 1%] + [gene-only loss minus whole-genome V2 rate]; only the second term is a context effect of the method.

Discrimination (P3b): D = recall of the D1 orthologue positives above the threshold minus the call rate of their paired LOSS natives above the same threshold (grades 50% and 75% pooled, per unit). A score that calls unedited natives, or that leans on raw similarity level, fails this by itself. Also printed: recall of the same positives without any deletion; the two-set AUROC (descriptive only).

Null: AUROC of D2.

**Cost.**

Stage 1, days 2-3. Search: about 6,300 orthologue proteins to 4,000 hits (15-30 minutes), no other search beyond the re-search of RI. Code: about 280 lines (loss clade and eligibility, graded deletion with refill, orthologue pairing, PSWAP on raw hits, MISLABEL, per-gene adaptive index). GPU: about 8 extra wide passes per edit chromosome.

**Limits.**

Random deletion of clade members is cleaner than real loss; real patchy genes are also faster-evolving and shorter, and PCn natives are by definition not patchy inside the genus: this measures susceptibility to the mechanism, not its frequency in nature (real patchy natives: stage 3, M35). Genes eligible for the paralogue swap are multi-copy in relatives, not a random sample. One clean false cell is simpler than block-wise reference contamination. Every edit is aimed at similarity inputs; DNA methods are untouched, so HV figures are read only as paired inflation of the same model and never as a rate to compare with a model that does not read the table. These generators are test-only for future training. TRAINING NEED: natives with gene-specific masking of near rows as explicit negatives (today rows are hidden for the whole genome only); several hits per reference proteome as input; single-cell label noise in reference rows; undecidable trials removed or given target 0.5 (about a quarter of the E2 trials).

### XK. Cross-kingdom regression check and repair of the old spike-in evaluation

**Purpose.**

Does a method keep the cross-kingdom detection (bacterium or archaeon into eukaryote and back) that every current method already has?

**Labels.**

Constructed labels only.

(a) Protein form, descriptive with audit: the held-out E2 trials by recipient and donor kind, all cells including eukaryote from eukaryote; label 1 = gene of another genome, label 0 = a random native gene of the recipient (unlabelled with respect to real transfers; in eukaryotes this includes organelle-derived genes that E_deep counts as true).

(b) Protein form, binding for protein-reading methods on bacterial recipients: E1 on E. coli K-12 restricted to unmutated spikes (divergence 0) against the 464 natives that map to backbone regions of DH10B. The 'other' natives of E1 are never negatives again. Entamoeba and Legionella: the only certified natives are the 'core' ribosomal and RNA-polymerase genes, few and trivially native; they give a harness regression number flagged 'saturated', nothing more.

(c) DNA form, binding for any DNA-reading model: cell XD of component SP (archaeal and eukaryotic CDS spliced at DNA level into bacterial chromosomes, with relocation placebos), and from stage 2c bacterial CDS spliced into eukaryote scaffolds (component EU). E1 and E2 are protein sets without DNA or gene order and cannot score such a model.

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail | metric | stage).

M12 Cross-kingdom transfer (regression only) | T by construction under E_deep | E2 cross-domain cells; E1 unmutated spikes against backbone-mapped natives | mutated-native placebo | none: saturated (wide 0.98-0.99, fixed rule 0.90-0.97; average precision 0.915-0.916 for all methods against backbone negatives) | AUROC per E2 cell inside a tolerance band (leaving it triggers an audit); average precision in E1 (binding) | stage 1, unsealing U1.

M12-DNA | T by construction | cell XD of SP | relocated natives | composition should pass (distinct composition), similarity should pass (far donor); a model that lost either input fails | R@1% against fuse, margin 0.05 | stage 1, unsealing U2; eukaryote recipients stage 2c.

M19-euk (descriptive) | T / C / N | the existing construct of bench/contamination.py (Encephalitozoon cuniculi with 6 Legionella fragments of 60 kb as extra contigs) plus the same fragments inserted inside E. cuniculi chromosomes and one native E. cuniculi segment as its own contig | integrated against contig placement of identical genes | proteins-only methods cannot use linkage; this is the one place in stage 1 where the pipeline's contig test (eukaryotes only, at least 3 genes) can fire | share of contaminant genes reported as confident transfers; three-class confusion | U2.

**Genomes.**

E2: the 71 held-out genomes (43 eukaryotes, 15 archaea, 12 bacteria, 1 virus). E1: E. coli K-12 (binding), Entamoeba histolytica and Legionella pneumophila (harness numbers). E. cuniculi with Legionella fragments. All of this is development material; no sealed genome is involved except through cell XD of SP.

**Controls.**

1. Mutated-native placebo: mutate() of bench/spikein.py applied to 200 backbone-mapped K-12 natives at every spike divergence level; their call rate at the 1% threshold must stay within 2 percentage points of the unmutated versions, otherwise random substitution itself creates signal and every mutated-spike figure is void.

2. Certified lower bound on the contamination of the old 'other' negatives: a >= max over thresholds of [tail(other) - tail(backbone)] / [1 - tail(backbone)], printed per method.

3. Retired for good: average precision against all natives and recall at a false-positive rate or count measured on unlabelled natives.

4. The E2 cells are also printed with a strict-threshold figure (recall at 1% of the label-0 trials), marked 'negatives unlabelled'.

**Metrics.**

E2: AUROC per cell. Band = the larger of 0.02 and 3 times the SD over wide's seeds (0.02 while only seed 0 exists). A new method version that falls below the current wide value by more than the band in a cross-domain cell triggers an AUDIT of that cell (which label-0 genes it ranks high), not an automatic failure. E1 (binding for protein-reading methods): average precision of unmutated spikes against backbone-mapped natives not below the current value by more than 0.02. XD (binding for DNA-reading models): R@1% not below fuse by more than 0.05.

**Cost.**

Stage 1, day 2, about 2 hours, plus the contamination construct through the pipeline on day 5 (one pipeline run). No new genome; about 800 placebo proteins (the fixed cost of a small search is measured on day 0); about 120 lines of code.

**Limits.**

E2 label-0 genes are random natives. The band at AUROC 0.98-0.99 is loose in terms of misrankings and says nothing at strict thresholds; the binding cells are therefore E1 with backbone-mapped natives and XD. Stage 1 adds NO new evidence on eukaryote recipients: old trials in saturated cells and one descriptive contamination construct. The eukaryote arm is component EU (stage 2c). TRAINING NEED: none; these cells are saturated.

### VR. Viral genomes as recipients: natives of intact phage genomes

**Purpose.**

On an intact phage genome, does a method keep genes that the whole viral genus shares below the call rate it has on cellular natives?

**Labels.**

PRIMARY NATIVES, defined for every method class: viral genus-core genes (VGC) = genes with a hit (at least 40% identity over at least 70% of the gene) in at least 80% of at least 3 other genomes of the same viral genus (a panel hidden from every method). Estimand: 'not gained since the viral genus diverged'. This class needs no functional annotation and no library hit, so it includes orphans.

SECONDARY STRATUM: virion and packaging hallmark genes = proteins matching PHROG families of the categories 'head and packaging', 'connector' and 'tail' (MMseqs profile search, E <= 1e-10; fallback if the download fails: product words terminase, portal, major capsid, tail tape measure, tail fiber, tail sheath, flagged annotation-derived). Stated dependence: a hallmark label is a profile-similarity label against families built from phage AND prophage proteins; selecting on it guarantees library hits and enriches for families with prophage copies filed under bacterial taxa, i.e. it INFLATES the false-call rate of taxonomy methods relative to viral natives at large. Every hallmark figure reads 'on genes with phage-family homology'.

All other viral genes are unlabelled. No positives in stage 1 (constructed host-gene splices and mosaics: component VM).

'Temperate' = the genome carries a gene matching a PHROG integrase family or a product word 'integrase' (annotation-derived, flagged).

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage).

M13 Viral genome as recipient: natives | N under the viral E_recent | VGC genes of 120 intact tailed-phage genomes with the viral genus hidden | the same genes with only the species hidden (V0); the hallmark stratum; the unlabelled genes printed beside | ai_all, rule logic, place.node: the closest homologues of the genes of temperate phages are prophage proteins filed under bacterial taxa (reference labelling at scale); wide's call is not a claim of cellular origin and is printed with the share of its pointer rows that are cellular; composition scores have a tiny background (30-150 genes) | call rate on VGC genes at the cellular threshold frozen on development | stage 1, unsealing U1.

M14 Viral regime: huge gene pool, small genomes, short genes | no label | all genes of the same genomes | - | a third of viral proteins have no cell outside their own proteome and rank last for every similarity method; 11% are under 100 aa; wide abstains under 14 genes | scorable share per method; share of genes under 100 aa that are scored | U1.

**Genomes.**

120 RefSeq genomes of tailed phages (class Caudoviricetes) with at least 14 CDS, a host genus recorded in Virus-Host DB (matched by the taxid lineage of the recorded host, not by name), and a genus-rank taxid whose genus holds at least 4 complete genomes (the scored genome and at least 3 for the hidden panel). 40 host genera in ascending order of SHA-256('<host genus>|vr-v1') among host genera with phages of at least 3 such viral genera; 3 genomes from 3 different viral genera per host genus (smallest SHA-256 of the accession). In hash order: 10 host genera development, 20 Test A, 10 Reserve B (this viral reserve exists whatever happens to the island reserve). About 8,400 proteins plus at least 360 small panel genomes. Hidden set of a phage: its viral genus, every viral library proteome in which at least 40% of the phage's proteins have a hit at 80% identity or more (distance radius), and its own species' proteome. Prophage copies inside bacterial library proteomes stay visible, as in reality. Unit of replication: the host genus.

**Controls.**

V0 beside V1. Every rate is printed with three denominators: all VGC genes; VGC genes with any visible hit; hallmark genes. The threshold is the cellular one frozen on development PCn natives, not one taken from the viral genome. Trivial scores (n_rows, viral_share, best_id) on the same genes. For wide the call rate on VGC genes is shown beside its call rate on the other genes of the same genome. A replication statement is written before unsealing: taxonomy-based methods above 5% on temperate phages.

**Metrics.**

Call rate on VGC natives per host genus at the frozen cellular 1% threshold, for every method (ai_all, ai_narrow, ai_gene, place.node, rule logic, wide, codon, kmer); t-interval over host genera. A method 'fails' if the lower limit exceeds 5% on all VGC genes. Coverage: share of viral genes with any score, overall and under 100 aa. Not a primary endpoint.

**Cost.**

Stage 1, days 2-3. Search: about 8,400 proteins (15 minutes); panel searches against 3 or more small genomes per phage (seconds each). Code: about 200 lines. Downloads: about 480 small genome records, the Virus-Host DB table, PHROG profiles (under 1 GB).

**Limits.**

Requiring 4 genomes per viral genus biases the sample toward well-studied phage genera; phages without a genus assignment are excluded. VGC says nothing about cellular against viral ancestry (that is E_cell, stage 1b). The threshold is transported from cellular genomes; for composition scores computed on 30-150 genes this transport is poor and is printed as such. Hosts are isolation hosts at genus level. Phages under 14 genes and eukaryotic, archaeal and RNA viruses are postponed to stage 1b. TRAINING NEED: complete viral genomes as DNA in gene order (not 100 sampled proteins); prophage regions of reference proteomes assigned to a 'mobile' row instead of the host taxon. Whether more viral training data helps is NOT decided by this component (see decision rules, 'owner's first question').

### GP. Genus panel: positional-core natives, the gate of the island labels, flags, and the audit of confident calls

**Purpose.**

Which genes of a test chromosome are kept in place by the other species of its genus (certified natives well beyond the published backbone), which island labels do those species contradict, and where do each method's confident calls on unlabelled genes fall?

**Labels.**

Everything here comes from genomes of the own genus, which every method has hidden; it uses neither composition nor the library, and no method output.

PANEL per genus: RefSeq assemblies of level Complete Genome or Chromosome that are neither in the island set nor behind a library proteome: up to 5 of the same species as the test chromosome, up to 5 of OTHER species of the genus (different species taxid, one assembly per species) and 4 of sibling genera (other genera of the family, one each; filled from the order if the family has fewer than 4); always by ascending SHA-256('<assembly accession>|panel-v2'). Files: protein FASTA plus feature table (gene order).

SEARCH: every gene of the test chromosome against every panel genome (DIAMOND --sensitive); best hit with at least 40% identity over at least 70% of the gene.

POSITIONAL ORTHOLOGUE of gene g in panel genome P: g has a best hit h in P, and at least one of the 2 genes upstream of g and at least one of the 2 genes downstream of g has its own best hit in P within 5 genes of h on the same replicon.

CLASSES. f = share of the other-species panel genomes with a positional orthologue (needs at least 3 such genomes; 'at least 80%' means 3 of 3, 4 of 4, 4 of 5). PC = f >= 0.8, single member of its within-genome family, not island-labelled. PCn = PC and not within 5 genes of an island-labelled gene: the negative class of RI and the threshold class of HV, SP, FR. Gated island gene = island-labelled and f <= 0.5: the positive class of RI. Label conflict = island-labelled and f >= 0.8. Everything else is 'unresolved' (multi-copy families, presence without position, 0.5 < f < 0.8) and is never counted as native or as transferred.

FLAGS (strata only): absent-from-genus = island gene without any hit (40% identity over 50% of the gene) in any other-species genome (defines S-a); genus-pool = island gene with a hit of at least 70% identity over 70% of the gene outside its position in at least one other species; deep = PCn native with a hit in at least 2 of the 4 sibling-genus genomes, else genus-restricted (defines S-f); FAST = PCn natives in the lowest tenth, within the chromosome, of mean identity to their positional orthologues; secondary-replicon positional core = the same PC test on genes of secondary replicons.

NATIVE BLOCK = 11 consecutive genes, none island-labelled or within 5 genes of an island-labelled gene, at least 9 of them PCn, tiled without overlap.

ACCEPTANCE GATES (label against label, no score), on the development units with at least 3 other-species genomes (Escherichia+Shigella, Yersinia; Salmonella has only S. bongori and is expected to have no PC class): at least 80% of the single-copy published backbone genes are PC; at most 5% of the published island genes are PC; at most 5% of the genes of the literature-curated islands of Sakai and CFT073 and of the 9 cryptic prophages of K-12 are PC. If a gate fails after at most half a day of fixes on development, the PC class is withdrawn and fallback F1 of RI applies. Per sealed genus, before unsealing: under 70% of single-copy backbone genes PC = 'panel unreliable'; conflict share c above 20% = 'label implausible'; either removes the genus from the primary population of P1 and P2.

SCENARIO MATRIX ROW (truth | case | matched control | expected to fail | metric | stage).

M15 Confident calls of unknown truth | no exact truth; island/k is the only bound that is reported as a bound | the top 1% and top 3% of genes of each method on each Test A chromosome | the same shares for the trivial scores and a random score | a method whose top calls are positional core is making calls that are false under E_recent | three shares at fixed k: gated island; positional core; unresolved | stage 1, unsealing U1.

**Genomes.**

All 18 sealed genera (labels only; needed before the split) and the 3 development genera with panels. At most 14 assemblies per genus, about 300 assemblies, about 450 MB. Genera with fewer than 3 other-species assemblies have no PC class, are not P1-eligible and are printed as such (risk: Prochlorococcus, Rhodopseudomonas).

**Controls.**

The panel excludes the genomes that defined the published labels and every library proteome. The agreement tables (backbone genes that are PC; island genes that are PC) are printed for every genus before unsealing. k is fixed in advance (1% and 3% of genes). The words 'upper bound' and 'panel-supported bound' of the first version are removed: presence without position (transposases, prophage genes that sit independently in most congeners) is 'unresolved', not native, because E_recent counts gains from the genus pool as transfers. The audit enters no decision rule. Label code never reads a method output; label files are hashed at freeze F1.

**Metrics.**

Per genus: number of PCn natives, gated island genes, conflicts, native blocks, gated island regions, c, panel sizes, median identity to the panel. Per method and Test A chromosome, among its k top-ranked genes: share gated island (the assumption-free lower bound of precision under E_recent), share positional core (false under E_recent), share unresolved; median over genera.

**Cost.**

Stage 1, days 1-2 (downloads start on day 0). Searches against at most 14 small genomes per genus: minutes. Code: about 220 lines (panel selection from metadata, best hits, positional test, classes, flags, native blocks, gates). Downloads about 450 MB; assembly lists through per-genus metadata queries.

**Limits.**

New label code by the same engineer now carries the negatives of both real-label endpoints; its checks are label-against-label only. No empty-site test: 'accessory' is not 'gained'. Positional conservation in at least 80% of the other species is a strong but not complete certificate (in-place replacement by homologous recombination stays N). The other species of a genus are at very different distances in different genera (near-identical in Brucella, deep in Bradyrhizobium); the median identity is printed, not corrected. Species are NCBI species taxids, which is uneven. TRAINING NEED: none; panel classes must not be used as training targets for the benchmark genera.

### SP. DNA splices of real elements and genes into real chromosomes (prophage, plasmid, single genes at four donor distances, cross-domain genes, contaminant contigs)

**Purpose.**

Which input class detects which kind of exactly labelled transfer: block or single gene, near or far donor, mobile or cellular, integrated or contaminant?

**Labels.**

Truth is set by construction from real, unmutated sequences. Every inserted protein is searched once. The donor species is absent from the library.

SINGLE-GENE CLASSES, by donor distance only; NO composition filter (the GC3-matched and GC3-distinct donor selections of the first version are REMOVED: they selected positives on the score of a method under test). NEAR = another genus of the recipient's family; MID = another family of the same order; FAR = another class or phylum of the same domain; MOB = single genes of the phages and plasmids below (8 phage, 8 plasmid); XD = another domain (8 archaeal CDS, 8 eukaryotic CDS without introns): the DNA form of the cross-kingdom regression. Donor genomes are complete genomes by ascending SHA-256('<accession>|sp-v2') among candidates that meet the class rule; GC3 and codon distance to the recipient are recorded as covariates.

WHICH DONOR GENES: 16 per recipient and class. For NEAR, MID, FAR: 8 from the donor's accessory fraction and 8 at random from all its genes, where accessory = no hit (50% identity over 70% of the gene) in ONE other complete genome of another species of the donor's genus. Every class is also reported by breadth tercile of the inserted protein (number of visible proteomes with a covering hit; a stratum, not a label).

COMPOSITION STRATA, post hoc on all single genes: 'composition-typical' = the inserted gene's codon AND kmer scores under the recipient's model lie inside the central 80% of the recipient's genes; 'composition-distinct' = outside the central 98% for either. Inside a stratum the figures of codon and kmer are printed as 'by construction'; the strata are informative for similarity methods, fusions and future models only. The stratum is named 'composition-typical', NOT 'old transfer'.

BLOCKS, two per recipient. B1 = one intact temperate phage whose recorded host (Virus-Host DB, taxid lineage) is the recipient's genus, else its family, else the unit leaves P5: a RefSeq Caudoviricetes record with an annotated integrase, 30-150 CDS, species absent from the library, smallest SHA-256('<accession>|sp-b1'); + one 10-40 kb segment of a plasmid of a species from another class; + one native block of 10-40 kb relocated (placebo). B2 = one intact phage of a host from another phylum (its recorded host genus is hidden as well; descriptive) + 2 of 4 broad-host-range elements (RP4 of IncP-1, RSF1010 of IncQ, R388 of IncW, the conjugative transposon Tn916; assigned by hash) + one relocated native block.

RELOCATED NATIVES (placebo and companion negatives): 80 single native genes per recipient through the same code (60 untouched, 20 sham-ameliorated), 1:1 with the 80 foreign singles.

PLACEMENT is independent of the trial class. Single genes go, CDS only, into the middle of intergenic gaps of at least 100 bp. Half of the single sites are 'context-clean' (the 5 genes on each side are all PCn natives), half are drawn among all eligible gaps; the site type is assigned independently of whether the gene is foreign or relocated. At least 22 genes between sites; at most floor(G/30) single sites per composite assembly (G = genes of the chromosome); composites are added until the 160 single genes are placed (2 composites for G >= 2,400, 3 for G >= 1,600, 6 for G = 937); inserted genes at most 5% of the chromosome's genes per composite (a phage is cut to a contiguous part if needed). Half of the block sites lie next to tRNA genes, for foreign and placebo blocks alike.

CONTIG VERSION of B1 and B2: the same foreign segments as separate contigs (class C, 0.5-5% of the assembly) and the native block as its own contig (class N); identical proteins.

AMELIORATION (secondary, gated): FAR, XD and MOB singles also exist with codons resampled from the recipient's usage conditional on the amino acid and the previous codon; the same resampling is applied to the 20 sham natives.

VISIBILITY PARITY: for every gene of a recipient's composites, natives included, the hidden set is H1 of the recipient plus the genus and distance set of every donor of that recipient (viral genus for phages).

FLAGS per trial from construction only (strata): I_sim (some visible reference differs by at least 0.05, in the profile units of research/array/ceiling.py, in its distance to donor and to recipient; needs a search of 100 proteins per donor genome); I_comp = not composition-typical.

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage; all in stage 1, unsealing U2).

M16 Viral donors into cells: prophages | T, viral donor, block | intact host-matched temperate phage spliced whole (B1); mismatched phage (B2, descriptive) | relocated native block; the same phage with only its species hidden | ai_all and rule logic on host-matched phages (best hits are prophage proteins filed under bacteria of the recipient's phylum: a difference of scope, reported as such); composition on host-matched phages; every similarity method on phage orphans | R@1% on phage genes (P5); region found at LR@2%; boundary error in genes.

M17 Single-gene transfers | T, one gene, four donor distances | CDS-only splices | relocated single natives | composition on single genes (variance, no block); similarity on NEAR and MID with the donor genus hidden (0.70 within bacteria in E2); rules and ai_all on in-phylum classes (scope) | R@1% per class (P4), never pooled into one scalar.

M18 Composition-typical transfers (the nearest thing to old transfers that stage 1 can build) | T; age is a coordinate | the composition-typical stratum of all single genes; ameliorated FAR, XD, MOB singles (gated) | the composition-distinct stratum; sham-ameliorated natives | DNA-only scores by construction; a DNA-only learned model | R@1% of similarity methods and fusions inside the stratum; complementarity table | real old gains: stage 3 (M35).

M19 Contamination of the assembly (harness check on bacterial recipients) | T / C / N; U = a contig of only foreign genes in a fragmented assembly, same-species contaminants, a plasmid contig against a contaminant plasmid | 2 x 2 on identical genes | the integrated placement and the native contig | every proteins-only method and every W1 DNA score gives T and C the same score (known in advance; what is reported is the size of the problem); the pipeline's contig test cannot fire on bacteria (it fires in M19-euk of XK) | share of contaminant genes above the threshold; three-class confusion | real case: stage 2 (M32).

M20 Fast exchange across hosts | T, recent, mobile pool; M when the element is its own replicon | plasmid segment from a distant class; the same broad-host-range element in several hosts | relocated native block | ai_all and rule logic whenever a library proteome of the recipient's phylum carries the element (the taxon of a reference is not the ancestry of a mobile gene); composition for elements that match the host | R@1% by host; across-host variance of recall per element | real labels: stage 2 (M30).

M21 Identifiability | T by construction; U where neither I_sim nor I_comp holds | single cellular genes by flag cell | the 'neither' cell is the standing null | every method near 0.5 in 'neither' | AUROC and R@1% per cell; headline figures use ALL trials.

**Genomes.**

Recipients: the 4 development chromosomes with a deep search and every Test A genus with at least 500 PCn natives (Reserve B, or the stage-3a genera if there is no Reserve B, later with the frozen code and disjoint donors). Per recipient: 3 cellular donor genomes (NEAR, MID, FAR) and 3 congeners for the accessory test; a shared XD pool (2 archaea + 2 eukaryotes for development, 4 + 4 for Test A); 2 phage genomes; plasmids from donor records or RefSeq; 4 broad-host-range elements by name. Donor genera are disjoint between development, Test A and later sets. A recipient whose family has no other genus with a complete genome of a non-library species has no NEAR class (flagged). With 13 Test A recipients: 1,040 single foreign genes (208 per class), 1,040 relocated natives, 26 phages, 13 plasmid segments, 26 element insertions, 26 relocated blocks, and the blocks again as contigs. Unit lists of P4 and P5 are computed before freeze F2; P5 needs at least 9 units with a host-matched temperate phage, otherwise it is descriptive.

**Controls.**

1. One code path: relocated natives go through the splice code. Gate on development, per method and trivial score: the interval of the AUROC of relocated natives against untouched PCn natives overlaps 0.45-0.55. The trivial score 'top-hit row equals its neighbours'' is printed and not gated (operon structure makes it fail legitimately).

2. Amelioration gate: the same criterion for sham-ameliorated natives, for gc3, codon, kmer, their W11 versions and a gradient-boosted classifier on non-label features. If it fails, every ameliorated cell is removed as 'generator rejected'.

3. Thresholds: the score exceeded by 1% of the PCn natives of the edited assembly that lie at least 6 genes from every splice site. Companion number without that choice: AUROC of spliced foreign genes against relocated natives (same code path, same site distribution).

4. Context methods (Wk scores, any neighbourhood model) are evaluated on the context-clean sites only.

5. Null: the 'neither' cell (interval overlapping 0.45-0.55). A trivial score outside is a generator leak; a real method outside is investigated for at most 4 hours and reported.

6. Input contract for DNA-reading methods: nucleotide sequence of every contig, CDS coordinates and strand, contig membership; identifiers renumbered by position; no product text. Recipient and donor records come from different annotation pipelines: this is a flag, not a filter, and the length distribution of inserted genes against natives is printed. Before any learned DNA model is scored on a spliced cell, genes are called again on all assemblies with one caller (stage 2b).

7. Extra trivial scores: distance to the contig end; gene length.

8. A cell that fails a placebo on Test A is reported as withdrawn; it may be rerun only on Reserve B (or on the stage-3a genera) with fresh donors and seeds.

9. Stage 1b: an island-excised version of each edit chromosome (zero-transfer control).

**Metrics.**

R@1% per class and cell; block detection as LR@2% with native blocks of the edited assembly; three-class confusion at the 1% threshold; AUROC per flag cell; complementarity table (codon, wide, fuse); scorable share; the composition and breadth strata. Per-gene protein methods score each inserted protein once; composites are built for wide and for DNA methods.

**Cost.**

Stage 1, days 4-5. Search: about 5,100 inserted proteins and about 7,000 donor-profile proteins (20 minutes); accessory tests in seconds. Code: about 550 lines (donor selection, insertion at GenBank level, relocation, contig emission, context-preserving resampling, truth table with class, donor distance, block id, site type, flags). Downloads: about 110 cellular genomes, 12 XD records, 34 phage records, plasmids (under 600 MB compressed). GPU: 4-12 composites per recipient for wide.

**Limits.**

Clean junctions and sites that are not natural integration hotspots; no change of the protein after transfer; contaminant contigs with clean ends and no read coverage; 4 clinically studied broad-host-range elements; isolation hosts. Stage 1 contains NO measurement of real old transfers: the composition-typical stratum is selected on two statistics and says what similarity retains when composition is silent, nothing more. 'Accessory' donor genes are defined by absence from one congener. A model trained on splices of this kind is in-distribution here; these cells restrict claims and set floors, they never show superiority. TRAINING NEED: whole replicons as DNA in gene order; plasmids as separate replicons; phage genera outside the exclusion list as splice donors with host pairs; three-class examples with contig membership; protein similarity kept as an input next to DNA.

### FR. Fragmented presentation of the same labelled chromosomes

**Purpose.**

How much of each method's strict-threshold recall survives when the same labelled genome is presented as short contigs or as a single contig without genome-wide context?

**Labels.**

Labels are unchanged; only the presentation changes. F-bag: the edit chromosome (real gated islands) and its first composite with the B1 blocks are cut into contigs with N50 of 100, 20 and 5 kb; half of the breakpoints at annotated repeats (transposase genes, rRNA operons, tRNA genes), half at random positions from a fixed seed; only CDS lying entirely inside a contig are kept; contig order and orientation are random. F-single: single contigs holding 3, 6, 14 or 30 genes, 50 of each size per chromosome, sampled without regard to labels. The harness hides the V1 rows in every case; in F-single only the taxon label and the rest of the genome are withheld from the method.

SCENARIO MATRIX ROW (truth | case | matched control | expected to fail and why | metric | stage).

M22 Highly fragmented genomes | labels unchanged; for a contig with only native genes the right output is no call; a contig with only foreign genes has no native frame (U, reported apart) | F-bag and F-single | the unfragmented chromosome | bag-of-proteins methods must be bit-identical in F-bag (a check of the harness) but need the genome's lineage, which F-single withholds; wide needs at least 14 genes and a genome-wide profile; composition scores lose their background in F-single and their window versions fall back toward gene level at 5 kb | paired change in R@1% and LR@2%; coverage; calls per native-only contig; call rate of PCn natives in the 5-kb presentation at the unfragmented threshold (a natural hard native for every input class) | stage 1, unsealing U2.

**Genomes.**

The edit chromosomes of the development and Test A recipients of SP: 3 N50 levels x 2 assemblies and 200 single contigs per chromosome.

**Controls.**

The unfragmented chromosome is the paired control. Cut points come from annotation features and a fixed seed, never from labels or scores. Bit-identity of bag-of-proteins scores in F-bag is asserted. Single contigs are sampled regardless of labels.

**Metrics.**

Paired change per genus in R@1% (per-gene methods) and LR@2% (W11) for each N50; coverage; for F-single, recall on mixed contigs and calls per native-only contig at the thresholds frozen on development.

**Cost.**

Stage 1, day 5. No search. Code: about 100 lines inside the splice module. GPU: 3-4 extra wide passes per chromosome.

**Limits.**

Simulated breakpoints; no misassemblies, no partial genes, no missing fraction; gene calling on fragments is not tested. F-single is the first item to drop in stage 1. TRAINING NEED: genomes presented as contigs of random length, incomplete genomes, a variable number of genes per example.

### PX. Protein edits on certified natives and on spliced genes (stage 1b)

**Purpose.**

Does a method confuse a property of the sequence itself (compositional bias, short length, a foreign block inside the gene, recent duplication) with foreign origin?

**Labels.**

Exact by construction; both classes always go through the same edit. Base natives are PCn natives; thresholds are the unedited chromosome's 1% PCn threshold under V1. Test A is already unsealed when this runs, so every Test A figure of this component is marked exploratory.

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage 1b).

M23 Sequence convergence and spurious similarity (low complexity, repeats, compositional bias, shared short domain) | N: a hit to a distant group without common ancestry | into 100 PCn proteins per recipient that have no hit outside the home family, insert at 1/3 or 2/3 of the length a 40-120 aa low-complexity or repeat segment of a far-clade library protein, REVERSED (composition and periodicity kept, homology destroyed) | the reversed globular segment of equal length; the unedited gene; the forward segment (a real domain graft = chimeric positive) | rule logic and ai_all if an unmasked search admits the hit; wide and place.node if new low-coverage far cells count | call rate and its ratio to the globular control; AUROC of forward against reversed grafts.

M24 Short genes, matched | labels unchanged | 120 spliced foreign and 120 PCn genes per recipient, each replaced by random in-frame windows of 50 and 100 aa, searched as genes; both classes cut identically | the full-length gene | all similarity methods below about 60 aa at moderate divergence; rule tiers by threshold design (a difference of log E-values scales with length); single-gene composition | R@1% by length; L50 = the length at which recall is half its full-length value.

M25 Recombination inside a gene (chimeric genes) | per residue: the replaced block is T, the rest N | for 100 PCn genes per recipient a contiguous aligned block of 50% of the gene is replaced by the residues and codons of the homologue from the MID donor and from the FAR donor | self-chimera through the same code; the reversed graft of M23 | one-hit-per-gene summaries; wide has one cell per proteome; gene-level composition is diluted by half | gene-level R@1% by block share; breakpoint error for methods with positional output.

M26 Family expansion by duplication | N | 30 PCn genes per recipient duplicated in tandem into 3 and 10 copies diverged by BLOSUM62-conditional substitution at 0.15, 0.3, 0.5 substitutions per site | the unduplicated gene; equal divergence without duplication | best-hit rules if diverged copies lose home hits first; wide if within-genome redundancy bends the table | call rate by divergence and copy number | first to drop.

**Genomes.**

The edit chromosomes of the development and Test A recipients of SP; donors from SP.

**Controls.**

Reversal as the decoy for bias; identical truncation of both classes; self-chimera; equal-divergence control for duplicates; hidden-set parity as in SP; trivial scores in both orientations on every cell.

**Metrics.**

As listed per row; t-intervals over genera.

**Cost.**

Stage 1b (days 6-8): about 1.5 days of code (about 250 lines) and at most 2 h of search (about 40,000 mostly short proteins).

**Limits.**

Decoys test composition- and periodicity-driven similarity only; structural convergence and promiscuous real domains cannot be given exact labels. A window of a long protein is a domain fragment, not a natural small protein. One donor and clean breakpoints per chimera. In-silico divergence is cruder than real paralogue evolution. TRAINING NEED: decoy negatives; a per-residue coverage channel; short queries in proportion (5% of trials are under 100 aa today); genes cut into windows with per-window labels.

### VM. Viral genomes as recipients, constructed: host-derived genes and mosaic genomes (stage 1b)

**Purpose.**

Can a method find a host-derived gene inside a viral genome and recover a known partition of a mosaic genome, in excess of the structure already present in the intact parents?

**Labels.**

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage 1b).

M27 Host-derived genes in viral genomes | under E_cell: host-derived against viral | per phage genome of VR, 6 genes from a held-out species of its recorded host genus and 3 from a distant cellular genome, inserted between viral genes | 3 own genes relocated (placebo); 3 genes from a phage of another viral genus (scored under M28); the VGC natives of VR | wide under 14 genes (abstains; small phages are added and counted as abstentions); every similarity method on the third of viral proteins without outside cells; composition (host and phage are coupled); rule logic cannot separate host-derived genes from native genes whose homologues are filed under bacteria | R@1% with the threshold from the VGC natives of the genome's host-genus group (not from hallmark genes, which are selected for having homologues), by host match and genome size.

M28 Mosaic genomes without a vertical backbone; recombination between genomes | no N/T label unless one co-ancestry class holds at least 80% of the genes; otherwise the truth is the partition | 100 artificial mosaics from 2-3 real phage genomes of different viral genera with the same host genus, joined at gene boundaries with parental shares 50:50, 80:20, 90:10, plus single-gene swaps; each parent's proteome and genus hidden | every gene is also scored in its intact parent: the statistic is the cross-parent pair dissimilarity in the mosaic MINUS that of the same pairs in the intact parents; sham self-joins; trivial pair scores (distance in gene positions, same top-hit row) | rules and ai_all (all genes inside one viral home group); wide (the profile is a mixture, 'foreign' has no polarity); composition (the parents share a host) | pairwise co-origin AUROC in excess of the parents; breakpoints in excess of sham joins; binary recall only for 90:10 and single swaps | natural mosaics: stage 3 (M36).

**Genomes.**

The 120 phage genomes of VR (same assignment by host genus), plus up to 40 small phages (under 14 genes) for the abstention count; host-gene donors are held-out species of the recorded host genus.

**Controls.**

Relocation placebo; viral-to-viral control; difference against the intact parents; sham self-joins; trivial pair scores must not explain the co-origin result.

**Metrics.**

As listed per row. For scalar methods the pair dissimilarity is the absolute score difference; for wide the distance between pointer distributions; for a segmenting method its declared segments. t-intervals over host genera.

**Cost.**

Stage 1b: about 1.5 days of code (about 300 lines on top of the splice module) and about 14,000 proteins of search (25 minutes).

**Limits.**

Joins in module order are tidier than illegitimate recombination; parents are limited to well-studied host genera; the history of natural mosaics stays unlabelled until stage 3. Real host-derived genes are stage 2 (M31) and are not independent of similarity. TRAINING NEED: complete phage genomes in gene order as recipients; host-virus pairs; host-gene-in-phage splices; genome-as-mixture examples with a pairwise same-origin output.

### RA. Further real anchors with a known, opposite bias (stage 2)

**Purpose.**

Do the conclusions drawn from the island anchor hold on real labels whose known bias runs the other way (labels found by recognising similarity rather than by absence from relatives)?

**Labels.**

SCENARIO MATRIX ROWS (truth | case | control | bias and expected failures | metric | stage).

M29 Prophages, manually curated per CDS | T = CDS flagged prophage; unflagged genes are NOT negatives | 30 bacterial genomes of the curated set outside the 18 island genera and outside Enterobacterales, at most 2 per genus, by ascending SHA-256('<accession>|prophage-v1') | the K-12 cryptic prophages on development | curation recognised phage-like genes, so this anchor favours similarity methods; before use the source's curation method is recorded, and if composition evidence was part of it the 'opposite bias' statement is weakened in the report | recall within a FIXED CALL BUDGET: among each method's top 1% and top 3% of genes per genome, gene and region level | stage 2a.

M30 Fast exchange, real single genes | 'recently shared with a plasmid of another host family AND absent from the congener panel': a chromosomal CDS with at least 99% nucleotide identity over at least 90% of its length (at least 500 bp) to a database plasmid whose host family differs, with no positional orthologue in any other-species panel genome (without the second condition the plasmid may have captured a native gene) | positives only | library entries at least 95% identical to the plasmid's proteins are hidden for every method; counts collapsed by family | favours similarity methods | recall within the top 1% and 3% | stage 2e.

M31 Real host-derived genes in viruses | cyanophage photosystem genes (psbA, psbD, hli) and captured host genes of herpes- and poxviruses from published tables | VGC natives | labels are annotation- and phylogeny-derived: recall of known cases only | recall within the top 1% and 3% of each genome | stage 2g.

M32 Real contamination | C against N scaffolds of the first tardigrade assembly | the curated re-assembly | scaffold labels are built from read-coverage evidence of the independent library only; if only labels resting on GC or taxonomic hits are available, that dependence is printed beside every figure. This is NEW label code: bench/contamination.py is a synthetic construct and is not reused here | share of contaminant genes reported as confident transfers | stage 2f.

**Genomes.**

M29: 30 genomes, split by genus hash into 20 opened for the frozen methods in stage 2a and 10 held for the DNA generation. Before planning, the size of the published set (49 genomes in its first version, 78 claimed in the second) and its licence are checked. M30: the Test A chromosomes. M31: phage and virus genomes outside the VR genera. M32: one eukaryote assembly pair.

**Controls.**

The direction of the label bias is printed next to every figure. No threshold is set or transported to these genomes: all recall figures use a fixed call budget (the number of genes each method would call at its transported threshold is printed separately as a transport diagnostic). For M30 the near-identical library entries are hidden. Trivial scores (viral_share, n_rows) on the same genes.

**Metrics.**

Recall within the genome-relative top 1% and top 3%; differences between methods paired per genus, t-interval over genera; no false-positive rate, since no clean negatives exist here.

**Cost.**

Stage 2: 2a curated prophages (half a day of code; about 120,000 proteins, 3.5 h of search; DNA methods need none); 2e plasmid identity labels (one day; a few GB; one nucleotide search); 2f tardigrade (one to two days: new label code); 2g host-derived viral genes (one day).

**Limits.**

Positives only, hence no ranking by precision; each set was built with similarity evidence and cannot rank similarity methods against each other fairly; cross-class statements need M29 and the island anchor side by side. TRAINING NEED: the held genomes of M29 and their genera enter the exclusion list as soon as the source list is verified.

### EU. Eukaryote recipients beyond regression (stage 2c, moved ahead of the training and plasmid items)

**Purpose.**

Do the hard conditions that matter most in eukaryotes (loss in relatives, paralogy, within-domain donors, viral insertions, contaminant contigs) break the methods on eukaryote genomes?

**Labels.**

NATIVES: positional orthologues against ONE hidden genome of a sister species or genus (the collinear test of GP; that genome is excluded from every method), single copy, stratified into marker (lineage single-copy marker families) and non-marker. No intron criterion (the 'at least 3 introns' rule of the first version is REMOVED: it fails in intron-poor lineages and is a gene-structure signal that a DNA-reading model reads directly). Marker natives are expected to be saturated and set no threshold.

INSERTS: foreign genes and relocated native genes pass through one splice path with the same gene structure (spliced CDS as a single-exon gene for both), so structure carries no label; thresholds for insert cells come from RELOCATED natives (at least 300 per genome), never from untouched natives.

LOSS: the library holds one proteome per family for non-mammal eukaryotes (seeddrift/donors.py, policy 'broad'), so hiding the genus already empties the family; the loss clade is defined as in HV (narrowest of family, order, class with at least 5 visible proteomes) and will usually be the order or class.

SCENARIO MATRIX ROWS (truth | case | matched control | expected to fail and why | metric | stage 2c).

M33 Eukaryote recipients: hard natives and within-domain transfers | N = positional natives, as they are and under the edits M8-M10 (thresholds from the non-marker natives); T = unmutated transplants from held-out eukaryotes (another order of the class; another phylum) and from bacteria (regression, DNA form: bacterial CDS spliced into scaffolds) | the paired controls of HV; 'far' is defined per recipient by the rule's own home_group() | rule logic inside a kingdom-wide home group; loss and paralogy are frequent; eukaryote from eukaryote is the unsaturated cell (0.83 against 0.77 in E2) | paired inflation; R@1% by donor class.

M34 Viral genes and contaminant contigs in eukaryotes | T (constructed endogenous viral element inside a scaffold), C (the same genes, or a bacterial segment, as their own contig), N | integrated against contig placement of identical genes | the pipeline's contig test needs 3 genes; proteins-only methods cannot use linkage | three-class confusion | real elements: stage 3 (M37).

**Genomes.**

10 eukaryote genomes with CDS and scaffold coordinates, at most 20,000 protein-coding genes, species absent from the library, a sister genome available, one per lineage (Ascomycota, Basidiomycota, Mucoromycota, Apicomplexa, Euglenozoa, Stramenopiles, Chlorophyta, Amoebozoa, Arthropoda, Nematoda), by ascending SHA-256('<taxid>|euk-v2'), from classes held out of training where one exists. 6 opened in stage 2c (all 6 with the same sign gives a two-sided p of 0.031), 4 held for the DNA generation. Named in the registry when the candidate lists are built, not downloaded before.

**Controls.**

As in HV and SP: paired edits, hidden-set parity, relocation placebo through the same code, trivial scores in both orientations. Only edited natives, non-marker natives and transplants are used for contrasts.

**Metrics.**

Inflation on edited natives; R@1% of transplants by donor class; three-class confusion; per-genome values, a t-interval over 6 genomes and the sign count. With fewer than 6 usable genomes the arm is declared descriptive.

**Cost.**

Stage 2c: about 2 days of code; about 72,000-120,000 proteins (2-3.5 h of search); 10 sister genomes; marker models with hmmer through conda.

**Limits.**

No real eukaryotic positives that are independent of similarity exist; literature lists are similarity-selected and are not used. Positional natives against one sister genome are a weaker certificate than the bacterial panel. Single-exon inserts are not natural eukaryotic genes; taking thresholds from relocated natives removes the leak and also the realism. 6 + 4 genomes support direction, not fine ranking. Until this arm is reported nothing is claimed for eukaryote recipients. TRAINING NEED: eukaryote genomes as scaffolds with gene structure; within-domain eukaryote donors; contaminant contigs as a third class.

### LD. Postponed heavy label building (stage 3)

**Purpose.**

What are the real rates on old gains, real patchy natives, natural mosaics and real viral insertions, which construction can only bound, and on which untouched genera can the next model be confirmed if there is no island reserve?

**Labels.**

SCENARIO MATRIX ROWS (truth | real data and construction of labels | safeguards | stage).

M35 Real older gains and real patchy natives | ladder of downloaded relatives: R = absent from all other species of the genus with an intact empty site; O = shared by the genus without a positional orthologue in neighbouring genera, with empty sites; patchy N = conserved flanks in a relative outside the radius | R is defined from congeners only; presence in library-represented genera is a stratum, never the label; a gene with a positional orthologue in exactly one outgroup genus is 'unresolved'; no relabelling from mobile-element callers, product words or composition | stage 3a.

M36 Natural mosaic phages | 'present here, absent from every other member of the subcluster, flanks conserved' from actinobacteriophage gene-family tables | use restrictions of unpublished genomes checked first | stage 3b.

M37 Real endogenous viral elements in eukaryotes | giant-virus insertions in green algae (76-1,782 genes each), polydnavirus loci | integration evidence required | stage 3c.

M38 Contaminated reference proteomes | published lists of contaminated database entries mapped to the library; a cleaned library with separate rows for mobile elements; audit of each method's calls that rest on a single row | stage 3d.

M39 Archaeal recipients | no real labels exist: the HV and SP generators on archaeal chromosomes | stage 3e.

M40 Phylogenetic audit of confident calls of unknown truth | blind trees for at most 300 confident unlabelled calls on which methods disagree | the evidence type is shared with similarity methods: a bound, in no ranking | stage 3f.

**Genomes.**

3a: first the 18 island-set genera (gene-resolution labels with empty sites), then 30-40 further genera with composition-free labels, the only way to resolve differences of about 0.05 (29 lineages are needed to state 'in at least 90% of lineages' with 95% confidence). If stage 1 ends without an island Reserve B, the further genera of 3a, labelled blind BEFORE the DNA model is frozen, are the confirmatory set of the bar. Others as named per row.

**Controls.**

Acceptance gates on development genomes before any use: at most 3% of published island genes labelled vertical, at most 2% of PCn natives labelled recent gain, at least 85% of literature-curated island genes labelled gain or unresolved. Label code never reads a method output; label files are hashed before scores are joined.

**Metrics.**

The primary metrics of RI with gene-resolution labels (R@1% against vertical genes, by age tier R and O); call rate on real patchy natives; the three shares of GP's audit tightened by empty-site evidence.

**Cost.**

Stage 3, 1-2 weeks, dominated by code and curation: 3a ladder (about 600 lines; hundreds of relatives), 3b-3f about one to two days each.

**Limits.**

New label code is itself a risk, which is why nothing here is needed for stages 1 and 2. Ladder labels cannot polarise gain against loss without empty sites and need at least 4 outgroup genera for the old tier.

## Primary endpoints

- P1 (component RI; unsealing U1; the P1-eligible genera of Test A, floor 9; regime V1; per-gene methods; window W1): per-genus R@1% = share of gated island genes above the score exceeded by 1% of the PCn natives of the same chromosome (ties at expectation). Contrasts, Holm over 3: codon minus wide; fuse minus codon; wide minus the linear rules (intersection-union over ai_all, ai_narrow, ai_gene, wide_rule_ai: wide must beat each). Printed beside each: the comparison with every trivial W1 score, the call split by class, mean recall over 0-5%, recall per island region. Context methods are reported in the context track P1c (same classes and metric, against all 9 context baselines), which is the gene-level endpoint of the bar for a neighbourhood-reading model and carries no stage-1 verdict.

- P2 (component RI; U1; the P2-eligible genera of Test A, floor 8; V1; window W11 with scored-only averaging; unit = region): per-genus LR@2% = share of gated island regions whose score exceeds the score exceeded by 2% of the chromosome's native blocks. Contrasts, Holm over 2: codon_nb minus wide_nb; fuse_nb minus codon_nb. Printed beside each: every trivial W11 score, the block-permutation null, the versions at 1% and with no native block allowed, and S-w.

- P3 (component HV; U1; Test A units with at least 100 eligible base genes; loss grades 50% and 75% pooled; one absolute threshold per method = its unedited V1 1% PCn threshold). (a) Paired inflation of wide's call rate on graded-LOSS natives: 'robust' if the upper limit is below 2 points for all eligible genes AND for those that keep a visible hit; 'susceptible' only if the lower limit exceeds 2 points inside the deep-vertical stratum; otherwise unresolved. (b) D(wide) = recall of the D1 orthologue positives minus the call rate of their paired LOSS natives: against 0, with a point estimate of at least 0.05 and above the same statistic of every trivial score in both orientations. (c) D(wide) minus D of each linear rule (intersection-union). Holm over 3.

- P4 (component SP; unsealing U2; Test A recipients; V1 with donor parity; per-gene methods): per-class R@1% on single spliced genes in the four distance classes NEAR, MID, FAR, MOB (16 genes per recipient and class, threshold from the PCn natives of the edited assembly); no pooled scalar. Contrasts per class, Holm over 8: wide minus the linear rules (intersection-union); fuse minus codon. The rule figures in in-phylum classes are marked as differences of scope. Descriptive beside: breadth terciles, composition strata, context-clean sites for window scores, cell XD.

- P5 (component SP; U2; Test A recipients with a host-matched temperate phage, floor 9 units, otherwise descriptive): R@1% on the genes of the spliced phage of block B1. Contrasts, Holm over 2: wide minus codon; wide minus the linear rules (intersection-union; ai_all's blindness to phage genes filed under the home phylum is reported as scope).

- Not primary, by decision: every other cell of the scenario matrix, the V2 regime (a required concordance, not an endpoint), the native call rates of VR, the audit of GP, threshold transport, rule tiers, cell XD and the eukaryote contamination construct (regression), all of stages 1b to 3. The three outcomes registered in bench/islands/README.md are withdrawn in the dated amendment before any run.

## Decision rules

UNIT AND INFERENCE. Unit of replication: the genus (one chromosome each) for cellular genomes, the host genus for viral genomes; genera of one family are never split between Test A and Reserve B. All differences are paired within unit. Intervals: two-sided 95% t-interval on the per-unit paired differences (the percentile bootstrap of the first version under-covers at 6-13 units and is removed); for 'equivalent', 'robust' and non-inferiority statements the interval obtained by inverting the sign-flip test. Tests: exact sign-flip permutation over units; Holm inside each primary endpoint; the smallest attainable p for the realised n is printed beside every Holm result. A contrast against a SET (all trivial scores; the four linear rules; the nine context baselines) is an intersection-union test: point estimate = the smallest mean difference over the set, p = the largest p over the set; every member must be beaten, so nothing is selected and no multiplicity penalty arises. Thresholds are per chromosome, so no pooling of raw scores occurs anywhere. wide = mean over its existing seeds and groupings.

RESOLUTION, FOR THE REALISED n. Half-width of a paired difference = t(n-1) x s / sqrt(n). For s = 0.10 / 0.12: n = 7: 0.092 / 0.111; n = 8: 0.084 / 0.100; n = 9: 0.077 / 0.092; n = 10: 0.072 / 0.086; n = 11: 0.067 / 0.081; n = 12: 0.064 / 0.076; n = 13: 0.060 / 0.073. Smallest two-sided sign-flip p = 2 / 2^n: 0.0156 (7), 0.0078 (8), 0.0039 (9), 0.0020 (10), 0.0010 (11). Holm over 3 needs 0.0167, over 2 needs 0.025, over 8 needs 0.00625 (n >= 9). The realised n, the realised s and the minimum detectable difference are printed beside every interval. A true difference of 0.06 (n = 10, s = 0.10) passes the first three conditions below in roughly one case in three, a difference of 0.10 in roughly four in five, before Holm and before the concordance conditions; differences under about 0.08 will mostly read 'unresolved', and the old gaps of 0.01-0.03 stay unresolved by design. No genus is added after an unsealing.

WHEN A DIFFERENCE COUNTS. 'A better than B' on a recall-type endpoint needs all of: (1) the interval excludes 0; (2) point estimate at least 0.05; (3) Holm-adjusted sign-flip p below 0.05; (4) for RI, the same sign in the three analyses that address a named bias: S-pub, S-a, S-b; (5) if A or B reads similarity: same sign and an interval excluding 0 under V2 as well; where V1 and V2 disagree the V2 result is reported and V1 is marked 'comparators possibly visible'; (6) if A and B belong to different input classes: an interval excluding 0 also with thresholds from deep PCn natives only (S-f); (7) A exceeds every registered trivial score of its window on that endpoint, otherwise the verdict reads 'not shown to detect origin beyond a trivial proxy'. 'Equivalent at this resolution' = the inverted interval lies inside (-0.05, +0.05). Everything else = 'unresolved'. P3a and P3b as worded in the endpoint.

HOW COMPONENTS ARE COMBINED. (1) Only the real-label endpoints support a ranking: P1 at gene level, P2 at block level. An unqualified 'A better than B' needs both to count; one alone is reported as 'at gene level only' or 'at block level only'; opposite signs mean no winner. Alpha is 0.05 per endpoint with Holm inside; error across the five endpoints is not controlled, and this is stated in the report. (2) Constructed endpoints (P3-P5) never create a ranking. They restrict one: a claim 'A better than B' from P1/P2 is printed with the qualifier 'except <endpoint or class>' wherever B is better than A by the counts rule on a constructed primary endpoint. The non-dominated set is computed on the five primary endpoints only: A dominates B if A is better (counts rule) on at least one and B on none. No scalar over cells is computed. (3) Hard natives. (i) Edits of the hit table (HV) are judged only as paired inflation of the SAME model (edited minus unedited, absolute unedited threshold, margin 2 points, ratio printed); they are never set against a model that does not read the table. The union-of-worst-cases rule of the first version is REMOVED. (ii) Across input classes only natural strata that exist for every class are compared: FAST, HEG, PCn natives under 100 aa, PCn natives without a hit, positional-core genes of secondary replicons, and PCn natives in the 5-kb presentation (after U2); each stratum's call rate is taken at the threshold that passes 1% of the PCn natives OUTSIDE the stratum (so no rate is capped); A is 'not worse on hard natives' if the upper limit of [A minus B] is at most 5 points in every stratum. (iii) A cross-class verdict 'better for the user' is issued only if, on development, at least one of these strata reaches a call rate of 3% or more for codon, kmer or fuse; otherwise the report says 'no hard native measured for DNA inputs' and issues no such verdict in stage 1. (iv) Over-calling: because thresholds come from PCn natives, every method calls exactly 1% of the positional-core natives; what each method calls among unresolved genes is printed (call split, GP audit) and enters no rule, since its truth is unknown. (4) Every DNA-versus-similarity statement carries 'island anchor only' (until M29 is run) and 'own genus hidden'. (5) Rule-versus-model differences where the positives lie outside the rule's design scope (in-phylum donors, phage genes filed under the home phylum) are reported as differences of scope next to the in-scope figures (FAR, XD, component XK). (6) Cross-kingdom: leaving the E2 band triggers an audit; the binding regression cells are E1 with backbone-mapped natives (protein-reading methods) and XD (DNA-reading models).

NULLS, PLACEBOS, REPLICATION TABLE. Null cells (D2 of HV; 'neither' of SP; relocated and sham natives) pass when their interval overlaps 0.45-0.55. A trivial score outside is a generator leak: the cell is withdrawn and fixed on development genomes only. A cell withdrawn on Test A is reported as withdrawn and may be rerun only on Reserve B, or on the stage-3a genera, with fresh donors and seeds; it is never rerun on Test A. A real method outside the band is investigated for at most 4 hours and reported as a finding. Before each unsealing a replication table is committed (one line per method and matrix row: fail = inflation above 2 points or recall below half of the matched control; pass otherwise; and, per row, which outcome would falsify the GENERATOR rather than the method). It is written after development results and is therefore a replication check, not a prediction. The 3 deviations with the largest standardised size are investigated before the report; the rest are listed as open.

THE BAR FOR A LEARNED DNA-READING MODEL, FIXED NOW. Scope: bacterial chromosomes and tailed phages. 'Better than fuse' needs all of: (a) on Test A, the gene-level endpoint (P1 for a per-gene model; the context track P1c against all nine context baselines, with block-permutation null and edge analysis, for a model that reads the neighbourhood) and P2 against fuse_nb, each with an interval excluding 0 and a point estimate of at least 0.05; at most 3 evaluations of Test A per model family, each logged with its count printed; (b) on Reserve B, if it exists with at least 5 P1-eligible genera: a mean difference of at least 0.05 with ALL eligible genera positive (one-sided p = 1/2^n: 0.031 for 5, 0.016 for 6); if there is no Reserve B, the same criterion on the stage-3a genera labelled blind before the model is frozen, and until then the claim reads 'not confirmed on untouched genomes'; (c) on the 10 held prophage genomes of M29: recall within the top 1% and top 3% of genes per genome not below fuse by more than 0.05 (a fixed call budget; the transported-threshold criterion of the first version is removed); (d) hard natives as in (3)(ii), and paired inflation under the HV edits if the model reads similarity; (e) cross-kingdom regression in DNA form: R@1% on cell XD not below fuse by more than 0.05, generated with fresh donors on the confirmatory genera; any eukaryote claim needs the held eukaryote genomes of EU; (f) a floor on single genes: non-inferiority to fuse (margin 0.05) in each class NEAR, MID, FAR, MOB at context-clean sites, on the confirmatory genera with the frozen generator; it is a floor, never evidence of superiority, because splices are in-distribution for a trained model; (g) genes called by one caller on all assemblies (stage 2b) before it is scored on any spliced cell; (h) the library used for the confirmatory set is the frozen one, or every old method is rescored on the new one first.

THE OWNER'S FIRST QUESTION (should the virus direction be added to training). Stage 1 answers only 'what does the training set cover' (manifest in the build order) and reports M13 and M16 as descriptive; neither cell varies the training data. The decision is made by an intervention in the stage-2d training slot: wide retrained (i) without the 69 viral genomes, (ii) as now, (iii) with the viral share raised from 9% to about 25% of genomes by adding phage genera outside EXCLUDE.tsv, sampled like the others. All three are scored on development units only: spliced prophages (M16), viral natives (M13), host genes in phages (M27) and the E2 viral trials. 'More viral proteomes help' requires: (iii) minus (ii) at least +0.05 in R@1% on spliced prophage genes with the same sign in at least 3 of the 4 development recipients, the same sign in M27, (ii) minus (i) not negative, and no increase of the call rate on viral natives above 1 point. Otherwise the virus direction enters the next generation only as DNA with gene order and re-filed mobile rows.

DECISIONS WHERE THE CRITIQUES DISAGREED OR WHERE A PROPOSED REPAIR WAS CHANGED (one sentence each).

- 'Core-like by presence' (first skeptic's gate) against 'presence is not native' (second skeptic): the gate and the new negative class both use POSITIONAL orthologues, and presence without position is unresolved.

- Panel-core natives as a guard beside backbone thresholds, or as the threshold class itself: they are the threshold class, because that charges over-calling directly, removes the 3-14-gene thresholds, and is the only way to more than 9 evaluable genera; the published labels stay as a required concordance.

- Composition-typical donors selected on gene-level composition scores, or no composition filter at all: no filter in the primary classes; the composition strata are post hoc and their composition-method figures are printed as 'by construction'.

- Whether 'backbone under V2' is a cross-class hard native: it is not (a DNA method is untouched by construction); it is reported as the sparse-lineage penalty of similarity methods, counting only deep natives as false.

- One unsealing or two: two pre-declared, logged unsealings (U1 real labels, views and viral natives on day 3; U2 splices and fragmentation on day 5), because the three-day plan was not executable and the owner asked for a usable hard benchmark at about three days.

- 12/6 split fixed now, or a split among evaluable genera: the rule and the hash order are fixed now, the outcome is computed from label files on day 2, and without at least 14 eligible genera there is no island reserve.

REVISION LOG (blocking issues; S1, S2, S3 = the three skeptics). S1.1 distant-comparator positives: gene-level gate by the hidden panel, conflict share above 20% removes the genus, one chromosome per genus. S1.2 any-age detections charged: sibling-genus 'deep' flag, S-f required for cross-class claims, V2 penalty counted on deep natives only (bracketed, not removed). S1.3 over-calling invisible: PCn natives are the threshold class; call split printed. S1.4 abstentions in windows: scored-only averaging, S-w as sensitivity. S1.5 window parity at gene level: receptive-field declaration and context track. S1.6 and S2.10(4) bar without single genes and cross-kingdom: cell XD, single-gene floor, fixed call budget. S1.7 and S2.8 asymmetric hard natives: union rule removed, paired inflation, natural strata only, cross-class user verdict conditional. S1.8, S2.3, S2.4 P3: graded loss, orthologue-paired positives, one absolute threshold, deep-vertical stratum, one-sided verdict, both orientations. S1.9 and S2.5 P4: distance-only classes, accessory half, per-class Holm, pooled scalar removed. S1.10, S2.9, S3.10 trivial gate and comparators: all trivial scores must be beaten on Test A; hiding by distance; taxid units; Spearman gate of 0.9; ai_gene and computed rule figures in P3. S1.11, S3.1, S3.4 unit counts: eligibility from label files before the freeze, floors, split ladder, fallback F1, bar criterion on the realised n. S2.1 comparators visible: H1 by distance, Escherichia with Shigella on development, V2 concordance, restatement diagnostic. S2.2 estimand wording: rewritten; standardised precision deleted; README lines corrected. S2.6 bracket: only island/k is a bound; three shares. S2.7 viral natives: viral genus-core class for all methods; hallmark dependence stated. S2.11 bar (c): fixed call budget. S3.2 pooling: removed with the extra chromosomes; quantiles before fusion and windows. S3.3 README outcomes: withdrawn in a dated commit; commit order stated. S3.5 capped strata: thresholds from natives outside the stratum. S3.6 unit eligibility of P3-P5: loss clade of at least 5 proteomes, at least 500 PCn natives, site capacity rule, unit lists in the freeze. S3.7 phage selection: operational definition and fallback. S3.8 day plan: 5 days, top-4,000 search instead of exclusion runs, adapters costed, pipeline budget by measurement. S3.9 wide: existing checkpoints, capped groupings. S3.11 combination: as above. S3.12 owner's first question: training intervention. S3.13 replicon rule: largest replicon only.

## Sealed test policy

THREE TIERS.

Development (open, looked at freely): the 24 Enterobacterales chromosomes of the island set (3 units); E. coli K-12 MG1655 with the DH10B-mapped labels, cryptic prophages and literature-curated islands; E1 and E2 in full; Legionella, Entamoeba, E. cuniculi; the 10 development host genera of the viral set; the 4 development recipients of HV, SP, FR with their donors. Everything that is tuned is tuned here: generator parameters, the gates, the fusion candidate, the frozen thresholds, the replication table. On development Escherichia and Shigella are always hidden together.

Test A (two pre-declared, logged unsealings): by the split rule of component RI, either 13 genera (when at least 14 genera are P1-eligible) or all 18. U1 (end of day 3) joins scores with labels for RI (P1, P1c, P2), HV (P3), XK, GP and the 20 Test A host genera of VR. U2 (end of day 5) opens only the constructed cases built on the Test A recipients: SP (P4, P5, cell XD), FR and the eukaryote contamination construct. Every parameter of SP and FR is fixed in this protocol; between U1 and U2 their code is run on development genomes only. After U1 Test A is 'exposed at design level' for every later method version; a later model family may be evaluated on it at most 3 times, each time logged.

Reserve B: exists for island labels only if at least 14 genera are P1-eligible (then 5 family units, by default Mycobacterium, Acinetobacter, Streptococcus, Staphylococcus, Brucella); otherwise the confirmatory set of the bar is the stage-3a genera, labelled blind before the model is frozen. Independently of that: the 10 Reserve B host genera of the viral set, and, named when their source lists are verified, 10 curated-prophage genomes (M29) and 4 eukaryote genomes (EU). Reserve B is opened once, when the first DNA-reading model is frozen; all methods, old and new, are then scored on it together, on the frozen library (or all old methods are rescored on the new library first). Its constructed cases are generated at that time with the frozen code and donor genera disjoint from development and Test A. Reserve B shares orders with Test A (it is independent at family level only).

WHAT MAY EXIST BEFORE A FREEZE. Label files, panels, eligibility tables, the split, deep searches, pipeline runs and generated edits of sealed genomes may be produced before the freeze, because none is an outcome (the search of the 18 selected chromosomes is running; panel labels of all 18 genera are needed to compute the split). What is sealed is the join of scores with labels. Scoring scripts write scores without access to label files; no printout of a sealed genome before its unsealing contains product text (gene products approximate the island label). Reserve B's hit tables, labels and scores live in a directory that the evaluator refuses to read without the flag --unseal B.

THE FREEZES. F1 (before U1) and F2 (before U2) are tagged commits. F1 holds: bench/hard/PROTOCOL.md (this protocol) and the dated amendment superseding bench/islands/README.md (registered in commit 620f18c); SHA-256 of the library FASTA and DIAMOND database, the taxonomy dump, the existing wide checkpoints and the other learned methods' checkpoints, the rule tag rules-R0, all label and panel files, the hidden sets H1 per genus, the evaluator; the eligibility tables, the split, the unit lists of P1, P2, P3; the frozen thresholds; the fusion candidate; the replication table; the exact byte strings of every SHA-256 seed. F2 adds the generated SP and FR cases, their truth tables, the unit lists of P4 and P5 and the gate results on development. The evaluator is run once per unsealing (--unseal A1, --unseal A2), writes only the registered tables and appends date, commit and hashes to results/unseal.log. Anything computed afterwards that is not registered is labelled exploratory (this includes all stage-1b figures on Test A).

EXCLUSION LIST FOR FUTURE TRAINING (bench/hard/EXCLUDE.tsv, a dated append-only file started now; no training run starts before the entries of the stage it depends on are complete). The 18 island genera and their families as training genomes or queries; every panel genome; every donor species, phage genus, viral recipient genus and plasmid used by the benchmark; the genera of the held prophage and eukaryote genomes once named. Test-only generators: gene-specific graded loss, the false reference cell, the relocation placebo; training code may build its own transplants and splices but may not import bench/hard code, seeds or donors. For the current models the report marks which sealed genera, families and classes were among the 732 training genomes; one retraining of wide with the families of the 18 genera excluded, rescored on Test A, is the stage-2d measurement of that exposure.

REPOSITORY. The repository is public: fetch scripts and SHA-256 values are committed; third-party tables (island set, curated prophages, PHROG, Virus-Host DB, plasmid database) and per-gene labels derived from them are committed only after each licence has been checked.

## Build order (superseded by patch 19)

- Day 0 (half a day, before any score meets a label): commit bench/hard and the modified bench/islands/evaluate.py as they are; write bench/hard/PROTOCOL.md and the ONE dated commit that supersedes bench/islands/README.md (largest-replicon rule, one chromosome per genus, taxid units, corrected label wording, withdrawal of the three registered outcomes, V1 by distance, windows, receptive fields); tag rules-R0; record checksums of checkpoints, library, taxonomy; start EXCLUDE.tsv; read replicon lengths from assembly metadata and fetch the Brucella replacement.

- Day 0, measurements that the schedule depends on: time a 40-protein and a 4,000-protein deep search (fixed against marginal cost); time a top-4,000 search on 500 genes with full lists; time one rule-pipeline run on a development chromosome. Correct every search and pipeline estimate below from these numbers before continuing.

- Day 0, in the background: downloads in this order: genus panels for the 18 sealed and 3 development genera (protein FASTA and feature tables, about 300 assemblies, 450 MB), viral set (about 480 small records, Virus-Host DB table, PHROG profiles), donor genomes and congeners for HV and SP (about 110 genomes, 12 XD records, 34 phage records, plasmids, under 600 MB). Queue of searches with fixed priority: Brucella replacement; 8 further development chromosomes; top-4,000 re-search of full lists (development first, then the 18 sealed chromosomes). Rule-pipeline runs one at a time at night, development first, then Test A in hash order.

- Day 0 adapters (about 120 lines): job file from a taxid without a pipeline run (research/placement/prep.py reads summary.json today); explicit hidden-row mask and a cap of 2 edited genes per 64-gene table for the genome scorer of research/array/wide.py (it hides the genus only today); profile misfit for whole genomes (research/array/ceiling.py handles saved trials only).

- Day 1 - RI on development: hidden sets H1 by distance with Escherichia and Shigella together, top-4,000 views (V2 flagged inexact until the re-search has finished), ai_gene, trivial scores, scored-only windows at 5, 11, 21 genes, composition scores, the 4 fusion candidates, evaluator (t-intervals, sign-flip tests, intersection-union contrasts, block-permutation null, context track). GP code on the development panels: positional orthologues, PCn natives, native blocks, acceptance gates. First numbers: the 4 development chromosomes with a finished search x V0/V1/V2 x all methods and trivial scores, with published labels and with gated labels. Measure: share of full and exact lists, variance of wide over groupings, Spearman of table and pipeline ai_all (gate 0.9).

- Day 2 - labels only for all 18 sealed genera: panel classes, gates per genus, eligibility tables, T, the split by the fixed rule, lock of Reserve B if it exists (or fallback F1 if the development gates failed). Then on development: HV (loss clades, graded LOSS, orthologue positives and their search of about 6,300 proteins, PSWAP, MISLABEL), XK (E2 by cell, E1 against backbone-mapped natives, mutated-native placebo), VR (select 120 phages and their genus panels, search about 8,400 proteins, score the 10 development host genera).

- Day 3 - remaining 8 development chromosomes scored; thresholds, fusion candidate and replication table fixed; unit lists of P1, P2, P3 written; freeze F1; unsealing U1: evaluator --unseal A1 writes docs/hard-results.md part 1 (P1, P1c, P2, P3 with verdicts, matrix rows M1-M15, per-genus tables with realised n and minimum detectable difference, call splits, null cells, replication table against outcomes, coverage, rule tiers for the chromosomes whose pipeline run has finished). This is the three-day deliverable: a hard benchmark on real labels with gated negatives, views of hard natives and viral natives.

- Day 4 - splice module (about 550 lines): donor selection by distance class, accessory test against one congener, block and single insertion at class-independent sites with context-clean half, relocation placebos, contig versions, context-preserving resampling with sham, truth table with flags; fragmentation views. Search inserted proteins and donor profiles (about 12,000 proteins). Run on development only: relocation gate, amelioration gate, null cells; fix what fails on development only.

- Day 5 - unit lists of P4 and P5; freeze F2; unsealing U2: evaluator --unseal A2 writes part 2 (P4, P5, cell XD, rows M16-M22, the eukaryote contamination construct through the pipeline, remaining rule tiers). Stage 1 totals: about 6-9 h of search, 15-30 GPU-hours for wide (nights), about 1,800 lines of new code on top of the 563 that exist (RI 350, adapters 120, GP 220, HV 280, VR 200, XK 120, SP and FR 650), about 2 GB of downloads.

- Drop order inside stage 1 if time runs short (fixed now): F-single; amelioration cells; broad-host-range panel; PSWAP; the eukaryote contamination construct; window sizes 5 and 21 (keep 11); rule tiers beyond 4 development and 6 Test A chromosomes; block B2. Never dropped: RI under V1 and V2 with gated labels, S-pub, window parity, composition, trivial and fusion scores; the GP gates, eligibility and split; HV graded LOSS with orthologue positives and MISLABEL; XK protein cells; VR genus-core natives; SP single genes in four classes, cell XD, host-matched prophages and relocation placebos; both freezes and both logged unsealings. If SP slips past day 6, U2 is postponed, never split further.

- Stage 1b (days 6-8, fixed order; Test A figures exploratory): PX M24 matched truncation (50 and 100 aa); VM M27 host genes in phages; PX M25 chimeric genes (50%); VM M28 mosaics; PX M23 reversed grafts; zero-transfer control (island-excised chromosomes); PX M26 tandem expansion. About 3 days of code, under 3 h of search. No new primary endpoint.

- Stage 2 (about 9 working days, fixed order): 2a curated prophage genomes, M29 (0.5 day, 3.5 h search); 2b one gene caller on every assembly and re-search of the proteins that change (0.5 day; precondition for scoring any learned DNA model on a spliced cell); 2c eukaryote arm EU on 6 opened genomes with sister genomes (2 days, 2-3.5 h search; moved ahead of the items below); 2d training interventions in one slot: the three viral-share arms of the owner's first question, one retraining of wide with the families of the 18 genera excluded, and the pipeline hook that applies visibility and edit filters to the main, deep and home searches so that the shipped rule tiers are scored under V2 and the HV edits (2 days and 4 training runs); 2e plasmid identity labels with the panel-absence condition, M30 (1 day); 2f real contamination with coverage-based labels, M32 (1-2 days); 2g real host-derived viral genes, M31 (1 day).

- Stage 3 (1-2 weeks, fixed order): 3a relatives ladder with empty sites on the island-set genera, then 30-40 further genera labelled blind (the route to resolving differences near 0.05 and, without an island reserve, the confirmatory set of the bar); 3b natural mosaic phages; 3c real endogenous viral elements; 3d contaminated-reference audit and a cleaned library with mobile rows; 3e archaeal recipients; 3f blind tree audit of at most 300 confident unlabelled calls.

- Training-data manifest for the next model generation (the owner's first question; written with the U1 report, each matrix row naming its training need). Today: 732 complete proteomes (352 bacteria, 264 eukaryotes, 44 archaea, 69 viruses of which 44 phages), 100 sampled proteins each, proteins only, transplants between cellular genomes; held-out trials with a viral recipient 108 of 7,668, with a viral donor 60; held-out test genomes 43 eukaryotes, 15 archaea, 12 bacteria, 1 virus; 5% of queries under 100 aa; a third of viral proteins without any cell outside their own proteome; no DNA, no gene order, no mobile-element context, no fragments, no third class, no loss or paralogy negatives. Candidate additions, each tied to a measurement: whole replicons as DNA with coordinates (about 2,000 prokaryote genomes with plasmids as separate replicons); phage genera outside the exclusion list as recipients and as splice donors with host pairs; gene-specific loss, paralogue and false-cell negatives; windowed genes; contig presentation; three classes; undecidable trials removed. Whether the virus direction helps as more proteomes is decided by the stage-2d intervention with its pre-stated minimum gain, not by stage-1 cells.

## Ideas that were dropped

- Proteins-only input with shuffled gene order as the primary condition: it excludes DNA and gene-order methods, which the owner requires as a legitimate class, and hides the strongest baseline on recent blocks (codon usage with neighbours 0.97 against 0.59-0.72).

- Certified core natives from marker genes, synteny in other genera or panel trees as primary negatives for protein methods: clean conserved natives are measured to be saturated and the panel-tree test is the taxonomy rule acting as judge. (The positional-core class adopted now is different: it is defined inside the hidden genus and is not restricted to conserved marker families.)

- A relatives ladder with empty sites over 10-32 relatives in stage 1: postponed to stage 3; stage 1 uses a positional presence test on at most 14 panel genomes per genus.

- A judge that labels transfers from composition windows and mobile-element callers, and any relabelling of negatives from provirus callers or product words: labels would be derived from signals of methods under test.

- A second 30-GB database and about 19,000 trees for pooled adjudication: the evidence type is the methods' own and it does not fit the budget; replaced by the genus panel now and a 300-gene tree audit in stage 3.

- Average precision against all natives and recall at a false-positive rate or count measured on unlabelled natives: both decrease when a method finds more real transfers.

- Random substitution of positives (the mutation arm) and clade grafts: random substitution is kept only as a placebo on natives.

- V0 as a ranked regime: the visible same-genus proteome restates the label.

- A scalar ranking over cells and the bar 'beat the fusion on that scalar': the cell menu decides the order.

- A smoothed-noise control for block structure: it is 0.5 under any label layout; replaced by the block-permutation null and the same window for every score.

- Independent codon resampling of positives only: one-sided synthetic DNA; replaced by a gated, context-preserving resampling with sham natives.

- Matching transplants to natives on functions of the methods' input and post-stratification that drops strata without both labels: both remove the narrow, few-hit and orphan genes that really transfer.

- The implied-foreign-fraction consistency check and precision at a competitor's call count.

- Fully simulated sequences and histories.

- Literature lists of transfers into eukaryotes as a ranking or precision set: found by best-hit screens in the saturated cross-domain cell.

- 'No structure should be found in intact phage parents' as the mosaic control: natural phages are mosaic; replaced by the difference against the parents and sham self-joins.

- F-single scored only on contigs that hold both classes: it leaks prevalence and ignores over-calling on native-only contigs.

- Removed in this revision - up to 3 chromosomes per genus with pooled thresholds: strains repeated the same orthologues, a pooling rule was needed, and positional-core natives make it unnecessary.

- Removed in this revision - published backbone genes as the threshold class of P1 and published backbone regions as the negative units of P2: saturated, 3-14 genes at the threshold, only 9 evaluable genera; kept as the required concordance S-pub.

- Removed in this revision - the fixed 12/6 split by backbone-count triplets and the statements 'expected evaluable Test A genera: 9 to 12' and '5 of 6 Reserve B genera positive': the certain count was 6 and 3, and 5 of 6 has probability 0.11 under no difference.

- Removed in this revision - V1 defined by genus name alone, the substring replicon rule, and units taken from the first word of the 2019 organism string.

- Removed in this revision - V3 (order hidden), the taxonomy-enabled DIAMOND database and per-genome search-time exclusion runs: 30-60 small runs each streaming the 16-GB library; replaced by one top-4,000 search filtered per view.

- Removed in this revision - windows that rank unscored genes last as the primary definition: abstentions of similarity methods cluster in islands and prophages and were read as nativeness; kept as sensitivity S-w.

- Removed in this revision - the 'best trivial score' chosen on development and the percentile bootstrap over 6-13 units: replaced by 'beat every registered trivial score on Test A' and by t-intervals with sign-flip tests.

- Removed in this revision - 'precision standardised to a prevalence of 5%', and the words 'upper bound' and 'panel-supported bound' in the congener audit: only island/k is a bound; presence without position is not evidence of nativeness.

- Removed in this revision - the three outcomes registered in bench/islands/README.md and the pipeline's ai_family runs: withdrawn in a dated commit before any run (they need all 18 chromosomes, a degenerate family home group and saturated negatives).

- Removed in this revision - LOSS as deletion of the whole family judged against a two-sided margin, random donor proteins as its matched positives, the two-set AUROC as the primary figure of P3b, and rule figures printed as 'by construction'.

- Removed in this revision - donor classes selected by GC3 (NEAR-matched, FAR-matched, FAR-distinct) and the equal-weight pooled scalar of P4; the label 'old transfer' for a composition-matched cell.

- Removed in this revision - the cross-class 'worst hard-native rate over the union' rule: every synthetic edit was aimed at similarity inputs, so a DNA-only model passed for free.

- Removed in this revision - hallmark genes as the only viral natives and as the threshold class of M27; 'temperate' without a definition; 'hallmark false-call rates near 1% make more viral proteomes the first training step' as a decision rule.

- Removed in this revision - recall at a transported frozen threshold on genomes without negatives (M29, M30, M31, bar criterion c): replaced by a fixed call budget.

- Removed in this revision - eukaryote natives defined by 'at least 3 introns', thresholds from marker natives, 4 opened eukaryote genomes, and the claim that bench/contamination.py can be reused for the tardigrade assembly.

- Removed in this revision - a single unsealing after three days and an unbounded rule for the number of wide seeds: replaced by two pre-declared unsealings over five days and by the existing checkpoints with a cap of 9 passes.

- Postponed as before: eukaryotic, archaeal and RNA viruses and phages under 14 genes (stage 1b and later); tolerance bounds, beta-binomial dispersion and conformal thresholds in the evaluator.

## Open risks

- NOT REPAIRED, only bounded - any-age detectors are still charged for genus-root and family-root gains among the positional-core natives. S-f (deep natives only) brackets the effect and is required for cross-class claims, but a method that is right about an old transfer in a conserved position still loses budget at the 1% threshold; the size is measured only by the stage-3 ladder.

- NOT REPAIRED - stage 1 has no real measurement of old transfers and no real single-gene transfers. The composition-typical stratum and the single-gene splices are constructions; real old gains and patchy natives wait for stage 3a.

- NOT REPAIRED - eukaryote recipients. Stage 1 adds only regression cells and one descriptive contamination construct; the scope clause says so. The arm with hard natives is stage 2c with 6 opened genomes, enough for a sign statement only.

- NOT REPAIRED - the published comparison-genome lists are probably not retrievable. Hiding by measured distance, the V2 concordance and the single-row restatement diagnostic stand in for the exact list; a comparator that is distant by median identity yet restates labels in a few regions can still slip through under V1.

- NOT REPAIRED - the deployment condition. Every verdict holds for 'own genus hidden'; composition keeps all its evidence while similarity loses the congeners that define the estimand, and a method designed to use congeners cannot be ranked here.

- The number of evaluable genera is still unknown: T (P1-eligible genera) is computed from label files on day 2 and lies between 0 and 17. Eight genera are uncertain (six count-flagged distant-comparator genera, plus Staphylococcus and Prochlorococcus near the 40-gene limit or without enough panel species). Fewer than 14 eligible genera means no island Reserve B and no untouched real-label genomes for the next model until stage 3a; fewer than 9 means P1 and P2 are reported without verdicts.

- Both real-label endpoints now rest on new panel code written by the same engineer (positional orthologues, PCn natives, native blocks). Its safeguards are label-against-label gates on two development units (Salmonella has too few species for a panel) and the required concordance with the published labels on 7-9 genera. If the gates fail, fallback F1 returns to published labels with 9 genera and the saturated-negative problem.

- Panel depth differs by genus (near-identical species in Brucella, deep divergence in Bradyrhizobium) and species are NCBI taxids; 'accessory in the panel' is not 'gained' without an empty-site test, so loss in congeners still produces false positives in the positive class.

- Few and dependent units: at most 13 genera in Test A, 5 of the 18 genera in one order, about 12 orders overall; Reserve B shares orders with Test A. Differences under about 0.08 will read 'unresolved', very likely including wide against the linear rules on real labels; that contrast will be informative mainly on constructed endpoints, which cannot create a ranking.

- The conjunction of conditions for a claim (Holm, three bias-specific concordances, V2, S-f, every trivial score) is strict by design: a true difference of 0.06 will usually not be declared. The report must present 'unresolved' as the expected outcome for small differences, not as evidence of equality.

- The DNA side may have no hard native that bites (highly expressed genes may not trouble likelihood statistics; no multi-replicon genome exists on development to test secondary replicons before unsealing). In that case no cross-class 'better for the user' verdict is issued in stage 1, and the comparison of input classes on native-side errors stays open.

- Generator realism and in-distribution training: clean junctions, random clade deletion and a single false cell are tidier than nature, and a model trained on similar edits can learn them. Placebos through the same code, test-only generators and the exclusion of constructed cells from superiority claims are partial protection.

- Annotation-style differences between recipient, donor and phage records remain until one gene caller is applied (stage 2b); harmless for the stage-1 methods, a leak for any learned DNA model scored earlier.

- Exposure of the learned models is only marked in stage 1: sealed genera lie in classes used for training, the development order was held out and the sealed genera were not, and wide was trained with real islands of training genomes labelled native. The direction of the net bias is unknown until the stage-2d retraining.

- wide is defined by the checkpoints that exist; if only seed 0 exists at the freeze there is no seed variance in any interval and the regression band is the fixed 0.02.

- Rule tiers are the deployed pipeline on unedited genomes with the genus hidden only; on every hard cell the rule family is a table re-implementation without home-search rescue, repeat filter or trees, and its fidelity is checked only by a Spearman gate on development. The tree step of the pipeline is exercised on no hard cell before the stage-2 hook.

- Viral natives: requiring 4 genomes per viral genus selects well-studied phages; the cellular threshold transports poorly to genomes of 30-150 genes, especially for composition scores; hosts are isolation hosts; hallmark labels depend on similarity to phage and prophage families.

- Schedule: stage 1 is 5 days with two unsealings instead of 3 days with one; search and pipeline times are measured on day 0 and may still push the splice part past day 6. About 1,800 new lines by one engineer leave little slack; the drop order is fixed.

- Two unsealings weaken the seal: Test A's real-label results are known while the splice code is finished. Protection is procedural (all SP and FR parameters fixed here, development-only runs between U1 and U2, hashes, a locked directory for Reserve B, logged evaluations); the same engineer builds labels, runs methods and writes the evaluator.

- Server reality: about 2 GB in many small downloads, package installation (tantan, hmmer or pyhmmer, PHROG profiles), a shared GPU, and pipeline runs competing with searches for 16 cores. Only some items have a stated fallback (PHROG product words, dropping the masked-fraction stratum, fewer pipeline runs).

- Family membership and current genus names come from the server's taxonomy dump, which was not read here; if the two two-genus families differ from the assumption, the split order must be recomputed by the stated rule before any score exists.

- The owner's first question is answered only at stage 2d (three training arms on development units, 4 recipients and 10 host genera); that sample can show a large effect of adding viral genomes and cannot exclude a small one.

## Patches of the second verification

Stage 1 can be frozen after the 20 patches below. They close all 15 blocking items and every item raised by two or more checkers. The first freeze (F1, unsealing U1, 4 working days) holds RI and GP only, with primary endpoints P1 and P2. P3, P4, P5 and P1c cannot be computed inside 4 days (donor selection, extra library runs, about 1,800 wide tables per HV chromosome, a 650-line splice module) and are moved to stage 1b under one second freeze F2 and unsealing U2. I checked the patches against the code in bench/hard and research/array; nothing was run and no file was changed. Not measured anywhere: the time of one wide pass on the shared GPU; the plan measures it on day 1 and has a fixed fallback.

### Scope of the first freeze and what moves to stage 1b

FIRST FREEZE (F1, unsealing U1): 4 working days (day 0 to day 3), 32 engineer hours, 3 nights of machine time.

A. CONTENT

- Components: RI (reduced as below); GP (panel classes, gates, eligibility, split, audit M15); HV graded LOSS on development only (descriptive, first to drop).

- Genomes. Development: the 4 development chromosomes with a finished deep search for protein methods (3 units; Salmonella with backbone thresholds, flagged); all 24 development chromosomes for DNA scores and panel labels. Sealed: 18 genera, Brucella replaced by a largest-replicon record; split by the fixed rule on day 2: Test A = 13 genera and Reserve B locked if n_P1 >= 14, otherwise all 18 are Test A; fallback PUB (no verdicts) if n_P1 < 9 or the development gates fail.

- Regimes: V1 (ranked), V2 (concordance only). Not V0.

- Windows: W1 and W11 (scored-only; S-w printed beside). Not W5, W21.

- Methods. DNA: gc3, codon, kmer. Similarity: wide (seed 0, 3 groupings), wide_rule_ai. Similarity plus taxonomy: table ai_all, ai_narrow, ai_gene. Both: fuse (1 of 4 candidates, chosen on development), fuse_triv. Trivial: short, nohit, n_rows, absent_near, best_id, sim_anom, copies, viral, random. W11 versions: codon_nb, kmer_nb, wide_nb, fuse_nb and every trivial score. Rule tiers (pipeline rules-R0): 4 development chromosomes plus the sealed runs finished before F1 (at most 6), counts only.

- Primary endpoints with verdicts: P1 (gene level, W1, V1, floor 9 genera; codon minus wide, fuse minus codon, wide minus the linear rules; Holm over 3). P2 (block level, W11, V1, floor 8 genera; codon_nb minus wide_nb, fuse_nb minus codon_nb; Holm over 2).

- Registered analyses at U1: S-pub, S-a, S-b, S-f, V2 concordance, S-w, the set test against every trivial score, label-shift null at gene and region level, call split, transported thresholds, single-row restatement diagnostic.

- Matrix rows measured on Test A: M1 to M7 and M15. On development only: M8 at loss grades 50% and 75% (paired inflation, no donor).

- Report: docs/hard-results.md part 1; a coverage table of the 22 matrix rows (measured on Test A / on development / existing development result not rerun / not yet measured, with its stage); the training-data manifest.

B. BUILD ORDER WITH HOURS (E = engineer hours)

Day 0 (8 h)

1. 1.0 h. Commit bench/hard and the modified bench/islands/evaluate.py as they are; bench/hard/PROTOCOL.md with these patches; the one dated commit that supersedes bench/islands/README.md; checksums of library, taxonomy, wide checkpoint; EXCLUDE.tsv.

2. 1.0 h. Replicon lengths; fetch the Brucella replacement; labels.py fix (gene at least 50% in both region sets = unlabelled; CDS under 30 residues counted) and rerun; genes/<accession>.tsv without label and product columns.

3. 0.5 h E, 2-4 h in the background. Run bench/hard/panel.py for the 18 sealed and 24 development chromosomes (about 1 GB from ENA).

4. 1.0 h E, 3-4 h machine. ONE library run at 4,000 hits: all proteins of the Brucella replacement and the full-list genes of the 4 searched development and the 17 other sealed chromosomes (about 46,000 proteins).

5. 3.0 h. views.py: merge the top-4,000 lists, first 1,000 visible hits, flag 'exact', H1 with Escherichia and Shigella together, one filtered hit file per regime. wide adapters: columns rebuilt per regime from that file, profile from the unfiltered table, H1 rows as hidden mask, job file from a taxid.

6. 1.5 h. ai_gene, absent_near, random, single-row restatement diagnostic.

Night 0: 4 development rule-pipeline runs, about 1 h each, after the search.

Day 1 (8 h)

7. 2.5 h. GP on development: panel_labels.py with the corrected region rule and the f-based gates; gates decided (at most half a day of fixes, else fallback PUB).

8. 2.5 h. Evaluator part 1: per-chromosome metrics and unit means, scored-only W11, the 4 fusion candidates, LR@2% counting rule, t-intervals, exact sign-flip, Holm per named family, set contrasts, label-shift null at gene and region level; synthetic tests.

9. 2.0 h E plus GPU. The 4 searched development chromosomes under V1 and V2: table, trivial and composition scores, wide, wide_rule_ai. The time of one wide pass and of one column build is measured here.

10. 1.0 h. Development numbers with published and gated labels; share of full and exact lists; SD of wide over groupings; Spearman of table against pipeline ai_all (naming only).

Night 1: rule-pipeline runs of sealed chromosomes in ascending SHA-256 of the accession, outputs unread.

Day 2 (8 h)

11. 2.0 h. Labels only for the 18 sealed genera: panel classes, per-genus checks, eligibility tables, n_P1, split; Reserve B material moved unread to the locked directory; or fallback PUB.

12. 3.0 h. Evaluator part 2: eligibility from the frozen tables, S-pub, S-a, S-b, S-f, V2 concordance with floors, strata of M2-M7, call split, transport, audit M15, --unseal with log and refusal of Reserve B, report writer.

13. 1.0 h. Fixed on development: fusion candidate, transported thresholds, replication table for M1-M7 and M15.

14. 1.5 h E plus about 1,200 wide tables. HV graded LOSS 50% and 75% on the development recipients, no donor, seed 0, one grouping. Descriptive. FIRST TO DROP.

15. 0.5 h E, night 2 machine. Test A scored without label files: filtered hit files and columns for V1 and V2, table, trivial and composition scores, wide, fusion.

Day 3 (8 h)

16. 1.5 h. Coverage check of the scores (no label read), unit lists of P1 and P2, hashes.

17. 0.5 h. Freeze F1.

18. 0.5 h. Unsealing U1, evaluator --unseal A1, once.

19. 3.5 h. Report part 1.

20. 1.5 h. Coverage table of the 22 matrix rows and the training-data manifest.

21. 0.5 h. The 3 largest deviations from the replication table.

Totals: about 1,100 new lines (evaluator 550, views and table scores 200, wide adapters and fusion 130, GP additions 150, HV development 100); 1 library run; 4 development and at most 6 sealed pipeline runs; about 120-150 wide passes.

Drop order inside the first freeze: the HV development figure; sealed rule tiers; two of the three wide groupings under V2. If one column build per regime does not fit, V2 uses the V1 build with the family rows masked and wide's V2 figures are flagged 'inexact'. Never dropped: P1 and P2 under V1 with gated labels, V2 concordance, S-pub, S-a, S-b, S-f, composition, trivial and fusion scores, GP gates, eligibility and split, the freeze and the logged unsealing. If F1 is not complete at the end of day 3, U1 moves as a whole.

C. MOVED TO STAGE 1b (about 6 working days after U1; fixed order; development first in every item; one freeze F2 and one unsealing U2 at the end)

1. HV with donors on development and Test A: donor selection and fetch, orthologue positives and 100 profile proteins per donor, graded LOSS, MISLABEL, D2, PSWAP. Primary endpoint P3 (a, b, c).

2. VR: Virus-Host DB and PHROG tables, selection and fetch of the viral set, genus-core natives (M13, M14).

3. XK: E2 by cell, E1 against backbone-mapped natives, mutated-native placebo (M12).

4. The 8 further development chromosomes (search at 4,000 hits directly, scores).

5. SP: donors, single genes in four classes (primary endpoint P4), host-matched prophages (primary endpoint P5), cell XD, contig versions, relocation and amelioration gates (M16-M21).

6. FR (M22).

7. Eukaryote contamination construct (M19-euk).

8. Remaining Test A rule tiers; P1c with W5 and W21; V0; place.node; profile misfit; secondary replicons.

The former stage 1b (PX, VM) becomes stage 1c, order unchanged.

D. PRIMARY ENDPOINTS MOVED, NOT KEPT: P3 (was U1), P4 and P5 (U2, now the end of stage 1b). P1c (no verdict) also moves. The first freeze has exactly two primary endpoints: P1 and P2.

### Patch 1. RI / Genomes (SPLIT bullet 'If T < 9 ...'; ELIGIBILITY line of P3-P5); the names 'fallback F1' and count 'T' wherever they occur (RI Genomes, GP Labels, Build order, Open risks)

*Problem.* Blocking. Raised by verdict1 N1, verdict2 NEW 1, verdict3 N2 and consistency; the Salmonella case also by executable and consistency. The fallback promised verdicts in one bullet and none in the next bullet and in Open risks. With the PC class withdrawn, HV, SP, FR and VR had no base genes and no threshold; Salmonella on development is in that state today. 'F1' named the freeze and the fallback, 'T' a class and a count. Conflict 1: consistency and verdict2 keep verdicts under the fallback with reworded conditions; verdict1 drops them. Chosen: no verdicts. Two of the three existing sentences already say so, it needs neither S-a nor S-f (not computable once the panel class is withdrawn), and the protocol itself retired these labels as saturated. Conflict 2: verdict2 would use backbone genes also where a PC class stands with fewer than 500 PCn natives; not taken, it mixes threshold classes inside one endpoint. Conflict 3: consistency would cut the development recipients to 3 chromosomes in 2 units; the backbone stand-in (verdict1, verdict3, executable) keeps all 4 searched chromosomes computable from data that exist, so it is chosen.

*Replaces.* '- If T < 9, or if the PC class fails its development gates (GP): fallback F1 = published labels are the primary classes, all 18 genera are Test A (9 evaluable for P1 and P2), verdicts carry the mark ... no Reserve B.' and 'Constructed endpoints P3-P5: at least 500 PCn natives (island labels are not needed).'

*New text.*

- If n_P1 < 9, or if the PC class fails its development gates (GP): fallback PUB. All 18 genera are Test A and there is no Reserve B. P1 and P2 are then REPORTED ON THE PUBLISHED LABELS WITHOUT VERDICTS. Gene level: published island against published backbone genes in the genera with at least 300 backbone and 50 island genes (9 today); R@1% and, as the headline, mean recall over 0-5% (only 3-14 backbone genes sit at the 1% point). Block level: published regions in the genera with at least 20 backbone and 5 island regions, one backbone region allowed to pass. Per-genus values, paired differences, t-intervals and sign-flip p are printed under the mark 'published labels, not gated'; no 'better than' statement is made and no concordance condition is evaluated. Where the PC class stands, the gated-label figures of the eligible genera are printed beside, descriptive.

- A chromosome without a PC class (class withdrawn under PUB; fewer than 3 other-species panel genomes, as for Salmonella on development): its single-copy published backbone genes are the base genes, the threshold class and the 'context-clean' neighbours (5 backbone genes on each side) in HV, SP, FR, VR and PX if there are at least 300 of them; otherwise the chromosome is no unit. Every such figure carries 'backbone thresholds'. Eligibility of P3-P5 reads: at least 500 PCn natives, or, without a PC class, at least 300 single-copy published backbone genes. The development recipients of HV, SP and FR stay the 4 searched chromosomes in 3 units, Salmonella flagged; the transported thresholds and the fusion candidate use the units with a PC class only.

- Names: the fallback is 'PUB' wherever this protocol says 'fallback F1' (F1 is the first freeze only). The number of P1-eligible genera is 'n_P1' wherever this protocol uses T as a count (T is the class 'transferred' only).

### Patch 2. RI / Labels (PRIMARY CLASSES, block track); GP / Labels (NATIVE BLOCK)

*Problem.* Blocking (consistency). 'Labelled' was undefined, and the label code gives the region id only to gated genes: a region with 3 island genes of which 2 are gated qualifies, has 2 genes, and is always 'missed' under the fewer-than-3 rule. The tiling origin of native blocks was not stated. Code change needed in bench/hard/panel_labels.py line 218: count gated genes in the 'at least 3' test.

*Replaces.* 'positive unit = gated island region (published island region with at least 3 labelled genes of which at least 50% are gated island genes); negative unit = native block (11 consecutive genes, none island-labelled or within 5 genes of an island-labelled gene, at least 9 of them PCn, tiled without overlap)'

*New text.*

positive unit = gated island region = a published island region with at least 3 gated island genes that are at least 50% of its island-labelled genes; only its gated island genes carry the region and enter its score. Negative unit = native block = 11 consecutive genes of one contig, none island-labelled or within 5 genes of an island-labelled gene, at least 9 of them PCn; only its PCn genes enter its score. Tiling as in bench/hard/panel_labels.py: from the first gene of each contig, 11 genes forward after a block, 1 gene forward after a window that fails, no wrap.

### Patch 3. RI / Labels (AMENDMENTS 1 and 7)

*Problem.* Blocking (consistency): codon_nb, wide_nb, fuse_nb were never defined and three computations carry the name (bench/islands/composition.py mean of raw scores; metrics.rank_window mean rank over all genes; the protocol's scored-only W11). Raised by three checkers (verdict1 N4, verdict2 NEW 5, verdict3 B7): the secondary-replicon stratum was 'scored' but has no search and no background rule. S-w is redefined as what the existing code does, so it equals the existing baselines and needs no new code.

*Replaces.* 'every other replicon is the stratum 'secondary replicon' (scored, descriptive)' and 'Registered sensitivity S-w: unscored genes enter with q = 0 (the first version's rule). Development prints both.'

*New text.*

(1), last clause: every other replicon is the stratum 'secondary replicon': descriptive, in no decision rule, scored in stage 1b once its proteins have been searched (composition background = the test chromosome; quantiles within the replicon).

(7), last sentences: Registered sensitivity S-w = metrics.rank_window as it is today (unscored genes tie at the lowest rank among all genes; mean over all 2k+1 genes). Development prints both. A window is circular only on a closed chromosome. NAMES: x_nb = W11 of x by this rule (codon_nb, kmer_nb, wide_nb and every trivial score at W11). fuse_nb = W11 applied to the quantile of fuse (fuse first, window second), with the same fusion candidate as fuse. The *_nb columns written by bench/islands/composition.py (mean of the raw score over 11 genes) are a different quantity and are never passed to the evaluator.

### Patch 4. RI / Labels (METHODS BY DECLARED INPUT)

*Problem.* Blocking (consistency), also verdict1 NB2, verdict2 NEW 3, verdict3 N4: fuse had no value for a gene that wide does not score (wide writes no row for a gene without a visible member), and the selection rule named no window, regime, labels, aggregation or tie-break. Blocking (consistency), also verdict2 B9 and executable: the members of 'linear rules' changed with the Spearman gate. Conflict on the missing component: verdict2 fills q = 0.5 (verdict1 fills normal score 0 in the means); verdict3, consistency and verdict1 for the maximum give the gene its one component. Chosen: the one component, with every candidate re-expressed as a quantile so that both groups of genes sit on the same 0-1 scale; a neutral fill would park every no-hit island gene at or above the median for the maximum and near the middle for the means. Receptive fields are declared here because amendment (8) asks for a list and none existed.

*Replaces.* ''Linear rules' = the set {ai_all, ai_narrow, ai_gene, wide_rule_ai}. fuse = one of 4 candidates (maximum of the two quantiles; mean of normal scores with weights codon:wide 1:1, 3:1, 1:3), the one with the highest development R@1% is fixed at freeze F1.'

*New text.*

'Linear rules' = the set {ai_all, ai_narrow, ai_gene, wide_rule_ai}: the three TABLE scores of bench/hard/views.py and wide_rule_ai, computed under the endpoint's regime on all eligible units. This is the set of every primary contrast, whatever the Spearman check of control 7 says. The pipeline's own ai_all and the profile misfit are printed beside and enter no test. ai_narrow: home group = the narrowest of family, order, class with at least 10 visible genera besides the own, else the phylum. ai_gene: the same index with the home rank chosen per gene = the narrowest of family, order, class, phylum in which the gene has a visible hit covering at least 40% of it.

RECEPTIVE FIELDS (amendment 8). Per-gene: gc3, codon, kmer, every table and trivial score, wide (it scores a gene in random groups of 64 genes of the genome, never by position), wide_rule_ai, fuse, fuse_triv, place.node, profile misfit. Context: every Wk score.

fuse. q_x = (rank - 0.5) / n of score x among the n genes of the chromosome that x scores, ties at the average rank. Candidates, in this order: (1) max(q_codon, q_wide); (2), (3), (4) the weighted mean of the normal scores of q_codon and q_wide with weights codon:wide 1:1, 3:1, 1:3. Each candidate is computed on the genes that BOTH components score and is then expressed as a quantile among those genes. A gene scored by one component only takes that component's q. A gene scored by neither is unscored. The candidate fixed at freeze F1 is the one with the highest mean, over the development units with a PC class (per-chromosome values averaged inside a unit; chromosomes with a deep search), of W1 R@1% under V1 with gated labels; ties go to the first candidate in the listed order. fuse_nb and fuse_triv use the same candidate.

### Patch 5. RI / Controls (3, 4, 7)

*Problem.* Blocking (executable): absent_near needs a self score that no search output holds, so a registered trivial score would be defined after seeing data; it feeds condition (7), P3b and fuse_triv. Blocking (consistency), also verdict2 B9 and executable: if the Spearman gate failed, the rule comparator became the pipeline's ai_all, which exists for at most 6 Test A chromosomes, under another hidden set and not under V2, so P1 contrast 3 could not be computed; the gate also named no hidden set. Control 4 put the profile misfit 'in every rule-versus-model contrast' although it is in no set. The 2 bits per residue normalisation is the one in research/array/build.py and wide.py.

*Replaces.* 'absent_near (1 minus the best bit score per residue relative to the self score among the 12 nearest visible references)' and 'A method must exceed EVERY registered trivial score ... printed as information).'; control 4 whole; in control 7 'Table ai_all against the pipeline's ai_all: median Spearman of at least 0.9 ... wherever a pipeline run exists.'

*New text.*

3. Trivial scores are methods under test in every window (higher = more foreign), under the column names of the code: short (minus log length; 'length' in this text), nohit, n_rows, absent_near, best_id, sim_anom, copies, viral ('viral_share' in this text), random = the first 8 bytes of SHA-256('<gene id>|random-v1') read as a number. absent_near = 1 minus the best value of [bit score / (2 x gene length in residues)], clipped to 0-1, over the gene's visible hits in the 12 visible reference proteomes with the highest s(r). The self score is taken as 2 bits per residue (the normalisation of research/array/build.py and wide.py) because the search output holds no self hit. A gene without a hit in those 12 proteomes has the value 1; if fewer than 12 visible proteomes have a defined s(r), all of them are used. A method must beat EVERY registered trivial score of its window on Test A by the set rule of decision-rule condition (7); nothing is selected on development, and the development ranking is printed as information.

4. Fixed untrained taxonomy-free comparators: wide_rule_ai (rule_ai of research/array/model.py: the gene's best similarity outside the genome's nearest rows minus its best inside them, on the rows wide is shown) is a member of the linear rules. The profile misfit (statistic own_all of research/array/ceiling.py) is printed beside from stage 1b and enters no test.

7, second sentence: Table ai_all against the pipeline's ai_all: Spearman over the genes that have both values, per development chromosome with a pipeline run, with the table computed under the pipeline's own hidden set (views.py --by-name); median over those chromosomes. Whatever the result, the linear rules stay the table scores. If the median is below 0.9 they are printed under the names ai_table_all, ai_table_narrow, ai_table_gene and every verdict on 'wide minus the linear rules' reads 'table rules, not the pipeline'. The pipeline's ai_all is never a comparator before the stage-2d hook.

### Patch 6. RI / Metrics (block track)

*Problem.* Blocking (consistency): LR@2% had no counting rule (with 75 blocks, 2% is 1.5) and bench/hard/evaluate.py lines 146-147 always allow exactly one negative unit; which genes enter a region or block score was open. Also consistency: the 'block-permutation null' of the text is a single label shift in the code and exists at region level only; 'no operating point below 0.3%' contradicted the printed version with no native block allowed. Code change: compute 'allowed' per chromosome and pass it to regions_found and shift_null; add the gene-level null.

*Replaces.* 'Region score = median W11 value of the region's genes. LR@2% = share of gated island regions above the score exceeded by 2% of the chromosome's native blocks (at least 1 block of at least 50). Printed beside: the same at 1% and with no native block allowed; the realised region-level rate; region AUROC; the block-permutation null.'

*New text.*

Block track (P2). Region score = median W11 value over the region's gated island genes; native block score = median W11 value over the block's PCn genes; a unit with fewer than 3 scored genes among those is unscored (amendment 7). LR@2%: allowed = max(1, floor(0.02 x number of native blocks of the chromosome)); a gated island region is found when its score is strictly above the (allowed + 1)-th highest native block score; no interpolation (metrics.regions_found with this 'allowed', which is also passed to the null). Published regions (S-pub, fallback PUB) use the same formula, which gives allowed = 1 for their 13-90 backbone regions. Printed beside, descriptive: the same with allowed = floor(0.01 x blocks) and with allowed = 0 (the sentence 'No operating point below 0.3% is reported' concerns the gene track only); the realised region-level rate; region AUROC; the label-shift null. The 'block-permutation null' of this protocol is everywhere the label-shift null of metrics.shift_null (all labels moved together by one random offset, 1,000 offsets); its gene-level form recomputes R@1% at Wk under the same shifted labels.

### Patch 7. Cost fields of RI, HV, XK, VR, SP, VM; VR / Metrics (method list); RI / Genomes (development line)

*Problem.* Raised by verdict2 B7, executable (four items) and consistency: search times of 15-30 minutes ignore the measured fixed cost of 30-40 minutes per library run; the search of the 18 chromosomes is finished, not running; the HV GPU figure of 8 passes contradicts the cap of 2 edited genes per table (about 1,800 tables per chromosome, also verdict3 N5); XK was under-costed; 'rule logic' in VR could mean 90 pipeline runs.

*Replaces.* RI Cost from 'Stage 1, days 0-3. Searches: none for the 18 selected chromosomes (running)' to 'Downloads: 1 GenBank record.'; HV Cost whole; XK Cost whole; '(15 minutes)' in VR; '(20 minutes)' in SP; '(25 minutes)' in VM; 'rule logic' in VR Metrics

*New text.*

RI Cost: 'First freeze, days 0-3. The deep search of the 18 selected chromosomes is finished. Every run against the library costs 30-40 min before the first protein (measured), so all query sets of one day go into ONE DIAMOND run at 4,000 hits. Day 0: one run (all proteins of the Brucella replacement and the full-list genes of the 4 searched development and the 17 other sealed chromosomes, about 46,000 proteins, about 3-4 h). The 8 further development chromosomes are searched in stage 1b at 4,000 hits directly; until then protein methods run on the 4 searched development chromosomes. Rule pipeline (each run holds its own library searches, about 1 h per chromosome on the shared cores): 4 development chromosomes; sealed chromosomes one at a time at night in ascending SHA-256 of the accession, at most 6 counted before U1, the rest in stage 1b. Code: bench/hard holds 1,126 lines (views 218, metrics 140, evaluate 257, panel 246, panel_labels 265); about 1,100 new lines for the first freeze. Downloads: 1 GenBank record and the panels.'

HV Cost: 'Stage 1b (graded LOSS on development without donors is already in the first freeze). Search: orthologue positives (at most 300 per recipient) and 100 profile proteins per donor in the merged runs of stage 1b, two runs (development, then Test A), about 1.5-2 h. Code about 280 lines. GPU: (edited genes / 2) tables per recipient, about 1,800 tables (about 30 passes of a 4,000-gene chromosome) for the full edit list, scored with seed 0 and ONE grouping; if the GPU budget is exceeded, base genes are cut to 150 per chromosome before anything else is dropped.'

XK Cost: 'Stage 1b, about half a day; the 800 placebo proteins ride in a merged library run; the eukaryote construct needs two pipeline runs (contig and integrated version); about 120 lines.'

VR Cost: replace '(15 minutes)' by '(in a merged run; about 45 minutes alone)'. VR Metrics: delete 'rule logic' from the method list; the table scores stand for the rule family, as in HV.

SP Cost: replace '(20 minutes)' by '(two merged runs, development then Test A, about 1.5 h)'. VM Cost: replace '(25 minutes)' by '(in a merged run)'.

### Patch 8. HV / Labels (matched positives for M8) and HV / Metrics

*Problem.* Raised by verdict3 N5 and consistency: in P3b the genes were scored under H1 plus the donor set but thresholded under V1, so the threshold no longer passed 1% of the natives. Blocking (executable): the donor's 'distance set' had no source search; (c) was missing from HV Metrics; 'reciprocal best hit' had no numbers. Conflict: verdict3 puts every HV cell under H1 plus the donor set; consistency keeps P3a under H1 and gives P3b its own threshold under the wider set. Chosen: consistency's split, because P3a under H1 needs no donor and is the only HV figure that can be computed in the first freeze from data that exist.

*Replaces.* 'Visibility = the recipient's H1 plus the donor's genus and its distance set, applied to natives and positives alike. Cell D1 = this.' and 'All at ONE absolute threshold per method: the score exceeded by 1% of the PCn natives of the unedited chromosome under V1.'; adds the P3c line

*New text.*

Labels, replacing the visibility sentence: 'Visibility in D1 and D2 = the recipient's H1 plus the donor's genus and the donor's distance set, applied to the positives and to their paired LOSS natives alike. Donor distance set = rules (b) and (c) of H1 applied to the donor, with s(r) computed from a search of 100 donor proteins (smallest SHA-256('<donor accession>|<protein id>|hv-v1')) that rides in the same library run as the orthologue positives. Reciprocal best hit = best hit in both directions with at least 40% identity over at least 70% of both proteins. Cell D1 = this.'

Metrics, replacing the first sentence: 'One absolute threshold per method and hidden set. P3a (inflation): every gene under H1; threshold = the score exceeded by 1% of the PCn natives of the unedited chromosome under H1. P3b, P3c and D2: positives and their paired LOSS natives under H1 plus the donor's genus and distance set; threshold = the score exceeded by 1% of the unedited PCn natives under that SAME hidden set (as P4 does with donor parity). A paired LOSS native therefore has two scores, one per hidden set, and each is used only with its own threshold.'

Metrics, added after 'Discrimination (P3b)': 'Rule contrast (P3c): D(wide) minus D of each linear rule, intersection-union.'

### Patch 9. GP / Labels (PANEL), GP / Genomes, GP / Cost

*Problem.* Raised by executable and consistency: the text describes RefSeq assemblies with protein FASTA and feature tables (450 MB); bench/hard/panel.py takes INSDC assemblies from ENA as EMBL flat files with submitter annotation, excludes by taxid and by two annotation filters, and transfers about 1 GB. Panel selection was listed as code still to write although it exists.

*Replaces.* 'PANEL per genus: RefSeq assemblies of level Complete Genome or Chromosome ... Files: protein FASTA plus feature table (gene order).'; 'about 300 assemblies, about 450 MB'; GP Cost whole

*New text.*

PANEL per test chromosome, as bench/hard/panel.py builds it: INSDC assemblies from ENA of level complete genome or chromosome with full representation, with the submitters' annotation (not one annotation pipeline). Left out: assemblies behind a library proteome (by assembly accession); assemblies whose taxid equals that of the test chromosome or of any island-set chromosome; assemblies with fewer than 300 translated CDS or with under 80% of their CDS translated. Chosen by ascending SHA-256('<assembly accession>|panel-v2'): up to 5 of the same species, up to 5 of OTHER species of the genus (one assembly per species), 4 of sibling genera (one per genus; filled from the order if the family has fewer than 4). Files: EMBL flat files of all replicons (gene order and sequence).

Genomes: 'about 300 assemblies, about 1 GB'.

Cost: 'First freeze, days 0-2. panel.py and panel_labels.py exist (511 lines); about 150 lines to add (f-based gates, eligibility tables, split, the region rule). Downloads about 1 GB, started on day 0 by running panel.py. Searches against at most 14 small genomes per chromosome: minutes.'

### Patch 10. GP / Labels (CLASSES, FLAGS, ACCEPTANCE GATES) and GP / Controls (agreement tables); RI / Labels (STRATA, family size)

*Problem.* Raised by verdict1 N2, verdict3 N1 and consistency: PC is defined as 'not island-labelled', so the gate 'at most 5% of the published island genes are PC' is always 0 and cannot fail; gate 1 is passed by code that calls everything conserved; c had no denominator. Conflict: verdict1 and consistency keep a flat 5%; verdict3 uses 5% where backbone genes outnumber island genes and 20% elsewhere. Chosen: verdict3. Two Yersinia chromosomes have more island than backbone genes (216/121, 246/118); a flat 5% there would withdraw the PC class for all 18 genera over the label noise that a sealed genus is allowed up to 20%, and P1 and P2 would lose their verdicts. The cryptic-prophage clause needs MG1655, which no step fetches (executable); no coordinate table of the literature islands exists in the repository. 'Single copy' is aligned with the code (pairwise DIAMOND, not mmseqs clusters) because it decides the PCn class of every endpoint.

*Replaces.* 'at least 80% of the single-copy published backbone genes are PC; at most 5% of the published island genes are PC; at most 5% of the genes of the literature-curated islands of Sakai and CFT073 and of the 9 cryptic prophages of K-12 are PC. If a gate fails ... fallback F1 of RI applies.'; '(backbone genes that are PC; island genes that are PC)'; in RI STRATA '(mmseqs easy-cluster, 40% identity, 80% coverage: 1, 2-4, 5 or more)'

*New text.*

ACCEPTANCE GATES (label against label, no score), on the development units with at least 3 other-species genomes (Escherichia+Shigella, Yersinia). Gate 1: at least 80% of the single-copy published backbone genes have f >= 0.8. Gate 2: the conflict share c = island-labelled genes with f >= 0.8 / island-labelled genes of the chromosome, computed per development chromosome; the median c over the chromosomes of a unit is at most 5% for the chromosomes whose published backbone genes outnumber their island genes and at most 20% for the others (the tolerance a sealed genus gets). Gate 3: at most 5% of the genes of the literature-curated islands of Sakai and CFT073 have f >= 0.8; it is computed only if their coordinate table is committed on day 0 and is otherwise printed as 'not computed'. The clause on the 9 cryptic prophages of K-12 is deleted. If gate 1 or 2 fails after at most half a day of fixes on development, the PC class is withdrawn and fallback PUB of RI applies.

DEFINITIONS AS CODED (bench/hard/panel_labels.py): single copy = no other gene of the chromosome aligns with at least 40% identity over at least 80% of both genes (DIAMOND --sensitive, E 1e-5); the family-size stratum (1, 2-4, 5 or more) = connected components of the same alignments, mmseqs is not used; hit for the 'deep' flag = 40% identity over 70% of the gene.

Controls: the agreement tables are 'backbone genes with f >= 0.8' and 'island genes with f >= 0.8'.

### Patch 11. Primary endpoints (P2)

*Problem.* Blocking (consistency): P2 used LR@2% without a counting rule and three undefined score names; Primary endpoints never mentioned the fallback.

*Replaces.* 'Contrasts, Holm over 2: codon_nb minus wide_nb; fuse_nb minus codon_nb.'

*New text.*

LR@2% is counted by the rule under RI Metrics: allowed = max(1, floor(0.02 x native blocks)); a region is found when strictly above the (allowed + 1)-th highest native block; region score over the gated island genes, block score over the PCn genes. Contrasts, Holm over 2: codon_nb minus wide_nb; fuse_nb minus codon_nb, where x_nb = W11 of x by amendment (7) and fuse_nb = W11 of the quantile of fuse. Under fallback PUB (RI, SPLIT) P1 and P2 are reported without verdicts.

### Patch 12. Primary endpoints (P3, P4, P5)

*Problem.* Blocking (consistency): 'Holm over 3' cannot be computed because P3a is an interval verdict without a p-value. Blocking (executable): no step generates, searches or scores the HV cells of Test A before U1, so P3 cannot be a U1 endpoint; it moves to stage 1b. Raised by verdict1 N5 and consistency: P3 and P4 had no unit floor, P3b no minimum of pairs, the deep-vertical verdict no minimum of genes.

*Replaces.* '(component HV; U1; Test A units with at least 100 eligible base genes; loss grades 50% and 75% pooled; one absolute threshold per method = its unedited V1 1% PCn threshold)'; 'Holm over 3.' at the end of P3; adds one sentence to P4

*New text.*

P3, header: '(component HV; stage 1b, unsealing U2; Test A units with at least 100 eligible base genes, floor 7 units, below it descriptive; loss grades 50% and 75% pooled; thresholds per hidden set as under HV Metrics; a unit enters (b) and (c) with at least 50 orthologue pairs; the 'susceptible' verdict of (a) needs at least 30 deep-vertical genes per unit)'.

P3, last sentence: 'P3a is judged by its interval alone and is outside the Holm family; Holm over 2 for (b) and (c).'

P4, after the contrasts: 'A class with fewer than 9 units is descriptive; the Holm family stays 8.'

P4 and P5: 'unsealing U2' is the end of stage 1b.

### Patch 13. Decision rules (UNIT AND INFERENCE)

*Problem.* Raised by verdict2 B1, verdict3 N4 and consistency: development units hold 5 to 14 chromosomes and no sentence said how a unit value is formed, although the fusion candidate, the frozen thresholds and the 3% trigger are development quantities; bench/hard/evaluate.py line 159 still pools the chromosomes of a genus. 'No pooling anywhere' contradicted the pooled transported thresholds. Consistency (blocking part): for a set contrast condition (1) did not say whose interval.

*Replaces.* 'Thresholds are per chromosome, so no pooling of raw scores occurs anywhere.'

*New text.*

Thresholds are per chromosome. On development a unit's value is the mean over its chromosomes of the per-chromosome metric, each with the threshold of its own PCn natives (Escherichia and Shigella are one unit); on Test A a genus is one chromosome. The only pooled quantity is the transported threshold (Transport, VR, F-single), which is descriptive. For a set contrast inside a Holm family, condition (1) below uses the smallest lower interval limit over the set.

### Patch 14. Decision rules (WHEN A DIFFERENCE COUNTS, conditions 4 to 7)

*Problem.* Blocking (consistency), also verdict1 N5 and NB3, verdict2 NEW 4 and B9, verdict3 N3, NB4 and B8. (a) Conditions (5) and (6) were not limited to RI, but HV and SP exist under V1 only and have no S-f form, so no P3c, P4 or P5 difference could ever count. (b) 'Exceeds every trivial score' had no test. (c) No minimum of genera for S-pub, S-a, S-b, V2, S-f and no rule for genera that are 'V2 not evaluable'. (d) 'Input class' was undefined. (e) S-a and S-f had no block-level form for P2. Conflict on 'exceeds': verdict2 and consistency use the t-interval of each member, verdict3 the largest sign-flip p; chosen the t-interval (two checkers; it is the interval the protocol already defines). Conflict on floors: 7 genera (verdict1, verdict3), 6 (verdict2), 5 (consistency); chosen 7 for the two conditions that need an interval excluding 0 and 5 for the same-sign analyses.

*Replaces.* from '(4) for RI, the same sign in the three analyses that address a named bias: S-pub, S-a, S-b;' to '... otherwise the verdict reads 'not shown to detect origin beyond a trivial proxy'.'

*New text.*

(4) for P1 and P2: the same sign in the three analyses that address a named bias, S-pub, S-a and S-b. Sign = sign of the mean paired difference over the genera evaluable in that analysis (S-pub: its own eligibility rule; S-a: at least 20 positives at gene level, at least 3 regions at block level; S-b: the eligible genera that remain). An analysis with fewer than 5 genera is 'not evaluable': the claim is printed without that concordance and says so. (5) for P1 and P2, if A or B reads similarity: same sign and an interval excluding 0 under V2, computed on the eligible genera that are V2-evaluable (the others are left out, never copied from V1); with fewer than 7 such genera the condition is same sign only and the claim carries 'V2 not tested'; where V1 and V2 disagree the V2 result is reported and V1 is marked 'comparators possibly visible'. (6) for P1 and P2, if A and B belong to different input classes: an interval excluding 0 also under S-f, on the genera with at least 300 deep PCn natives (P2: at least 20 native blocks whose PCn genes are all deep); with fewer than 7 such genera, same sign only and 'S-f not tested'. Input classes are DNA, similarity (with or without taxonomy) and both; so (6) applies to codon against wide and to fuse against codon, not to wide against the linear rules. (7) A beats the set of registered trivial scores of its window on that endpoint: for every member the mean paired difference is above 0 and its 95% t-interval excludes 0 (no margin; outside every Holm family); otherwise the verdict reads 'not shown to detect origin beyond a trivial proxy'. SCOPE: P3c, P4 and P5 need (1), (2), (3) and (7) only, under V1 with donor parity. 'The counts rule' elsewhere in this protocol = conditions (1)-(7) with this scope. BLOCK-LEVEL FORMS for P2: S-pub = published regions with one backbone region allowed; S-a = the gated island regions whose gated genes are at least 50% absent-from-genus; S-f = threshold from the native blocks whose PCn genes are all deep.

### Patch 15. Decision rules (HOW COMPONENTS ARE COMBINED, 3 ii and iii); RI / Labels (STRATA, HEG)

*Problem.* Raised by verdict1 N4 and B7, verdict2 NEW 5, verdict3 B7. 'Every stratum' contained secondary replicons, which have no search, no background rule, at most about 5 units and no development genome, so 'not worse on hard natives' could never hold; the 5-kb stratum is 1% by construction for per-gene scores; no minimum stratum size; the trigger 'codon, kmer or fuse' lets FAST natives fire through fuse's similarity half; 'better for the user' lost its definition with the union rule. Conflict on size: 20 genes (verdict1), 30 genes (verdict3), 30 genes in at least 6 units (verdict2); chosen 30 genes and 6 units, the smallest n at which the inverted sign-flip interval exists. The HEG family list matched neither set of research/placement/markers.py; the --strict set is named.

*Replaces.* (3)(ii) and (3)(iii) from '(ii) Across input classes only natural strata ...' to '... issues no such verdict in stage 1.'; in RI STRATA '(Pfam families Ribosomal_, RNA_pol_, EFG_, GTP_EFTU; E <= 1e-10)'

*New text.*

(3) Hard natives. 'A better for the user than B' = 'A better than B' under (1) AND A 'not worse on hard natives' under (ii). (i) unchanged. (ii) Across input classes only natural strata that exist for every class are compared: FAST, HEG, PCn natives under 100 aa, PCn natives without a hit. A genus enters a stratum with at least 30 genes in it, and a stratum enters the rule only if at least 6 units do. Each stratum's call rate is taken at the threshold that passes 1% of the PCn natives OUTSIDE the stratum (so no rate is capped). A is 'not worse on hard natives' if the upper limit of [A minus B] is at most 5 points in every stratum that entered. Secondary replicons and the 5-kb presentation are descriptive and enter no rule (no search of secondary replicons exists before stage 1b; in the 5-kb presentation per-gene scores are identical by construction). (iii) A cross-class verdict 'better for the user' is issued only if, on development (mean over the units with a PC class), at least one of these strata reaches a call rate of 3% or more for codon or kmer; otherwise the report says 'no hard native measured for DNA inputs' and issues no such verdict in stage 1.

RI STRATA: HEG = PCn natives matched by research/placement/markers.py --strict (Pfam families Ribosomal_ and RNA_pol_; E <= 1e-10).

### Patch 16. Decision rules (NULLS, PLACEBOS, first sentence); it also governs SP / Controls 1, 2 and 5

*Problem.* Raised by verdict2 NB5 and verdict3 NB16: 'interval overlaps 0.45-0.55' named no interval. The only interval defined is the t-interval over units; the development gates have 4 recipients in 3 units, where it overlaps the band whatever the generator does, so the relocation and amelioration gates could not fail. Conflict: verdict2 adds a point band 0.42-0.58; verdict3 adds a half-width limit of 0.05 with 'gate not powered' beyond it. Both are taken, with the limit at 0.07: at 0.05 the 80 sham-ameliorated natives of the 4 development recipients (half-width about 0.06) could never pass, which would remove every ameliorated cell without a test.

*Replaces.* 'Null cells (D2 of HV; 'neither' of SP; relocated and sham natives) pass when their interval overlaps 0.45-0.55.'

*New text.*

Null cells and gates (D2 of HV; 'neither' of SP; relocated and sham natives). Statistic: AUROC on the genes pooled over the recipients of the tier (the development recipients for a gate). Interval: 95% bootstrap over genes, stratified by recipient, 2,000 resamples. Pass = the point estimate lies inside 0.42-0.58 AND the interval overlaps 0.45-0.55 AND its half-width is at most 0.07. A wider interval reads 'gate not powered': the cell is descriptive and supports no verdict. On Test A the same statistic is also printed per unit with a t-interval over units.

### Patch 17. Decision rules (THE BAR FOR A LEARNED DNA-READING MODEL: a, b, and a note on e, f)

*Problem.* Raised by verdict1 N3 and verdict2 B3: (b) named no statistic, and 'the same criterion' on the stage-3a set meant all of 30-40 genera positive, which no real model meets; no Reserve B is the likely case. (a) requires P2, which can be below its floor. Conflict: verdict1 asks for one-sided p below 0.05, verdict2 for p at most 0.031; chosen p below 0.05, which equals 'all 5 positive' at n = 5 and stays attainable at 30-40 genera. (e) and (f) cannot be computed as worded (8 context-clean genes per class and recipient; no 95% interval at n = 5) and depend on SP, which is now built in stage 1b: they are marked provisional instead of being frozen unattainable.

*Replaces.* '(b) on Reserve B, if it exists with at least 5 P1-eligible genera: a mean difference of at least 0.05 with ALL eligible genera positive ... the claim reads 'not confirmed on untouched genomes';'; adds one sentence to (a) and one after (h)

*New text.*

(a), added at the end: 'If P2 is below its floor on Test A, (a) is judged at gene level and the claim reads 'at gene level only'.'

(b) on the confirmatory set (Reserve B if it exists with at least 5 P1-eligible genera, otherwise the stage-3a genera labelled blind before the model is frozen; until then the claim reads 'not confirmed on untouched genomes'): statistic = the per-genus difference on the gene-level endpoint (P1 R@1% against fuse for a per-gene model; for a context model P1c against each of the nine context baselines, intersection-union: smallest mean difference, largest p). Criterion: mean difference at least 0.05 AND one-sided exact sign-flip p below 0.05. With 5 genera this means all 5 positive (p = 0.031); with the 30-40 genera of stage 3a it does not require every genus to be positive;

Added after (h): 'Criteria (e) and (f) are PROVISIONAL until freeze F2: SP is built in stage 1b, and their statistic, margin and unit floor are fixed there on development evidence, before any DNA-reading model exists. Until then no model can be declared to pass or fail them.'

### Patch 18. Sealed test policy (THREE TIERS: Test A; WHAT MAY EXIST BEFORE A FREEZE; THE FREEZES); RI / Labels (AMENDMENT 6, identifiers)

*Problem.* Blocking (executable): U1 listed P3, XK and VR, for which nothing on Test A is generated, searched or scored before day 3. Raised by verdict2 B7 and executable: no slip rule for U1; 'hash order' of the pipeline queue named no hash and used Test A before the split. Also verdict2 NEW 2 (SP and FR have choices without a rule and their code does run on Test A between U1 and U2) and consistency (the scoring scripts read the label file, which carries label, region and product columns; identifiers are locus tags, not renumbered). The search of the 18 chromosomes is finished.

*Replaces.* 'U1 (end of day 3) joins scores with labels for RI (P1, P1c, P2), HV (P3), XK, GP and the 20 Test A host genera of VR. U2 (end of day 5) opens only ... their code is run on development genomes only.'; '(the search of the 18 selected chromosomes is running; ...)'; 'Scoring scripts write scores without access to label files;'; 'the unit lists of P1, P2, P3'; 'F2 adds the generated SP and FR cases, their truth tables, the unit lists of P4 and P5 and the gate results on development.'; in RI amendment (6) 'identifiers renumbered by position'

*New text.*

Test A, replacing the U1 and U2 sentences: 'U1 (end of day 3, after freeze F1) joins scores with labels for RI (P1, P2, the analyses S-pub, S-a, S-b, S-f, V2 and S-w, the strata of rows M2-M7, transport, the rule tiers whose pipeline run has finished) and GP (audit M15). U2 (end of stage 1b, after freeze F2) opens everything else built on Test A: HV (P3), the 20 Test A host genera of VR, XK, SP (P4, P5, cell XD), FR, the eukaryote contamination construct, P1c, the remaining rule tiers. What U1 contains is fixed at F1; if F1 is not complete at the end of day 3, U1 moves as a whole and no component is unsealed alone. Between U1 and U2 the generators of HV, SP and FR may change only on development evidence; Test A cases are generated once, by the code tagged after the development gates have passed, and no score of a Test A constructed case is joined to its truth table before U2. Every HV, SP or FR choice without a stated rule takes the candidate with the smallest SHA-256('<recipient accession>|<candidate id>|sp-v2') among those that meet the stated condition; a phage that does not fit under the 5% cap is replaced by the next in hash order, never cut. These strings are part of F1.'

What may exist: '(the search of the 18 selected chromosomes is finished; panel labels of all 18 genera are needed to compute the split). Pipeline runs of sealed chromosomes are queued in ascending SHA-256 of the accession; the outputs of chromosomes that the split sends to Reserve B are moved unread to the locked directory.' and 'Scoring scripts read genes/<accession>.tsv (gene_id, contig, start, end, strand only) and never a label file; label files carry the label, region and product columns and are opened only by the label code and by the evaluator at an unsealing.'

The freezes: F1 holds 'the unit lists of P1 and P2'; 'F2 adds the HV, VR, XK, SP and FR cases, their truth tables, the unit lists of P3, P4 and P5 and the gate results on development.'

RI amendment (6): 'identifiers are the pipeline's locus-tag names (renumbering by position stays in the SP input contract for DNA-reading models)'.

### Patch 19. Build order (the bullets from 'Day 0 (half a day ...' through 'Stage 1b (days 6-8 ...', including the drop order)

*Problem.* Blocking (executable, two items): day 0 scheduled downloads of the viral set and of HV and SP donors, whose lists are produced by code of days 2 and 4; nothing generated, searched or scored the Test A cells of HV, RI and VR before U1. Raised by verdict2 B7 and executable (six items) and consistency: three timing runs measure what is already known; small searches each pay 30-40 minutes; realistic duration 8-10 working days; code 3,000 lines, not 1,800; the wide adapter was a row mask although wide reads columns built from the first 1,000 stored hits; the S-b diagnostic, the genes table and the pipeline queue order were in no day. The smallest scope that answers the owner within 4 days (executable) is adopted: RI and GP at U1; everything else in stage 1b in a fixed order.

*Replaces.* the nine bullets from '- Day 0 (half a day, before any score meets a label): ...' through '- Stage 1b (days 6-8, fixed order; Test A figures exploratory): ...'

*New text.*

- This build order supersedes every day number elsewhere in this protocol. Freeze F1 and unsealing U1 are at the end of day 3 and hold RI and GP only (primary endpoints P1 and P2). Everything else of the former stage 1 is stage 1b, with ONE second freeze F2 and unsealing U2 at its end. Matrix rows M1-M7 and M15 are opened at U1; M8-M14 and M16-M22 at U2. The former stage 1b (PX, VM) is stage 1c.

- Search rule: every run against the library costs 30-40 min before the first protein, so all query sets of one day go into ONE DIAMOND run at 4,000 hits. No list is downloaded before the code that selects it has run.

- Day 0 (8 h). (1 h) Commit bench/hard and the modified bench/islands/evaluate.py as they are; bench/hard/PROTOCOL.md; the ONE dated commit that supersedes bench/islands/README.md (as listed before, plus: the 50/50 gene rule of the README is replaced by the eligibility rules of this protocol); checksums of library, taxonomy and the wide checkpoint; EXCLUDE.tsv. (1 h) Replicon lengths; fetch the Brucella replacement; labels.py fix (a gene at least 50% in both region sets is unlabelled; CDS under 30 residues are counted) and rerun; genes/<accession>.tsv without label and product columns for every scoring script. (0.5 h, then 2-4 h in the background) run bench/hard/panel.py for the 18 sealed and 24 development chromosomes. (1 h, then 3-4 h machine) the one library run: all proteins of the Brucella replacement and the full-list genes of the 4 searched development and the 17 other sealed chromosomes. (3 h) views.py: merged top-4,000 lists, first 1,000 visible hits, flag 'exact', H1 with Escherichia and Shigella together, one filtered hit file per regime; wide adapters: columns rebuilt per regime from that filtered file (build.py columns), genome profile from the unfiltered table, H1 rows as the hidden mask, job file from a taxid. (1.5 h) ai_gene, absent_near, random, the single-row restatement diagnostic (covering = at least 40% of the gene). Night 0: 4 development rule-pipeline runs (run.py --configs default), one at a time after the search.

- Day 1 (8 h). (2.5 h) GP on development: panel_labels.py with the corrected region rule and the f-based gates; gates decided. (2.5 h) Evaluator, part 1: per-chromosome metrics and unit means, scored-only W11, the 4 fusion candidates, the LR@2% counting rule, t-intervals, exact sign-flip, Holm per named family, set contrasts, label-shift null at gene and region level; tests on synthetic data. (2 h) The 4 searched development chromosomes under V1 and V2: table, trivial and composition scores, wide (3 groupings), wide_rule_ai; the time of one wide pass and of one column build is measured here. (1 h) Development numbers with published and gated labels; share of full and exact lists; SD of wide over groupings; Spearman of table against pipeline ai_all (naming only). Night 1: rule-pipeline runs of sealed chromosomes, one at a time, in ascending SHA-256 of the accession, outputs unread.

- Day 2 (8 h). (2 h) Labels only for the 18 sealed genera: panel classes, per-genus checks, eligibility tables, n_P1, the split by the fixed rule; Reserve B's hit tables, labels and pipeline outputs moved unread to the locked directory; or fallback PUB. (3 h) Evaluator, part 2: eligibility from the frozen tables, S-pub, S-a, S-b, S-f, V2 concordance with its floors, strata of rows M2-M7, call split, transport, audit M15, --unseal with log and refusal of Reserve B, report writer; frozen_precision_5 deleted. (1 h) Fixed on development: fusion candidate, transported thresholds, replication table for rows M1-M7 and M15. (1.5 h) HV graded LOSS at 50% and 75% on the development recipients (no donor, no download, no search; seed 0, one grouping, at most 2 edited genes per table): descriptive, FIRST TO DROP. (0.5 h, then night 2) Test A scored without label files: filtered hit files and columns for V1 and V2, table, trivial and composition scores, wide, fusion.

- Day 3 (8 h). (1.5 h) Coverage check of the scores, unit lists of P1 and P2, hashes. (0.5 h) Freeze F1. (0.5 h) Unsealing U1: evaluator --unseal A1, once. (3.5 h) docs/hard-results.md part 1: P1 and P2 with verdicts (or PUB without), per-genus tables with realised n and minimum detectable difference, concordances, call splits, strata M2-M7, audit M15, transport, restatement flags, the HV development figure, rule tiers of the finished runs (at most 6 sealed, descriptive). (1.5 h) Coverage table of the 22 matrix rows (measured on Test A / on development / existing development result not rerun / not yet measured, with its stage) and the training-data manifest. (0.5 h) The 3 largest deviations from the replication table.

- Drop order inside the first freeze: the HV development figure; sealed rule tiers; two of the three wide groupings under V2. If one column build per regime does not fit, V2 uses the V1 build with the family rows masked and wide's V2 figures are flagged 'inexact'. Never dropped: P1 and P2 under V1 with gated labels, the V2 concordance, S-pub, S-a, S-b, S-f, composition, trivial and fusion scores, the GP gates, eligibility and split, the freeze and the logged unsealing. If F1 is not complete at the end of day 3, U1 moves as a whole.

- First-freeze totals: 32 engineer hours; about 1,100 new lines (evaluator 550, views and table scores 200, wide adapters and fusion 130, GP additions 150, HV development 100); 1 library run (3-4 h); 4 development and at most 6 sealed pipeline runs (about 1 h each, at night); about 120-150 wide passes on the shared GPU.

- Stage 1b (about 6 working days after U1; fixed order; development first in every item; Test A cases generated once by the code tagged after the development gates; then F2 and U2). (1) HV with donors: select and fetch one donor genome per recipient; orthologue positives and 100 profile proteins per donor in one library run; graded LOSS, MISLABEL, D2, PSWAP: P3. (2) VR: Virus-Host DB and PHROG tables, select and fetch the viral set, genus-core natives. (3) XK: E2 by cell, E1 against backbone-mapped natives (commit the DH10B mapping script), mutated-native placebo. (4) The 8 further development chromosomes, searched at 4,000 hits directly. (5) SP: select and fetch donors, congeners, phages and plasmids; single genes (P4), host-matched prophages (P5), cell XD, contig versions, gates. (6) FR. (7) The eukaryote contamination construct (two pipeline runs). (8) Remaining Test A rule tiers, P1c with W5 and W21, V0 (needs a species column in the library table), place.node, profile misfit, secondary replicons. Drop order inside stage 1b: F-single; amelioration cells; broad-host-range panel; PSWAP; the eukaryote construct; block B2. If SP slips, U2 is postponed, never split. About 1,900 further lines and 4-5 library runs (about 8 h in all when merged per day).

- Stage 1c (the former stage 1b, order unchanged; Test A figures exploratory): PX M24, VM M27, PX M25, VM M28, PX M23, zero-transfer control, PX M26. No new primary endpoint.

### Patch 20. Estimands (MATCH TO EACH METHOD, wide)

*Problem.* Consistency: the text says wide compares a gene with 'the surrounding genes'. In research/array/wide.py (class Whole) genes are scored in random groups of 64, never by position. As written, wide would be a context method under amendment (8) and could not enter P1, where all three contrasts use it or fuse.

*Replaces.* 'wide: 'similarity pattern differs from that of the surrounding genes', taxonomy-free'

*New text.*

wide: 'similarity pattern differs from that of the other genes of the genome' (genes are scored in random groups of 64, never by position: a per-gene method under amendment 8), taxonomy-free

### Left open on purpose

- Bar criteria (e) and (f): statistic against relocated natives and a computable margin (verdict1 B6, N3). Not fixed: one checker, not stage 1, and SP is now built in stage 1b. They are marked provisional until F2. The U1 report must say in its section on the bar that no model can pass or fail (e) and (f) yet, and the F2 text must fix statistic, margin and unit floor before any DNA-reading model exists.

- Pass rate of the full conjunction of conditions (verdict1 NB3). Not simulated: one checker, and it needs a simulation the 4 days cannot hold. The report must print realised n and minimum detectable difference beside every verdict and state that 'unresolved' is the expected reading for true differences under about 0.08-0.10, not evidence of equality.

- SP and FR details still without a rule (verdict2 NEW 2 rest; consistency: P5 window and threshold wording, MOB donors, two tie rules under one name, 4 not 3 composites at G = 1,600, XD genes, FR contig lengths). Only the hash default and the generate-once rule are adopted now, because SP is stage 1b. They must be written into the F2 text before any Test A composite is generated; the U1 report must list SP and FR as 'not yet frozen'.

- The seal is weaker than one unsealing: the HV, SP and FR generators are written by the same engineer after Test A's real-label results are known. Inherent in the 4-day scope. The limits section of the U1 and U2 reports must say so in these words.

- Thin development base: fusion candidate, transported thresholds and the HV development figure rest on the 4 searched development chromosomes, of which at most 3 in 2 units have a PC class; the 8 further development chromosomes come in stage 1b. The report must say this beside the fusion candidate and the transport table.

- Secondary replicons are not searched and not scored in the first freeze (the smaller fix of verdict1 N4 was taken). The coverage table must show them as 'not measured' and the report makes no statement on multi-replicon genomes.

- Code-side differences listed by the consistency checker (percentile bootstrap, pooling per genus and frozen_precision_5 in bench/hard/evaluate.py, eligibility defaults, labels.py line 72, views.py column names and missing V0, fetch.py genus and replicon test, run.py double configuration). These are code tasks inside the build-order hours, not protocol text. The methods section must name the evaluator commit and state that the earlier development baselines (pooled per genus, old window) are not comparable with P1 and P2.

- Two tie rules: R@1% interpolates inside tie groups, the call split and transport use a strict threshold (consistency, one checker). Left as is; it matters only for tied scores such as nohit. A footnote under every call-split table must say that its counts can differ from R@1% times the class size.

- Variance of wide over seeds and groupings 'added to the intervals' has no formula (consistency, one checker); the proposed formula double counts, since the between-genus variance already contains grouping noise. Only seed 0 exists. The report must print the SD over groupings beside every wide interval and the note 'no seed variance'.

- Stage 1b is estimated at about 6 working days and the time of one wide pass on the shared GPU was never measured. The U1 report must give the measured pass time and the revised date of U2.

- Under fallback PUB there are no verdicts, and constructed endpoints fall to at most 9 genera with thresholds from 300 backbone genes (3 genes at the 1% point). Every table produced under PUB must carry 'published labels, not gated' or 'backbone thresholds' in its header.

- Wording gaps that enter no first-freeze endpoint (consistency, one checker): 'rule logic', 'pointer rows', E1, E2, the coverage number in the breadth terciles, and Shigella assemblies counted as a sibling genus of Escherichia in the panel. The U1 report needs a glossary, and the GP table must print panel roles and annotation source per genus.

## Corrections of the two judges

All corrections marked blocking are adopted. The two that concern the first freeze are already in PROTOCOL.md: no registered analysis is waived (condition 4),

`fuse_triv` joins the trivial set for fusions (condition 7), hard-native strata use only genes that both methods score, and the two-step rule for native blocks.

### Judge 1

Not ready: 4 blocking problems, 12 non-blocking. All are fixed by protocol text; none needs new data. Nothing was run and no file was changed. I read the protocol and the code in bench/hard (views.py, metrics.py, evaluate.py, panel.py, panel_labels.py), bench/islands (composition.py, labels.py, README.md) and research/array (wide.py, model.py).

Answers to the four questions:

- Decided by construction: no first-freeze contrast of P1 or P2 is fully decided by construction. Three rules do tilt a result by construction: the hard-native rule (blocking 3), the single-grouping scoring of HV edits (P3a), and the inexact V2 fallback (condition 5).

- Reward for abstaining: yes, in the hard-native rule of patch 15. A similarity method leaves no-hit natives unscored, so its call rate there is 0 and it is 'not worse' for free. The fuse rule of patch 4 does not reward abstention (a gene wide cannot score falls back to the codon quantile).

- Win without detecting transfers better: yes, in three places. (a) Patch 14 waives a required concordance exactly when most genera are flagged. (b) fuse_triv is defined but no rule uses it, so 'fuse better than codon' and the bar 'better than fuse' never test whether the learned model adds anything over codon plus a no-near-hit flag. In views.py best_id, n_rows and the table indices are empty for no-hit genes, while absent_near puts them at the top, so fuse_triv can plausibly beat fuse on these labels. (c) The new scope sentence lets a win on a constructed endpoint alone make a method 'dominate'.

- Contradictions: the move of P3-P5 into 'stage 1b' collides with two unpatched sentences that call all stage-1b figures exploratory and not primary (blocking 4). Others: patch 13 against patch 16 (pooling), patch 18 against the SP section (cutting a phage), patch 1 against patch 4 under PUB, patch 14 against RI control 6 and against the set-test sentence of UNIT AND INFERENCE.

Checked and found consistent with the code: the tiling and region rule (patch 2), S-w as metrics.rank_window (patch 3), the 2 bits per residue normalisation (patch 5), regions_found with 'allowed' (patch 6), the panel description (patch 9), single copy as coded (patch 10), wide as a per-gene method scored in random groups of 64 (patch 20), the sign-flip arithmetic of patches 15 and 17.

**Blocking: Decision rules (WHEN A DIFFERENCE COUNTS, condition 4); RI / Controls 6**

Lets a method win without detecting transfers better, and contradicts an unpatched section. The patch says an analysis with fewer than 5 genera is 'not evaluable' and the claim is printed without that concordance. S-b removes the genera flagged 'label visible' and the count-flagged distant-comparator genera. It falls under 5 genera exactly when most genera are flagged, so the check is dropped when the label bias is widest. This is plausible: under the default split Test A has about 6 unflagged P1-candidate genera (Bacillus, Burkholderia, Corynebacterium, Pseudomonas, Rhizobium, Sinorhizobium); two 'label visible' flags bring S-b to 4. RI control 6 (unpatched) still says S-pub, S-a and S-b are 'required for a claim'.

*Correction.* Replace the last sentence of (4) by: 'No analysis is waived. With fewer than 5 genera the sign is still taken over the genera the analysis has, and the claim carries "<analysis> on n genera". If S-pub, S-a or S-b has no genus, no "better than" verdict is issued on that endpoint; the result reads "unresolved: <analysis> not evaluable".' RI control 6 stays as it is.

**Blocking: Decision rules (WHEN A DIFFERENCE COUNTS, condition 7); RI / Controls 3; RI / Labels (METHODS BY DECLARED INPUT, fuse); bar criteria (a) and (b)**

Lets a method win without detecting transfers better. fuse_triv (codon plus absent_near) is defined, but it is not in the registered trivial list of control 3 and no rule uses it. Condition (7) compares fuse only with single trivial scores, which fuse beats through its codon half. So 'fuse better than codon' (a contrast of P1 and of P2) and the bar 'better than fuse' never test whether wide adds anything over a no-near-hit flag. On these labels the gap can be real: PCn natives are conserved in congeners and nearly always have hits, island genes often have none; absent_near puts every no-hit gene at the top, while fuse gives a no-hit gene only its codon quantile.

*Correction.* Add to (7): 'For A = fuse or fuse_nb, and for any method that reads both DNA and similarity, the set also contains fuse_triv at the same window (fuse_triv at W1; fuse_triv_nb = W11 applied to the quantile of fuse_triv). If A does not beat it, the verdict reads "fusion not shown to add to composition plus a trivial similarity proxy".' Add to control 3: 'fuse_triv and fuse_triv_nb are registered for this purpose.' In the bar, (a) and (b): replace 'against fuse' by 'against fuse and fuse_triv' and 'against fuse_nb' by 'against fuse_nb and fuse_triv_nb'.

**Blocking: Decision rules (HOW COMPONENTS ARE COMBINED, 3 ii)**

Rewards a method for abstaining. The patch keeps the strata 'PCn natives without a hit' and 'PCn natives under 100 aa' in the rule 'A not worse on hard natives'. A similarity method leaves a no-hit gene unscored; an unscored gene 'ranks last, is never called' (amendment 6). Its call rate in the no-hit stratum is therefore 0 by construction: wide is 'not worse' than codon for free, and codon is 'worse' whenever it calls more than 5 points of those genes. The short-gene stratum has the same effect in part. The stratum can enter: genus-restricted core genes have no hit once the genus is hidden, so 30 genes in 6 units is reachable.

*Correction.* Replace the two sentences on entry by: 'For a pair (A, B) the call rates are taken only on the stratum genes that BOTH methods score; a gene a method leaves unscored is an abstention, printed as coverage per method and stratum, and is never counted as a non-call. A genus enters a stratum for that pair with at least 30 such genes, and a stratum enters the rule for that pair only if at least 6 units do. "PCn natives without a hit" therefore enters no pair that contains a similarity-only method; it is reported under M3.'

**Blocking: Build order (first bullet: 'The former stage 1b (PX, VM) is stage 1c'); Primary endpoints (last bullet, unpatched); Sealed test policy (THE FREEZES, unpatched sentence); HV / Genomes; RI / Genomes (development line); Decisions where the critiques disagreed; Open risks**

Contradiction with unpatched sections on the status of three primary endpoints. P3, P4 and P5 are moved into 'stage 1b', but 12 unpatched places still use 'stage 1b' for the old content. Two of them change the meaning of P3-P5: Primary endpoints says 'Not primary, by decision: ... all of stages 1b to 3', and THE FREEZES says everything unregistered is exploratory, '(this includes all stage-1b figures on Test A)'. After the results are known either reading could be claimed. Smaller stale sentences: HV Genomes writes the unit list of P3 'into freeze F1' (patch 18 says F2); the RI development line still says protein methods run on 12 chromosomes; 'Decisions where the critiques disagreed' and two Open risks describe U1 as 'real labels, views and viral natives on day 3' and say 'all SP and FR parameters fixed here' (left_open says they are not).

*Correction.* Add to the first build-order bullet: 'In every section that these patches do not replace, "stage 1b" means stage 1c (PX, VM, E_cell, the zero-transfer control, small phages and other viruses).' Primary endpoints, last bullet: '... all of stages 1c to 3.' THE FREEZES: '(this includes all stage-1c figures on Test A; P3, P4 and P5 are registered here, are built in stage 1b and are opened once at U2).' HV Genomes: 'The unit list of P3 is written into freeze F2.' RI Genomes, development line: 'protein methods on the 4 searched chromosomes in the first freeze, on 12 from stage 1b.' Decisions: 'two pre-declared, logged unsealings: U1 (RI and GP, end of day 3) and U2 (everything else built on Test A, end of stage 1b).' Open risks: replace 'all SP and FR parameters fixed here' by 'SP and FR parameters fixed at F2 on development evidence only'.

**Note: Decision rules (WHEN A DIFFERENCE COUNTS, conditions 5 and 6)**

Below 7 genera both conditions fall back to 'same sign only', and no lower floor is given. With 0 genera the sign does not exist and the condition is passed by default. Condition (5) protects against a similarity method that reads a visible comparator; condition (6) protects similarity methods against thresholds set by genus-root gains. Each then disappears for the side it protects.

*Correction.* Add to (5): 'With no V2-evaluable genus, no "better than" verdict is issued when A reads similarity; when only B reads similarity the claim carries "V2 not evaluable".' Add to (6): 'With no genus that meets the S-f size rule, no "better than" verdict is issued when A reads DNA only; otherwise the claim carries "S-f not evaluable".'

**Note: Build order (drop order inside the first freeze: 'V2 uses the V1 build with the family rows masked and wide's V2 figures are flagged inexact')**

The fallback feeds condition (5) with a biased figure. Under it wide, wide_rule_ai and fuse read columns built from the first 1,000 V1-visible hits with the family masked, while the table rules read the exact V2 file. 'wide minus the linear rules' under V2 then compares unequal inputs, and 'codon minus wide' under V2 is tilted toward codon. The patch flags the figures but does not say how (5) is judged; 'the V2 concordance' is in the never-dropped list, so the inexact figures would be used.

*Correction.* Add: 'An inexact V2 figure of wide, wide_rule_ai, fuse or their W11 forms does not enter condition (5). For every contrast that contains one of them the condition reads "V2 not tested" and the claim carries that mark. The V2 figures of the table, composition and trivial scores are computed from the exact filtered file and printed as usual.'

**Note: Cost field of HV ('scored with seed 0 and ONE grouping'); Build order day 2 (HV development figure); HV / Controls**

Tilts P3a against wide by construction. Edited genes are scored with one grouping, but wide's ordinary score, which gives the unedited control C0 and the 1% threshold, is the mean over 3 groupings. A single-grouping score has more variance, so more edited genes cross a threshold set on averaged scores even if the edit does nothing. The paired inflation is then above 0 from noise alone. The same applies to the HV development figure of the first freeze.

*Correction.* Add to HV Controls and to the HV cost line: 'The unedited control C0 of every edited gene and the PCn natives that set the threshold are scored through the same adapter, with the same seed and the same single grouping (one unedited pass with that grouping). A score averaged over several groupings is never compared with a single-grouping score.'

**Note: Decision rules (NULLS, PLACEBOS, first sentence) against Decision rules (UNIT AND INFERENCE)**

Two patches contradict each other. Patch 13 says 'The only pooled quantity is the transported threshold'. Patch 16 defines the null and gate statistic as 'AUROC on the genes pooled over the recipients'. Raw scores of different assemblies are not on one scale, so a pooled AUROC can pass or fail on between-genome offsets.

*Correction.* In the null rule: 'Statistic: the AUROC computed inside each recipient (positives and negatives of the same assembly), averaged over the recipients of the tier with weights = number of positive-negative pairs; no score is compared across assemblies. Interval: 95% bootstrap over genes inside each recipient, 2,000 resamples.' The pass rule is unchanged.

**Note: Sealed test policy (THREE TIERS, Test A: 'a phage that does not fit under the 5% cap is replaced by the next in hash order, never cut') against SP / Labels (PLACEMENT)**

Contradiction with an unpatched section. SP PLACEMENT still says '(a phage is cut to a contiguous part if needed)'. P5 is defined on an intact phage spliced whole.

*Correction.* In SP PLACEMENT delete '(a phage is cut to a contiguous part if needed)' and write: 'a phage that does not fit under the 5% cap is replaced by the next in hash order; if none fits, the unit leaves P5'.

**Note: RI / Genomes (fallback PUB) against RI / Labels (METHODS BY DECLARED INPUT, fuse); bar criterion (a)**

Under PUB after failed development gates no development unit has a PC class. Patch 4 chooses the fusion candidate on 'the development units with a PC class ... with gated labels', and patch 1 takes the transported thresholds from the same units. Both are then undefined, although F1 must hold them. Bar (a) is also undefined under PUB, because P1 and P2 carry no verdict.

*Correction.* Add to the fuse rule: 'If no development unit has a PC class (PUB after failed gates), the candidate is the one with the highest mean, over the 3 development units, of W1 mean recall over 0-5% on the published labels (island against single-copy backbone genes), same tie rule; the transported thresholds then come from the single-copy published backbone genes of the development chromosomes with at least 300 of them.' Add to bar (a): 'Under PUB (a) cannot be judged on Test A; the claim reads "not shown on gated labels" until (b) is met.'

**Note: Decision rules (THE BAR, a); RI / Labels (AMENDMENT 7, fuse_nb)**

Lets a future model clear the block-level bar while being worse than plain composition. Patch 3 fixes fuse_nb as the window of a fusion whose candidate is chosen on gene-level W1 R@1%. That candidate can be worse than codon_nb at block level (development: codon_nb finds 0.89 of regions). Bar (a) asks for P2 'against fuse_nb' only, while the gene-level context track asks for all nine context baselines.

*Correction.* Bar (a): 'P2 against each of fuse_nb, fuse_triv_nb, codon_nb and kmer_nb (intersection-union: smallest mean difference, every interval excluding 0, point estimate at least 0.05).'

**Note: Decision rules (WHEN A DIFFERENCE COUNTS, the SCOPE sentence) against HOW COMPONENTS ARE COMBINED (2), unpatched**

The patch makes differences on P3c, P4 and P5 able to count (before, none could). The unpatched rule then says 'A dominates B if A is better (counts rule) on at least one [of the five primary endpoints] and B on none'. A win on spliced genes alone now removes B from the non-dominated set, although the same paragraph says constructed endpoints 'never create a ranking' and SP says splices 'never show superiority'.

*Correction.* In (2): 'A dominates B if A is better (counts rule) on P1 or P2 and B is better on none of the five primary endpoints. A difference on P3c, P4 or P5 alone never puts a method ahead; it adds the qualifier "except ..." or cancels a domination.'

**Note: RI / Genomes (second new bullet: chromosomes without a PC class; eligibility of P3-P5)**

Under PUB every unit of P3-P5 takes its 1% threshold from published backbone genes. The protocol retired that class as saturated for similarity inputs, with 3 genes at the 1% point. A threshold from it is lenient for similarity-reading methods, so the cross-class contrasts of P4 (fuse minus codon) and P5 (wide minus codon) are tilted toward similarity and still carry verdicts. Outside PUB the same bullet mixes backbone-threshold and PCn-threshold units inside one endpoint, which the patch's own note rejects.

*Correction.* Add: 'In a unit with backbone thresholds the cross-class contrasts of the constructed endpoints (P4 fuse minus codon; P5 wide minus codon) are descriptive. Same-class contrasts (wide minus the linear rules; P3) keep their verdicts under the mark "backbone thresholds". Outside PUB a Test A unit with backbone thresholds is shown and enters no mean or test of P3-P5.'

**Note: RI / Metrics (block track); Primary endpoints (P2)**

'A unit with fewer than 3 scored genes among those is unscored' can be read two ways: scored by the method itself, or having a W11 value (amendment 7 gives an unscored gene the mean of its scored neighbours, so nearly every gene has one). The choice changes P2 for every method that abstains: under the first reading a region with 3 gated genes is 'missed' for wide as soon as one has no hit, whatever the other two score. The patch also does not say whether unscored native blocks count toward 'allowed'.

*Correction.* Add: '"Scored" here means scored by the method itself (it has a W1 value). Unscored native blocks stay in the count that sets "allowed". Beside every LR@2% the number of gated island regions and of native blocks that are unscored for that method is printed, and for each P2 contrast LR@2% is also printed over the regions that both methods score (descriptive).'

**Note: Decision rules (WHEN A DIFFERENCE COUNTS, condition 7) against UNIT AND INFERENCE (unpatched sentence on set contrasts); Primary endpoints (P3b)**

Two rules for one test. UNIT AND INFERENCE still says a contrast against all trivial scores is an intersection-union test with 'p = the largest p over the set'. Patch 14 judges the trivial set member by member with a t-interval, no p and no Holm. P3b says 'above the same statistic of every trivial score' and names no rule.

*Correction.* UNIT AND INFERENCE: 'A contrast against the four linear rules or the nine context baselines is an intersection-union test (...). The trivial set is judged member by member as in condition (7); its largest sign-flip p is printed, not required.' P3b: 'above = the member-wise rule of condition (7), in both orientations'.

**Note: RI / Labels (METHODS BY DECLARED INPUT, ai_gene)**

ai_gene has no home group for a gene whose only visible covering hits lie outside its phylum. Such genes are the far-donor positives, so the evaluator would have to invent their value after seeing data. ai_gene is a member of the linear rules in P1, P3c, P4 and P5.

*Correction.* Add: 'If the gene has no visible covering hit inside its phylum, the home group is the phylum (empty home side, E = 1), as for ai_all.'

### Judge 2

Not ready: one blocking problem (P2 eligibility). The build order itself is executable.

**Can every step run?** Yes, with the small corrections below. I read the protocol, the patch set and the code in the repository (bench/hard, bench/islands, research/array, research/placement, seeddrift/pipeline.py, seeddrift/tools.py). I ran nothing on the server and measured nothing.

- Every first-freeze step has its inputs from an earlier step or from data that exist. One library run is enough: the Brucella replacement plus the full-list genes.

- By F2 scaling that run takes roughly 1.5-5 h (46-184 million alignments), so "3-4 h" is fair. Its output is first needed on day 1 afternoon.

- Machine time elsewhere is light. `build.py columns` is one pass over a hit file. wide scores a 4,000-gene chromosome in about 63 tables per grouping and aligns nothing. Panel labels take minutes per chromosome.

- The tight resource is the engineer: 32 hours with no slack, and panel.py and panel_labels.py are not among the tested code of F3.

- Gaps: no step computes the HEG stratum (M6); gate 1 has no aggregation rule; the fusion candidate is undefined if the gates fail; the V0 control of M1 and M3 is not in the first freeze.

**Are P1 and P2 computable, with enough genera?** Both are computable in every branch. Whether they carry verdicts is not known before day 2.

- **P1 (floor 9):** by F6, Rickettsia is out. Eight genera have at least 76 island genes and are not count-flagged. The ninth must come from Bacillus, Staphylococcus, Prochlorococcus (they need 66-77% of their island genes gated) or from a count-flagged genus. If n_P1 is below 9, the patched fallback PUB gives no verdict at all.

- **P2 (floor 8):** probably below its floor. Three genera have fewer than 5 published island regions, and only four genera are clear of every count-based hurdle. The rule of 50 native blocks needs at least 450 PCn natives packed 9 per 11 genes, a density I expect in tight genera only (my estimate; no panel has been built). Development is Escherichia, Shigella and Yersinia and will not show this.

The P2 rule can be changed from label files today and cannot be changed after the freeze. The owner should either adopt the ladder in the first problem or accept in writing that the first result is at gene level only.

**Blocking: RI / Genomes (ELIGIBILITY, P2 bullet; Floors); Primary endpoints (P2); stage1_scope A ('Primary endpoints with verdicts')**

P2 will probably miss its floor of 8 genera, and no checker counted it. Facts from bench/islands/results/genomes.tsv and F6: (a) 'at least 5 gated island regions' is impossible for Haemophilus (4 published island regions), Staphylococcus (4) and Rickettsia (2). That leaves at most 15 genera, and at most 11 of the 13 Test A genera under the default Reserve B. (b) Six more genera have only 6-9 published island regions (Prochlorococcus 6, Acinetobacter 7, Bacillus 7, Sinorhizobium 8, Streptococcus 8, Corynebacterium 9) and need 5 of them gated with at least 3 gated genes each. Five others are count-flagged (Bradyrhizobium, Brucella, Rhodopseudomonas, Shewanella, Vibrio), so their c condition is uncertain. Only Burkholderia, Mycobacterium, Pseudomonas and Rhizobium are clear of every count-based hurdle. (c) 50 native blocks need at least 450 PCn natives packed 9 per 11 consecutive genes, while P1 asks for 500 PCn natives anywhere. With independent placement a 4,000-gene chromosome gives about 52 blocks if 45% of its genes are PCn, about 22 at 40% and about 2 at 30%. Clustering of core genes lowers the needed share somewhat. I expect that share in tight genera and not in species-rich genera whose 5 panel species are picked by hash (Pseudomonas, Bacillus, Streptococcus, Shewanella); this is an estimate, nothing was measured. Development is Escherichia, Shigella and Yersinia, so day 1 will not show the problem. It appears on day 2, after PROTOCOL.md is committed.

*Correction.* ELIGIBILITY, P2 (replaces the P2 bullet): 'the panel and c conditions; at least 5 gated island regions; native blocks by a two-step ladder, evaluated from label files only, on day 2, after the split and before any score is joined. Step 1: a native block has at least 9 PCn genes among its 11; a genus needs at least 50 blocks. If fewer than 8 Test A genera are P2-eligible under step 1, step 2 is used for ALL genera: a native block has at least 7 PCn genes among its 11 (still no gene island-labelled or within 5 genes of an island-labelled gene; still only the PCn genes enter the block score; same tiling), and a genus needs at least 50 blocks. The step in force is written into F1; under step 2 every P2 figure carries the mark relaxed blocks. If step 2 also gives fewer than 8 genera, P2 is reported without verdicts under step 1 and every P1 verdict reads at gene level only. Day 1 prints the number of native blocks under both steps for the 24 development chromosomes. Day 2 prints, per sealed genus, published island regions, gated island regions, PCn natives and native blocks under both steps. Known today: Haemophilus, Staphylococcus and Rickettsia cannot be P2-eligible.' GP NATIVE BLOCK and the P2 endpoint refer to this ladder. If the owner keeps the strict rule instead, stage1_scope A must say: 'P2 is expected to fall below its floor; the first report is then at gene level only.'

**Note: stage1_scope A ('Primary endpoints with verdicts'); RI / Genomes (Floors bullet)**

P1 verdicts are not assured, and the scope reads as if they were. By F6: Rickettsia (18 island genes) can never reach 40 gated island genes. Staphylococcus (52), Prochlorococcus (55) and Bacillus (61) need 77%, 73% and 66% of their island genes to pass the gate. Six genera are count-flagged (Bradyrhizobium, Brucella, Haemophilus, Rhodopseudomonas, Shewanella, Vibrio). Only 8 genera have at least 76 island genes and no flag (Acinetobacter, Burkholderia, Corynebacterium, Mycobacterium, Pseudomonas, Rhizobium, Sinorhizobium, Streptococcus). Each of them must still pass the panel checks; Streptococcus has 2,001 genes in all and needs 500 PCn natives. So n_P1 >= 9 needs all 8 plus at least one doubtful genus. Below 9, the patched fallback PUB gives no verdict, so the first freeze can end without any verdict. Every branch is defined and computable, so this does not block.

*Correction.* stage1_scope A, replacing 'Primary endpoints with verdicts': 'Primary endpoints: P1 and P2. Verdicts depend on the day-2 eligibility tables, which do not exist today: P1 needs n_P1 >= 9, P2 needs 8 P2-eligible Test A genera. From the published counts, 8 genera have at least 76 island genes and are not count-flagged; the ninth must come from Bacillus, Staphylococcus, Prochlorococcus or a count-flagged genus. If n_P1 < 9 or the development gates fail, the first report contains no verdict (fallback PUB: 9 genera on published labels, descriptive). If only P2 is below its floor, P1 verdicts read at gene level only. The day-2 table prints, per genus, every eligibility count and the reason for each exclusion.'

**Note: Build order (Day 1 and Day 2 bullets); stage1_scope B steps 9 and 15**

No step produces the HEG stratum. The scope promises matrix row M6 on Test A, and decision rule (3)(iii) reads the HEG call rate of codon and kmer on development. research/placement/markers.py --strict exists and needs pyhmmer and Pfam-A.hmm, which the pipeline's data store holds. No day runs it for the 24 development or the Test A chromosomes.

*Correction.* Add to Day 1, inside the 2 h of the scoring step: 'research/placement/markers.py --strict on the protein FASTA of the 24 development chromosomes (no label read), output heg/<accession>.txt.' Add to the night-2 Test A scoring: 'the same for the Test A chromosomes.' Add to F1: 'the HEG lists'. Add: 'If pyhmmer or Pfam-A.hmm is missing on the server, the HEG stratum is reported as not measured, M6 moves to stage 1b, and rule (3)(iii) is evaluated on FAST, PCn natives under 100 aa and PCn natives without a hit.'

**Note: GP / Labels (ACCEPTANCE GATES), patch 10**

Step 7 ('gates decided') cannot be decided as worded. Gate 1 has no aggregation rule (per chromosome, per unit or pooled). Gate 2 mixes a unit median with two classes of chromosomes. The gates decide whether P1 and P2 fall back to PUB, so the rule must be exact before day 0.

*Correction.* 'Gate 1: per development chromosome, the share of its single-copy published backbone genes with f >= 0.8; the median over the chromosomes of each unit is at least 80%, in both units. Gate 2: per development chromosome, c = island-labelled genes with f >= 0.8 / island-labelled genes. In each unit, the median c over its chromosomes with more published backbone than island genes is at most 5%, and the median c over its other chromosomes, if it has any, is at most 20%. Both gates use all 19 chromosomes of the two units (Escherichia+Shigella 14, Yersinia 5), not only the searched ones.'

**Note: RI / Labels (METHODS BY DECLARED INPUT, fuse), patch 4; RI / Metrics (Transport); RI / Genomes (fallback PUB), patch 1**

If the development gates fail, the PC class is withdrawn. Then there is no 'development unit with a PC class' and there are no gated labels. The fusion candidate (and with it fuse, fuse_nb, fuse_triv) and the transported thresholds have no definition, yet fusion scores are 'never dropped' and are printed under PUB. Step 13 could not run in that branch.

*Correction.* Add to the fuse paragraph and to Transport: 'If the PC class is withdrawn on development (gates failed), the fusion candidate and the transported thresholds are fixed with published labels: published island genes against single-copy published backbone genes of the 4 searched development chromosomes (3 units), same metric, same aggregation, same tie-break. Both then carry the mark published labels, not gated.'

**Note: stage1_scope A ('Matrix rows measured on Test A: M1 to M7 and M15'); RI Controls 1; matrix rows M1 and M3**

V0 is not in the first freeze (it needs a species column in the library table; the patch moves it to stage 1b). Rows M1 and M3 name V0 as their matched control and control 1 says 'V0 is printed'. As written the scope claims two rows as fully measured that cannot be.

*Correction.* 'M1 and M3 are measured at U1 without their V0 control. The V0 column of both rows and control 1 are stage 1b. The coverage table marks them control not yet measured.'

**Note: Cost field of RI (rule pipeline), patch 7; Build order nights 0-2, patch 19**

'About 1 h' per rule-pipeline run is the lower end. In seeddrift/pipeline.py one run is a sensitive DIAMOND pass over the whole library (30-40 min before the first protein by F2), then a deep search (MMseqs2-GPU on the shared GPU when configured, otherwise DIAMOND ultra-sensitive over the library), then Pfam annotation and trees. On night 2 the sealed pipeline runs compete with the Test A scoring for cores and GPU. The text also does not say which 4 development chromosomes are run.

*Correction.* 'Rule pipeline: about 1-2 h per chromosome. The 4 development runs are the 4 development chromosomes with a deep search, one at a time after the library run. On night 2 the Test A scoring runs first; pipeline runs start only after it has finished. The sealed runs counted at U1 are those finished before F1, at most 6.'

**Note: Build order Day 0 (the one library run), patch 19; RI Cost, patch 7**

Two details decide whether the single run is complete; a second run costs at least 30-40 min plus its queries. (1) bench/hard/views.py sets full = 1 from the hits that map to library taxa; a full list with any hit to an unknown taxon is not flagged. (2) F2 gives a duration for the deep search of the 4 development chromosomes while the protocol still calls them 'queued'; the scope assumes stored lists exist.

*Correction.* Add: 'A list is full when the stored file holds at least 1,000 lines for the gene, counted before any taxon filter (the full column of views.py is not used for this selection). If a stored top-1,000 list is missing for any of the 4 development chromosomes, all of its proteins go into the same run. Expected duration by F2 scaling: 1.5-5 h. The output is first needed on day 1 afternoon.'

**Note: Build order (Day 0-2 hours; first-freeze totals), patch 19; GP Cost, patch 9**

The 32 hours have no slack. The 'at most half a day of fixes' for the GP gates is not in them. panel.py and panel_labels.py are untracked in git and are not among the tested code of F3; their first real runs are budgeted at 0.5 h and inside step 7. panel_labels.py parses up to 14 EMBL records and runs two DIAMOND searches per chromosome, with no cache, so every rerun searches again. I estimate roughly 3-4 min per chromosome, about 2-3 h for the 42 chromosomes.

*Correction.* Add to the build order: 'Days are working days. U1 follows F1 whenever F1 is complete; the half day of gate fixes and any debugging of the first real panel runs move U1 as a whole. panel_labels.py keeps its panel and self hits on disk and does not search again on a rerun. The 24 development chromosomes are labelled during day 1 (about 1.5 h of machine time, in parallel with the evaluator work). The 18 sealed chromosomes are labelled in night 1 with the code tagged after the development gates have passed.' GP Cost: replace 'minutes' by 'minutes per chromosome, about 2-3 h for all 42'.

**Note: Build order Day 0 (commit bullet), patch 19; stage1_scope B step 1**

The commit list does not match the repository. bench/islands/evaluate.py is already in commit b79ae8c. Uncommitted today: bench/hard/evaluate.py, metrics.py, views.py (modified), bench/hard/panel.py and panel_labels.py (untracked), and research/array/wide.py, ceiling.py, fit.py, report.py (modified). wide.py holds the genome scorer of a method under test and is not in the list.

*Correction.* 'Commit, as they are: the five files of bench/hard and the modified research/array/wide.py, ceiling.py, fit.py and report.py (bench/islands/evaluate.py is already in commit b79ae8c). Then PROTOCOL.md, the dated commit superseding the README, checksums, EXCLUDE.tsv.'

**Note: GP / Labels (ACCEPTANCE GATES, gate 3), patch 10**

The patch says no coordinate table of the literature islands exists and makes gate 3 optional. bench/islands/fetch.py already extracts SupTable3.txt from the island-set download into the data folder, and read_regions in bench/islands/labels.py reads its Lit_ID column. I did not open that table (it is on the server), so its content needs one check.

*Correction.* 'Gate 3 uses SupTable3.txt of the island-set download (the table with a Lit_ID column, already extracted by bench/islands/fetch.py), read with read_regions of bench/islands/labels.py. It is computed on day 1 if the table holds the Sakai and CFT073 islands; otherwise it is printed as not computed. Gate 3 does not trigger fallback PUB.'

**Note: Build order Day 0 (wide adapters), patch 19**

'3 groupings', 'SD of wide over groupings' and the drop item 'two of the three wide groupings under V2' need one output per grouping. score_genomes in research/array/wide.py builds all repeats in one dataset with the fixed seed 7 and writes only their mean.

*Correction.* Add to the wide adapters: 'a grouping-seed argument; one output column per grouping (wide_g1, wide_g2, wide_g3) and their mean; wide = the mean; the same for wide_rule_ai.'

**Note: Build order, 'replaces' field of patch 19**

The patch replaces 'the nine bullets' from 'Day 0 (half a day ...' through 'Stage 1b (days 6-8 ...'. The protocol has eleven bullets in that range (lines 595-605 of final.md: four Day 0 bullets, Day 1 to Day 5, the drop order, Stage 1b). An editor who counts nine would leave two old bullets in place.

*Correction.* 'replaces: the eleven bullets from "- Day 0 (half a day, before any score meets a label)" through "- Stage 1b (days 6-8, fixed order; Test A figures exploratory)", lines 595-605 of final.md.'

