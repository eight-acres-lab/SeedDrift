# Event registry: fields and grading rules (rules version 0.1, draft)

Status: draft for review. Nothing here has been used with any score; no event's role (pool, sealed, prospective) is
assigned. The numbers in this file (probability intervals, the minimum radius) are starting points. They change only
by a new, dated rules version, written before any score meets any label.

## 1. What is registered

- One record is one claimed event, not one gene. A transfer that was later amplified into paralogues is one event. A
  paper that claims many transfers and lists them only in a supplement is registered as one `claim_set` record until
  the list is copied, row by row, from the supplement. Supplements are downloaded on the servers only.
- Included: a peer-reviewed claim that a named recipient lineage acquired a gene or a DNA segment from another
  lineage, at any evidence grade, in any direction, any kingdom. Claims that were later disputed, overturned or
  retracted are included as well: they are the hard negatives of the catalogue.
- Inclusion never depends on any model output, on any score, or on whether the recipient's genomes are in a training
  corpus. Records are compiled without looking at any model output.
- Every fact is copied from the source, never from memory. The `verification` column says where each record's facts
  were read: an abstract, the open full text, or the supplement. `memory` marks a field that still has to be checked.

## 2. Fields

Lists inside a cell are separated by `;`. Unknown values are `?`.

| column | content |
|---|---|
| `event_id` | stable id, never reused |
| `record_kind` | `event` or `claim_set` |
| `claim` | one line: what is claimed |
| `recipient` | recipient species, strain or virus as the source states it |
| `recipient_group` | the corpus group of the recipient: `bacteria`, `archaea`, `mammals`, `other_vertebrates`, `invertebrates`, `plants_algae`, `fungi`, `protists`, `phage`, `eukaryotic_virus`, `giant_virus`, `plasmid` |
| `region` | genes, locus tags or coordinates as stated, with the assembly they refer to |
| `gene_family` | the gene family or function |
| `donor_claimed` | the donor lineage as claimed |
| `donor_group` | the corpus group of the donor (same vocabulary as `recipient_group`) |
| `direction` | `prok>prok`, `prok>euk`, `euk>prok`, `euk>euk`, `virus>host`, `host>virus`, `virus>virus` (charter, section 1) |
| `scenario_tags` | complex scenarios of charter section 1 (vocabulary in section 3) |
| `category` | `protein_coding`, or one of the classes the charter reports separately: `organelle_derived`, `eukaryotic_transposon`, `endogenous_viral_element` |
| `claims` | what is claimed, each `y` (claimed), `n` (not claimed, or not applicable, such as `germline` in a bacterium) or `?`: `foreign` (foreign DNA present), `integrated` (in the chromosome or genome), `germline`, `donor` (a donor identity), `timing` |
| `exclusion_taxon` | the exclusion radius: the deepest recipient taxon the source reports as sharing the event, at least the genus; `name (rank)` |
| `exclusion_lineage` | lineage of the exclusion taxon from the root, names separated by `;` (checked against NCBI taxonomy on the server) |
| `recipient_exclusion_taxid` | NCBI taxid of the exclusion taxon; empty until checked on the server |
| `exclusion_basis` | why that radius: the source's statement |
| `papers` | DOIs of the papers that make the claim; grouping uses them |
| `year` | year of the first claim |
| `db_date` | date of the databases the claim used, as stated |
| `assembly` | assembly, clone or sequence accessions the claim used |
| `discovery` | `S`: the candidate list starts from a best hit or an index threshold; `N`: another route (multi-strain comparison, experiment, long reads, synteny, recombination detection) |
| `ev_direct`, `ev_position`, `ev_phylogeny`, `ev_population` | each evidence line: `y`, `n` or `?`, then `:` and a few words |
| `position_rule` | which position evidence applies (section 3): `integration`, `replacement`, `virus_chimera`, `none` |
| `grade_transfer` | A, B, C or D for "a transfer happened" (section 4) |
| `grade_donor` | A, B, C, D for "the donor is who the claim says", or `-` when no donor is claimed |
| `status` | `standing`, `replicated`, `disputed`, `overturned`, `retracted`, `narrowed` (section 5) |
| `p_transfer`, `p_donor` | probability interval that the claim is true, `lo-hi`, from section 6; never 0 or 1 |
| `truth_class` | `transfer`, `contamination` (foreign DNA not integrated: contamination, symbionts, food), `native`, `artefact` (nothing in the assembly to score), `unresolved` |
| `recheckable` | evidence a fixed check can re-examine: `reads:<accessions>`, `second_assembly:<accession>`, `population:<what>`, or `none` |
| `later` | later papers: `support:<doi>`, `challenge:<doi>`, `reply:<doi>`, `comment:<doi>`, `erratum:<doi>`, `retraction:<doi>`, each with a few words in brackets; a paper whose DOI was not read is cited by journal, volume and page |
| `exposure` | exposure level L0-L5 of the recipient genomes in this project (section 8) |
| `in_pretraining` | whether genomes inside the exclusion radius are in the medium corpus: `y`, `n`, or `?` until checked on the server |
| `sealed_overlap` | whether the exclusion radius contains a sealed island genus (bench/hard/EXCLUDE.tsv) or a genome of the sealed test part of the medium corpus: `y`, `n`, `?` |
| `provisional_cover` | whether configs/exclusions/pilot_recipients.tsv already covers the exclusion taxon: `y` (the taxon or an ancestor is listed), `partial` (only some descendants are listed), `n` |
| `role` | `pool`, `sealed` or `prospective`; `unassigned` until the seeded hash of section 7 runs |
| `verification` | where the record's facts were read |
| `notes` | anything else |

## 3. Scenarios and position evidence

Scenario tags: `near` (within a phylum, genus or species), `replacement` (allelic or in-situ replacement), `ancient`
(the transfer predates the recipient clade, or the gene is assimilated), `donor_unsampled`, `partial` (a domain, a
fusion, part of a gene), `serial` (repeated transfers, loss or relocation after transfer), `high_foreign` (a genome with
a high foreign fraction), `chimeric_virus`, `contamination_vs_integration`, `multicopy`, `intron_gain`,
`large_insert` (a segment of many genes, an endosymbiont genome).

Position evidence is defined by scenario (charter, section 3):

| `position_rule` | scenarios | counts as position evidence |
|---|---|---|
| `integration` | an insertion into a recipient genome, any direction, including a host gene captured by a virus | recipient-native genes flanking the segment on the same contig or clone; insertion junctions; spliceosomal introns acquired in the recipient copy; the same insertion site in several species (orthologous insertion); the segment located on a recipient chromosome by FISH, or inherited with recipient markers in crosses (genetic mapping); for a gene in a virus, its position among viral genes in a genome sequenced from an isolate |
| `replacement` | allelic replacement, recombination between close relatives | recombination breakpoints in an alignment of the locus, or population-genomic evidence at the same locus |
| `virus_chimera` | a virus genome assembled from parts of unrelated viruses | breakpoints between the parental segments in a genome of one isolate or one assembled circle |
| `none` | a scenario with no applicable position evidence | grades A and B stay empty; the report says "no scorable positive in this scenario" |

## 4. Grades

The letters follow charter section 3; their definitions do not change.

- **A**: experimental or direct observation, plus position evidence.
- **B**: a phylogeny with support, plus position evidence.
- **C**: a phylogeny only (most ancient and assimilated transfers stop here).
- **D**: similarity only. Never a positive label.

"Experimental or direct observation" means physical evidence independent of the one assembly. Any of these counts:
PCR or sequencing across the junctions; FISH or Southern blots on recipient DNA; long reads spanning the junctions; an
independent second assembly or clone with the segment; presence in the germline, in eggs or across individuals of a
pedigree or population; a transformation that reproduces the event. These are the kinds of physical evidence the
review channel accepts. A functional assay (RNAi, a knockout, heterologous or transgenic expression) shows that the
gene works; it counts as direct observation only when the gene is also shown to be encoded in the recipient genome.

"A phylogeny with support" means the recipient copy nests inside the donor clade with reported support (bootstrap or
posterior), with the sampling of the donor and the recipient's relatives stated.

The grade is of the evidence for "a transfer into this recipient lineage happened". The donor is graded separately:

- **A**: the donor is identified by direct evidence: the segment is near-identical to a sequenced donor genome over a
  long stretch, or carries the donor's own structures, such as T-DNA borders;
- **B**: a phylogeny with support nests the recipient copy inside the claimed donor clade, with dense sampling;
- **C**: a phylogeny without support, or with sparse sampling;
- **D**: best hits only;
- `-`: no donor claimed.

## 5. Status

- `standing`: no published challenge; only the claimants' evidence.
- `replicated`: a team independent of the claimants reproduced a positive result, and no independent evidence argues
  against it.
- `disputed`: a team independent of the claimants reproduced a positive result, and other independent evidence argues
  against it. Not a positive label; only score distributions are reported.
- `overturned`: only the claimants support it, and independent evidence argues against it.
- `retracted`: an editorial act. It is not a ruling that the claim is false; the record keeps its grade.
- `narrowed`: later work kept a part of the claim (for example the transfer but not the donor, or fewer genes than
  claimed). The part that stands keeps its grade; the part that fell is a separate `overturned` record when the source
  lists it.

## 6. Probability intervals (starting points; deliberately wide)

The interval is the probability that the claim is true. Its other side is the assumed error rate e = 1 - p. Only blind
review narrows an interval; nothing else does.

| grade or status | p (transfer) | p (donor) |
|---|---|---|
| A, standing or replicated | 0.85-0.99 | 0.80-0.99 |
| B, standing or replicated | 0.70-0.97 | 0.60-0.95 |
| C, standing or replicated | 0.40-0.90 | 0.30-0.85 |
| D | 0.10-0.70 | 0.05-0.60 |
| disputed, any grade | 0.20-0.80 | 0.10-0.70 |
| overturned, any grade | 0.01-0.30 | 0.01-0.30 |
| retracted | the grade's lower bound minus 0.30 (at least 0.05) to the grade's upper bound | same rule |

`narrowed` records take the interval of their grade for the part that stands.

## 7. Exclusion radius, groups and the three parts

- The exclusion radius is the deepest recipient taxon the source reports as sharing the event, at least the genus.
  When the source reports presence in several taxa, the radius is their lowest common ancestor.
- Groups are the units of every split: two records are in the same group when one's exclusion taxon contains the
  other's (the radii overlap; in a tree two subtrees are either nested or disjoint), when they share a paper in
  `papers`, or when they share an assembly in `assembly`. Gene family and donor lineage are strata, never split units.
  `registry.py` computes the groups; the column is not hand-written.
- Each group's part (pool, sealed or prospective) is fixed by a seeded hash before any score, and changes only by a
  dated rule, never from pool to sealed. The sealed fraction and the seed are not set yet; until they are, every role is
  `unassigned`.
- The registry draft never lives at `bench/catalogue/events.tsv`: gate G2 reads that file and excludes every recipient
  in it from every training corpus.

## 8. Exposure

Exposure levels follow bench/hard/V2_REGISTER.md (L0 named only, up to L5 scores joined with labels) and are recorded
for the recipient genomes inside the exclusion radius, across both the old line and the relational-model line.
`in_pretraining` records separately whether such genomes are in the medium corpus, which is unlabelled pretraining
data. Results are stratified by both.

## 9. Double registration

A random subset of records, drawn by group, is registered a second time by someone who has not seen the first
registration, from the sources and these rules alone. The report gives their agreement on `grade_transfer`,
`grade_donor`, `status`, `discovery`, `exclusion_taxon` and `truth_class`.
