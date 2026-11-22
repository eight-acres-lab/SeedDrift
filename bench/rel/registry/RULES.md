# Event registry: fields and grading rules (rules version 0.2, draft)

Status: draft. Nothing here has been used with any score; no event's role (pool, sealed, prospective) is assigned. The
numbers in this file (probability intervals, the minimum radius) are starting points. They change only by a new, dated
rules version, written before any score meets any label.

Changes from version 0.1 (2026-11-22, after review of the 20-record sample):
- A and B need evidence of foreign origin (section 4). **Provisional, pending the user's confirmation**: it changes
  the grade definitions of charter section 3.
- In replacement scenarios a statistical recombination or breakpoint test can stand in for the supported phylogeny
  of grade B (section 4). **Provisional, pending the user's confirmation**, for the same reason.
- FISH and genetic mapping count as position evidence for insertions into eukaryotic nuclear genomes (section 3).
- Position evidence for a virus chimera is a physically confirmed molecule spanning the junction, plus the junction
  placed within a stated window (section 3).
- Discovery has three routes: S, P, N (section 2).
- The exclusion radius of a virus without an ICTV taxon (section 7).
- Grades use all documented evidence, radii use all papers on the event, status uses independence (sections 4, 5, 7).
- Independence is defined (section 5).
- Disputed, overturned and grade-D records, and claim sets, do not take part in grouping (section 7).
- Proposals for review with the next double registration: how to split a paper into events (section 1), the
  position rule of ancient transfers (section 3), and how the truth class follows from the status (section 10).

## 1. What is registered

- One record is one claimed event, not one gene. A transfer that was later amplified into paralogues is one event.
- Every fact is copied from the source, never from memory. Grades are given on the evidence that was read. When the
  full text could not be read, the record says so (`full_text`), and its grade is the grade of what was read. That is
  not a judgment on the claim; the grade moves when the full text is read.
- Inclusion never depends on any model output, on any score, or on whether the recipient's genomes are in a training
  corpus. Records are compiled without looking at any model output.
- Included: a peer-reviewed claim that a named recipient lineage acquired a gene or a DNA segment from another lineage,
  at any evidence grade, in any direction, any kingdom. Claims that were later disputed, overturned or retracted are
  included too: they are the hard negatives of the catalogue.
- **Splitting a paper into events (proposal, in use for this version, for review):**
  - Each acquisition the source infers is one event. Separate acquisitions are recognised by any of: different gene
    families from different donor clades; different insertion sites; separate trees that place the recipient copies in
    different donor clades; the source stating separate events.
  - Each recipient lineage is a separate event. A paper on several species gives one record per recipient lineage per
    acquisition, unless the source places one acquisition in their common ancestor; then it is one event with that
    ancestor as its radius.
  - Paralogues from one acquisition are one event. So are genes moved together on one fragment.
  - A paper whose events are listed only in a supplement stays one `claim_set` record until the list is copied, row by
    row, from the supplement. Supplements are downloaded on the servers only. Claim sets are counted, but they do not
    take part in grouping (section 7) and are never labels.

## 2. Fields

Lists inside a cell are separated by `;`. Unknown values are `?`.

| column | content |
|---|---|
| `event_id` | stable id, never reused |
| `record_kind` | `event` or `claim_set` |
| `claim` | one line: what is claimed |
| `recipient` | recipient species, strain or virus as the source states it |
| `recipient_group` | the corpus group of the recipient (configs/kingdom_map.yml): `bacteria`, `archaea`, `mammals`, `other_vertebrates`, `invertebrates`, `plants_algae`, `fungi`, `protists`, `phage`, `eukaryotic_virus`, `giant_virus`, `plasmid` |
| `region` | genes, locus tags or coordinates as stated, with the assembly they refer to |
| `gene_family` | the gene family or function |
| `donor_claimed` | the donor lineage as claimed, down to the lineage the evidence supports |
| `donor_group` | the corpus group of the donor (same vocabulary as `recipient_group`) |
| `direction` | `prok>prok`, `prok>euk`, `euk>prok`, `euk>euk`, `virus>host`, `host>virus`, `virus>virus` (charter, section 1) |
| `scenario_tags` | complex scenarios of charter section 1 (vocabulary in section 3) |
| `category` | `protein_coding`, or one of the classes the charter reports separately: `organelle_derived`, `eukaryotic_transposon`, `endogenous_viral_element`. A domesticated viral gene with a protein function is `protein_coding` (direction `virus>host`); an endogenous viral element that is not domesticated is `endogenous_viral_element` |
| `claims` | what is claimed, each `y` (claimed), `n` (not claimed, or not applicable, such as `germline` in a bacterium) or `?`: `foreign` (foreign DNA present), `integrated` (in the chromosome or genome), `germline`, `donor` (a donor identity), `timing` |
| `exclusion_taxon` | the exclusion radius (section 7), as `name (rank)` |
| `exclusion_lineage` | lineage of the exclusion taxon from the root, names separated by `;` (checked against NCBI taxonomy on the server) |
| `recipient_exclusion_taxid` | NCBI taxid of the exclusion taxon; empty until checked on the server |
| `exclusion_basis` | why that radius: the statements of the papers it rests on |
| `papers` | DOIs of the papers that make the claim; grouping uses them |
| `year` | year of the first claim |
| `db_date` | date of the databases the claim used, as stated |
| `assembly` | assembly, clone or sequence accessions the claim used |
| `discovery` | `S`: candidates found systematically by similarity or best hits (including a systematic search for a gene type, such as fully coding env genes); `P`: candidates chosen systematically by phylogeny or by function, without a best-hit screen; `N`: found by chance, or a gene first cloned for its function and only afterwards found to be foreign; `?` when the route was not read |
| `ev_direct`, `ev_position`, `ev_population` | each evidence line: `y`, `n` or `?`, then `:` and a few words |
| `ev_phylogeny` | `y`: a phylogeny with reported support (bootstrap, posterior, a topology test); `w`: a phylogeny without reported support, or support not read; `n`; `?`; then `:` and a few words |
| `ev_origin` | evidence of foreign origin other than a phylogeny, as `kind: words` items: `donor_identity` (near-identical to a sequenced donor genome over a long stretch), `recombination_test` (a statistical recombination or breakpoint test, with the test named), `donor_structure` (the donor's own structures, such as proviral LTRs or T-DNA borders; recorded, not yet accepted, section 11); or `n`, `?` |
| `position_rule` | which position evidence applies (section 3): `integration`, `replacement`, `virus_chimera`, `none` |
| `grade_transfer` | A, B, C or D for "a transfer happened" (section 4) |
| `grade_donor` | A, B, C, D for "the donor is who the claim says", or `-` when no donor is claimed |
| `status` | `standing`, `replicated`, `disputed`, `overturned`, `retracted`, `narrowed` (section 5) |
| `p_transfer`, `p_donor` | probability interval that the claim is true, `lo-hi`, from section 6; never 0 or 1 |
| `truth_class` | `transfer`, `contamination` (foreign DNA not integrated: contamination, symbionts, food), `native`, `artefact` (nothing in the assembly to score), `unresolved` (section 10) |
| `recheckable` | evidence a fixed check can re-examine: `reads:<accessions>`, `second_assembly:<accession>`, `population:<what>`, or `none` |
| `later` | later papers: `support:<doi>`, `challenge:<doi>`, `reply:<doi>`, `comment:<doi>`, `erratum:<doi>`, `retraction:<doi>`, each with a few words in brackets; a paper whose DOI was not read is cited by journal, volume and page |
| `full_text` | whether the full text of the claim papers was read: `y`, `partial` (some papers, or part of the text), `n` (abstracts only) |
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
a high foreign fraction), `chimeric_virus`, `contamination_vs_integration` (the claim had to tell integrated DNA from
contamination, symbionts or food), `multicopy`, `intron_gain`, `large_insert` (a segment of many genes, an endosymbiont
genome).

Position evidence is defined by scenario (charter, section 3):

| `position_rule` | scenarios | counts as position evidence |
|---|---|---|
| `integration` | an insertion into a recipient genome, any direction, including a host gene captured by a virus | recipient-native genes flanking the segment on the same contig or clone; insertion junctions; spliceosomal introns acquired in the recipient copy; the same insertion site in several species (orthologous insertion); for a eukaryotic nuclear genome, the segment located on a recipient chromosome by FISH, or inherited with recipient markers in crosses (genetic mapping); for a gene in a virus, its position among viral genes in a genome sequenced from an isolate |
| `replacement` | allelic replacement, recombination between close relatives | recombination breakpoints in an alignment of the locus, or population-genomic evidence at the same locus |
| `virus_chimera` | a virus genome assembled from parts of unrelated viruses | a physically confirmed molecule spanning the junction (PCR across the junction, or a single long read), plus the junction placed within a stated window. A formal statistical breakpoint test is not needed here; it counts as evidence of foreign origin (`ev_origin`, `recombination_test`) |
| `none` | a claim with no locus at all (for example a genome-wide count without gene positions) | grades A and B stay empty; the report says "no scorable positive in this scenario" |

**The position rule of ancient transfers (proposal, in use for this version, for review):** the position rule follows
the kind of genomic event, not its age. An ancient transfer is `integration` (or `replacement`) like any other.
Position evidence for it is then the gene's conserved place among native genes across the recipient clade (an
orthologous position), or acquired introns. Age is the `ancient` tag. `none` is kept for claims without loci.

## 4. Grades

The letters follow charter section 3. Each grade needs these evidence lines:

| grade | needs |
|---|---|
| **A** | experimental or direct observation, plus position evidence, plus evidence of foreign origin |
| **B** | a phylogeny with support, plus position evidence. In a `replacement` scenario, a statistical recombination or breakpoint test (named) can stand in for the phylogeny |
| **C** | a phylogeny only, with or without support, or evidence of foreign origin without position evidence (most ancient and assimilated transfers stop here) |
| **D** | similarity only. Never a positive label |

- "Evidence of foreign origin" (provisional, pending the user's confirmation) means: a phylogeny with support
  (`ev_phylogeny` `y`), or near-identity to a sequenced donor genome (`ev_origin` `donor_identity`), or, in a
  `replacement` or `virus_chimera` scenario, a statistical recombination or breakpoint test (`recombination_test`).
  The reason is the overturned tardigrade claim: junction PCR and linkage without foreign origin gave grade A under
  version 0.1.
- "Experimental or direct observation" means physical evidence independent of the one assembly. Any of these counts:
  - PCR or sequencing across the junctions;
  - FISH or Southern blots on recipient DNA;
  - long reads spanning the junctions;
  - an independent second assembly or clone with the segment;
  - presence in the germline, in eggs, or across individuals of a pedigree or population;
  - a transformation that reproduces the event.

  A functional assay (RNAi, a knockout, heterologous or transgenic expression) shows that the gene works. It counts as
  direct observation only when the gene is also shown to be encoded in the recipient genome.
- "A phylogeny with support": the recipient copy nests inside the donor clade with reported support, and the sampling
  of the donor and of the recipient's relatives is stated.
- **Which evidence counts:** all documented evidence on the event, including later work by other teams (the `papers`
  and `later` columns). Status (section 5) is decided separately, by independence.
- `registry.py` derives the grade from the evidence lines and warns when a record's grade differs.

The donor is graded separately:
- **A**: direct evidence: the segment is near-identical to a sequenced donor genome over a long stretch, or carries the
  donor's own structures, such as T-DNA borders or proviral structure;
- **B**: a phylogeny with support nests the recipient copy inside the claimed donor clade, with dense sampling;
- **C**: a phylogeny without support, or with sparse sampling;
- **D**: best hits only;
- `-`: no donor claimed. A captured viral gene has a donor: write the viral lineage the evidence supports and grade
  it on its own evidence.

## 5. Status

- `standing`: no published challenge, and no independent reproduction.
- `replicated`: a team independent of the claimants reproduced a positive result, and no independent evidence argues
  against it.
- `disputed`: a team independent of the claimants reproduced a positive result, and other independent evidence argues
  against it. Not a positive label; only score distributions are reported.
- `overturned`: only the claimants support it, and independent evidence argues against it.
- `retracted`: an editorial act. It is not a ruling that the claim is false; the record keeps its grade.
- `narrowed`: later work kept a part of the claim (for example the transfer but not the donor, or fewer genes than
  claimed). The part that stands keeps its grade; the part that fell is a separate `overturned` record when the source
  lists it.

Independence:
- A later paper is independent when it shares no author with the claim papers and its corresponding author is in a
  different laboratory. Partial author overlap is not independent.
- A paper that takes the transfer as given and studies the gene's function is not a reproduction of the transfer.

## 6. Probability intervals (starting points; deliberately wide)

The interval is the probability that the claim is true. Its other side is the assumed error rate e = 1 - p. Only the
label review channel changes an interval; nothing else does.

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

- The exclusion radius is the deepest recipient taxon reported as sharing the event, at least the genus, taken over
  all papers on the event: claim papers and later work alike. When presence is reported in several taxa, the radius
  is their lowest common ancestor.
- For a virus without an ICTV taxon: the radius is the lowest named taxon that contains the virus, at least that taxon
  (not the genus). Its taxid is that taxon's taxid in NCBI, `no rank` included; the rank is written as `(no rank)`.
- **Groups** are the units of every split. They are computed by `registry.py`, never hand-written.
  - Two records are in the same group when one's exclusion taxon contains the other's (the radii overlap: in a tree
    two subtrees are either nested or disjoint), when they share a paper in `papers`, or when they share an assembly
    in `assembly`.
  - Gene family and donor lineage are strata, never split units.
  - Disputed, overturned and grade-D records, and claim sets, do not take part in grouping until the recipient
    exclusion rule for them is settled (appendix B.2 of the programme). They are reported as ungrouped.
- Each group's part (pool, sealed or prospective) is fixed by a seeded hash before any score, and changes only by a
  dated rule, never from pool to sealed. The sealed fraction and the seed are not set yet; until they are, every role is
  `unassigned`.
- The registry draft never lives at `bench/catalogue/events.tsv`: gate G2 reads that file and excludes every recipient
  in it from every training corpus.

## 8. Exposure

Exposure levels follow bench/hard/V2_REGISTER.md (L0 named only, up to L5 scores joined with labels) and are recorded
for the recipient genomes inside the exclusion radius, across both the old line and the relational-model line.
`in_pretraining` records separately whether such genomes are in the medium corpus, which is unlabelled pretraining
data. Events whose genomes were in pretraining stay in the registry; results are stratified by both columns.

## 9. Double registration

A random subset of records, drawn by a seeded hash, is registered a second time by someone who has not seen the first
registration, from the sources and these rules alone. The report gives their agreement on `grade_transfer`,
`grade_donor`, `status`, `discovery`, `exclusion_taxon` and `truth_class`, taking the first registration as
compiled, before any reconciliation.

## 10. Truth class (proposal, in use for this version, for review)

| status | truth class |
|---|---|
| `standing`, `replicated`, `narrowed` (the part that stands), `retracted` | `transfer`, at any grade; grade D is still never a positive label |
| `disputed` | `unresolved` |
| `overturned` | the failure kind the refuting papers establish: `contamination` (contaminant, symbiont, food or other DNA not integrated), `native` (gene loss, sampling, a wrong inference), `artefact` (an assembly error: nothing in the assembly to score), or `unresolved` when they reject the claim without establishing an alternative |

## 11. Recorded, not accepted yet

`donor_structure` (proviral LTRs and retroviral genes, T-DNA borders) is recorded in `ev_origin`. It is not accepted
as evidence of foreign origin for grades A and B until the provisional rule of section 4 is settled. The report gives
the grades that would change if it were accepted.
