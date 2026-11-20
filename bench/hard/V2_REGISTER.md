# Benchmark v2: the register of the island reference set (rule, fixed before any register run)

The island reference set (Bertelli et al. 2019, `bench/islands/results/provenance.json`: 104 bacterial chromosomes in
22 genera) is split into parts. Benchmark v2 may seal only chromosomes nobody has looked at. This file fixes, before
any register run, how "looked at" is decided and what follows from it. It is not changed after the first register
run; a later correction is a new, dated section below, and it may only demote (a chromosome moves from candidate to
development, never back).

## Parts

| part | members |
|---|---|
| development | the chromosomes of `bench/islands/results/dev_genomes.tsv` (24) |
| development_unlisted | chromosomes of a development genus (`bench/islands/fetch.py` `DEV_GENERA`, by the organism's first word) not in `dev_genomes.tsv` |
| sealed_v1 | the chromosomes of `bench/hard/results/sealed.tsv` except NC_010167.1 (17) |
| sealed_secondary | NC_010167.1 (Brucella chromosome II; in sealed.tsv but outside every rule of PROTOCOL.md) |
| rest | every other chromosome of the set |

## Exposure levels

| level | meaning |
|---|---|
| L0 | named only: accession, organism, taxid, a fixture string |
| L1 | counted by code for a design decision, the counts never written, printed or read by a person |
| L2 | a count, statistic or replicon fact written, printed or read by a person, or the chromosome entered a design decision |
| L3 | its record, proteins, hits, panel or label files exist |
| L4 | it was scored |
| L5 | its scores were joined with labels (a trial, a figure, a design decision on results) |

## Rule

1. A `rest` chromosome at level L2 or above is `demoted` (to development); below L2 it is a `candidate` for sealing in
   benchmark v2.
2. `development_unlisted` chromosomes are development; they are never candidates.
3. The level of a chromosome is the highest of its declared exposures (`bench/hard/results/v2_exposure_declared.tsv`)
   and of the exposures the register's scans find. An exposure a scan finds and no declared row covers stops the
   register run until it is declared (with its level) in a new commit.
4. The register reads the island tables for accessions and organisms only: no row is counted and no coordinate kept.

## Scans

Register run 1 (machine i-1) scans names only, never file contents: every path under the island store and under the
machine's CI runs is matched against the accessions of the set (with or without version). A path naming a chromosome
is level L3 (its record, proteins, hits, panel or labels exist), L4 under a score folder (`scores/`, `scores_<acc>`,
`raw/`, `views/`, `dev_composition/`, `dev24_v/`, `dev_views/`), L5 under an evaluation folder (`eval/`). The scan
roots, the number of paths walked and every finding are part of the register's report. Later register runs add the
other machine, the git history of every branch and the developer machine's files; they may only demote.

Decided by the user on 2026-11-20 (`fetch.ranked()` counted the published island and backbone regions of every
chromosome of the 18 non-development genera, in memory only, to pick the sealed chromosome of each genus): this
counting is level L1, "not looked at". NC_009504.1 (the set's second Brucella record, also chromosome II) entered a
design decision (the Brucella replacement planned with DESIGN.md amendment (1); its replicon status checked, PROTOCOL.md
section 3): level L2, so it is demoted whatever the scans find.

Every candidate lies in a sealed_v1 genus and ranked below that genus's sealed chromosome under `fetch.ranked()` (a
known selection: fewer labelled regions), so candidates share units with sealed_v1 and are not independent of it.
