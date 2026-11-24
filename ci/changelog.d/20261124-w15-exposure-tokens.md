Package W15: the exposure scan's tokens by the reviewers' rules. Latin names: every `Genus epithet` pair
(`[A-Z][a-z]+ [a-z]{3,}`) in every clause of the recipient column, inside brackets too, stop words aside;
`Candidatus` / `Ca.` a prefix, never a token of its own; species-level names without an epithet (`Pseudomonas sp.`,
`Chaetothyriales sp. CBS 135597`) give nothing, never their genus or order; a lone word only for a name NCBI ranks
genus; every NCBI rank below a genus (subgenus, species group, species, varietas, isolate, ...) gives its binomials.
Nucleotide accessions are tokens too, matched as `register.scan_names` does (with or without version, bounded by
non-alphanumerics): those the record cites in any column (EV012's JN900499) and, with `exposure_islands` (the island
store), the island-set chromosomes inside the radius from `bench/hard/results/v2_reference_set.tsv` and the store's
taxid cache, leaving out the sealed chromosomes and the sealing candidates unused (the V2 register keeps their
exposure). The report gives the reference set's and the cache's sha256, the rows used and left out.
