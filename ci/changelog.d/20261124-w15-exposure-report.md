Package W15: the exposure scan's report names no corpus genome inside a radius. With the corpus manifest public, the
train and held-out genomes of a radius listed in full would show its test genomes by the gaps (programme lead's
review, 2026-11-24). The tokens from inside a radius (assembly numbers, species of Ensembl genomes, island-set
chromosomes) are kept in memory for matching and only counted in the report (`n_assemblies`, `n_species`,
`n_nucleotide`, `radius_tokens_matched`); no hash of the lists is given, since a small radius's list could be found by
trying the corpus's subsets. Only the record's own tokens from the draft are named. Examples and walk errors pass
through a redactor (`<assembly>`, `<accession>`, `<species>`) and then the last unseen check. Also: the sealed and
candidate island chromosomes are a skip list (counted only); the unseen check also finds species words written
joined (`NicotianaAttenuata`); the per-root digest covers only the paths kept; a record declared `?` with a scan hit
is listed as above its declared level; recipient binomials count only when their first word is an NCBI genus
(`japanese_smelt` is none); WGS accessions need their `0x` version (`SOAP18064315` is none); `candidatus`,
`uncultured` and the like are never tokens. The docstring says what the skip does not cover: test genomes named by
assembly are not skipped by species. Tests pin the last filter, the sorted walk and the redactor.
