# References and dependencies

SeedDrift's own code is MIT-licensed (see `LICENSE`). It ships **no third-party data and no
third-party binaries**: everything below is downloaded or installed on the machine that runs
SeedDrift, from the original source, by `./seeddrift.sh install` and `./seeddrift.sh data`.
Each item keeps its own license. If you publish results, cite the data and tools you used.

## Data (`$SEEDDRIFT_DATA`, described in `data/datasets.json`)

| Dataset | What it is used for | Source | License | Cite |
|---|---|---|---|---|
| `reference-proteomes` | Similarity search for every gene (DIAMOND). UniProt reference proteomes, taxonomically balanced (Gammaproteobacteria 1 per genus; other bacteria, archaea, mammals 1 per family; other eukaryotes 1 per order; viruses 1 per genus) | UniProt (EBI FTP `knowledgebase/reference_proteomes`) | CC BY 4.0 | The UniProt Consortium, *Nucleic Acids Res.* (release used) |
| `ncbi-taxonomy` | Lineages, donor groups and the genome's own taxon | NCBI Taxonomy `taxdump.tar.gz` | Public domain: NCBI places no restrictions on use or distribution | Schoch et al., *Database* 2020 |
| `uniprot-swissprot` | Taxonomy and names of the structures | UniProtKB/Swiss-Prot FASTA (EBI FTP) | CC BY 4.0 | The UniProt Consortium |
| `alphafold-swissprot` | Structure search for genes without sequence homologs (Foldseek) | AlphaFold DB v6, Swiss-Prot set (EBI FTP) | CC BY 4.0 | Jumper et al., *Nature* 2021; Varadi et al., *Nucleic Acids Res.* 2024 |
| `pfam-a` | Repeat-like families for the repeat filter | Pfam-A HMMs and annotation (EBI FTP, InterPro) | CC0 1.0 | Mistry et al., *Nucleic Acids Res.* 2021 |

Demo inputs (`data/demos.json`) are downloaded from Ensembl Bacteria (no restrictions on use) and
UniProt (CC BY 4.0).

## Models

| Model | Used for | Source | Revision | License | Cite |
|---|---|---|---|---|---|
| ESMFold v1 | Structure prediction (GPU) | `facebook/esmfold_v1` on Hugging Face (used as published, not mirrored) | `75a3841ee059df2bf4d56688166c8fb459ddd97a` | MIT | Lin et al., *Science* 2023 |

## Software (installed from conda-forge / bioconda / PyPI; versions pinned in `deploy/envs/*.yml`)

| Tool | Version | Used by | License |
|---|---|---|---|
| DIAMOND | 2.2.8 | tools service | GPL-3.0-or-later |
| Foldseek | 10.941cd33 | tools service | GPL-3.0 |
| COILS (ncoils, Lupas et al. 1991; Russell and Lupas 1999) | 2.2.1 | tools service | GPL |
| pyhmmer (HMMER 3 bindings) | 0.12.3 | tools service | MIT, BSD-3-Clause, BSD-2-Clause |
| pyrodigal (Prodigal bindings) | 3.x | API (gene calling) | GPL-3.0 |
| PyTorch | 2.4.0 (CUDA 12.1) | ESMFold service | BSD-3-Clause |
| transformers | 4.49.0 | ESMFold service | Apache-2.0 |
| Biopython | 1.88 | BLOSUM62 for tandem repeats | Biopython License (BSD-3-Clause-like) |
| NumPy | ≥ 1.26 | everywhere | BSD-3-Clause |
| FastAPI / Uvicorn | ≥ 0.110 / ≥ 0.29 | all services | MIT / BSD-3-Clause |

Used only by the multi-genome pipeline (`seeddrift prepare` / `scan` / `donors`): OrthoFinder 3.2
(GPL-3.0-only), MAFFT 7.526 (BSD-3-Clause), FastTree 2.2.0 (GPL-2.0-or-later), IQ-TREE 3.1.4
(GPL-2.0-or-later), MMseqs2 18-8cc5c (MIT).

SeedDrift calls these programs as separate processes and does not include or modify their code.
Container images built from `deploy/docker/` contain GPL programs: if you redistribute such images,
the GPL terms of those programs apply to them.

## Planned: data mirrors on Hugging Face (not done yet)

The original sources are slow from some networks. The plan is to mirror the datasets above,
unchanged and with their original license and attribution in each dataset card, under
`huggingface.co/datasets/PointEightAI/`, one repository per kind of data rather than per project:
`uniprot-reference-proteomes-balanced`, `uniprot-swissprot`, `ncbi-taxonomy`,
`alphafold-swissprot`, `pfam-a`. ESMFold is referenced from `facebook/esmfold_v1` directly.
`seeddrift data fetch` will then try the mirror first and fall back to the original source.
