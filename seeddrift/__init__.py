"""seeddrift: genome-wide horizontal gene transfer scan (prototype).

Evidence combined per candidate transfer:
  * gene tree vs species tree (DTL reconciliation over a sample of gene trees)
  * synteny (recipient-side insertion, donor-side co-transfer, neighbour co-transfer)
  * mobile elements near the recipient copy
  * recency (gene-tree distance vs genome background)
  * eukaryote contamination check (foreign gene on a contig with no native genes)
"""

__version__ = "0.1.0"
