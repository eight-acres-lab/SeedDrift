Package W15: registry batch 5 (`bench/rel/registry/events_v0.2-batch5.tsv`, 30 records), revised after review, and the
fifth server run's checks (run 20261123T163435: 138 DOIs looked up, 55 full texts, 54 of them from PMC's open-access
dataset, eight of nine supplements; train and held-out counts; no sealed genus).

- Taxids, corpus counts and sealed-genus overlap from that run are written into the 45 records it checked. Records
  whose radius that run did not check carry no taxid, and `?` for both (EV040, EV068, EV070 to EV072, EV074).
- Claim sets split from files that run stored, each record citing the file with its full sha256 and the table row or
  text section (rules 0.2 section 1, 2026-11-23: a split claim set's ID passes to the first event):
  - Dunning 2019 by fragment (Dataset S1A; EV006 becomes fragment A, EV046 to EV067 the others). Grade B, not A: the
    integration statement is general and SI Fig. S7 was not read per fragment. The donor grade of EV006 goes from B (as
    the claim set) to C: each fragment's donor group is only in Fig. 4D, and the 97.2% over 45.7 kb of fragment A does
    not name the relative in the text. Main-text facts that cite fragments by letter are held as `?` (notes keep the
    words) until the ASEM_C4 identifiers of the dataset are reconciled with the main text's and the assembly's; PCK is
    noted on fragments J and L. Population: Dataset S1B transcribed as it stands into `dunning2019_s1b.tsv` with the
    committed `xlsx_sheet.py`.
  - Bratke 2008 by gene (Table 3 and the text's sections): the two glutathione peroxidase captures (EV040, EV068,
    B on a supported tree), the two ribonucleotide reductase small-subunit captures (EV070, EV075), and claim sets for
    interleukin-10 (EV069), thymidine kinase and photolyase (EV071, EV072: D, no tree placing the pox copies among host
    genes) and the 19 single-capture families (EV074). Viral isolate genomes count as position evidence, not as direct
    observation. The LINE-1 capture paper is kept on every split record.
  - Christin 2012 is registered apart (EV073): what was read does not tie its transfers to Dunning's fragments.
- `registry.py`: source clusters (records joined by a shared DOI), per-cell events, sources and lineages with the
  largest share and a flag above 30% (rules 0.2 section 8a), a table counting each source once, claim sets apart, and
  the largest group per label. `large_insert` means three or more genes on one fragment.
- The stage tries Europe PMC for supplements when the open-access dataset holds none, and records that.
