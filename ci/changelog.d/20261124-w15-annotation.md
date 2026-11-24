Package W15: stage `w15_registry` item 5, the annotation lookup. With `annotation: {wgs_master, max_mb}` the stage links
the WGS master record to its assembly (E-utilities), takes the assembly's annotation report from NCBI Datasets page
by page (gene records only, never sequence; each page stored with its sha256) and reports which ASEM gene identifiers
the drafts cite are locus tags, the locus-tag prefixes, and the unmatched ones. A failed lookup fails the gate.
Profile `rel-dev-w15-annot-i1` (fetch off); request `configs/rel/requests/20261124-w15-annotation.yaml`.
- Run 20261123T222453 (rel-dev-w15-annot-i1): the WGS record links to assembly GCA_004135705.1 (ASEM_AUS1_V1.0), which
  has no annotation at NCBI (an empty annotation report; no annotation property). None of the 60 ASEM identifiers the
  drafts cite can be checked against NCBI locus tags; the fragment records say so, and the letter-cited facts stay `?`.
