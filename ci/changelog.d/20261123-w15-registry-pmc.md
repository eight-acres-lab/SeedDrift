# W15: the event registry after the fourth server run (since CI 0.5)

- From the fourth server run (20261122T185535: 111 of 111 lookups, 28 full texts, 3 of 8 supplements; Europe PMC
  still answered 34 full-text requests with 500 after the retries): the psbA claim set is regraded A on the supported
  trees of its full text, and the crucivirus record keeps C (the junction window is in a figure and the GenBank
  features, not in the text). A supplement over the size cap now records its error type (`TooLarge`).
- Full texts and supplements of open-access articles now come first from PMC's open-access dataset on AWS Open Data
  (`pmc-oa-opendata`: the latest version's article XML, text and metadata, and its supplementary files without the
  figures), the sanctioned route for automated retrieval; Europe PMC is the fallback. Articles outside the open
  subset are not retrieved automatically (PMC's terms bar automated bulk download from its own site).
- Batch 3 (`bench/rel/registry/events_v0.2-batch3.tsv`): the psbA claim set split by the acquisitions its full text
  names (four of psbA, two of psbD, Sullivan 2006); EV011 becomes the Synechococcus-myovirus psbA event and EV031 to
  EV035 the others, with radii from the NCBI taxdump of the server runs (Kyanoviridae; Sechaudvirinae for the
  podoviruses). The profile reads the three drafts and one more supplement.
