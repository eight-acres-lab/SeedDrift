# W3: the offline stage takes real tables only (and synthetic ones for tests)

- Stage rel_w3_offline refuses a ruler set, and a profile that asks for one: W18's implant tables are rendered by stage
  rel_w18_offline alone, with the masks of W18's passes (hashed on the triple), which this stage's plans would not
  reproduce (the programme lead: one rendering path per kind of table). The copy of a source's other files (index,
  counts, census) stays: tables of version 1 have them.
