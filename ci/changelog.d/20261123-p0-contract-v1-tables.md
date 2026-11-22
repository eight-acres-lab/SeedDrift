# P0: contract additions for tables of version 1 (API stays 1)

- `SHARD_FIELDS_ADDED` (`genome_of_row`: the genome of each row inside a table, usable by a model only as equality
  between rows) and `META_FIELDS_ADDED` (`row_gene`, `row_first`: where each row's segment lies in its own genome) for
  the tables built from the uncapped search. Kept apart from `SHARD_FIELDS` and `META_FIELDS`, so code written for
  tables of version 0 is unchanged; their defaults for version 0 are stated.
- From table version 1, `KIND_ABSENT` means searched without a homolog (an input only until round 1 decides) and
  `KIND_SELF_COPY` marks rows of the examined genome's own other copies (at most two per table).
- `MAX_ROWS_PER_GENOME` = 2, `MAX_SELF_COPY_ROWS` = 2, and the provisional distance tiers `TIER_EDGES`, for sampling and
  reporting only.
