# W5 baseline result verification

The baseline gate recomputes the full and subset TSVs from archived scores and the pinned held-out shards. It verifies counts and losses for each emitted group, stratum and component, requires the configured baselines and checks each row belongs in its file. The joint switching baseline supplies only total scores.

Independent review reproduced a cross-bin swap that preserved both the overall and group means. A separate check reproduced a missing or non-finite amino-acid array and a subset row moved into the full file. Checksums were updated with each edit. The regression suite exercises these cases and a clean control through the normal gate. This verifies the published aggregates. Scores permuted between positions in the same table can preserve all of them.
