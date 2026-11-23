# W6: the report reads over-fitting group by group

- `rel_w6_report.passes`: beside the curve of the mean over the groups, a table by group: the node of the group's lowest
  held-out bits (scattered; tables and passes), the change to the next node and to the last, the group's own seeds'
  difference (the largest spread of the seeds in that group at the last node they all reached) and the reading; a group
  that rose beyond it is a key point. The programme lead's ruling for round 1: a group that over-fits first is hidden in
  the mean.
- A group's threshold is at least 0.02 bits (one pair of seeds can differ by 0 in a group by chance).
- The trainer keeps every node's bits per held-out table and condition (`node_cells.npz`: the summed bits and the scored
  cells of every table of the nodes' subset), for paired intervals between two nodes of an arm on the same tables.
- A group's change from its lowest node to the last is read with the evaluator's paired bootstrap
  (`evalh/intervals.paired_bootstrap`: split units inside the group, 2,000 replicates, seed 20261123, 95%) on the
  trainer's `node_cells.npz`; the units are the examined genomes' split units (`bench/rel/implant.split_units` on the
  training-data index of `rel_w6_report.units_source`). A group over-fits (a key point) only beyond its threshold and with
  its interval above 0; either alone is listed only.
