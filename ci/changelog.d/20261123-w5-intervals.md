# W5: the evaluator's paired bootstrap over split units

- `seeddrift/train/rel/evalh/intervals.py`: paired_bootstrap(diff_sum, cells, group, unit) for two readings of the
  same cells (two arms, or one arm at two nodes): the evaluator's aggregate of the difference (per group over its cells,
  then the mean over groups; per block too) and its percentile interval over replicates that draw every group's split
  units with replacement (2,000 replicates, seed 20261123, protocol round1). Draws by the evaluator's 64-bit hash, the
  same in every numpy. table_sums turns per-cell bits into the per-table sums it takes. The tables carry no split
  unit: the caller supplies the examined genome's (an evaluation stage may take bench/rel/implant.split_units).
