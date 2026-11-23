# W18: a reading takes the offline placements of a passed run only

- w18_passes.load_offline (run_ruler's offline): the run of stage rel_w18_offline must have passed and its stage record
  must list offline.json with this content; the stage's own gate reads its placements before the run has passed.
