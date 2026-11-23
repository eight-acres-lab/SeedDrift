# W3: the renderer's version in every training and evaluation record

- The reader's render string (`data/tables.RENDER_VERSION`, part of the trainer's `versions()` in every summary,
  evaluation state and model index) carries the version of package W3's renderer: `stored-w3-2`, no longer the fixed
  `stored-w3`. The strict offline renderings (w3_offline, w18_offline) already record it in their manifests, and the
  leak-free read-outs as `w3-<version>`.
