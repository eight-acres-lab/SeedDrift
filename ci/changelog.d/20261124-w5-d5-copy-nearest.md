# W5 D5 phase 1 — reproducing the copy_nearest baseline

D5 compares the original 3,000 held-out tables with a paired set that adds close-relative rows. Both readings use one saved fit of `copy_nearest`. The following inputs are recorded in `ci/profiles/rel-dev-w5-baselines-v1-i1/config.yaml`.

| Original input | Passed run ID | Manifest path | Manifest SHA-256 |
| --- | --- | --- | --- |
| v1 training and held-out tables | `20261123T122205-exp-rel-c76dd7cd-rel-dev-w12-v1-k3-i1` | `data/rel_w12/v1/tableset.json` | `dfb612a560fc9cb8991cdb0ffce3541d1bb3ea79ab63489d58a881d5695202c1` |
| Strict offline rendering of the original held-out tables | `20261123T124714-exp-rel-8015afa8-rel-dev-w3-offline-v1-i1` | `data/rel_w3/offline/tableset.json` | `2f761265d52ac6a09dbea7285f5a90e1e6b0dbea99bb6603c5e6c260f862d496` |

**Fit provenance.** The baseline profile fits on the first 6,000 stored training tables. The separate chain baseline uses their first 1,500. The stage checks every fit table's split before calling `fit_all`. The fit seed is 20261123 and the evaluation seed is 20261122. The full fitted parameters, including the residue prior and rates for each mask, are stored in `data/rel_w5/baselines/params.json`. Its file hash is the `params` output of stage `rel_w5_baselines` in the source run record. The shorter `fit.numbers` in `results.json` omits the prior. An independent refit also writes its elapsed time into the parameter file.

**C0 handoff.** Pin the actual passed v5 baseline run ID, commit, and output hashes for `params.json`, `results.json` and `cells.npz`. The scheduled source commit is `4a4530a7dfc66d63385658e7cb0c705cf6644b0d`. Check the baseline gate and hash each file against the run record. Verify the fit table count, training split, seed and v1 table manifest against the original profile. Reuse the exact `params.json` bytes for both inputs and record the same hash beside both readings. The original v5 per-cell results can supply the original member of the pair for matching conditions.

**W2 handoff.** Pin the passed close-set table and strict offline rendering run IDs, manifest paths and hashes. Record the close-set purpose (`heldout`, `not_for_training`), its pairing with the original target rows, and W2's checks on retained training rows and tables with no added relatives. Use the same target coordinates and residues for each condition, with per-table pairing and a scored-cell digest on each side. The no-relatives control must reproduce the original bits table by table. Record which groups and conditions actually have paired observations.

Set `rel_w5_baselines.frozen_fit` to the pinned source run and the explicit fields documented in the stage. Scoring reads the saved parameters and writes the same bytes into its own output. The stage verifies the condition layouts and pairs the ordinary and no-relatives target digests with the original result. The repaired control includes the partner-row permutation in its digest and requires a separate eligibility comparison. The source run and the close set each retain their own stage record. The close-set inputs and v5 fit output hashes will come from their producing runs.
