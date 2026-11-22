# configs/rel: configuration of the relational-model line

One owner per file; a package that needs a change in a file it does not own adds a request file instead.

| Path | Content | Owner |
|---|---|---|
| `kinds.yaml` | the pipeline kinds (rel_tables, rel_traineval, rel_readout, rel_corpus, rel_events): lock, gates, every stage with its rule, output, module, gates and side | coordinator (C0) |
| `machines.json` | the machines (GPU, memory, CI root, trainings at once) that "this configuration belongs to this machine" checks | C0 |
| `sealed.json` | what the sealed check refuses: the pinned test split of the medium corpus, the sealed chromosome lists, the extra exclusion lists | C0 |
| `variants/<author>-<name>.yaml` | one trainable variant per file (keys: the contract, `seeddrift/train/rel/api.py`); never changed once a round used it | its author |
| `protocols/<round>.yaml` | a round's protocol: metrics, effect sizes, intervals, mask shares, coverage, slots (variants, seeds, budgets, machines), the `tolerance` block of `ci compare`; committed before the round's deadline | programme lead (P0) |
| `requests/<date>-<package>-<short>.yaml` | one request per file: a GPU slot, a CPU run over 15 minutes, a contract change, a request to another package | its author |
| `coordinator/` | the coordinator's state: merged branches, runs started, queued requests, tables copied | C0 |

Profiles live in `ci/profiles/`: `rel-<round>-<machine>/` (a round, written by C0 from the protocol) and
`rel-dev-<package>-<purpose>/` (a package's own smoke runs, each with the line `needs_gpu: false` and `train.device: cpu`;
removed once the feature is merged). Every rel profile sets `train.gpu_lease: false` and names its machine (`rel.machine`).
