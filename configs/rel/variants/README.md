# Variants

One trainable variant per file, `<author>-<name>.yaml`, with the keys of the contract (`seeddrift/train/rel/api.py`,
version 1): name, author, model module, objective module, mask, rendering, rows and columns, relatives or not,
optimizer changes, declared memory, table set, API version. Seeds, budgets and machines are not here: the round's
protocol assigns them. A variant used by a round is never changed again; a change is a new file.
