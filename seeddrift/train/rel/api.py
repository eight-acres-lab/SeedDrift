"""The contract between the packages of the relational-model line, version 1 (owned by the programme lead).

Pure Python: imports nothing, so it loads in the CI environment (no torch) and in the model interpreter.
Everyone else reads it; changes go through a request file (configs/rel/requests/).

What is not here on purpose: the table-set manifest (handoff.py) and the layout of shard files on disk, which has one
implementation, the reader and writer of package W2 (seeddrift/train/rel/tables/); the evaluation items, the strata and
the result-row writer, which have one implementation in seeddrift/train/rel/evalh/.
"""

API = 1
API_VERSION = API     # the name the foundation commit reads

# ---- vocabulary (codon ids as in the existing training arrays: 1..64 = codon in ACGT order + 1) ----
PAD = 0
CODON_MIN, CODON_MAX = 1, 64
OTHER = 65            # a codon with an ambiguous base, a stop inside a gene
GAP = 66              # alignment gap in the rendered grid
MASK = 67             # a hidden cell
ABSENT = 68           # searched, no homolog found (an input only until the uncapped search exists)
NOT_SEARCHED = 69     # never a target, never in a read-out
TRUNCATED = 70        # never a target, never in a read-out
BOS = 71              # start of a stretch (causal arms)
EXT_EMPTY = 72        # placed by extension, no content
VOCAB = 73

# A target cell is always a residue of its own row: tokens 1..65. Mass an arm puts elsewhere is lost.
TARGET_MIN, TARGET_MAX = 1, 65

# ---- blocks of a table, in own-coordinate storage order (relation blocks first, target gene last) ----
ROLE_RELATION, ROLE_LEFT, ROLE_RIGHT, ROLE_TARGET = 0, 1, 2, 3
N_RELATION, LEN_RELATION = 16, 32
LEN_NEIGHBOUR = 128
LEN_TARGET = 256
N_BLOCKS = N_RELATION + 3
ROW_LEN = N_RELATION * LEN_RELATION + 2 * LEN_NEIGHBOUR + LEN_TARGET      # 1,024 slots per row


def block_slices():
    """(role, start, stop) of every block inside a row of ROW_LEN slots."""
    out, at = [], 0
    for _ in range(N_RELATION):
        out.append((ROLE_RELATION, at, at + LEN_RELATION))
        at += LEN_RELATION
    for role in (ROLE_LEFT, ROLE_RIGHT):
        out.append((role, at, at + LEN_NEIGHBOUR))
        at += LEN_NEIGHBOUR
    out.append((ROLE_TARGET, at, at + LEN_TARGET))
    return out


# The 512-column shape is cut from the same shards: a 128-codon window of the target block around the scored
# stretch, 64 codons of each neighbour, the first 8 relation blocks.
SHAPE_512 = dict(target=128, neighbour=64, n_relation=8, relation=32)

# ---- kinds of a (row, block) ----
# KIND_NONE: nothing there (the table has fewer rows, or the examined genome has no such block).
# KIND_PRESENT: codons. KIND_ABSENT: searched without a cap, no homolog (cannot occur before the uncapped search).
# KIND_NOT_SEARCHED: no homolog row is stored; under a capped search this does not mean there is no homolog.
# KIND_TRUNCATED: a homolog row exists but none of its aligned codons lies inside the stored part of the block.
KIND_NONE = 0
KIND_PRESENT, KIND_ABSENT, KIND_NOT_SEARCHED, KIND_TRUNCATED, KIND_SELF_COPY = 1, 2, 3, 4, 5

# ---- the arrays of a shard (file layout and manifest: see the module docstring) ----
# N tables, R rows (row 0 is the examined genome; the others in a fixed order, distance tiers interleaved),
# B = N_BLOCKS blocks, ROW_LEN slots per row.
SHARD_FIELDS = {
    "tok":       ("uint8", "N R ROW_LEN", "codons of this row's segment of each block, in the row's own coordinates, left-justified in the block's slice; PAD after the segment ends"),
    "n":         ("int16", "N R B", "length of the segment"),
    "col":       ("int16", "N R ROW_LEN", "stored placement: for each codon, its column in the coordinates of frame_row, as a 0-based offset inside the same block (grid slot = block start + col); the frame row has col = i for its i-th codon; -1 = no column (insertion against the frame, outside the aligned span, or an empty slot); valid values strictly increase inside a block"),
    "frame_row": ("int8",  "N B", "the row whose coordinates col refers to (0 in table version 0)"),
    "frame_len": ("int16", "N B", "length of the frame row's segment"),
    "kind":      ("uint8", "N R B", "KIND_*"),
    "start":     ("int64", "N R B", "start of the segment on its own replicon; -1 = unknown"),
    "strand":    ("int8",  "N R B", "+1, -1, 0 = unknown"),
    "replicon":  ("int16", "N R B", "index of the replicon inside its genome; -1 = unknown"),
}
# Never given to a model: sampling and reporting only.
META_FIELDS = {
    "genome":    ("int32", "N R", "index of the genome of each row in the run's genome list; -1 = no row"),
    "group":     ("int8",  "N", "one of the 12 groups, of the examined genome"),
    "split":     ("int8",  "N", "0 train, 1 held out"),
    "gene":      ("int32", "N B", "gene index of each block in the examined genome"),
    "tier":      ("int8",  "N R", "distance tier of the row (defined after the census; -1 until then)"),
    "identity":  ("float32", "N R", "share of identical codons between the row and row 0 over the aligned part of the target block; nan = no cell"),
}

TABLE_REAL, TABLE_SYNTHETIC, TABLE_RULER = "real", "synthetic", "ruler"
LAYOUT_LEAKFREE, LAYOUT_STORED, LAYOUT_UNALIGNED = "leakfree", "stored", "unaligned"

# ---- conditions (names shared by mask schemes and by the evaluator) ----
COND_SCATTERED = "scattered"            # single cells
COND_SUFFIX = "suffix"                  # from a cut to the end of the stretch
COND_WHOLE_GENE = "whole_gene"          # the whole target block of the examined row
COND_NO_RELATIVES = "no_relatives"      # all other rows removed
COND_REPAIRED = "repaired"              # rows of every non-target block permuted, evaluation only
COND_ROWS_REMOVED = "rows_removed"      # the nearest tier of rows removed
COND_JOIN = "+"                         # a combined condition joins names in a fixed order, e.g. "whole_gene+no_relatives"


def condition(*parts):
    """The name of a combined condition: its parts joined by COND_JOIN, in the order given."""
    return COND_JOIN.join(parts)


def flops_per_cell(params, layers, width, attention_len):
    """Planning formula for training cost per token cell. For the axial trunk attention_len = row length + rows."""
    return 6 * params + 12 * layers * width * attention_len


# ---- what an arm is made of (documentation; modules are found by name through the registry) ----
# mask scheme : plan(meta, rng) -> visible, target      two boolean arrays shaped like tok; cells in neither are hidden and unscored
# renderer    : render(table, visible, layout, seed) -> rendered     shared by all arms
# objective   : loss(model, rendered, target, truth) -> scalar, per-cell losses with cell indices
# model       : logp(rendered, target, truth=None) -> for every target cell a normalised distribution over tokens 1..65
#               info = dict(params=..., flops_per_cell=..., factorisation="independent" | "chain",
#                           layout="any" | "gene_last", gpu_mb=..., common_version=...)
RESULT_ROW = (
    "run", "commit", "variant", "author", "module", "params", "seed", "table_set", "table_kind", "items_sha256",
    "item_half", "layout", "budget", "weights", "cells_seen", "target_cells", "cells_per_s", "peak_mb",
    "condition", "group", "tier", "stratum", "passes", "bits", "baseline", "bits_minus_baseline", "n_items",
    "status", "versions",
)
