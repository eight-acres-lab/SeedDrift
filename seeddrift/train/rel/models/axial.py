"""The reference trunk of the relational line: axial attention over a table of rows (genomes) by columns (codons of
loci), the base arm every other arm is compared with.

  - A cell is a token of the contract's vocabulary plus the role of its block (relation, left, right, target).
  - Row attention reads the whole row, focus and relation blocks alike. Position is rotary, on half of every head's
    dimensions, and counts inside a locus: every block has a lane of LANE positions, so cells of different loci are
    never neighbours; the other half of the dimensions carries no position, so a row can pick among loci by content.
  - Column attention is among the rows of a table at the same column, with a learned "no source" key; a row does not
    attend to itself. No row index is given anywhere: permuting the rows of a table permutes the output.
  - The number of rows varies from table to table; a micro-batch stacks the rows of its tables.
  - Output: a plain linear layer over the residues 1..65, computed in float32 for the target cells only; the trunk
    runs in bfloat16 on a GPU.

Sizes (SIZES): 10m is 8 layers of width 256, 30m 12 x 384, 100m 12 x 768.

`python -m seeddrift.train.rel.trainer.run selftest` calls every test_ function of this module on the CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from seeddrift.train.rel import api as A
from seeddrift.train.rel.models.common import layers as C

MODULE = "axial"
SIZES = {"10m": dict(layers=8, width=256), "30m": dict(layers=12, width=384), "100m": dict(layers=12, width=768)}
NOMINAL = {"10m": 10e6, "30m": 30e6, "100m": 100e6}
LANE = 512                                   # positions of a block's lane; larger than any block
N_OUT = A.TARGET_MAX - A.TARGET_MIN + 1      # residues 1..65
ROW_HEAD, COLUMN_HEAD, RATIO = 64, 128, 4


def column_plan(slices=None):
    """role [C] and rotary position [C] of every column of the grid."""
    slices = A.block_slices() if slices is None else slices
    role, pos = [], []
    for b, (r, s, e) in enumerate(slices):
        role += [r] * (e - s)
        pos += [b * LANE + p for p in range(e - s)]
    return torch.tensor(role, dtype=torch.long), torch.tensor(pos, dtype=torch.long)


def count_parameters(layers, width, vocab=A.VOCAB, roles=4, n_out=N_OUT, ratio=RATIO):
    """The number of parameters, from the architecture."""
    attention = 3 * width * width + 3 * width + width * width + width
    block = 2 * attention + 2 * width + (ratio * width * width + ratio * width) + (ratio * width * width + width) + 3 * 2 * width
    return layers * block + vocab * width + roles * width + 2 * width + n_out * width + n_out


class Axial(nn.Module):
    def __init__(self, layers, width, slices=None):
        super().__init__()
        self.layers, self.width = int(layers), int(width)
        role, pos = column_plan(slices)
        cos, sin = C.rotary_tables(pos, ROW_HEAD // 4)
        self.register_buffer("role_of_column", role, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.token = nn.Embedding(A.VOCAB, width)
        self.role = nn.Embedding(4, width)
        self.blocks = nn.ModuleList([C.AxialBlock(width, ROW_HEAD, COLUMN_HEAD, RATIO) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, N_OUT)
        C.init_weights(self, self.blocks)

    def trunk(self, grid, sizes):
        """grid long [N, C]: the rows of the tables of a micro-batch, table after table; sizes: rows per table."""
        if grid.shape[0] != sum(sizes) or grid.shape[1] != self.role_of_column.shape[0]:
            raise ValueError("grid %s does not fit %d rows of %d columns" % (tuple(grid.shape), sum(sizes), self.role_of_column.shape[0]))
        layout = C.table_layout(sizes, grid.device)
        x = self.token(grid) + self.role(self.role_of_column)[None]
        for block in self.blocks:
            x = block(x, self.cos, self.sin, layout)
        return x

    def forward(self, grid, sizes, row, col):
        """Logits float32 [M, 65] over the residues 1..65 for the cells (row[m], col[m])."""
        on_gpu = grid.device.type == "cuda"
        with torch.autocast(device_type=grid.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            x = self.trunk(grid, sizes)
        with torch.autocast(device_type=grid.device.type, enabled=False):
            return self.head(self.norm(x[row, col].float()))

    def logp(self, grid, sizes, row, col):
        """For every target cell a normalised distribution over the tokens 1..65: log-probabilities float32 [M, 65]
        (column c is token c + 1)."""
        return torch.log_softmax(self.forward(grid, sizes, row, col), dim=-1)


def logp(model, grid, sizes, row, col):
    """The contract's reading of an arm: for every target cell a normalised distribution over the tokens 1..65."""
    return model.logp(grid, sizes, row, col)


def build(size):
    """size: a name of SIZES or a dict with layers and width."""
    spec = SIZES[size] if isinstance(size, str) else size
    return Axial(int(spec["layers"]), int(spec["width"]))


def info(size, rows=16):
    spec = SIZES[size] if isinstance(size, str) else size
    layers, width = int(spec["layers"]), int(spec["width"])
    params = count_parameters(layers, width)
    return dict(module=MODULE, params=params, layers=layers, width=width, flops_per_cell=A.flops_per_cell(params, layers, width, A.ROW_LEN + rows),
                factorisation="independent", layout="any", common_version=C.COMMON_VERSION)


# ---- self-tests (CPU, small widths; called by the trainer's selftest mode in the model interpreter) ----

def _small(seed=0):
    torch.manual_seed(seed)
    return Axial(2, 256).eval()


def _random_grid(rows, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, A.VOCAB, (rows, A.ROW_LEN), generator=g)


def _all_cells(n):
    row = torch.arange(n).repeat_interleave(A.ROW_LEN)
    return row, torch.arange(A.ROW_LEN).repeat(n)


def test_parameter_count():
    for name, spec in SIZES.items():
        n = sum(p.numel() for p in Axial(1, spec["width"]).parameters())
        assert n == count_parameters(1, spec["width"]), (name, n, count_parameters(1, spec["width"]))
        full = count_parameters(spec["layers"], spec["width"])
        assert abs(full - NOMINAL[name]) < 0.2 * NOMINAL[name], (name, full)
    assert info("30m")["params"] == count_parameters(12, 384)


def test_rows_permute():
    """Permuting the rows of a table permutes the output: the model has no row index."""
    model = _small()
    sizes = [5, 3]
    grid = _random_grid(8, 1)
    perm = torch.tensor([3, 0, 4, 1, 2, 7, 5, 6])                      # within each table
    with torch.no_grad():
        a = model(grid, sizes, *_all_cells(8)).view(8, A.ROW_LEN, N_OUT)
        b = model(grid[perm], sizes, *_all_cells(8)).view(8, A.ROW_LEN, N_OUT)
    assert torch.allclose(a[perm], b, atol=2e-4), float((a[perm] - b).abs().max())
    assert not torch.allclose(a[0], a[1], atol=1e-3)


def test_tables_do_not_mix():
    """A table's output is the same alone and stacked with tables of other sizes (the padding of the column attention)."""
    model = _small()
    one, two, three = _random_grid(4, 2), _random_grid(1, 3), _random_grid(6, 4)
    with torch.no_grad():
        alone = [model(g, [g.shape[0]], *_all_cells(g.shape[0])) for g in (one, two, three)]
        both = model(torch.cat([one, two, three]), [4, 1, 6], *_all_cells(11))
    assert torch.allclose(torch.cat(alone), both, atol=2e-4), float((torch.cat(alone) - both).abs().max())


def test_column_and_row_reach():
    """A cell's output depends on the same column of another row of its table and on another locus of its own row,
    and not on a row of another table."""
    model = _small()
    grid = _random_grid(5, 5)
    sizes = [3, 2]
    row, col = torch.tensor([0, 0]), torch.tensor([900, 10])
    with torch.no_grad():
        base = model(grid, sizes, row, col)
        g = grid.clone()
        g[1, 900] = (g[1, 900] % 60) + 1                              # same column, another row of table 0
        assert (model(g, sizes, row, col)[0] - base[0]).abs().max() > 1e-5
        g = grid.clone()
        g[0, 40] = (g[0, 40] % 60) + 1                                # another locus of the same row
        assert (model(g, sizes, row, col)[0] - base[0]).abs().max() > 1e-5
        g = grid.clone()
        g[3] = (g[3] % 60) + 1                                        # a row of table 1
        assert torch.allclose(model(g, sizes, row, col), base, atol=2e-4)


def test_lone_row():
    """A table of one row (no relatives): the column attention has the "no source" key only."""
    model = _small()
    with torch.no_grad():
        out = model.logp(_random_grid(1, 6), [1], *_all_cells(1))
    assert torch.isfinite(out).all() and torch.allclose(out.exp().sum(-1), torch.ones(A.ROW_LEN), atol=1e-4)


def test_hidden_content_does_not_leak():
    """Through the reader: changing the residues of hidden cells changes neither the grid the model reads nor any
    of its outputs, while changing a visible cell does."""
    import numpy as np
    from seeddrift.train.rel.data import tables as T
    rng = np.random.default_rng(7)
    rows = 4
    tok = np.zeros((rows, A.ROW_LEN), np.uint8)
    n = np.zeros((rows, A.N_BLOCKS), np.int16)
    col = np.full((rows, A.ROW_LEN), -1, np.int16)
    for b, (_, s, e) in enumerate(A.block_slices()):
        n[:, b] = e - s
        tok[:, s:e] = rng.integers(1, 66, size=(rows, e - s))
        col[:, s:e] = np.arange(e - s)
    tab = dict(tok=tok, n=n, col=col, kind=np.full((rows, A.N_BLOCKS), A.KIND_PRESENT, np.uint8), frame_len=n[0].copy())
    cells = T.inside(tab)
    hide = cells & (rng.random(cells.shape) < 0.15)
    hide[0, T.T0:T.T1] = True                                        # and the whole target gene of row 0
    grid, truth = T.render(tab, cells & ~hide, hide)
    other = dict(tab, tok=np.where(hide, (tok % 64) + 1, tok).astype(np.uint8))
    grid2, _ = T.render(other, cells & ~hide, hide)
    assert (grid == grid2).all() and (grid[hide] == A.MASK).all()
    model = _small()
    row, col_ = np.nonzero(truth)
    row, col_ = torch.from_numpy(row), torch.from_numpy(col_)
    with torch.no_grad():
        a = model(torch.from_numpy(grid.astype(np.int64)), [rows], row, col_)
        b = model(torch.from_numpy(grid2.astype(np.int64)), [rows], row, col_)
        seen = dict(tab, tok=np.where(~hide & (T.POS[None, :] == 3), (tok % 64) + 1, tok).astype(np.uint8))
        grid3, _ = T.render(seen, cells & ~hide, hide)
        c = model(torch.from_numpy(grid3.astype(np.int64)), [rows], row, col_)
    assert torch.equal(a, b) and not torch.allclose(a, c, atol=1e-5)


def test_backward():
    torch.manual_seed(0)
    model = Axial(2, 256)
    grid = _random_grid(5, 8)
    row, col = torch.tensor([0, 2, 4]), torch.tensor([5, 800, 1023])
    loss = torch.nn.functional.cross_entropy(model(grid, [3, 2], row, col), torch.tensor([0, 10, 64]))
    loss.backward()
    missing = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing[:5]
    assert abs(float(loss) - 4.17) < 0.5, float(loss)                 # near ln 65 at the start
