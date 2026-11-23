"""The reference trunk of the relational line: axial attention over a table of rows (genomes) by columns (codons of
loci), the base arm every other arm is compared with.

  - A cell is a token of the contract's vocabulary plus the role of its block (relation, left, right, target).
  - Row attention reads the whole row, focus and relation blocks alike. Position is rotary, on half of every head's
    dimensions, and counts inside a locus: every block has a lane of LANE positions, so cells of different loci are
    never neighbours; the other half of the dimensions carries no position, so a row can pick among loci by content.
  - Column attention is among the rows of a table at the same column, with a learned "no source" key; a row does not
    attend to itself. No row index is given anywhere: permuting the rows of a table permutes the output.
  - Cells that hold PAD (beyond a segment, outside a relative's aligned span, a block that does not exist) take no part
    in either attention as keys, so nothing a model computes for any other cell depends on them.
  - The number of rows varies from table to table; a micro-batch stacks the rows of its tables.
  - Output: a plain linear layer over the residues 1..65, computed in float32 for the target cells only; the trunk
    runs in bfloat16 on a GPU. The hierarchical head (a variant) gives instead the distribution of the amino acid first
    and then that of the codon given the amino acid, in 22 groups: the 20 amino acids of the standard code, the three
    stop codons, and OTHER alone (its probability inside its group is 1); their product is again a normalised
    distribution over the residues 1..65, so the bits are still per codon and the objective and the evaluator read it
    as they read the plain head. It writes the standard genetic code into the model.

Sizes (SIZES): 10m is 8 layers of width 256, 30m 12 x 384, 100m 12 x 768, 200m 12 x 1024. A size given as a dict may add
gate: true (the switch of the reference trunk: relation-block cells attend along the row to relation blocks only),
head: hierarchical (the output head above; default flat) and genome_bias: true (tables of version 1: a learned bias of
the column attention between two rows of one genome, genome_of_row read as equal or not only).

`python -m seeddrift.train.rel.trainer.run selftest` calls every test_ function of this module on the CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from seeddrift.train.rel import api as A
from seeddrift.train.rel.models.common import layers as C

MODULE = "axial"
SIZES = {"10m": dict(layers=8, width=256), "30m": dict(layers=12, width=384), "100m": dict(layers=12, width=768), "200m": dict(layers=12, width=1024)}
NOMINAL = {"10m": 10e6, "30m": 30e6, "100m": 100e6, "200m": 200e6}
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


HEADS = ("flat", "hierarchical")
N_GROUPS = 22                                # 20 amino acids, stop, OTHER


def token_groups():
    """long [65]: the group 0..21 of every residue 1..65 (the amino acid 0..19 of a codon under the standard code, the
    stop codons 20, OTHER 21), from the evaluator's classes (the same code the parts of the bits use)."""
    from seeddrift.train.rel.evalh import score as SC
    g = torch.as_tensor(SC.amino_acid_classes()[A.TARGET_MIN:A.TARGET_MAX + 1] - 1, dtype=torch.long)
    if g.shape[0] != N_OUT or int(g.min()) != 0 or int(g.max()) != N_GROUPS - 1 or int((g == N_GROUPS - 1).sum()) != 1:
        raise ValueError("the residues 1..65 do not fall into 20 amino acids, stop and OTHER")
    return g


def hierarchical_logp(codon, group, groups):
    """log p of every residue [M, 65] float32: log p(group) + log p(codon | group). codon [M, 65] and group [M, 22] are
    logits; groups [65] the group of every residue. A group of one residue (OTHER) gives it probability 1 inside."""
    m = codon.shape[0]
    index = groups.unsqueeze(0).expand(m, -1)
    top = torch.full((m, N_GROUPS), float("-inf"), dtype=codon.dtype, device=codon.device)
    top = top.scatter_reduce(1, index, codon.detach(), reduce="amax", include_self=True)           # a shift per group, for stability
    total = torch.zeros((m, N_GROUPS), dtype=codon.dtype, device=codon.device).scatter_add(1, index, torch.exp(codon - top.gather(1, index)))
    inside = codon - (top + torch.log(total)).gather(1, index)                                      # log p(codon | its group)
    return torch.log_softmax(group, dim=-1).gather(1, index) + inside


def count_parameters(layers, width, vocab=A.VOCAB, roles=4, n_out=N_OUT, ratio=RATIO, head="flat"):
    """The number of parameters, from the architecture."""
    attention = 3 * width * width + 3 * width + width * width + width
    block = 2 * attention + 2 * width + (ratio * width * width + ratio * width) + (ratio * width * width + width) + 3 * 2 * width
    extra = N_GROUPS * width + N_GROUPS if head == "hierarchical" else 0
    return layers * block + vocab * width + roles * width + 2 * width + n_out * width + n_out + extra


def gate_columns(slices=None):
    """The number of leading columns that are relation blocks (they come first in the contract's column order)."""
    slices = A.block_slices() if slices is None else slices
    g = 0
    for r, s, e in slices:
        if r != A.ROLE_RELATION:
            break
        g = e
    if any(r == A.ROLE_RELATION and s >= g for r, s, e in slices):
        raise ValueError("the relation blocks are not the leading columns")
    return g


class Axial(nn.Module):
    def __init__(self, layers, width, slices=None, gate=False, head="flat", ext=False, genome_bias=False):
        """gate: the switch of the reference trunk (off by default): when on, the cells of the relation blocks attend
        along the row only to relation blocks, so what a row is elsewhere is not coloured by its focus locus; the cells
        of the focus blocks attend to every block. head: flat (65 classes) or hierarchical (amino acid, then codon).
        ext: a learned vector added to the cells the renderer laid by extension (a layout other than stored marks
        them; off for the stored layout, which marks none, so such an arm has neither the parameter nor the sum).
        genome_bias: in every column attention a learned bias per head between two rows of one genome (genome_of_row,
        tables of version 1), read as equal or not only."""
        super().__init__()
        if head not in HEADS:
            raise ValueError("head is one of %s, not %r" % (HEADS, head))
        self.layers, self.width = int(layers), int(width)
        self.gate = gate_columns(slices) if gate else None
        role, pos = column_plan(slices)
        cos, sin = C.rotary_tables(pos, ROW_HEAD // 4)
        self.register_buffer("role_of_column", role, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.token = nn.Embedding(A.VOCAB, width)
        self.role = nn.Embedding(4, width)
        self.genome_bias = bool(genome_bias)
        self.blocks = nn.ModuleList([C.AxialBlock(width, ROW_HEAD, COLUMN_HEAD, RATIO, genome_bias) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, N_OUT)
        self.head_kind = head
        if head == "hierarchical":
            self.group_head = nn.Linear(width, N_GROUPS)
            self.register_buffer("groups", token_groups(), persistent=False)
        self.ext = nn.Embedding(1, width) if ext else None             # registered last: the other weights draw as before
        C.init_weights(self, self.blocks)

    def trunk(self, grid, sizes, ext=None, genome=None):
        """grid long [N, C]: the rows of the tables of a micro-batch, table after table; sizes: rows per table; ext bool
        [N, C] or None: the cells laid by extension; genome long [N] or None: genome_of_row (read with genome_bias only)."""
        if grid.shape[0] != sum(sizes) or grid.shape[1] != self.role_of_column.shape[0]:
            raise ValueError("grid %s does not fit %d rows of %d columns" % (tuple(grid.shape), sum(sizes), self.role_of_column.shape[0]))
        layout = C.table_layout(sizes, grid.device)
        keys = grid != A.PAD                                          # PAD cells take no part in attention
        keys = keys | ~keys.any(dim=1, keepdim=True)                  # (a row of PAD only would have no key at all)
        x = self.token(grid) + self.role(self.role_of_column)[None]
        if ext is not None:
            if ext.shape != grid.shape:
                raise ValueError("ext %s does not fit the grid %s" % (tuple(ext.shape), tuple(grid.shape)))
            if self.ext is not None:
                x = x + ext.to(x.dtype)[..., None] * self.ext.weight[0]
            elif bool(ext.any()):
                raise ValueError("cells laid by extension, for a model built without ext (a stored-layout arm)")
        g = genome.long() if (genome is not None and self.genome_bias) else None
        for block in self.blocks:
            x = block(x, self.cos, self.sin, layout, keys, self.gate, g)
        return x

    def forward(self, grid, sizes, row, col, ext=None, genome=None):
        """Logits float32 [M, 65] over the residues 1..65 for the cells (row[m], col[m]); with the hierarchical head
        they are already normalised log-probabilities (so cross-entropy and log_softmax read them unchanged)."""
        on_gpu = grid.device.type == "cuda"
        with torch.autocast(device_type=grid.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            x = self.trunk(grid, sizes, ext, genome)
        with torch.autocast(device_type=grid.device.type, enabled=False):
            h = self.norm(x[row, col].float())
            if self.head_kind == "hierarchical":
                return hierarchical_logp(self.head(h), self.group_head(h), self.groups)
            return self.head(h)

    def logp(self, grid, sizes, row, col, ext=None, genome=None):
        """For every target cell a normalised distribution over the tokens 1..65: log-probabilities float32 [M, 65]
        (column c is token c + 1)."""
        return torch.log_softmax(self.forward(grid, sizes, row, col, ext, genome), dim=-1)


def logp(model, grid, sizes, row, col, ext=None, genome=None):
    """The contract's reading of an arm: for every target cell a normalised distribution over the tokens 1..65."""
    return model.logp(grid, sizes, row, col, ext, genome)


def build(size):
    """size: a name of SIZES or a dict with layers and width (and optionally gate, head, ext, genome_bias)."""
    spec = SIZES[size] if isinstance(size, str) else size
    return Axial(int(spec["layers"]), int(spec["width"]), gate=bool(spec.get("gate", False)), head=str(spec.get("head", "flat")),
                 ext=bool(spec.get("ext", False)), genome_bias=bool(spec.get("genome_bias", False)))


def info(size, rows=16):
    spec = SIZES[size] if isinstance(size, str) else size
    layers, width = int(spec["layers"]), int(spec["width"])
    params = count_parameters(layers, width, head=str(spec.get("head", "flat"))) + (width if spec.get("ext") else 0) \
        + (layers * (width // COLUMN_HEAD) if spec.get("genome_bias") else 0)
    return dict(module=MODULE, params=params, layers=layers, width=width, gate=bool(spec.get("gate", False)), head=str(spec.get("head", "flat")),
                ext=bool(spec.get("ext", False)), genome_bias=bool(spec.get("genome_bias", False)),
                flops_per_cell=A.flops_per_cell(params, layers, width, A.ROW_LEN + rows),
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


def test_pad_cells_take_no_part():
    """The output of every cell that is not PAD is the same whatever the embedding of PAD is: PAD cells are never keys."""
    model = _small()
    grid = _random_grid(5, 9)
    grid[:, 100:140] = A.PAD
    grid[1, 700:] = A.PAD
    grid[3] = A.PAD                                                   # a row of PAD only
    grid[3, 0] = 5
    sizes = [3, 2]
    row, col = _all_cells(5)
    with torch.no_grad():
        a = model(grid, sizes, row, col).view(5, A.ROW_LEN, N_OUT)
        model.token.weight[A.PAD] += torch.randn(model.width)
        b = model(grid, sizes, row, col).view(5, A.ROW_LEN, N_OUT)
    real = grid != A.PAD
    assert torch.allclose(a[real], b[real], atol=2e-4), float((a[real] - b[real]).abs().max())
    assert not torch.allclose(a[~real], b[~real], atol=1e-3)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_masked_blocks_stay_finite():
    """A row whose first 600 cells are PAD (a whole block of keys masked, as in a one-row table of a genome with few
    genes): outputs and gradients stay finite."""
    torch.manual_seed(0)
    model = Axial(1, 256)
    grid = _random_grid(1, 10)
    grid[0, :600] = A.PAD
    row, col = torch.zeros(200, dtype=torch.long), torch.arange(800, 1000)
    out = model(grid, [1], row, col)
    loss = torch.nn.functional.cross_entropy(out, torch.zeros(200, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(out).all() and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_column_attention_in_chunks():
    """The column attention in chunks of sequences (the GPU's grid limit) gives what it gives in one call."""
    model = _small()
    grid = torch.cat([_random_grid(1, 20 + k) for k in range(5)])
    grid[:, :100] = A.PAD
    sizes = [1, 1, 1, 1, 1]
    row, col = _all_cells(5)
    with torch.no_grad():
        whole = model(grid, sizes, row, col)
        keep = C.COLUMN_CHUNK
        try:
            C.COLUMN_CHUNK = 1000                                     # 5 x 1,024 sequences in six calls
            parts = model(grid, sizes, row, col)
        finally:
            C.COLUMN_CHUNK = keep
    assert torch.allclose(whole, parts, atol=1e-5), float((whole - parts).abs().max())


def test_gate_keeps_relation_blocks_apart():
    """With the gate on, nothing a relation-block cell computes depends on a focus-block cell (of its row or, through
    the column attention, of another row); with it off, it does. Focus cells still read relation cells."""
    torch.manual_seed(0)
    gated = Axial(2, 256, gate=True).eval()
    plain = Axial(2, 256).eval()
    plain.load_state_dict(gated.state_dict())
    g = gate_columns()
    assert g == A.N_RELATION * A.LEN_RELATION
    grid = _random_grid(4, 11)
    sizes = [3, 1]
    row, col = _all_cells(4)
    changed = grid.clone()
    changed[:, g:] = (changed[:, g:] % 60) + 1
    with torch.no_grad():
        a, b = (gated(x, sizes, row, col).view(4, A.ROW_LEN, N_OUT) for x in (grid, changed))
        c, d = (plain(x, sizes, row, col).view(4, A.ROW_LEN, N_OUT) for x in (grid, changed))
        rel = grid.clone()
        rel[:, :g] = (rel[:, :g] % 60) + 1
        e = gated(rel, sizes, row, col).view(4, A.ROW_LEN, N_OUT)
    assert torch.allclose(a[:, :g], b[:, :g], atol=1e-5) and not torch.allclose(c[:, :g], d[:, :g], atol=1e-4)
    assert not torch.allclose(a[:, g:], e[:, g:], atol=1e-4)
    assert sum(p.numel() for p in gated.parameters()) == count_parameters(2, 256)


def test_hierarchical_head():
    """The hierarchical head is a normalised distribution over the residues 1..65 whose group marginals are the
    softmax of its group logits; OTHER alone in its group, the three stop codons in one; the trunk is the flat one's."""
    groups = token_groups()
    assert [int((groups == g).sum()) for g in (20, 21)] == [3, 1] and int(groups[A.OTHER - A.TARGET_MIN]) == 21
    assert sorted(set(groups.tolist())) == list(range(N_GROUPS))
    torch.manual_seed(0)
    model = Axial(2, 256, head="hierarchical").eval()
    flat = Axial(2, 256).eval()
    flat.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("group_head.")})
    assert sum(p.numel() for p in model.parameters()) == count_parameters(2, 256, head="hierarchical") == count_parameters(2, 256) + 22 * 257
    assert info(dict(layers=12, width=384, head="hierarchical"))["params"] == count_parameters(12, 384) + 22 * 385
    grid = _random_grid(5, 9)
    sizes = [3, 2]
    row, col = _all_cells(5)
    with torch.no_grad():
        lp = model.logp(grid, sizes, row, col)
        out = model(grid, sizes, row, col)
        h = model.norm(model.trunk(grid, sizes)[row, col].float())
        marg = torch.zeros(lp.shape[0], N_GROUPS).scatter_add(1, groups.expand(lp.shape[0], -1), lp.exp())
    assert torch.allclose(lp.exp().sum(dim=-1), torch.ones(lp.shape[0]), atol=1e-5) and torch.allclose(out, lp, atol=1e-5)
    assert torch.allclose(marg, torch.softmax(model.group_head(h), dim=-1), atol=1e-5)
    assert torch.allclose(lp[:, A.OTHER - A.TARGET_MIN], torch.log_softmax(model.group_head(h), dim=-1)[:, 21], atol=1e-5)
    with torch.no_grad():
        assert torch.allclose(model.trunk(grid, sizes), flat.trunk(grid, sizes))
    model.train()
    loss = torch.nn.functional.cross_entropy(model(grid, sizes, torch.tensor([0, 2, 4]), torch.tensor([5, 800, 1023])), torch.tensor([0, 10, 64]))
    loss.backward()
    missing = [k for k, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing and abs(float(loss) - 4.17) < 0.6, (missing[:5], float(loss))
    try:
        Axial(1, 256, head="deep")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown head was accepted")


def test_ext_cells():
    """The ext vector: with no cell laid by extension a model built with ext computes what the same weights compute
    without it; a cell laid by extension changes the output; a model built without ext refuses such cells."""
    torch.manual_seed(0)
    with_ext = Axial(2, 256, ext=True).eval()
    plain = Axial(2, 256).eval()
    plain.load_state_dict({k: v for k, v in with_ext.state_dict().items() if not k.startswith("ext.")})
    assert sum(p.numel() for p in with_ext.parameters()) == count_parameters(2, 256) + 256
    assert info(dict(layers=12, width=384, ext=True))["params"] == count_parameters(12, 384) + 384
    grid = _random_grid(4, 13)
    sizes = [3, 1]
    row, col = _all_cells(4)
    none = torch.zeros_like(grid, dtype=torch.bool)
    some = none.clone()
    some[1, 300:340] = True
    with torch.no_grad():
        a = plain(grid, sizes, row, col)
        assert torch.allclose(with_ext(grid, sizes, row, col, none), a, atol=1e-6) and torch.allclose(plain(grid, sizes, row, col, none), a)
        b = with_ext(grid, sizes, row, col, some).view(4, A.ROW_LEN, N_OUT)
        assert not torch.allclose(b[1, 300:340], a.view(4, A.ROW_LEN, N_OUT)[1, 300:340], atol=1e-4)
    try:
        plain(grid, sizes, row, col, some)
    except ValueError:
        pass
    else:
        raise AssertionError("a model built without ext read cells laid by extension")


def test_genome_bias_is_equivariant():
    """The same-genome bias: zero at the start (the output equals the model's without it); once set, the grouping of
    rows into genomes changes the output; permuting the rows of a table (their genome numbers with them) permutes the
    output, and renumbering the genomes changes nothing: the numbers are read as equal or not only."""
    torch.manual_seed(0)
    model = Axial(2, 256, genome_bias=True).eval()
    plain = Axial(2, 256).eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if "same_genome" not in k})
    assert sum(p.numel() for p in model.parameters()) == count_parameters(2, 256) + 2 * 2 == info(dict(layers=2, width=256, genome_bias=True))["params"]
    grid = _random_grid(6, 21)
    sizes = [5, 1]
    row, col = _all_cells(6)
    genome = torch.tensor([0, 3, 3, 7, 0, 0])                  # rows 1 and 2 of one genome, row 4 a self-copy of row 0
    with torch.no_grad():
        assert torch.allclose(model(grid, sizes, row, col, genome=genome), plain(grid, sizes, row, col), atol=1e-6)
        for blk in model.blocks:
            blk.column.same_genome.fill_(1.5)
        a = model(grid, sizes, row, col, genome=genome).view(6, A.ROW_LEN, N_OUT)
        apart = model(grid, sizes, row, col, genome=torch.arange(6)).view(6, A.ROW_LEN, N_OUT)
        assert not torch.allclose(a[:5], apart[:5], atol=1e-4)                            # the grouping is read
        perm = torch.tensor([3, 1, 4, 0, 2])                                              # the rows of the first table
        full = torch.cat([perm, torch.tensor([5])])
        b = model(grid[full], sizes, row, col, genome=genome[full]).view(6, A.ROW_LEN, N_OUT)
        assert torch.allclose(b[:5], a[perm], atol=1e-5)
        renamed = torch.tensor([9, 1, 1, 4, 9, 2])                                         # the same grouping, other numbers
        c = model(grid, sizes, row, col, genome=renamed).view(6, A.ROW_LEN, N_OUT)
        assert torch.allclose(c[:5], a[:5], atol=1e-5) and torch.allclose(c[5], a[5], atol=1e-5)
        assert torch.allclose(model(grid, sizes, row, col).view(6, A.ROW_LEN, N_OUT), plain(grid, sizes, row, col).view(6, A.ROW_LEN, N_OUT),
                              atol=1e-6)                                                  # no genome given: no bias


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
