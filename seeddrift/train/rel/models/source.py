"""Arms S1 and S2 of decision D7 (package W5 on W6's common layers): output layers built around "one source at a
time". A source of a cell is another row of its table whose cell in the same column holds a residue (1..65), or "no
source"; a cell is predicted by copying its source's codon through a substitution distribution, or from its own row
alone. Other rows reach the prediction only through which source is copied and the codon copied.

The trunk is the reference arm's (models/axial.py: the same blocks, embeddings, rotary lanes and initialisation),
split in two so that one branch cannot see other rows:
  - the first self_layers blocks run every row as a table of its own (the column attention sees only its learned
    "no source" key): their output h_self depends on the row's own cells only;
  - the remaining blocks run over the table as usual: their output h_full sees every row.
A step costs what the reference trunk of the same layers and width costs; no row index is given anywhere.

Shared output pieces, at a target cell (float32):
  p_self(x)          softmax(a(h_self)), the cell's distribution from its own row alone ("no source");
  sub_l(x | c)       softmax(a(h_self) + g_l(h_self) * B_l[c]) for l = 1..LEVELS, the substitution of a copied codon
                     c: the own-row distribution tilted towards what c says, more or less strongly by level (g_l > 0,
                     learned per cell from h_self; B_l a learned 65 x 65 table). It reads no other row but through c.
  score(r, l)        <Q_l h_full[cell], K h_full[row r, same column]> / sqrt(KEY) for every other row r of the
                     table; score(none) = a linear function of h_full[cell]. A source is available only where row r's
                     cell holds a residue 1..65: GAP, PAD, MASK, ABSENT, NOT_SEARCHED, TRUNCATED, EXT_EMPTY never are.
                     "No source" always is.

S2 (learned copy model): p(x) = w_none p_self(x) + sum over available (r, l) of w_rl sub_l(x | c_r), the weights a
  softmax over the available (r, l) and none. In the programme's notation lambda = 1 - w_none, w_r = sum_l w_rl / lambda.

S1 (source path): the source of a row's cells is a chain along the columns of each block, restarted at every block
  (another locus). Its log potentials, from the grid only:
    unary(r) = logsumexp_l score(r, l) [+ evidence], unary(none) = score(none) [+ evidence], -1e4 where unavailable;
    pair(s, s') = 1 + (exp(kappa) - 1) [s == s'], the stay bonus kappa = softplus(linear(h_full)) given by the trunk
    per position, at most KAPPA_MAX (the edge into a cell; none at a block's first column);
    evidence (switch, on by default): at a visible residue x of the row itself, log E(x | c_r) for source r and
    log E0(x) for none, E a learned 65 x 65 substitution table with no context (so no cell's own token is ever read
    through a context that has seen it). With the trunk's scores zero this is the classical switching copy model with
    learned rates and table: S1 nests it.
  The marginal posterior P(s_cell | grid) comes from the forward and backward passes in log space; a target cell is
  predicted by summing over sources: p(x) = P(none) p_self(x) + sum_r P(r) sum_l v_l(r) sub_l(x | c_r), with
  v(r) = softmax_l score(r, l). With every stay bonus zero and no evidence S1 is S2.

Both arms give every target cell its own normalised distribution over the residues 1..65 from the grid alone: the
factorisation is independent, as the reference arm's, so their bits compare with it directly and the objective (mlm)
and the evaluator read them unchanged (forward returns normalised log-probabilities, as the hierarchical head does).
The joint bits of a row's hidden cells (the forward algorithm with the true codons as emissions) and the source
posterior given those codons are read-outs only: readout().

Sizes: a dict {layers, width, arm: s1 | s2, self_layers, levels (default 3), evidence (S1 only, default true)} or a name
of SIZES; ext is added by the trainer for a layout other than stored, as for axial.

`python -m seeddrift.train.rel.trainer.run selftest` calls every test_ function of this module on the CPU.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seeddrift.train.rel import api as A
from seeddrift.train.rel.models import axial as AX
from seeddrift.train.rel.models.common import layers as C

MODULE = "source"
SOURCE_VERSION = "2"                         # 2: the own-row blocks without their unneeded column attention, the chain in probability space
ARMS = ("s1", "s2")
N_OUT = A.TARGET_MAX - A.TARGET_MIN + 1      # residues 1..65
LEVELS = 3
KEY = 64                                     # dimension of the pair scores
NEG = -1.0e4                                 # log potential of an unavailable source (finite: no -inf - -inf in the chain)
STAY0 = 3.0                                  # the stay bonus at the start
KAPPA_MAX = 20.0                             # the stay bonus at most (a stay probability of 1 - 2e-9 per position)
STRENGTH0 = (0.5, 5.0)                       # the copy strengths of the weakest and strongest level at the start
EVIDENCE0 = 2.0                              # the identity log-odds of the evidence table at the start
SIZES = {"s1-30m": dict(layers=12, width=384, arm="s1", self_layers=4), "s2-30m": dict(layers=12, width=384, arm="s2", self_layers=4),
         "s1-10m": dict(layers=8, width=256, arm="s1", self_layers=3), "s2-10m": dict(layers=8, width=256, arm="s2", self_layers=3)}
KNOWN = {"layers", "width", "arm", "self_layers", "levels", "evidence", "ext"}


def spec_of(size):
    spec = dict(SIZES[size]) if isinstance(size, str) else dict(size)
    unknown = set(spec) - KNOWN
    if unknown:
        raise ValueError("unknown keys of a source size: %s" % sorted(unknown))
    arm = str(spec.get("arm", ""))
    if arm not in ARMS:
        raise ValueError("arm is one of %s, not %r" % (ARMS, arm))
    layers, self_layers = int(spec["layers"]), int(spec["self_layers"])
    if not 1 <= self_layers < layers:
        raise ValueError("self_layers must be at least 1 and below layers (%d), not %d" % (layers, self_layers))
    evidence = bool(spec.get("evidence", arm == "s1"))
    if evidence and arm != "s1":
        raise ValueError("evidence is a switch of arm s1")
    return dict(layers=layers, width=int(spec["width"]), arm=arm, self_layers=self_layers, levels=int(spec.get("levels", LEVELS)),
                evidence=evidence, ext=bool(spec.get("ext", False)))


def count_parameters(layers, width, arm, levels=LEVELS, evidence=None, ext=False, vocab=A.VOCAB, roles=4):
    """The number of parameters, from the architecture (the blocks are the reference trunk's)."""
    evidence = arm == "s1" if evidence is None else evidence
    block = AX.count_parameters(1, width) - AX.count_parameters(0, width)
    n = layers * block + vocab * width + roles * width + 2 * 2 * width                     # embeddings, the two norms
    n += (N_OUT * width + N_OUT) + (levels * width + levels) + levels * N_OUT * N_OUT       # own-row logits, strengths, B
    n += (width * levels * KEY + levels * KEY) + (width * KEY + KEY) + (width + 1)        # queries, keys, no source
    if arm == "s1":
        n += width + 1                                                                   # stay bonus
        if evidence:
            n += N_OUT * N_OUT + N_OUT
    return n + (width if ext else 0)


def chain_log(u, lk):
    """The chain of sources along positions, in log space (the reference of chain, for the tests).

    u: [K, P, S] log unary potentials; lk: [K, P] log(exp(kappa) - 1) of the edge into each position (NEG: no
    coupling, as at a chain's first position or a restart). The pair potential is 1 + (exp(kappa) - 1) [s == s'].
    Returns the log posterior marginals [K, P, S] and log Z [K]. Forward and backward messages are renormalised at
    every position."""
    K, P, S = u.shape
    zero = torch.zeros(K, 1, dtype=u.dtype, device=u.device)
    a = u[:, 0]
    z = torch.logsumexp(a, dim=-1)
    a = a - z[:, None]
    logz, alphas = z, [a]
    for p in range(1, P):
        cur = u[:, p] + torch.logaddexp(lk[:, p, None] + a, zero)
        z = torch.logsumexp(cur, dim=-1)
        a = cur - z[:, None]
        logz = logz + z
        alphas.append(a)
    b = torch.zeros(K, S, dtype=u.dtype, device=u.device)
    betas = [b]
    for p in range(P - 1, 0, -1):
        t = u[:, p] + b
        b = torch.logaddexp(lk[:, p, None] + t, torch.logsumexp(t, dim=-1, keepdim=True))
        b = b - torch.logsumexp(b, dim=-1, keepdim=True)
        betas.append(b)
    post = torch.log_softmax(torch.stack(alphas, 1) + torch.stack(betas[::-1], 1), dim=-1)
    return post, logz


def chain(u, c):
    """The chain of sources along positions, in probability space with the messages renormalised at every position:
    about half the operations of the log-space recursion (chain_log), the same result.

    u: [K, P, S] log unary potentials (NEG: an unavailable state, probability exactly 0); c: [K, P] exp(kappa) - 1 >= 0
    of the edge into each position (0: no coupling, as at a chain's first position or a restart). The pair potential
    is 1 + c [s == s']. Returns the log posterior marginals [K, P, S] (NEG where 0) and log Z [K]."""
    K, P, S = u.shape
    top = u.max(dim=-1, keepdim=True).values
    phi = torch.exp(u - top)                                       # the largest state of every position is 1
    ones = torch.ones(K, S, dtype=u.dtype, device=u.device)
    c = c[..., None]
    a = phi[:, 0]
    z = a.sum(dim=-1, keepdim=True)
    a = a / z
    alphas, zs = [a], [z]
    for p in range(1, P):                                           # sum_s psi(s, s') a(s) = 1 + c a(s'), a summing to 1
        a = phi[:, p] * torch.addcmul(ones, c[:, p], a)
        z = a.sum(dim=-1, keepdim=True)
        a = a / z
        alphas.append(a)
        zs.append(z)
    b = ones / S
    betas = [b]
    for p in range(P - 1, 0, -1):
        t = phi[:, p] * b
        b = torch.addcmul(t.sum(dim=-1, keepdim=True), c[:, p], t)
        b = b / b.sum(dim=-1, keepdim=True)
        betas.append(b)
    joint = torch.stack(alphas, 1) * torch.stack(betas[::-1], 1)
    post = joint / joint.sum(dim=-1, keepdim=True)
    logpost = torch.where(post > 0, torch.log(post.clamp_min(1e-38)), torch.full_like(post, NEG))
    logz = torch.log(torch.cat(zs, dim=1)).sum(dim=1) + top[..., 0].sum(dim=1)
    return logpost, logz


def strengths(levels):
    """The copy strengths of the levels at the start, geometric from STRENGTH0[0] to STRENGTH0[1]."""
    lo, hi = STRENGTH0
    return [math.sqrt(lo * hi)] if levels == 1 else [lo * (hi / lo) ** (k / (levels - 1)) for k in range(levels)]


def log_expm1(k):
    """log(exp(k) - 1) for k > 0, stable at both ends."""
    return k + torch.log(-torch.expm1(-k))


def block_plan(slices=None):
    """The columns of every block padded to the longest: column [B, P] (the block's first column where padded) and
    real [B, P]."""
    slices = A.block_slices() if slices is None else slices
    P = max(e - s for _, s, e in slices)
    column = torch.zeros(len(slices), P, dtype=torch.long)
    real = torch.zeros(len(slices), P, dtype=torch.bool)
    for b, (_, s, e) in enumerate(slices):
        column[b, :e - s] = torch.arange(s, e)
        column[b, e - s:] = s
        real[b, :e - s] = True
    return column, real


class Source(nn.Module):
    def __init__(self, layers, width, arm, self_layers, levels=LEVELS, evidence=None, ext=False, slices=None):
        super().__init__()
        spec = spec_of(dict(layers=layers, width=width, arm=arm, self_layers=self_layers, levels=levels,
                            evidence=arm == "s1" if evidence is None else evidence, ext=ext))
        self.layers, self.width, self.arm = spec["layers"], spec["width"], spec["arm"]
        self.self_layers, self.levels, self.evidence = spec["self_layers"], spec["levels"], spec["evidence"]
        role, pos = AX.column_plan(slices)
        cos, sin = C.rotary_tables(pos, AX.ROW_HEAD // 4)
        column, real = block_plan(slices)
        self.register_buffer("role_of_column", role, persistent=False)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("block_column", column, persistent=False)
        self.register_buffer("block_real", real, persistent=False)
        self.token = nn.Embedding(A.VOCAB, width)
        self.role = nn.Embedding(4, width)
        self.blocks = nn.ModuleList([C.AxialBlock(width, AX.ROW_HEAD, AX.COLUMN_HEAD, AX.RATIO) for _ in range(layers)])
        self.norm_self, self.norm_full = nn.LayerNorm(width), nn.LayerNorm(width)
        L = self.levels
        self.own = nn.Linear(width, N_OUT)                       # a(h_self): the own-row logits
        self.strength = nn.Linear(width, L)                      # g_l(h_self) before softplus
        self.sub = nn.Parameter(torch.zeros(L, N_OUT, N_OUT))    # B_l[c, x]
        self.query = nn.Linear(width, L * KEY)
        self.key = nn.Linear(width, KEY)
        self.none = nn.Linear(width, 1)
        if self.arm == "s1":
            self.stay = nn.Linear(width, 1)
            if self.evidence:
                self.ev_table = nn.Parameter(torch.zeros(N_OUT, N_OUT))
                self.ev_none = nn.Parameter(torch.zeros(N_OUT))
        self.ext = nn.Embedding(1, width) if spec["ext"] else None
        self.version1 = False                                    # the paths of version 1 (reference_paths; the tests only)
        C.init_weights(self, self.blocks)
        with torch.no_grad():
            eye = torch.eye(N_OUT)
            self.sub.copy_(eye.expand(L, N_OUT, N_OUT))
            self.strength.bias.copy_(torch.tensor([math.log(math.expm1(g)) for g in strengths(L)]))
            if self.arm == "s1":
                self.stay.bias.fill_(math.log(math.expm1(STAY0)))
                if self.evidence:
                    self.ev_table.copy_(EVIDENCE0 * eye)

    # ---- the trunk ----
    def trunk(self, grid, sizes, ext=None):
        """(h_self, h_full) [N, C, D]: after the first self_layers blocks, every row a table of its own; after all."""
        if grid.shape[0] != sum(sizes) or grid.shape[1] != self.role_of_column.shape[0]:
            raise ValueError("grid %s does not fit %d rows of %d columns" % (tuple(grid.shape), sum(sizes), self.role_of_column.shape[0]))
        layout = C.table_layout(sizes, grid.device)
        keys = grid != A.PAD
        keys = keys | ~keys.any(dim=1, keepdim=True)
        x = self.token(grid) + self.role(self.role_of_column)[None]
        if ext is not None:
            if ext.shape != grid.shape:
                raise ValueError("ext %s does not fit the grid %s" % (tuple(ext.shape), tuple(grid.shape)))
            if self.ext is not None:
                x = x + ext.to(x.dtype)[..., None] * self.ext.weight[0]
            elif bool(ext.any()):
                raise ValueError("cells laid by extension, for a model built without ext (a stored-layout arm)")
        alone = C.table_layout([1] * int(grid.shape[0]), grid.device) if self.version1 else None
        h_self = None
        for k, block in enumerate(self.blocks):
            if k == self.self_layers:
                h_self = x
            if k >= self.self_layers:
                x = block(x, self.cos, self.sin, layout, keys)
            elif self.version1:                                  # version 1: the column attention run on lone rows
                x = block(x, self.cos, self.sin, alone, keys)
            else:
                x = self.alone_block(block, x, keys)
        return h_self, x

    def alone_block(self, block, x, keys):
        """An AxialBlock on rows that are tables of their own: the column attention of a lone row sees only its learned
        "no source" key (a row never attends to itself), so its output is out(no-source value) at every cell, exactly.
        That vector is added without running the attention (its query, key and value projections and its norm are not
        needed: their parameters stay, unused, so the block is the reference trunk's). The same as
        block(x, cos, sin, table_layout([1] * N), keys) (test_alone_block_is_the_block_on_lone_rows)."""
        x = x + block.row(block.norm_row(x), self.cos, self.sin, keys, None)
        col = block.column
        x = x + col.out(col.no_source_value.to(x.dtype))
        return x + block.ff_out(F.gelu(block.ff_in(block.norm_ff(x))))

    # ---- the pieces of the output ----
    def _tables(self, grid, sizes):
        index, valid, _ = C.table_layout(sizes, grid.device)
        tab = torch.repeat_interleave(torch.arange(len(sizes), device=grid.device), torch.as_tensor(sizes, device=grid.device))
        start = torch.cumsum(torch.as_tensor([0] + list(sizes[:-1]), device=grid.device), 0)
        local = torch.arange(grid.shape[0], device=grid.device) - start[tab]
        return index, valid, tab, local

    def pieces(self, grid, sizes, row, col, ext=None):
        """Everything the output is made of (float32), for the forward pass, the tests and the read-outs."""
        on_gpu = grid.device.type == "cuda"
        with torch.autocast(device_type=grid.device.type, dtype=torch.bfloat16, enabled=on_gpu):
            h_self, h_full = self.trunk(grid, sizes, ext)
        with torch.autocast(device_type=grid.device.type, enabled=False):
            return self._pieces(grid, sizes, row, col, h_self, h_full)

    def _pieces(self, grid, sizes, row, col, h_self, h_full):
        index, valid, tab, local = self._tables(grid, sizes)
        T, R = index.shape
        L = self.levels
        tok = grid[index]                                                          # [T, R, C]
        residue = (tok >= A.TARGET_MIN) & (tok <= A.TARGET_MAX) & valid[:, :, None]
        hf = self.norm_full(h_full.float())
        q = self.query(hf)[index].view(T, R, -1, L, KEY)                            # [T, R, C, L, KEY]
        k = self.key(hf)[index]                                                    # [T, R, C, KEY]
        none = self.none(hf)[index][..., 0]                                        # [T, R, C]
        score = torch.einsum("ticlk,tjck->ticjl", q, k) / math.sqrt(KEY)          # [T, R(i), C, R(j), L]
        other = valid[:, :, None] & valid[:, None, :] & ~torch.eye(R, dtype=torch.bool, device=grid.device)[None]
        avail = other[:, :, None, :] & residue.permute(0, 2, 1)[:, None]           # [T, R(i), C, R(j)]
        score = score.masked_fill(~avail[..., None], NEG)
        t, i = tab[row], local[row]
        hs = self.norm_self(h_self[row, col].float())
        own = self.own(hs)                                                         # [M, 65]
        g = F.softplus(self.strength(hs))                                          # [M, L]
        c = tok[t, :, col]                                                         # [M, R]: the codon of every other row
        cc = c.clamp(A.TARGET_MIN, A.TARGET_MAX) - A.TARGET_MIN
        tilt = self.sub[:, cc].permute(1, 2, 0, 3)                                 # [M, R, L, 65]
        sub = torch.log_softmax(own[:, None, None, :] + g[:, None, :, None] * tilt, dim=-1)
        out = dict(index=index, valid=valid, tab=tab, local=local, t=t, i=i, col=col, tok=tok, avail=avail, score=score, none=none,
                   own=torch.log_softmax(own, dim=-1), sub=sub, strength=g, cell_avail=avail[t, i, col], cell_score=score[t, i, col],
                   cell_none=none[t, i, col])
        if self.arm == "s1":
            unary = torch.cat([torch.logsumexp(score, dim=-1), none[..., None]], dim=-1)          # [T, R, C, R + 1]
            if self.evidence:
                x = tok.clamp(A.TARGET_MIN, A.TARGET_MAX) - A.TARGET_MIN                            # the row's own token
                src = tok.clamp(A.TARGET_MIN, A.TARGET_MAX) - A.TARGET_MIN                          # every other row's token
                table = torch.log_softmax(self.ev_table, dim=-1)                                   # [c, x]
                ev = table[src[:, None, :, :].expand(T, R, R, -1).transpose(2, 3), x[:, :, :, None].expand(T, R, -1, R)]
                ev0 = torch.log_softmax(self.ev_none, dim=-1)[x]
                seen = residue[..., None]
                unary = unary + torch.cat([ev, ev0[..., None]], dim=-1) * seen
            unary = torch.cat([unary[..., :R].masked_fill(~avail, NEG), unary[..., R:]], dim=-1)
            kappa = F.softplus(self.stay(hf)[index][..., 0])                                       # [T, R, C]
            out.update(unary=unary, kappa=kappa)
        return out

    def chains(self, p, extra=None):
        """S1: the log posterior of every cell's source [T, R, C, R + 1] and log Z of every (row, block) chain
        [T, R, B]; extra [T, R, C, R + 1] (log emissions of the true codons, read-outs only) is added to the unary."""
        u = p["unary"] if extra is None else p["unary"] + extra
        T, R, Cn, S = u.shape
        col, real = self.block_column, self.block_real                                            # [B, P]
        B, P = col.shape
        up = u[:, :, col]                                                                          # [T, R, B, P, S]
        up = torch.where(real[None, None, :, :, None], up, torch.zeros((), dtype=up.dtype, device=up.device))
        first = torch.zeros_like(real)
        first[:, 0] = True
        if self.version1:                                        # version 1: the log-space recursion
            lk = log_expm1(p["kappa"][:, :, col].clamp(1e-6, KAPPA_MAX)).masked_fill((~real | first)[None, None], NEG)
            post, logz = chain_log(up.reshape(T * R * B, P, S), lk.reshape(T * R * B, P))
        else:
            cc = torch.expm1(p["kappa"][:, :, col].clamp(max=KAPPA_MAX))
            cc = cc.masked_fill((~real | first)[None, None], 0.0)
            post, logz = chain(up.reshape(T * R * B, P, S), cc.reshape(T * R * B, P))
        post = post.view(T, R, B, P, S)
        cell = torch.zeros(T, R, Cn, S, dtype=post.dtype, device=post.device)
        cell[:, :, col[real]] = post[:, :, real]
        return cell, logz.view(T, R, B)

    def weights(self, p):
        """log weights of the components at every target cell: [M, R, L] for (row, level) and [M] for no source."""
        if self.arm == "s2":
            M, R, L = p["cell_score"].shape
            w = torch.log_softmax(torch.cat([p["cell_score"].reshape(M, R * L), p["cell_none"][:, None]], dim=1), dim=1)
            return w[:, :R * L].view(M, R, L), w[:, R * L]
        post, _ = self.chains(p)
        cell = post[p["t"], p["i"], p["col"]]
        R = p["cell_score"].shape[1]
        level = torch.log_softmax(p["cell_score"], dim=-1)
        return cell[:, :R, None] + level, cell[:, R]

    @staticmethod
    def mix(logw, logw_none, sub, own):
        """log p [M, 65]: logsumexp over components of log weight + log component."""
        M, R, L, X = sub.shape
        lw = torch.cat([logw.reshape(M, R * L), logw_none[:, None]], dim=1)
        comp = torch.cat([sub.reshape(M, R * L, X), own[:, None]], dim=1)
        return torch.logsumexp(lw[:, :, None] + comp, dim=1)

    def forward(self, grid, sizes, row, col, ext=None):
        """log-probabilities float32 [M, 65] over the residues 1..65 for the cells (row[m], col[m]), normalised."""
        p = self.pieces(grid, sizes, row, col, ext)
        logw, logw_none = self.weights(p)
        return self.mix(logw, logw_none, p["sub"], p["own"])

    def logp(self, grid, sizes, row, col, ext=None):
        return torch.log_softmax(self.forward(grid, sizes, row, col, ext), dim=-1)


class reference_paths:
    """with reference_paths(model): the model computes by the paths of version 1 (the own-row blocks run their column
    attention on lone rows; S1's chain in log space). For the tests of version 2 against version 1 only."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.model.version1 = True
        return self.model

    def __exit__(self, *exc):
        self.model.version1 = False
        return False


def logp(model, grid, sizes, row, col, ext=None):
    """The contract's reading of an arm: for every target cell a normalised distribution over the tokens 1..65."""
    return model.logp(grid, sizes, row, col, ext)


@torch.no_grad()
def readout(model, grid, sizes, row, col, ext=None, token=None):
    """The source read-out of an arm at the target cells (row, col), in eval mode.

    Returns a dict of tensors: rows [M, R] (the global row of every source column, -1 where the table has fewer rows),
    weights [M, R + 1] (the weight of every other row and, last, of no source; summed over levels), lam [M] (1 - the
    weight of no source). S1 adds path [N, C, R + 1] (the posterior of the source of every cell of every row, given the
    grid; columns as in rows of the cell's table) and, when token [M] (the true codons 1..65) is given, given_truth
    [M, R + 1] (the posterior given the grid and every target cell's true codon) and joint_nats (minus the log
    probability of all target codons together under the chain, the forward algorithm; a read-out, never compared with
    an independent arm's bits)."""
    was = model.training
    model.eval()
    p = model.pieces(grid, sizes, row, col, ext)
    logw, logw_none = model.weights(p)
    index, valid, t = p["index"], p["valid"], p["t"]
    rows = torch.where(valid, index, torch.full_like(index, -1))[t]
    weights = torch.cat([torch.logsumexp(logw, dim=-1), logw_none[:, None]], dim=1).exp()
    out = dict(rows=rows, weights=weights, lam=1.0 - logw_none.exp())
    if model.arm == "s1":
        post, logz = model.chains(p)
        T, R = index.shape
        N = grid.shape[0]
        path = torch.zeros(N, grid.shape[1], R + 1, dtype=post.dtype, device=post.device)
        path[index[valid]] = post[valid].exp()
        out["path"] = path
        if token is not None:
            x = (token.long() - A.TARGET_MIN).clamp(0, N_OUT - 1)
            level = torch.log_softmax(p["cell_score"], dim=-1)
            em_rows = torch.logsumexp(level + p["sub"].gather(3, x[:, None, None, None].expand(-1, R, model.levels, 1))[..., 0], dim=-1)
            em = torch.cat([em_rows, p["own"].gather(1, x[:, None])], dim=1)                        # [M, R + 1]
            extra = torch.zeros_like(p["unary"])
            extra[p["t"], p["i"], col] = em
            post2, logz2 = model.chains(p, extra)
            out["given_truth"] = post2[p["t"], p["i"], col].exp()
            out["joint_nats"] = -(logz2 - logz).sum()
    model.train(was)
    return out


def build(size):
    s = spec_of(size)
    return Source(s["layers"], s["width"], s["arm"], s["self_layers"], levels=s["levels"], evidence=s["evidence"], ext=s["ext"])


def unused_parameters(width, self_layers):
    """The parameters version 2 never uses: in each own-row block the column attention's query-key-value projection,
    its norm and its "no source" key (its output is out(no-source value), alone_block)."""
    return int(self_layers) * (3 * width * width + 3 * width + 2 * width + width)


def flops_per_cell(params, layers, width, self_layers, rows):
    """The planning formula of the contract (api.flops_per_cell: 6 x parameters + 12 x layers x width x attention
    length) for this trunk: the parameters it uses, the row attention over the row in every block, the column attention
    over the rows in the blocks across the table only."""
    used = params - unused_parameters(width, self_layers)
    return 6 * used + 12 * width * (layers * A.ROW_LEN + (layers - self_layers) * int(rows))


def measured_flops(model, rows=32, share=0.15, seed=0):
    """The calibration of the analytic count (protocol round1, batch 3): the floating-point operations PyTorch's
    FlopCounterMode counts for one table of `rows` rows (a random grid, `share` of its cells hidden and scored),
    forward and backward through the cross-entropy, on the CPU. The counter counts matrix products only (not the
    attention kernels, not embedding look-ups), so it is below the analytic count; the ratio of two arms is what is
    reported beside it. Any model of the registry with the forward of models/axial.py; returns dict(total, by_op)."""
    from torch.utils.flop_counter import FlopCounterMode
    g = torch.Generator().manual_seed(int(seed))
    grid = torch.randint(A.TARGET_MIN, A.TARGET_MAX + 1, (int(rows), A.ROW_LEN), generator=g)
    m = int(int(rows) * A.ROW_LEN * float(share))
    row, col = torch.randint(0, int(rows), (m,), generator=g), torch.randint(0, A.ROW_LEN, (m,), generator=g)
    grid[row, col] = A.MASK
    token = torch.randint(0, N_OUT, (m,), generator=g)
    model = model.cpu().train()
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fc:
        F.cross_entropy(model(grid, [int(rows)], row, col), token).backward()
    model.zero_grad(set_to_none=True)
    return dict(total=int(fc.get_total_flops()), by_op={str(k): int(v) for k, v in fc.get_flop_counts().get("Global", {}).items()})


def info(size, rows=16):
    s = spec_of(size)
    params = count_parameters(s["layers"], s["width"], s["arm"], s["levels"], s["evidence"], s["ext"])
    return dict(module=MODULE, params=params, params_used=params - unused_parameters(s["width"], s["self_layers"]), layers=s["layers"],
                width=s["width"], arm=s["arm"], self_layers=s["self_layers"], levels=s["levels"], evidence=s["evidence"], ext=s["ext"],
                head="source-" + s["arm"], flops_per_cell=flops_per_cell(params, s["layers"], s["width"], s["self_layers"], rows),
                factorisation="independent", layout="any", common_version=C.COMMON_VERSION, source_version=SOURCE_VERSION)


# ---- self-tests (CPU, small widths; called by the trainer's selftest mode in the model interpreter) ----

SPECIALS = (A.GAP, A.PAD, A.MASK, A.ABSENT, A.NOT_SEARCHED, A.TRUNCATED, A.EXT_EMPTY)


def _small(arm, seed=0, **kw):
    torch.manual_seed(seed)
    return Source(2, 256, arm, 1, **kw).eval()


def _grid(rows, seed, specials=0.2):
    """Random residues with a share of cells that are no residue (gaps, padding, masks, markers)."""
    g = torch.Generator().manual_seed(seed)
    grid = torch.randint(A.TARGET_MIN, A.TARGET_MAX + 1, (rows, A.ROW_LEN), generator=g)
    pick = torch.rand(rows, A.ROW_LEN, generator=g) < specials
    kind = torch.tensor(SPECIALS)[torch.randint(0, len(SPECIALS), (rows, A.ROW_LEN), generator=g)]
    return torch.where(pick, kind, grid)


def _cells(n, rows, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, rows, (n,), generator=g), torch.randint(0, A.ROW_LEN, (n,), generator=g)


def test_parameter_count():
    for arm in ARMS:
        for ext in (False, True):
            for evidence in ((True, False) if arm == "s1" else (False,)):
                m = Source(2, 256, arm, 1, evidence=evidence, ext=ext)
                n = sum(p.numel() for p in m.parameters())
                assert n == count_parameters(2, 256, arm, evidence=evidence, ext=ext), (arm, ext, evidence, n)
    for name, spec in SIZES.items():
        i = info(name)
        assert i["params"] == count_parameters(spec["layers"], spec["width"], spec["arm"]) and i["factorisation"] == "independent"
        ref = AX.count_parameters(spec["layers"], spec["width"])
        assert abs(i["params"] - ref) < 0.03 * ref, (name, i["params"], ref)                  # the reference trunk's size
    assert info(dict(SIZES["s1-30m"], ext=True))["params"] == count_parameters(12, 384, "s1") + 384
    for arm in ARMS:                                                              # the parameters used: those a backward reaches
        torch.manual_seed(43)
        model = Source(3, 256, arm, 2)
        grid = _grid(5, 44)
        model(grid, [3, 2], *_cells(50, 5, 45)).logsumexp(-1).sum().backward()
        reached = sum(p.numel() for p in model.parameters() if p.grad is not None)
        i = info(dict(layers=3, width=256, arm=arm, self_layers=2))
        assert reached == i["params_used"] == i["params"] - unused_parameters(256, 2), (arm, reached, i["params_used"])
    for P_, L_, W_, R_ in ((28463681, 12, 384, 32), (1000, 2, 256, 1)):           # the contract's formula, the reference arm's:
        assert flops_per_cell(P_, L_, W_, 0, R_) == A.flops_per_cell(P_, L_, W_, A.ROW_LEN + R_)   # with no own-row block, equal
    ref = AX.info("30m", rows=32)["flops_per_cell"]
    assert ref == A.flops_per_cell(AX.info("30m")["params"], 12, 384, A.ROW_LEN + 32)
    for name in ("s1-30m", "s2-30m"):
        f = info(name, rows=32)["flops_per_cell"]
        assert 0.9 * ref < f < ref, (name, f, ref)                               # a little less than the reference trunk's


def test_chain_against_enumeration():
    """Forward and backward in log space give the posterior marginals and log Z of enumerating every path, with an
    unavailable state, a restart and positions of equal potentials."""
    import itertools
    torch.manual_seed(3)
    K, P, S = 3, 5, 3
    u = torch.randn(K, P, S, dtype=torch.float64) * 2
    u[0, 2, 1] = NEG
    u[2, 3:] = 0.0
    lk = log_expm1(torch.rand(K, P, dtype=torch.float64) * 4 + 0.1)
    lk[:, 0] = NEG
    lk[1, 3] = NEG
    post, logz = chain(u, torch.exp(lk))
    post_l, logz_l = chain_log(u, lk)
    assert torch.allclose(post.exp(), post_l.exp(), atol=1e-12) and torch.allclose(logz, logz_l, atol=1e-9)
    for k in range(K):
        terms, paths = [], list(itertools.product(range(S), repeat=P))
        for path in paths:
            w = sum(float(u[k, p, path[p]]) for p in range(P))
            w += sum(math.log1p(math.exp(float(lk[k, p]))) for p in range(1, P) if path[p] == path[p - 1])
            terms.append(w)
        z = torch.logsumexp(torch.tensor(terms, dtype=torch.float64), 0)
        assert abs(float(z - logz[k])) < 1e-9, (k, float(z), float(logz[k]))
        marg = torch.zeros(P, S, dtype=torch.float64)
        for path, w in zip(paths, terms):
            for p in range(P):
                marg[p, path[p]] += math.exp(w - float(z))
        assert torch.allclose(marg, post[k].exp(), atol=1e-9), (k, marg, post[k].exp())
    assert float(post[0, 2, 1].exp()) == 0.0 and float(post_l[0, 2, 1].exp()) < 1e-300
    u32 = torch.randn(64, 256, 33) * 3                                           # long chains in float32, a strong stay bonus:
    u32[:, :, 5] = NEG                                                           # the two recursions agree, gradients finite
    lk32 = log_expm1(torch.full((64, 256), KAPPA_MAX))
    lk32[:, 0] = NEG
    u32.requires_grad_(True)
    a, za = chain(u32, torch.exp(lk32))
    b, zb = chain_log(u32, lk32)
    assert torch.allclose(a.exp(), b.exp(), atol=1e-4) and torch.allclose(za, zb, rtol=1e-4)
    (a.exp()[..., :4].sum() + za.sum()).backward()
    assert torch.isfinite(u32.grad).all() and (u32.grad[:, :, 5] == 0).all()


def test_version_2_is_version_1():
    """Version 2 (the own-row blocks without their column attention, S1's chain in probability space) gives what version
    1 gives, in fp32: the distributions, the loss and the gradient of every parameter (zero in version 1 where version 2
    leaves a parameter unused: the own-row blocks' column projections and norms)."""
    for arm in ARMS:
        for ext in (False, True):
            torch.manual_seed(37)
            model = Source(3, 256, arm, 2, ext=ext)
            grid = _grid(9, 38, specials=0.3)
            sizes = [5, 1, 3]
            row, col = _cells(400, 9, 39)
            grid[row, col] = A.MASK
            mark = (torch.rand(grid.shape, generator=torch.Generator().manual_seed(40)) < 0.1) if ext else None
            token = torch.randint(0, 65, (400,), generator=torch.Generator().manual_seed(41))
            out = {}
            for version in (1, 2):
                model.zero_grad(set_to_none=True)
                if version == 1:
                    with reference_paths(model):
                        lp = model(grid, sizes, row, col, mark)
                else:
                    lp = model(grid, sizes, row, col, mark)
                loss = F.cross_entropy(lp, token)
                loss.backward()
                grads = {k: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for k, p in model.named_parameters()}
                out[version] = (lp.detach(), loss.item(), grads)
            (lp1, l1, g1), (lp2, l2, g2) = out[1], out[2]
            assert torch.allclose(lp1.exp(), lp2.exp(), atol=1e-6) and abs(l1 - l2) < 1e-5, (arm, ext, float((lp1 - lp2).abs().max()), l1, l2)
            for k in g1:
                scale = float(g1[k].abs().max()) + 1e-8
                assert torch.allclose(g1[k], g2[k], atol=2e-5 * max(1.0, scale), rtol=1e-3), (arm, ext, k, float((g1[k] - g2[k]).abs().max()), scale)
            assert not model.version1


def test_chain_float32_long_sequences_do_not_underflow():
    """The chain in probability space, fp32, renormalised at every position with its log scale summed: on 512
    positions of 33 states with potentials spread over +-60 nats, unavailable states, restarts and stay bonuses from 0
    to the cap, its posteriors and log Z are those of the log-space recursion in float64 (no posterior that is not
    negligible lost to underflow; log Z finite)."""
    g = torch.Generator().manual_seed(42)
    K, P, S = 16, 512, 33
    u = torch.randn(K, P, S, generator=g, dtype=torch.float64) * 20
    u[torch.rand(K, P, S, generator=g) < 0.3] = NEG
    u[:, :, -1] = torch.randn(K, P, generator=g, dtype=torch.float64) * 20          # one state always available
    kappa = torch.rand(K, P, generator=g, dtype=torch.float64) * KAPPA_MAX
    kappa[:, ::97] = 0.0                                                           # restarts
    c = torch.expm1(kappa)
    lk = torch.where(kappa > 0, log_expm1(kappa.clamp_min(1e-12)), torch.full_like(kappa, NEG))
    ref_post, ref_z = chain_log(u, lk)                                             # float64, log space
    post, z = chain(u.float(), c.float())
    assert torch.isfinite(z).all() and torch.allclose(z.double(), ref_z, rtol=1e-5), float((z.double() - ref_z).abs().max())
    p, rp = post.double().exp(), ref_post.exp()
    assert torch.allclose(p.sum(-1), torch.ones(K, P, dtype=torch.float64), atol=1e-5)
    assert torch.allclose(p, rp, atol=1e-4), float((p - rp).abs().max())
    assert not ((rp > 1e-6) & (p == 0)).any()                                      # nothing that counts underflowed


def test_alone_block_is_the_block_on_lone_rows():
    """The own-row blocks add the column attention's output without running it: the same as the reference block run
    with every row a table of its own, and h_self of the trunk the same as the attention path gives."""
    model = _small("s2", seed=35)
    grid = _grid(6, 36)
    keys = grid != A.PAD
    alone = C.table_layout([1] * 6, grid.device)
    with torch.no_grad():
        x = model.token(grid) + model.role(model.role_of_column)[None]
        for block in model.blocks:
            want = block(x, model.cos, model.sin, alone, keys)
            got = model.alone_block(block, x, keys)
            assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())
            x = want


def test_normalised_and_finite():
    """Every target cell gets a normalised distribution over the residues 1..65, finite, with finite gradients, for
    grids full of cells that are no residue (and a row of PAD only)."""
    for arm in ARMS:
        model = _small(arm)
        model.train()
        grid = _grid(9, 1, specials=0.4)
        grid[4] = A.PAD
        grid[4, 7] = 12
        sizes = [4, 1, 4]
        row, col = _cells(400, 9, 2)
        lp = model(grid, sizes, row, col)
        assert torch.isfinite(lp).all() and torch.allclose(lp.exp().sum(-1), torch.ones(len(row)), atol=1e-5), arm
        assert torch.allclose(model.logp(grid, sizes, row, col), lp, atol=1e-5)
        lp[:, 3].sum().backward()
        bad = [k for k, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, (arm, bad)


def test_self_branch_sees_no_other_row():
    """The own-row branch (p_self, the strength of every level of the substitution, h_self) is the same whatever the
    other rows of the table hold: its attention never reaches them. The prediction itself does change."""
    for arm in ARMS:
        model = _small(arm, seed=4)
        grid = _grid(7, 5)
        sizes = [4, 3]
        row, col = torch.zeros(300, dtype=torch.long), torch.randint(0, A.ROW_LEN, (300,), generator=torch.Generator().manual_seed(6))
        changed = grid.clone()
        changed[1:4] = _grid(3, 7, specials=0.5)
        changed[4:] = _grid(3, 8)
        with torch.no_grad():
            a, b = model.pieces(grid, sizes, row, col), model.pieces(changed, sizes, row, col)
            h_a, _ = model.trunk(grid, sizes)
            h_b, _ = model.trunk(changed, sizes)
            lp_a, lp_b = model(grid, sizes, row, col), model(changed, sizes, row, col)
        assert torch.allclose(h_a[0], h_b[0], atol=1e-6), float((h_a[0] - h_b[0]).abs().max())
        assert torch.allclose(a["own"], b["own"], atol=1e-6) and torch.allclose(a["strength"], b["strength"], atol=1e-6), arm
        assert not torch.allclose(lp_a, lp_b, atol=1e-4), arm


def test_other_rows_only_through_copying():
    """Where no other row of the table holds a residue in the target's column, the prediction is p_self exactly, and
    stays so whatever else the other rows hold: other rows reach a prediction only through the codons copied."""
    for arm in ARMS:
        model = _small(arm, seed=8)
        grid = _grid(5, 9)
        sizes = [5]
        col = torch.tensor([3, 40, 300, 700, 900, 1000])
        row = torch.zeros(len(col), dtype=torch.long)
        for r, kind in zip(range(1, 5), (A.GAP, A.MASK, A.PAD, A.EXT_EMPTY)):
            grid[r, col] = kind
        changed = grid.clone()
        keep = torch.zeros(A.ROW_LEN, dtype=torch.bool)
        keep[col] = True
        changed[1:, ~keep] = _grid(4, 10)[:, ~keep]
        with torch.no_grad():
            lp, own = model(grid, sizes, row, col), model.pieces(grid, sizes, row, col)["own"]
            lp2 = model(changed, sizes, row, col)
        assert torch.allclose(lp, own, atol=1e-6) and torch.allclose(lp2, own, atol=1e-6), (arm, float((lp - own).abs().max()))


def test_sources_are_residues_only():
    """A source has a residue 1..65 in the target's column and is another row of the same table; GAP, PAD, MASK,
    ABSENT, NOT_SEARCHED, TRUNCATED and EXT_EMPTY cells never carry weight; no source always does."""
    for arm in ARMS:
        model = _small(arm, seed=11)
        grid = _grid(9, 12, specials=0.5)
        sizes = [6, 3]
        row, col = _cells(500, 9, 13)
        r = readout(model, grid, sizes, row, col, token=torch.randint(1, 66, (500,)) if arm == "s1" else None)
        rows, w = r["rows"], r["weights"]
        assert torch.allclose(w.sum(-1), torch.ones(500, dtype=w.dtype), atol=1e-5) and (w[:, -1] > 0).all()
        src = grid[rows.clamp_min(0), col[:, None]]
        ok = (rows >= 0) & (rows != row[:, None]) & (src >= A.TARGET_MIN) & (src <= A.TARGET_MAX)
        assert (w[:, :-1][~ok] == 0).all() and (w[:, :-1][ok] > 0).all(), arm
        same = rows[torch.arange(500), torch.argmax(w[:, :-1], dim=1)]
        assert (same // 6 == row // 6)[ok.any(1)].all()                                   # sources stay inside the table


def test_rows_permute():
    """Permuting the rows of a table permutes the output: no row index anywhere."""
    for arm in ARMS:
        model = _small(arm, seed=14)
        grid = _grid(8, 15)
        sizes = [5, 3]
        perm = torch.tensor([3, 0, 4, 1, 2, 7, 5, 6])
        inv = torch.argsort(perm)
        row, col = _cells(300, 8, 16)
        with torch.no_grad():
            a = model(grid, sizes, row, col)
            b = model(grid[perm], sizes, inv[row], col)
        assert torch.allclose(a, b, atol=2e-4), (arm, float((a - b).abs().max()))


def test_tables_do_not_mix():
    for arm in ARMS:
        model = _small(arm, seed=17)
        parts = [_grid(4, 18), _grid(1, 19), _grid(6, 20)]
        row, col = _cells(200, 11, 21)
        with torch.no_grad():
            both = model(torch.cat(parts), [4, 1, 6], row, col)
            alone = torch.empty_like(both)
            at = 0
            for g in parts:
                m = (row >= at) & (row < at + g.shape[0])
                alone[m] = model(g, [g.shape[0]], row[m] - at, col[m])
                at += g.shape[0]
        assert torch.allclose(alone, both, atol=2e-4), (arm, float((alone - both).abs().max()))


def test_lone_row():
    """A table of one row: the prediction is p_self and lambda is 0."""
    for arm in ARMS:
        model = _small(arm, seed=22)
        grid = _grid(1, 23)
        row, col = torch.zeros(100, dtype=torch.long), torch.arange(0, 1000, 10)
        r = readout(model, grid, [1], row, col)
        with torch.no_grad():
            lp, own = model(grid, [1], row, col), model.pieces(grid, [1], row, col)["own"]
        assert torch.allclose(lp, own, atol=1e-6) and (r["lam"].abs() < 1e-6).all(), arm


def test_hidden_content_does_not_leak():
    """Through the reader: changing the residues of hidden cells changes neither the grid the model reads nor any of
    its outputs, while changing a visible cell does."""
    import numpy as np
    from seeddrift.train.rel.data import tables as T
    rng = np.random.default_rng(24)
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
    hide[0, T.T0:T.T1] = True
    grid, truth = T.render(tab, cells & ~hide, hide)[:2]
    other = dict(tab, tok=np.where(hide, (tok % 64) + 1, tok).astype(np.uint8))
    grid2 = T.render(other, cells & ~hide, hide)[0]
    seen = dict(tab, tok=np.where(~hide & (T.POS[None, :] == 3), (tok % 64) + 1, tok).astype(np.uint8))
    grid3 = T.render(seen, cells & ~hide, hide)[0]
    assert (grid == grid2).all() and (grid[hide] == A.MASK).all()
    r, c = (torch.from_numpy(x) for x in np.nonzero(truth))
    for arm in ARMS:
        model = _small(arm, seed=25)
        with torch.no_grad():
            a, b, d = (model(torch.from_numpy(g.astype(np.int64)), [rows], r, c) for g in (grid, grid2, grid3))
        assert torch.equal(a, b) and not torch.allclose(a, d, atol=1e-5), arm


def test_s1_without_stay_and_evidence_is_s2():
    """S1 with no stay bonus and no evidence term is S2 (the same weights): the chain then factorises by position."""
    s2 = _small("s2", seed=26)
    s1 = Source(2, 256, "s1", 1, evidence=False).eval()
    missing, unexpected = s1.load_state_dict(s2.state_dict(), strict=False)
    assert sorted(missing) == ["stay.bias", "stay.weight"] and not unexpected
    with torch.no_grad():
        s1.stay.weight.zero_()
        s1.stay.bias.fill_(-40.0)                                                         # kappa ~ 0: no coupling
    grid = _grid(6, 27)
    row, col = _cells(300, 6, 28)
    with torch.no_grad():
        a, b = s1(grid, [6], row, col), s2(grid, [6], row, col)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


def test_s1_readouts_agree_with_the_prediction():
    """With one target cell per chain the joint bits are the cells' own, the posterior given the true codon is the
    prior weights times the emission, and the source path of every cell sums to one."""
    model = _small("s1", seed=29)
    grid = _grid(6, 30)
    sizes = [4, 2]
    slices = A.block_slices()
    row = torch.tensor([0, 1, 2, 4, 0, 5])
    col = torch.tensor([slices[-1][1] + 7, slices[-1][1] + 100, slices[3][1] + 4, slices[-2][1] + 9, slices[0][1] + 1, slices[-3][1] + 50])
    grid[row, col] = A.MASK
    token = torch.tensor([5, 64, 1, 33, 65, 20])
    with torch.no_grad():
        lp = model(grid, sizes, row, col)
        r = readout(model, grid, sizes, row, col, token=token)
        p = model.pieces(grid, sizes, row, col)
    assert abs(float(r["joint_nats"]) + float(lp.gather(1, (token - 1)[:, None]).sum())) < 1e-3
    level = torch.log_softmax(p["cell_score"], dim=-1)
    x = (token - 1)[:, None, None, None].expand(-1, p["sub"].shape[1], model.levels, 1)
    em = torch.cat([torch.logsumexp(level + p["sub"].gather(3, x)[..., 0], dim=-1), p["own"].gather(1, (token - 1)[:, None])], dim=1).exp()
    want = r["weights"] * em
    assert torch.allclose(r["given_truth"], want / want.sum(-1, keepdim=True), atol=1e-5)
    real = grid != A.PAD
    assert torch.allclose(r["path"].sum(-1)[real], torch.ones(int(real.sum())), atol=1e-5)
    assert torch.allclose(r["path"][row, col], r["weights"], atol=1e-5)


def test_backward_and_start():
    for arm in ARMS:
        torch.manual_seed(31)
        model = Source(2, 256, arm, 1)
        grid = _grid(5, 32)
        row, col = torch.tensor([0, 2, 4]), torch.tensor([5, 800, 1023])
        loss = F.cross_entropy(model(grid, [3, 2], row, col), torch.tensor([0, 10, 64]))
        loss.backward()
        unused = ["blocks.0.norm_column.weight", "blocks.0.norm_column.bias", "blocks.0.column.qkv.weight", "blocks.0.column.qkv.bias",
                  "blocks.0.column.no_source_key"]                         # the own-row block's column attention is not run
        missing = [k for k, p in model.named_parameters() if (p.grad is None) != (k in unused) or (p.grad is not None and not torch.isfinite(p.grad).all())]
        assert not missing, (arm, missing[:5])
        v = loss.item()
        assert 3.0 < v < 6.0, (arm, v)                                                    # near ln 65 at the start


def test_masked_blocks_stay_finite():
    """A one-row table whose first 600 cells are PAD: outputs and gradients stay finite."""
    for arm in ARMS:
        torch.manual_seed(33)
        model = Source(1 + 1, 256, arm, 1)
        grid = _grid(1, 34, specials=0.0)
        grid[0, :600] = A.PAD
        row, col = torch.zeros(200, dtype=torch.long), torch.arange(800, 1000)
        out = model(grid, [1], row, col)
        loss = F.cross_entropy(out, torch.zeros(200, dtype=torch.long))
        loss.backward()
        assert torch.isfinite(out).all() and torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_measured_flops_calibrate_the_analytic_count():
    """The calibration counts the matrix products of one table, forward and backward: positive, below the analytic
    count (which also counts attention and embeddings), and lower for a source arm than for the reference trunk of the
    same layers and width, as the analytic count is."""
    torch.manual_seed(46)
    ref, src = AX.Axial(2, 256), Source(2, 256, "s2", 1)
    a, b = measured_flops(ref, rows=4), measured_flops(src, rows=4)
    assert 0 < b["total"] < a["total"] and "aten.mm" in a["by_op"]
    assert a["total"] < 4 * A.ROW_LEN * AX.info(dict(layers=2, width=256), rows=4)["flops_per_cell"]
    assert b["total"] < 4 * A.ROW_LEN * info(dict(layers=2, width=256, arm="s2", self_layers=1), rows=4)["flops_per_cell"]


def test_sizes_are_checked():
    for bad in (dict(layers=2, width=256, arm="s3", self_layers=1), dict(layers=2, width=256, arm="s1", self_layers=2),
                dict(layers=2, width=256, arm="s2", self_layers=1, evidence=True), dict(layers=2, width=256, arm="s1", self_layers=1, gate=True)):
        try:
            spec_of(bad)
        except ValueError:
            continue
        raise AssertionError("accepted %r" % (bad,))
