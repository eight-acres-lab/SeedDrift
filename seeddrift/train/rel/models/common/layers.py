"""Layers shared by the model modules of the relational line. COMMON_VERSION goes into every result row; a layer a
round has used is not changed afterwards (new behaviour is a switch or a new module).

  RowAttention      attention along a row over all its columns; rotary position on half of every head's dimensions
  ColumnAttention   attention among the rows of one table at the same column, with a learned "no source" key
  AxialBlock        row attention, column attention, feed-forward (pre-norm, residual)
  table_layout      which rows of a micro-batch belong to which table, padded for the column attention

Attention is torch's scaled_dot_product_attention. Nothing here knows a row's index: rows are a set.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COMMON_VERSION = "1"


def rotary_tables(positions, pairs, base=10000.0):
    """cos and sin [C, pairs] of the rotary angles of integer positions [C]."""
    freq = base ** (-torch.arange(pairs, dtype=torch.float64) / pairs)
    angle = positions.to(torch.float64)[:, None] * freq[None, :]
    return angle.cos().float(), angle.sin().float()


def rotate(x, cos, sin):
    """x [..., C, head]: the first 2 * pairs dimensions rotated by the position of the column (pair i is dimensions i
    and pairs + i), the rest left as they are."""
    m = cos.shape[-1]
    a, b = x[..., :m], x[..., m:2 * m]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([a * cos - b * sin, a * sin + b * cos, x[..., 2 * m:]], dim=-1)


def table_layout(sizes, device):
    """sizes: rows of each table of a micro-batch, whose N = sum(sizes) rows are stacked table after table.
    Returns index [T, R] (row of N at each place, 0 where padded), valid [T, R], and the attention mask [T, R, R + 1]
    of the column attention: a row sees the other rows of its table and the "no source" key (the last), never itself."""
    T, R = len(sizes), max(sizes)
    index = torch.zeros(T, R, dtype=torch.long)
    valid = torch.zeros(T, R, dtype=torch.bool)
    at = 0
    for t, k in enumerate(sizes):
        index[t, :k] = torch.arange(at, at + k)
        valid[t, :k] = True
        at += k
    mask = valid[:, None, :] & ~torch.eye(R, dtype=torch.bool)[None]
    mask = torch.cat([mask, torch.ones(T, R, 1, dtype=torch.bool)], dim=2)
    return index.to(device), valid.to(device), mask.to(device)


class RowAttention(nn.Module):
    def __init__(self, width, head=64):
        super().__init__()
        if width % head or head % 4:
            raise ValueError("width %d is not a multiple of the row head %d (itself a multiple of 4)" % (width, head))
        self.heads, self.head = width // head, head
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, cos, sin):
        N, C, d = x.shape
        q, k, v = self.qkv(x).view(N, C, 3, self.heads, self.head).permute(2, 0, 3, 1, 4)          # each [N, heads, C, head]
        y = F.scaled_dot_product_attention(rotate(q, cos, sin), rotate(k, cos, sin), v)
        return self.out(y.transpose(1, 2).reshape(N, C, d))


class ColumnAttention(nn.Module):
    def __init__(self, width, head=128):
        super().__init__()
        if width % head or width // head < 2:
            raise ValueError("width %d does not give at least two column heads of %d" % (width, head))
        self.heads, self.head = width // head, head
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.no_source_key = nn.Parameter(torch.zeros(width))
        self.no_source_value = nn.Parameter(torch.zeros(width))

    def forward(self, x, layout):
        index, valid, mask = layout
        N, C, d = x.shape
        T, R = index.shape
        H, h = self.heads, self.head
        q, k, v = self.qkv(x)[index].view(T, R, C, 3, H, h).permute(3, 0, 2, 4, 1, 5).reshape(3, T * C, H, R, h)
        k = torch.cat([k, self.no_source_key.to(k.dtype).view(1, H, 1, h).expand(T * C, H, 1, h)], dim=2)
        v = torch.cat([v, self.no_source_value.to(v.dtype).view(1, H, 1, h).expand(T * C, H, 1, h)], dim=2)
        m = mask[:, None, None].expand(T, C, 1, R, R + 1).reshape(T * C, 1, R, R + 1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=m)                                   # [T * C, H, R, h]
        y = y.view(T, C, H, R, h).permute(0, 3, 1, 2, 4).reshape(T, R, C, d)
        return self.out(y[valid])


class AxialBlock(nn.Module):
    def __init__(self, width, row_head=64, column_head=128, ratio=4):
        super().__init__()
        self.norm_row, self.norm_column, self.norm_ff = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.row = RowAttention(width, row_head)
        self.column = ColumnAttention(width, column_head)
        self.ff_in, self.ff_out = nn.Linear(width, ratio * width), nn.Linear(ratio * width, width)

    def forward(self, x, cos, sin, layout):
        x = x + self.row(self.norm_row(x), cos, sin)
        x = x + self.column(self.norm_column(x), layout)
        return x + self.ff_out(F.gelu(self.ff_in(self.norm_ff(x))))

    def residual_outputs(self):
        return (self.row.out, self.column.out, self.ff_out)


def init_weights(module, blocks):
    """Normal(0, 0.02) weights and zero biases; the last layer of every residual branch scaled by the number of
    branches, so the depth does not change the scale of the stream at the start."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, ColumnAttention):
            nn.init.normal_(m.no_source_key, std=0.02)
            nn.init.normal_(m.no_source_value, std=0.02)
    outs = [o for b in blocks for o in b.residual_outputs()]
    for o in outs:
        nn.init.normal_(o.weight, std=0.02 / math.sqrt(len(outs)))
