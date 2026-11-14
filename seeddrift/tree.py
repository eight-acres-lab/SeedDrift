"""Minimal Newick utilities and an indexed species tree (stdlib + numpy only)."""
from __future__ import annotations

import random
import re

import numpy as np

_NUMERIC = re.compile(r"^[0-9.eE+\-]+(/[0-9.eE+\-]+)*$")
_SAFE = re.compile(r"^[^\s,():;'\[\]]+$")


class Node:
    __slots__ = ("name", "length", "children", "parent", "support")

    def __init__(self, name=None, length=None):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None
        self.support = None

    def add(self, child):
        child.parent = self
        self.children.append(child)
        return child

    def is_leaf(self):
        return not self.children

    def preorder(self):
        stack = [self]
        while stack:
            nd = stack.pop()
            yield nd
            stack.extend(reversed(nd.children))

    def postorder(self):
        stack = [(self, False)]
        while stack:
            nd, done = stack.pop()
            if done:
                yield nd
            else:
                stack.append((nd, True))
                for ch in reversed(nd.children):
                    stack.append((ch, False))

    def leaves(self):
        return [nd for nd in self.preorder() if not nd.children]


# ----------------------------------------------------------------------------
# Newick I/O
# ----------------------------------------------------------------------------

def _strip_comments(s):
    out, depth = [], 0
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _set_label(node, label):
    if not label:
        return
    if node.children and _NUMERIC.match(label):
        try:
            node.support = float(label.split("/")[-1])
            return
        except ValueError:
            pass
    node.name = label


def parse_newick(text):
    s = _strip_comments(text).strip()
    if s.endswith(";"):
        s = s[:-1]
    if not s:
        raise ValueError("empty Newick string")
    root = Node()
    node = root
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            node = node.add(Node())
            i += 1
        elif ch == ",":
            if node.parent is None:
                raise ValueError("unbalanced Newick: ',' at top level")
            node = node.parent.add(Node())
            i += 1
        elif ch == ")":
            if node.parent is None:
                raise ValueError("unbalanced Newick: too many ')'")
            node = node.parent
            i += 1
        elif ch == ":":
            j = i + 1
            while j < n and s[j] not in ",():;":
                j += 1
            txt = s[i + 1:j].strip()
            node.length = float(txt) if txt else None
            i = j
        elif ch.isspace():
            i += 1
        else:
            if ch == "'":
                j, buf = i + 1, []
                while j < n:
                    if s[j] == "'":
                        if j + 1 < n and s[j + 1] == "'":
                            buf.append("'")
                            j += 2
                            continue
                        break
                    buf.append(s[j])
                    j += 1
                label = "".join(buf)
                j += 1
            else:
                j = i
                while j < n and s[j] not in ",():;":
                    j += 1
                label = s[i:j].strip()
            _set_label(node, label)
            i = j
    if node is not root:
        raise ValueError("unbalanced Newick: missing ')'")
    return root


def read_trees(path):
    """Return every ';'-terminated Newick string in a file."""
    with open(path) as fh:
        text = _strip_comments(fh.read())
    return [chunk.strip() + ";" for chunk in text.split(";") if chunk.strip()]


def _quote(name):
    if name is None:
        return ""
    return name if _SAFE.match(name) else "'" + name.replace("'", "''") + "'"


def to_newick(root, lengths=True, internal_names=True, support=False):
    parts = {}
    for nd in root.postorder():
        if nd.children:
            s = "(" + ",".join(parts.pop(id(c)) for c in nd.children) + ")"
            if internal_names and nd.name:
                s += _quote(nd.name)
            elif support and nd.support is not None and nd.parent is not None:
                s += "%g" % nd.support
        else:
            s = _quote(nd.name)
        if lengths and nd.length is not None and nd.parent is not None:
            s += ":%.6g" % nd.length
        parts[id(nd)] = s
    return parts[id(root)] + ";"


# ----------------------------------------------------------------------------
# Tree operations
# ----------------------------------------------------------------------------

def copy_tree(root):
    new = {}
    for nd in root.postorder():
        m = Node(nd.name, nd.length)
        m.support = nd.support
        for c in nd.children:
            m.add(new.pop(id(c)))
        new[id(nd)] = m
    return new[id(root)]


def prune_to(root, keep):
    """Copy of the tree restricted to leaves named in `keep`; unary nodes collapsed."""
    new = {}
    for nd in root.postorder():
        if not nd.children:
            new[id(nd)] = Node(nd.name, nd.length) if nd.name in keep else None
            continue
        kids = [k for k in (new.pop(id(c)) for c in nd.children) if k is not None]
        if not kids:
            new[id(nd)] = None
        elif len(kids) == 1:
            k = kids[0]
            if nd.length is not None or k.length is not None:
                k.length = (k.length or 0.0) + (nd.length or 0.0)
            new[id(nd)] = k
        else:
            m = Node(nd.name, nd.length)
            m.support = nd.support
            for k in kids:
                m.add(k)
            new[id(nd)] = m
    r = new[id(root)]
    if r is not None:
        r.length = None
    return r


def resolve_polytomies(root, rng, root_max=2):
    """Randomly resolve multifurcations (zero-length new branches)."""
    for nd in list(root.postorder()):
        limit = root_max if nd is root else 2
        while len(nd.children) > limit:
            a, b = rng.sample(nd.children, 2)
            nd.children.remove(a)
            nd.children.remove(b)
            m = Node(length=0.0)
            m.add(a)
            m.add(b)
            nd.add(m)


def has_support(root):
    return any(nd.support is not None for nd in root.preorder() if nd.children and nd.parent is not None)


def contract_weak(root, min_support, min_len=2e-6):
    """Contract internal branches with support < min_support or length <= min_len (in place).

    Unsupported resolutions are what make a conserved family look transferred; once they are
    multifurcations, a species-tree-guided resolution removes the spurious incongruence.
    Returns the number of contracted branches.
    """
    n = 0
    for nd in list(root.postorder()):
        if nd is root or not nd.children:
            continue
        weak = (nd.support is not None and nd.support < min_support) or \
               (nd.length is not None and nd.length <= min_len)
        if not weak:
            continue
        par = nd.parent
        pos = par.children.index(nd)
        for c in nd.children:
            c.parent = par
            c.length = (c.length or 0.0) + (nd.length or 0.0)
        par.children[pos:pos + 1] = nd.children
        n += 1
    return n


def resolve_polytomies_guided(root, species_of, sp, rng, root_max=2):
    """Resolve multifurcations the way the species tree suggests.

    Greedily joins the two subtrees whose species sets have the most recent common ancestor
    (fewest species under the LCA); disjoint species sets (a speciation) win ties over
    overlapping ones (a duplication), remaining ties are random. species_of(leaf name) ->
    species index in `sp`.
    """
    def lca(sset):
        it = iter(sset)
        x = next(it)
        rest = list(it)
        while rest and not all(sp.desc[x, y] for y in rest):
            x = sp.parent[x]
        return x

    spset = {}
    for nd in root.postorder():
        if nd.children:
            s = set()
            for c in nd.children:
                s |= spset[id(c)]
            spset[id(nd)] = frozenset(s)
        else:
            spset[id(nd)] = frozenset([species_of(nd.name)])
    for nd in list(root.postorder()):
        limit = root_max if nd is root else 2
        if len(nd.children) <= limit:
            continue
        kids = list(nd.children)
        while len(kids) > limit:
            best, best_key = None, None
            for i in range(len(kids)):
                si = spset[id(kids[i])]
                for j in range(i + 1, len(kids)):
                    sj = spset[id(kids[j])]
                    key = (len(sp.leafsets[lca(si | sj)]), 0 if si.isdisjoint(sj) else 1, rng.random())
                    if best_key is None or key < best_key:
                        best, best_key = (i, j), key
            i, j = best
            a, b = kids[i], kids[j]
            m = Node(length=0.0)
            m.add(a)
            m.add(b)
            spset[id(m)] = spset[id(a)] | spset[id(b)]
            kids = [k for t, k in enumerate(kids) if t not in (i, j)] + [m]
        nd.children = []
        for k in kids:
            nd.add(k)


def unroot(root):
    """Turn a bifurcating root into a trifurcation (in place)."""
    if len(root.children) != 2:
        return root
    a, b = root.children
    if b.is_leaf():
        a, b = b, a
    if b.is_leaf():
        return root
    root.children = [a]
    a.length = (a.length or 0.0) + (b.length or 0.0)
    for c in b.children:
        root.add(c)
    return root


def random_nni(root, rng):
    """Swap a random child of an internal edge with a sibling (one NNI move)."""
    internal = [nd for nd in root.preorder() if nd.children and nd.parent is not None]
    if not internal:
        return False
    v = rng.choice(internal)
    u = v.parent
    sibs = [c for c in u.children if c is not v]
    if not sibs:
        return False
    s, c = rng.choice(sibs), rng.choice(v.children)
    u.children[u.children.index(s)] = c
    c.parent = u
    v.children[v.children.index(c)] = s
    s.parent = v
    return True


def unrooted_adjacency(root):
    """Return (adj, names) for the unrooted version of the tree.

    adj[i] is a list of (neighbour, branch_length); names[i] is the leaf name or None.
    A bifurcating root is suppressed (its two edges are merged); it remains as an
    isolated index with no neighbours.
    """
    r = root
    while len(r.children) == 1:
        r = r.children[0]
    nodes = list(r.preorder())
    idx = {id(nd): i for i, nd in enumerate(nodes)}
    adj = [[] for _ in nodes]
    names = [nd.name if not nd.children else None for nd in nodes]
    skip_root = len(r.children) == 2
    for nd in nodes:
        if nd is r or (skip_root and nd.parent is r):
            continue
        a, b = idx[id(nd)], idx[id(nd.parent)]
        L = nd.length or 0.0
        adj[a].append((b, L))
        adj[b].append((a, L))
    if skip_root:
        c1, c2 = r.children
        a, b = idx[id(c1)], idx[id(c2)]
        L = (c1.length or 0.0) + (c2.length or 0.0)
        adj[a].append((b, L))
        adj[b].append((a, L))
    return adj, names


# ----------------------------------------------------------------------------
# Species tree with integer indices
# ----------------------------------------------------------------------------

class SpeciesTree:
    """Rooted binary species tree. Nodes are indexed in postorder (root = n - 1)."""

    def __init__(self, root, rng=None):
        rng = rng or random.Random(0)
        leaf_names = [nd.name for nd in root.leaves()]
        if any(not nm for nm in leaf_names):
            raise ValueError("every species-tree leaf needs a name")
        if len(set(leaf_names)) != len(leaf_names):
            raise ValueError("duplicate species names in the species tree")
        root = prune_to(root, set(leaf_names))
        if len(root.children) > 2:
            print("[seeddrift] warning: species tree root is not bifurcating; resolving randomly")
        resolve_polytomies(root, rng, root_max=2)

        nodes = list(root.postorder())
        n = len(nodes)
        index = {id(nd): i for i, nd in enumerate(nodes)}
        self.n = n
        self.root = n - 1
        self.parent = [-1] * n
        self.children = [[] for _ in range(n)]
        self.names = [None] * n
        self.lengths = [nd.length for nd in nodes]
        used = set(leaf_names)
        for i, nd in enumerate(nodes):
            for c in nd.children:
                ci = index[id(c)]
                self.children[i].append(ci)
                self.parent[ci] = i
            if nd.children:
                nm = nd.name
                if not nm or nm in used:
                    nm = "N%d" % i
                    while nm in used:
                        nm += "_"
                used.add(nm)
                self.names[i] = nm
            else:
                self.names[i] = nd.name
        self.is_leaf = [not ch for ch in self.children]
        self.index = {nm: i for i, nm in enumerate(self.names)}
        self.leaf_index = {nm: i for i, nm in enumerate(self.names) if self.is_leaf[i]}
        self.depth = [0] * n
        for i in range(n - 1, -1, -1):
            p = self.parent[i]
            self.depth[i] = 0 if p < 0 else self.depth[p] + 1
        self.desc = np.zeros((n, n), dtype=bool)
        for i in range(n):
            self.desc[i, i] = True
            for c in self.children[i]:
                self.desc[i] |= self.desc[c]
        self.leafsets = [
            tuple(self.names[j] for j in np.flatnonzero(self.desc[i]) if self.is_leaf[j])
            for i in range(n)
        ]

    def has_lengths(self):
        return all(self.lengths[i] is not None and self.lengths[i] > 0
                   for i in range(self.n) if i != self.root)

    def branch_intervals(self):
        """(start, end) time of every branch, root = 0, deepest leaf = 1; leaves extend to 1."""
        t = [0.0] * self.n
        for i in range(self.n - 1, -1, -1):
            p = self.parent[i]
            t[i] = 0.0 if p < 0 else t[p] + (self.lengths[i] or 0.0)
        H = max(t) or 1.0
        start = np.array([t[self.parent[i]] / H if self.parent[i] >= 0 else 0.0 for i in range(self.n)])
        end = np.array([1.0 if self.is_leaf[i] else t[i] / H for i in range(self.n)])
        return start, end

    def sibling(self, i):
        p = self.parent[i]
        if p < 0:
            return -1
        a, b = self.children[p]
        return b if a == i else a

    def related(self, a, b):
        """True if a and b lie on one root-to-leaf lineage."""
        return bool(self.desc[a, b] or self.desc[b, a])

    def label(self, i):
        if self.is_leaf[i]:
            return self.names[i]
        ls = self.leafsets[i]
        head = ",".join(ls[:3]) + (",..." if len(ls) > 3 else "")
        return "%s{%s;n=%d}" % (self.names[i], head, len(ls))

    def to_newick(self):
        nodes = [Node(self.names[i], self.lengths[i]) for i in range(self.n)]
        for i in range(self.n):
            for c in self.children[i]:
                nodes[i].add(nodes[c])
        return to_newick(nodes[self.root])
