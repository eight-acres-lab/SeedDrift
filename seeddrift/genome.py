"""Genome context: gene order (synteny), mobile elements, contamination check."""
from __future__ import annotations

import bisect
import csv
import math
import random
import zlib

# Mobile-element type weights by recipient domain. Unknown types get DEFAULT_ME_WEIGHT.
ME_WEIGHTS = {
    "prok": {"integrase": 1.0, "ice": 1.0, "plasmid": 1.0, "integron": 0.9, "transposase": 0.9,
             "is": 0.9, "insertion_sequence": 0.9, "phage": 0.8, "prophage": 0.8,
             "genomic_island": 0.8, "recombinase": 0.7, "trna": 0.5},
    "euk": {"giant_virus": 0.9, "virus": 0.8, "eve": 0.8, "te": 0.6, "transposon": 0.6,
            "ltr": 0.6, "dna": 0.6, "helitron": 0.6, "line": 0.5, "sine": 0.4, "trna": 0.2},
}
DEFAULT_ME_WEIGHT = 0.4
ME_WINDOW = {"prok": 5000, "euk": 20000}   # bp


def me_weight(t, domain):
    t = t.strip().lower()
    table = ME_WEIGHTS.get(domain, ME_WEIGHTS["prok"])
    if t in table:
        return table[t]
    if t.startswith("is") and (len(t) == 2 or t[2:3].isdigit()):
        return table.get("is", DEFAULT_ME_WEIGHT)
    for k, v in table.items():
        if len(k) > 2 and k in t:
            return v
    return DEFAULT_ME_WEIGHT


class Gene:
    __slots__ = ("gid", "species", "family", "contig", "start", "end", "idx")

    def __init__(self, gid, species, family, contig, start, end):
        self.gid, self.species, self.family = gid, species, family
        self.contig, self.start, self.end = contig, start, end
        self.idx = -1


def _need(row, cols, path):
    missing = [c for c in cols if c not in row]
    if missing:
        raise ValueError("%s: missing column(s) %s" % (path, ", ".join(missing)))


def load_genes(path):
    out = []
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            _need(row, ["gene_id", "species", "family", "contig", "start", "end"], path)
            s, e = int(float(row["start"])), int(float(row["end"]))
            out.append(Gene(row["gene_id"], row["species"], row["family"], row["contig"],
                            min(s, e), max(s, e)))
    return out


def load_mes(path):
    out = []
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            _need(row, ["species", "contig", "start", "end", "type"], path)
            s, e = int(float(row["start"])), int(float(row["end"]))
            out.append((row["species"], row["contig"], min(s, e), max(s, e), row["type"]))
    return out


def load_composition(path):
    """TSV with gene_id, gc, gc3 (fractions); other columns ignored."""
    out = {}
    with open(path) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            _need(row, ["gene_id", "gc", "gc3"], path)
            try:
                out[row["gene_id"]] = (float(row["gc"]), float(row["gc3"]))
            except ValueError:
                continue
    return out


def load_species_meta(path):
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            _need(row, ["species", "domain"], path)
            out[row["species"]] = row["domain"].strip().lower()
    return out


class GenomeIndex:
    def __init__(self, genes, mes=(), domains=None, default_domain="prok", window=4):
        self.w = window
        self.genes = {g.gid: g for g in genes}
        self.domains = domains or {}
        self.default_domain = default_domain
        self.order = {}
        self.species_genes = {}
        self.by_fam_sp = {}
        for g in genes:
            self.order.setdefault((g.species, g.contig), []).append(g)
            self.species_genes.setdefault(g.species, []).append(g)
            self.by_fam_sp.setdefault((g.family, g.species), []).append(g)
        for lst in self.order.values():
            lst.sort(key=lambda g: (g.start, g.end))
            for i, g in enumerate(lst):
                g.idx = i
        me = {}
        for species, contig, s, e, t in mes:
            me.setdefault((species, contig), []).append((s, e, t))
        self.me = {}
        for key, lst in me.items():
            lst.sort()
            self.me[key] = ([x[0] for x in lst], [x[1] for x in lst], [x[2] for x in lst],
                            max(x[1] - x[0] for x in lst))
        self._nf = {}
        self._syn_bg = {}
        self._me_bg = {}

    def domain(self, species):
        return self.domains.get(species, self.default_domain)

    # -- gene order --------------------------------------------------------
    def neighbors(self, g, w=None):
        w = self.w if w is None else w
        lst = self.order[(g.species, g.contig)]
        lo, hi = max(0, g.idx - w), min(len(lst), g.idx + w + 1)
        return [lst[i] for i in range(lo, hi) if i != g.idx]

    def neighbor_families(self, g):
        if g.gid not in self._nf:
            self._nf[g.gid] = frozenset(n.family for n in self.neighbors(g)) - {g.family}
        return self._nf[g.gid]

    def _side(self, g, step, skip=frozenset()):
        lst = self.order[(g.species, g.contig)]
        out, i = [], g.idx + step
        while 0 <= i < len(lst) and len(out) < self.w:
            if lst[i].family != g.family and lst[i].gid not in skip:
                out.append(lst[i])
            i += step
        return out

    def _anchors(self, g, Y, step, skip=frozenset(), k=2, max_scan=30):
        """Nearest genes on one side of g whose family genome Y also has (orthologous flanks).

        Genes Y lacks are passed over: they belong to the same island or to other insertions
        and would otherwise leave the test without anchors.
        """
        lst = self.order[(g.species, g.contig)]
        out, i, n = [], g.idx + step, 0
        while 0 <= i < len(lst) and len(out) < k and n < max_scan:
            h = lst[i]
            if h.family != g.family and h.gid not in skip and (h.family, Y) in self.by_fam_sp:
                out.append(h)
            i += step
            n += 1
        return out

    def _flank_pair(self, g, Y, skip):
        """Closest pair of g's left/right anchors on one contig of Y: (distance, a, b) or None."""
        best = None
        for hl in self._anchors(g, Y, -1, skip):
            for hr in self._anchors(g, Y, +1, skip):
                if hr.family == hl.family:
                    continue
                for a in self.by_fam_sp[(hl.family, Y)]:
                    for b in self.by_fam_sp[(hr.family, Y)]:
                        if a.contig == b.contig and abs(a.idx - b.idx) <= 2 * self.w + 1:
                            if best is None or abs(a.idx - b.idx) < best[0]:
                                best = (abs(a.idx - b.idx), a, b)
        return best

    def insertion_test(self, g, relatives, skip=frozenset()):
        """Count relatives where g's flanks are adjacent with an empty site vs. with g's family present.

        Returns (n_empty, n_conserved, locus_families). Empty site = insertion in g's lineage
        (HGT-like); conserved = the locus already carried this family (vertical-like).
        locus_families = families around the orthologous locus in the relatives (ancestral
        neighbourhood), used to tell co-transferred neighbours from ancestral ones.
        `skip` = gene ids that must not serve as flank anchors (neighbours transferred
        together with g: in the donor they sit next to g's family and fake a conserved locus).
        """
        empty = conserved = 0
        locus = set()
        for Y in relatives:
            if Y == g.species:
                continue
            hit = self._flank_pair(g, Y, skip)
            if hit is None:
                continue
            _, a, b = hit
            lo, hi = sorted((a.idx, b.idx))
            lst = self.order[(Y, a.contig)]
            if any(lst[k].family == g.family for k in range(lo + 1, hi)):
                conserved += 1
            else:
                empty += 1
            for k in range(max(0, lo - self.w), min(len(lst), hi + self.w + 1)):
                locus.add(lst[k].family)
        return empty, conserved, locus

    def site_test(self, g, relatives, skip=frozenset()):
        """For a family absent from the relatives: is g's position an empty site there?

        Returns (n_clean, n_occupied). Clean = g's flanks are adjacent in the relative, or only
        genes of the relative's own (rare) families sit between them: an insertion in g's lineage
        (or a clean deletion / a hotspot used twice). Occupied = a gene sits between the flanks
        whose family g's genome lacks but other genomes share: replacement, or the same gene
        under another family id after an orthogroup split.
        """
        if not hasattr(self, "_fam_nsp"):
            self._fam_nsp = {}
            for fam, _ in self.by_fam_sp:
                self._fam_nsp[fam] = self._fam_nsp.get(fam, 0) + 1
        clean = occupied = 0
        for Y in relatives:
            if Y == g.species:
                continue
            best = self._flank_pair(g, Y, skip)
            if best is None:
                continue
            _, a, b = best
            lo, hi = sorted((a.idx, b.idx))
            lst = self.order[(Y, a.contig)]
            occ = any((h.family, g.species) not in self.by_fam_sp and self._fam_nsp.get(h.family, 0) >= 2
                      for h in lst[lo + 1:hi])
            if occ:
                occupied += 1
            else:
                clean += 1
        return clean, occupied

    # -- composition / length ----------------------------------------------
    def set_composition(self, comp):
        """comp: {gene_id: (gc, gc3)}; per-genome robust centre and spread."""
        self.comp = comp
        self._comp_stats = {}
        for sp_, genes in self.species_genes.items():
            stats = []
            for k in (0, 1):
                v = sorted(comp[g.gid][k] for g in genes if g.gid in comp)
                if len(v) < 20:
                    stats = None
                    break
                med = v[len(v) // 2]
                mad = sorted(abs(x - med) for x in v)[len(v) // 2] * 1.4826
                stats.append((med, max(mad, 0.005)))
            if stats:
                self._comp_stats[sp_] = stats

    def comp_score(self, g):
        """Mean robust |z| of GC and GC3 (0 when composition is unknown)."""
        st = getattr(self, "_comp_stats", {}).get(g.species)
        c = getattr(self, "comp", {}).get(g.gid)
        if not st or c is None:
            return 0.0
        z = [abs(c[k] - st[k][0]) / st[k][1] for k in (0, 1)]
        return min(6.0, sum(z) / 2.0)

    def len_score(self, g):
        """log2(gene length / genome median), clipped to [-4, 3]."""
        if not hasattr(self, "_len_med"):
            self._len_med = {}
            for sp_, genes in self.species_genes.items():
                v = sorted(x.end - x.start + 1 for x in genes)
                self._len_med[sp_] = v[len(v) // 2] if v else 1
        L = g.end - g.start + 1
        return max(-4.0, min(3.0, math.log2(max(1, L) / float(self._len_med[g.species]))))

    def jaccard(self, g, h):
        A, B = self.neighbor_families(g), self.neighbor_families(h)
        if not A or not B:
            return 0.0
        return len(A & B) / float(len(A | B))

    def syn_background(self, X, Y, n=150):
        """Mean neighbourhood Jaccard between single-copy orthologs of genomes X and Y."""
        key = (X, Y) if X < Y else (Y, X)
        if key not in self._syn_bg:
            rng = random.Random(zlib.crc32(("%s|%s" % key).encode()))
            genes = self.species_genes.get(key[0], [])
            vals = []
            for g in rng.sample(genes, min(n, len(genes))):
                hs = self.by_fam_sp.get((g.family, key[1]))
                if hs and len(hs) == 1:
                    vals.append(self.jaccard(g, hs[0]))
            self._syn_bg[key] = sum(vals) / len(vals) if vals else 0.0
        return self._syn_bg[key]

    def context_shift(self, rec_genes, donor_species, family):
        """How much less the recipient copy shares its neighbourhood with the donor-clade
        copies than single-copy orthologs of the same genome pair usually do
        (background - best Jaccard). ~0 for vertically inherited genes misplaced by tree
        error; > 0 when the copy sits in a new genomic context. None if no donor copies.
        """
        vals = []
        for g in rec_genes[:5]:
            best = None
            for Y in donor_species:
                for h in self.by_fam_sp.get((family, Y), [])[:3]:
                    v = self.syn_background(g.species, Y) - self.jaccard(g, h)
                    best = v if best is None else min(best, v)
            if best is not None:
                vals.append(best)
        return sum(vals) / len(vals) if vals else None

    def donor_synteny(self, rec_genes, donor_species, family, ancestral):
        """Co-transfer evidence: neighbours shared by recipient and donor copies that are NOT
        part of the recipient lineage's ancestral neighbourhood (ancestral[gid] = families
        around the orthologous locus in relatives). Returns mean over copies of min(n, 3) / 3.
        """
        vals = []
        for g in rec_genes[:5]:
            foreign = self.neighbor_families(g) - ancestral.get(g.gid, frozenset())
            if not foreign:
                vals.append(0.0)
                continue
            best = 0
            for Y in donor_species:
                for h in self.by_fam_sp.get((family, Y), [])[:3]:
                    best = max(best, len(foreign & self.neighbor_families(h)))
            vals.append(min(best, 3) / 3.0)
        return sum(vals) / len(vals) if vals else 0.0

    # -- mobile elements ---------------------------------------------------
    def me_near(self, g):
        rec = self.me.get((g.species, g.contig))
        if not rec:
            return []
        starts, ends, types, maxlen = rec
        W = ME_WINDOW.get(self.domain(g.species), 5000)
        lo = bisect.bisect_left(starts, g.start - W - maxlen)
        hi = bisect.bisect_right(starts, g.end + W)
        out = []
        for k in range(lo, hi):
            gap = max(0, starts[k] - g.end, g.start - ends[k])
            if gap <= W:
                out.append((gap, types[k]))
        return out

    def me_score(self, g):
        dom = self.domain(g.species)
        lam = ME_WINDOW.get(dom, 5000) / 3.0
        best, types = 0.0, set()
        for gap, t in self.me_near(g):
            best = max(best, me_weight(t, dom) * math.exp(-gap / lam))
            types.add(t.lower())
        return best, types

    def me_background(self, species, n=500):
        if species not in self._me_bg:
            genes = self.species_genes.get(species, [])
            rng = random.Random(zlib.crc32(species.encode()))
            sample = rng.sample(genes, min(n, len(genes)))
            self._me_bg[species] = (sum(self.me_score(g)[0] for g in sample) / len(sample)) if sample else 0.0
        return self._me_bg[species]

    # -- contamination -----------------------------------------------------
    def native_count(self, g, foreign, cap=2):
        """Number of other genes on g's contig that are not foreign (stops at cap)."""
        n = 0
        for h in self.order[(g.species, g.contig)]:
            if h.gid != g.gid and h.gid not in foreign:
                n += 1
                if n >= cap:
                    break
        return n
