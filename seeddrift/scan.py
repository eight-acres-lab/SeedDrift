"""Genome-wide scan: reconcile every family, attach genome-context evidence, score."""
from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

from .genome import GenomeIndex, load_composition, load_genes, load_mes, load_species_meta
from .model import ALL_FEATURES, DONOR_GAMMA, load_model, predict, softmax_donors, tier
from .reconcile import DTLModel, background_distances, reconcile_family, split_paralogs
from .tree import SpeciesTree, parse_newick, prune_to, read_trees, to_newick

TREE_EXT = (".nwk", ".newick", ".tre", ".tree", ".treefile", ".ufboot", ".boottrees", ".trees")

EVENT_COLUMNS = [
    "event_id", "family", "recipient", "recipient_label", "recipient_domain", "n_genes",
    "recipient_genes", "P_HGT", "tier", "source", "p_direction", "alt_recipient", "alt_genes",
    "donor1", "p_donor1", "donor2", "p_donor2", "donor3", "p_donor3"] + ALL_FEATURES + ["me_types", "flags"]


def log(msg):
    print("[seeddrift] " + msg, file=sys.stderr, flush=True)


def _priority(path):
    p = path.lower()
    if p.endswith((".ufboot", ".boottrees", ".trees")):
        return 2
    if p.endswith(".treefile"):
        return 0
    return 1


def discover_trees(folder):
    fams = {}
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if fn.startswith(".") or not os.path.isfile(path) or not fn.lower().endswith(TREE_EXT):
            continue
        fams.setdefault(fn.split(".")[0], []).append(path)
    return {f: sorted(p, key=_priority) for f, p in fams.items()}


def subsample(trees, k):
    if len(trees) <= k:
        return trees
    rest = trees[1:]
    step = len(rest) / float(k - 1)
    return [trees[0]] + [rest[int(i * step)] for i in range(k - 1)]


def _sample(seq, k, key):
    seq = list(seq)
    if len(seq) <= k:
        return seq
    return random.Random(zlib.crc32(key.encode())).sample(seq, k)


# -- worker process -----------------------------------------------------------
_W = {}


def _init(sp_newick, names, costs, dated, n_bt, seed, bg, collapse=0.0, split=0):
    sp = SpeciesTree(parse_newick(sp_newick))
    if sp.names != names:
        raise RuntimeError("species tree indexing differs between processes")
    _W.update(model=DTLModel(sp, *costs, dated=dated), n_bt=n_bt, seed=seed, bg=bg, collapse=collapse,
              split=split)


def _pool_map(fn, tasks, threads, init_args, label):
    out = []
    step = max(1, len(tasks) // 10)
    if threads > 1:
        with ProcessPoolExecutor(threads, initializer=_init, initargs=init_args) as ex:
            for i, r in enumerate(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (threads * 8))), 1):
                out.append(r)
                if label and i % step == 0:
                    log("%s %d/%d families" % (label, i, len(tasks)))
    else:
        _init(*init_args)
        for i, t in enumerate(tasks, 1):
            out.append(fn(t))
            if label and i % step == 0:
                log("%s %d/%d families" % (label, i, len(tasks)))
    return out


def _split_work(task):
    """Split one family at ancient duplications; returns [(name, trees, gene_sp)]."""
    fam, trees, gene_sp = task
    groups = split_paralogs(fam, trees[0], gene_sp, _W["model"], _W["split"], _W.get("collapse", 0.0),
                            _W["seed"])
    if len(groups) == 1:
        return [task]
    out = []
    for k, grp in enumerate(groups, 1):
        keep = set(grp)
        sub = []
        for txt in trees:
            try:
                t = prune_to(parse_newick(txt), keep)
            except ValueError:
                continue
            if t is not None and len(t.leaves()) >= 2:
                sub.append(to_newick(t, support=True))
        out.append(("%s.%d" % (fam, k), sub, {g: gene_sp[g] for g in grp}))
    return out


def _work(task):
    fam, trees, gene_sp = task
    return reconcile_family(fam, trees, gene_sp, _W["model"], _W["n_bt"], _W["seed"], _W["bg"],
                           _W.get("collapse", 0.0))


# -- pipeline -----------------------------------------------------------------

def build_events(results, min_freq):
    events = []
    for res in results:
        if not res["runs"]:
            continue
        dcost = math.log1p(max(0.0, (res["cost_dl"] or 0.0) - (res["cost_dtl"] or 0.0)))
        for R, e in res["events"].items():
            f = e["runs"] / float(res["runs"])
            f_opt = e["opt"] / float(res["n_trees"])
            if max(f, f_opt) < min_freq:
                continue
            tot = float(sum(e["donors"].values()))
            thr = 0.5 * e["occ"]
            genes = sorted(g for g, c in e["genes"].items() if c >= thr)
            if not genes:
                genes = [max(e["genes"], key=e["genes"].get)]
            events.append({
                "family": res["family"], "R": R, "f_T": f, "f_opt": f_opt, "dcost": dcost,
                "alt": max(0.0, res.get("alt_cost", {}).get(R, 0.0)),
                "donors": {d: c / tot for d, c in e["donors"].items()},
                "genes": genes,
                "rec_by_donor": {d: -sum(v) / len(v) for d, v in e["rec"].items() if v},
                "source": "tree", "clock": 0.0, "dir": 0.0, "other_genes": [],
            })
    return events


def _robust(v):
    v = sorted(v)
    med = v[len(v) // 2]
    mad = sorted(abs(x - med) for x in v)[len(v) // 2] * 1.4826
    return med, mad


def build_clock_events(results, sp, zthr, tree_events, min_per_node=20, mad_floor=0.05):
    """Candidates from molecular-clock anomalies at speciation nodes.

    Each node's null distribution comes from all families (mostly vertical): z =
    (log ratio - median) / MAD. A node with z <= -zthr gives two candidate directions
    (either child lineage may be the recipient); `dir` later decides between them.
    """
    by_node = {}
    for res in results:
        for s, lr, _, _ in res.get("clock", []):
            by_node.setdefault(s, []).append(lr)
    pooled = [v for vs in by_node.values() for v in vs]
    if len(pooled) < 30:
        return [], {}
    gmed, gmad = _robust(pooled)
    stats = {}
    for s, vs in by_node.items():
        med, mad = _robust(vs) if len(vs) >= min_per_node else (gmed, gmad)
        stats[s] = (med, max(mad, mad_floor), len(vs))
    have = set((e["family"], e["R"]) for e in tree_events)
    explained = {}          # genes already placed by a well-supported reconciliation transfer
    for e in tree_events:
        if max(e["f_T"], e["f_opt"]) >= 0.5:
            explained.setdefault(e["family"], set()).update(e["genes"])
    events = []
    for res in results:
        dcost = math.log1p(max(0.0, (res["cost_dl"] or 0.0) - (res["cost_dtl"] or 0.0)))
        done = explained.get(res["family"], set())
        for k, (s, lr, g0, g1) in enumerate(res.get("clock", [])):
            med, mad, _ = stats[s]
            z = (lr - med) / mad
            if z > -zthr or done.intersection(g0) or done.intersection(g1):
                continue
            c0, c1 = sp.children[s]
            pair = "%s|%d|%d" % (res["family"], s, k)   # same node can be tested once per paralog
            for R, donor, genes, other in ((c0, c1, g0, g1), (c1, c0, g1, g0)):
                if (res["family"], R) in have:
                    continue
                events.append({"pair": pair,
                    "family": res["family"], "R": R, "f_T": 0.0, "f_opt": 0.0, "dcost": dcost,
                    "donors": {donor: 1.0}, "genes": sorted(genes),
                    "rec_by_donor": {donor: -lr}, "source": "clock", "clock": -z, "dir": 0.0,
                    "other_genes": sorted(other),
                })
    return events, stats


COMP_TAU = 0.3        # amelioration time scale, in species-tree height units


def _base_family(fam):
    head, _, tail = fam.rpartition(".")
    return head if head and tail.isdigit() else fam


def _lca(sp, nodes):
    nodes = list(nodes)
    x = nodes[0]
    while not all(sp.desc[x, y] for y in nodes[1:]):
        x = sp.parent[x]
    return x


def gain_nodes(sp, present, gain_cost=3.0, loss_cost=1.0):
    """Wagner-style gain/loss parsimony of a presence pattern on the species tree.

    Returns {gain node: [present species it explains]}; a gain on the root branch means the
    family is ancestral and is left out. A patchy family (homologous genes acquired independently
    by distant lineages, e.g. mobile or island genes) gets several recent gains instead of one
    deep gain followed by many losses.
    """
    n = sp.n
    A, Pc, gainA, keepP = [0.0] * n, [0.0] * n, [False] * n, [True] * n
    for v in range(n):                         # postorder: children before parents
        if sp.is_leaf[v]:
            here = v in present
            Pc[v], A[v] = (0.0, gain_cost) if here else (loss_cost, 0.0)
            gainA[v], keepP[v] = here, here
            continue
        sa = sum(A[c] for c in sp.children[v])
        sp_ = sum(Pc[c] for c in sp.children[v])
        keepP[v] = sp_ <= loss_cost + sa
        Pc[v] = sp_ if keepP[v] else loss_cost + sa
        gainA[v] = gain_cost + sp_ <= sa       # tie: one older gain rather than several
        A[v] = gain_cost + sp_ if gainA[v] else sa
    out, stack = {}, [(sp.root, "A", None)]
    while stack:
        v, state, src = stack.pop()
        if state == "A" and gainA[v]:
            state, src = "P", v
        elif state == "P" and not keepP[v]:
            state, src = "A", None
        if sp.is_leaf[v]:
            if v in present and src is not None:
                out.setdefault(src, []).append(v)
            continue
        for c in sp.children[v]:
            stack.append((c, state, src))
    out.pop(sp.root, None)
    return out


def build_gain_events(gi, sp, gain_cost=3.0, loss_cost=1.0):
    """Candidates for transfers from unsampled donors: (parts of) families gained inside the tree.

    Reconciliation sees such genes as an origin, not a transfer (no donor in the data).
    Gain nodes come from gain/loss parsimony of the family's presence pattern. Alternatives the
    gain model has to weigh: vertical descent with losses in every lineage branching off between
    the root and the clade (`g_alt` of them), a de novo or spurious ORF (short, no mobile
    elements), or an orthogroup split (the locus is occupied in relatives).
    """
    fams = {}
    for g in gi.genes.values():
        if g.species in sp.leaf_index:
            fams.setdefault(g.family, []).append(g)
    # subfamilies from --split-paralogs: which genomes carry a sister subfamily (a paralog)?
    sib_sp = {}
    for fam, genes in fams.items():
        base = _base_family(fam)
        if base != fam:
            for g in genes:
                sib_sp.setdefault(base, {}).setdefault(g.species, set()).add(fam)
    events = []
    for fam, genes in fams.items():
        present = set(sp.leaf_index[g.species] for g in genes)
        gains = gain_nodes(sp, present, gain_cost, loss_cost)
        base = _base_family(fam)
        sibs = sib_sp.get(base, {}) if base != fam else {}
        for R, spp in gains.items():
            spp = set(spp)
            mine = [g for g in genes if sp.leaf_index[g.species] in spp]
            d, x = 0, R
            while x != sp.root:
                x = sp.parent[x]
                d += 1
            events.append({
                "family": fam, "R": R, "f_T": 0.0, "f_opt": 0.0, "dcost": 0.0, "donors": {},
                "genes": sorted(g.gid for g in mine), "rec_by_donor": {}, "source": "gain",
                "clock": 0.0, "dir": 0.0, "other_genes": [], "n_gains": len(gains),
                "g_alt": float(d), "g_patchy": 1.0 - len(spp) / float(len(sp.leafsets[R])),
                "g_size": math.log2(len(mine) / float(len(spp))),
                # share of gaining genomes that already hold a paralog (sister subfamily): an
                # origin by duplication rather than acquisition
                "g_para": sum(1 for g in mine if sibs.get(g.species, set()) - {fam}) / float(len(mine)),
            })
    return events


def annotate(events, sp, gi):
    G = DONOR_GAMMA
    by_gene, foreign, moved_into = {}, set(), {}
    for k, ev in enumerate(events):
        special = ev["source"] in ("clock", "gain")
        if ev["f_T"] >= 0.3 or special:
            for g in ev["genes"]:
                by_gene.setdefault(g, []).append(k)
        if ev["f_T"] >= 0.5 or (ev["source"] == "gain" and ev["g_alt"] >= 2):
            foreign.update(ev["genes"])
        if max(ev["f_T"], ev["f_opt"]) >= 0.3 or special:
            moved_into.setdefault(ev["R"], set()).update(ev["genes"])
    # share of each genome's genes moved into each branch: background for block enrichment
    moved_bg = {}
    for R, gids in moved_into.items():
        for gid in gids:
            g = gi.genes.get(gid)
            if g is not None:
                moved_bg[R, g.species] = moved_bg.get((R, g.species), 0) + 1
    for (R, s_), n in list(moved_bg.items()):
        moved_bg[R, s_] = n / float(len(gi.species_genes[s_]))
    # composition only speaks for recent acquisitions (foreign signatures ameliorate): weight it
    # by exp(-age / COMP_TAU), age = time before present of the recipient branch's midpoint
    if sp.has_lengths():
        st_, en_ = sp.branch_intervals()
        comp_w = [math.exp(-(1.0 - 0.5 * (st_[i] + en_[i])) / COMP_TAU) for i in range(sp.n)]
    else:
        comp_w = [1.0] * sp.n
    fam_copies = {}
    for g in gi.genes.values():
        c = fam_copies.setdefault(g.family, [0, set()])
        c[0] += 1
        c[1].add(g.species)

    for k, ev in enumerate(events):
        for f in ("g_alt", "g_patchy", "g_size", "site", "site_n", "g_len", "alt", "g_para"):
            ev.setdefault(f, 0.0)
        R, fam, donors = ev["R"], ev["family"], ev["donors"]
        rec = [gi.genes[g] for g in ev["genes"] if g in gi.genes]
        sib = sp.sibling(R)
        ev["near"] = sum(p for d, p in donors.items() if sib >= 0 and sp.desc[sib, d])

        # recipient-side synteny: is the locus an empty site in the sister clade and in the
        # next outgroup? (sister alone cannot tell insertion in R from deletion in the sister)
        rel = _sample(sp.leafsets[sib], 8, fam + sp.names[R]) if sib >= 0 else []
        par = sp.parent[R]
        out_grp = sp.sibling(par) if par >= 0 else -1
        if out_grp >= 0:
            rel += _sample(sp.leafsets[out_grp], 6, fam + sp.names[out_grp])
        skip = moved_into.get(R, set())
        ne = nc = 0
        ancestral = {}
        for g in rec[:5]:
            a, b, locus = gi.insertion_test(g, rel, skip)
            ne, nc = ne + a, nc + b
            ancestral[g.gid] = locus
        n = ne + nc
        ev["ins"] = (ne - nc) / float(n) if n else 0.0
        ev["ins_n"] = n
        ev["depth"] = math.log2(len(sp.leafsets[R]))
        ev["comp"] = comp_w[R] * (sum(gi.comp_score(g) for g in rec[:10]) / len(rec[:10]) if rec else 0.0)
        fc = fam_copies.get(fam)
        ev["copies"] = fc[0] / float(len(fc[1])) if fc else 1.0
        gain = ev["source"] == "gain"
        if gain:
            cl = oc = 0
            for g in rec[:5]:
                a, b = gi.site_test(g, rel, skip)
                cl, oc = cl + a, oc + b
            ev["site"] = (cl - oc) / float(cl + oc) if cl + oc else 0.0
            ev["site_n"] = cl + oc
            ev["g_len"] = sum(gi.len_score(g) for g in rec[:10]) / len(rec[:10]) if rec else 0.0

        # clock candidates: which side holds the inserted copy? outgroup-only insertion test
        if ev["source"] == "clock" and out_grp >= 0:
            rel_o = _sample(sp.leafsets[out_grp], 8, fam + "|og|" + sp.names[out_grp])
            other = [gi.genes[g] for g in ev["other_genes"] if g in gi.genes]
            sib_skip = moved_into.get(sib, set())
            ev["dir"] = (_ins_score(gi, rec, rel_o, skip) - _ins_score(gi, other, rel_o, sib_skip))

        # neighbours transferred into the same branch (co-transferred block)
        nb = []
        for g in rec[:10]:
            hits = set()
            for h in gi.neighbors(g):
                for k2 in by_gene.get(h.gid, ()):
                    if k2 != k and events[k2]["R"] == R and events[k2]["family"] != fam:
                        hits.add(k2)
            nb.append(hits)
        ev["cotx"] = sum(1 for s in nb if s) / float(len(nb)) if nb else 0.0
        if gain:
            # block enrichment: neighbours gained by / moved into the same branch, minus the
            # genome-wide share (many families are absent only from a distant outgroup)
            vals = []
            for g in rec[:10]:
                hs = gi.neighbors(g)
                if hs:
                    inside = sum(1 for h in hs if h.gid in moved_into.get(R, ()) and h.family != fam)
                    vals.append(inside / float(len(hs)) - moved_bg.get((R, g.species), 0.0))
            ev["cotx"] = sum(vals) / len(vals) if vals else 0.0

        # mobile elements near the recipient copies
        vals, types = [], set()
        for g in rec[:10]:
            s, t = gi.me_score(g)
            vals.append(s - gi.me_background(g.species))
            types |= t
        ev["me"] = sum(vals) / len(vals) if vals else 0.0
        ev["me_types"] = sorted(types)

        # eukaryote contamination: foreign gene on a contig without native genes
        contam = 0
        if sp.is_leaf[R] and gi.domain(sp.names[R]) == "euk" and rec:
            bad = sum(1 for g in rec if gi.native_count(g, foreign) < 2)
            contam = int(2 * bad > len(rec))
        ev["contam"] = contam
        if gain:
            ev.update(near=0.0, donor_probs={}, donor_syn=0.0, recency=0.0, ctx=0.0)
            continue

        # donor ranking
        scores, syn_vals, rw, rs, cw, cs = {}, [], 0.0, 0.0, 0.0, 0.0
        for d, p in sorted(donors.items(), key=lambda x: -x[1])[:5]:
            dsp = _sample(sp.leafsets[d], 10, fam + sp.names[d])
            syn = gi.donor_synteny(rec, dsp, fam, ancestral)
            ctx = gi.context_shift(rec, dsp, fam)
            if ctx is not None:
                cw, cs = cw + p, cs + p * ctx
            cot = (sum(1 for s in nb if any(events[k2]["donors"].get(d, 0) >= 0.2 for k2 in s))
                   / float(len(nb))) if nb else 0.0
            r = ev["rec_by_donor"].get(d)
            rr = max(-3.0, min(3.0, r)) if r is not None else 0.0
            if r is not None:
                rw, rs = rw + p, rs + p * rr
            shared = 0
            if types:
                for Y in dsp:
                    if any(gi.me_score(h)[1] & types for h in gi.by_fam_sp.get((fam, Y), [])[:3]):
                        shared = 1
                        break
            if p >= 0.1:
                syn_vals.append(syn)
            scores[d] = (math.log(p) + G["syn"] * syn + G["cotx"] * cot
                         + G["recency"] * rr + G["me_shared"] * shared)
        ev["donor_probs"] = softmax_donors(scores)
        ev["donor_syn"] = max(syn_vals) if syn_vals else 0.0
        ev["recency"] = rs / rw if rw else 0.0
        ev["ctx"] = cs / cw if cw else 0.0


def merge_clock_pairs(events):
    """One 'exchange' event per clock test: recipient = side with the stronger insertion
    evidence, p_direction = confidence in that orientation (0.5 = unresolved)."""
    out, pairs = [], {}
    for ev in events:
        if ev["source"] == "clock":
            pairs.setdefault(ev["pair"], []).append(ev)
        else:
            ev.update(p_direction=1.0, alt_R=None, alt_genes=[])
            out.append(ev)
    for evs in pairs.values():
        if len(evs) == 1:       # other orientation already covered by a reconciliation event
            a, b = evs[0], None
            delta = a["dir"] + 0.25 * a["ins"]
        else:
            a, b = evs
            delta = (a["dir"] - b["dir"]) / 2.0 + 0.25 * (a["ins"] - b["ins"])
            if delta < 0:
                a, b, delta = b, a, -delta
            for f in ("ins", "ctx", "me", "cotx", "ins_n"):   # a trace on either side counts
                a[f] = max(a[f], b[f])
        a["p_direction"] = 1.0 / (1.0 + math.exp(-3.0 * delta))
        a["dir"] = abs(delta)
        a["alt_R"] = next(iter(a["donors"]))             # the other child lineage
        a["alt_genes"] = a["other_genes"]
        out.append(a)
    return out


def _ins_score(gi, genes, relatives, skip):
    ne = nc = 0
    for g in genes[:5]:
        a, b, _ = gi.insertion_test(g, relatives, skip)
        ne, nc = ne + a, nc + b
    return (ne - nc) / float(ne + nc) if ne + nc else 0.0


def flags_for(ev):
    f = []
    if ev["source"] == "gain":
        f.append("unsampled_donor")
        if ev["site_n"] >= 2 and ev["site"] >= 0.5:
            f.append("clean_insertion_site")
        if ev["site_n"] >= 2 and ev["site"] <= -0.5:
            f.append("locus_occupied")
        if ev["g_len"] <= -1.5:
            f.append("short_orf")
        if ev["g_patchy"] >= 0.5:
            f.append("patchy")
        if ev.get("n_gains", 1) > 1:
            f.append("recurrent_gain")
        if ev["g_para"] >= 0.5:
            f.append("paralog_present")
    if ev["comp"] >= 2.0:
        f.append("composition_outlier")
    if ev["source"] == "tree" and ev.get("copies", 1.0) >= 1.5:
        f.append("multicopy_family")          # hidden paralogy can fake (deep) transfers
    if ev["source"] == "clock":
        f.append("clock_anomaly")
        if ev["dir"] == 0:
            f.append("direction_unresolved")
    if ev["contam"]:
        f.append("possible_contamination")
    if ev["near"] >= 0.5:
        f.append("sister_clade_donor")
    if ev["source"] == "tree" and ev["f_T"] < 0.5:
        f.append("weak_tree_support")
    if ev["ins_n"] >= 2 and ev["ins"] >= 0.5:
        f.append("insertion_site")
    if ev["ins_n"] >= 2 and ev["ins"] <= -0.5:
        f.append("recipient_locus_conserved")
    if ev["ctx"] >= 0.2:
        f.append("context_shift")
    if ev["cotx"] >= 0.5:
        f.append("co_transferred_block")
    if ev["me"] >= 0.3:
        f.append("mobile_element_nearby")
    if ev["donor_syn"] >= 0.2:
        f.append("donor_synteny")
    if ev["recency"] >= 1.0:
        f.append("recent")
    return f


def _fmt(x):
    return "%.3f" % x if isinstance(x, float) else str(x)


def write_outputs(events, sp, gi, outdir, model):
    os.makedirs(outdir, exist_ok=True)
    events.sort(key=lambda e: -e["P_HGT"])
    with open(os.path.join(outdir, "hgt_events.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(EVENT_COLUMNS)
        for i, ev in enumerate(events, 1):
            ev["event_id"] = "E%05d" % i
            top = sorted(ev["donor_probs"].items(), key=lambda x: -x[1])[:3]
            top += [(None, 0.0)] * (3 - len(top))
            R = ev["R"]
            row = [ev["event_id"], ev["family"], sp.names[R], sp.label(R),
                   gi.domain(sp.names[R]) if sp.is_leaf[R] else "-", len(ev["genes"]),
                   ",".join(ev["genes"]), _fmt(ev["P_HGT"]), ev["tier"], ev["source"],
                   _fmt(float(ev["p_direction"])),
                   sp.label(ev["alt_R"]) if ev["alt_R"] is not None else "", ",".join(ev["alt_genes"])]
            if ev["source"] == "gain":
                top = [("unsampled", 1.0), (None, 0.0), (None, 0.0)]
            for d, p in top:
                row += [d if isinstance(d, str) else (sp.label(d) if d is not None else ""), _fmt(p)]
            row += [_fmt(float(ev[f])) for f in ALL_FEATURES]
            row += [",".join(ev["me_types"]), ",".join(ev["flags"])]
            w.writerow(row)

    # gene-level P = P(event) x P(this side is the recipient); clock exchanges split by p_direction
    best = {}
    for ev in events:
        sides = [(ev["genes"], ev["R"], ev["p_direction"], next(iter(ev["donor_probs"]), None))]
        if ev["alt_genes"]:
            sides.append((ev["alt_genes"], ev["alt_R"], 1.0 - ev["p_direction"], ev["R"]))
        for genes, R, pdir, donor in sides:
            pg = ev["P_HGT"] * pdir
            for g in genes:
                if g not in best or pg > best[g][0]:
                    best[g] = (pg, ev, R, donor)
    with open(os.path.join(outdir, "hgt_genes.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "species", "family", "P_HGT", "tier", "event_id", "source", "recipient",
                    "donor1", "p_donor1", "flags"])
        for g, (pg, ev, R, donor) in sorted(best.items(), key=lambda x: -x[1][0]):
            gene = gi.genes.get(g)
            if R == ev["R"] and ev["donor_probs"]:
                donor, pd = max(ev["donor_probs"].items(), key=lambda x: x[1])
            else:
                pd = 1.0
            w.writerow([g, gene.species if gene else "", gene.family if gene else ev["family"],
                        _fmt(pg), tier(pg), ev["event_id"], ev["source"], sp.label(R),
                        "unsampled" if ev["source"] == "gain" else (sp.label(donor) if donor is not None else ""),
                        _fmt(pd), ",".join(ev["flags"])])

    with open(os.path.join(outdir, "species_tree.labeled.nwk"), "w") as fh:
        fh.write(sp.to_newick() + "\n")

    tiers = {}
    for ev in events:
        tiers[ev["tier"]] = tiers.get(ev["tier"], 0) + 1
    per_sp = {}
    for g, (pg, ev, _, _) in best.items():
        if pg >= 0.6 and g in gi.genes:
            s = gi.genes[g].species
            per_sp[s] = per_sp.get(s, 0) + 1
    lines = ["model: %s" % model.get("source", "?"),
             "candidate events (f_T >= min_freq): %d" % len(events),
             "  " + ", ".join("%s=%d" % (t, tiers.get(t, 0)) for t in ("high", "medium", "low", "unlikely")),
             "genes in medium/high events: %d" % sum(per_sp.values()),
             "", "medium/high HGT genes per genome:"]
    for s, c in sorted(per_sp.items(), key=lambda x: -x[1]):
        lines.append("  %-20s %d" % (s, c))
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def run_scan(a):
    t0 = time.time()
    with open(a.species_tree) as fh:
        sp = SpeciesTree(parse_newick(fh.read()))
    domains = load_species_meta(a.species_meta) if a.species_meta else {}
    genes_all = load_genes(a.genes)
    genes = [g for g in genes_all if g.species in sp.leaf_index]
    if len(genes) < len(genes_all):
        log("ignored %d genes from species absent in the species tree" % (len(genes_all) - len(genes)))
    mes = load_mes(a.me) if a.me else []
    gene_by_id = {g.gid: g for g in genes}
    log("species: %d, genes: %d, mobile elements: %d" % (len(sp.leaf_index), len(genes), len(mes)))

    tasks, skipped = [], 0
    for fam, paths in sorted(discover_trees(a.gene_trees).items()):
        trees = []
        for p in paths:
            trees.extend(read_trees(p))
        try:
            first = parse_newick(trees[0]) if trees else None
        except ValueError:
            first = None
        if first is None:
            skipped += 1
            continue
        gene_sp = {lf.name: sp.leaf_index[gene_by_id[lf.name].species]
                   for lf in first.leaves() if lf.name in gene_by_id}
        if len(gene_sp) < 2:
            skipped += 1
            continue
        tasks.append((fam, subsample(trees, a.samples), gene_sp))
    log("gene families: %d (skipped %d with <2 mappable genes or unreadable trees)" % (len(tasks), skipped))
    if not tasks:
        raise SystemExit("no usable gene trees found in %s" % a.gene_trees)

    costs = tuple(float(x) for x in a.costs.split(","))
    dated = a.dated == "on" or (a.dated == "auto" and sp.has_lengths())
    if a.dated == "on" and not sp.has_lengths():
        raise SystemExit("--dated on needs branch lengths (time units) on every species-tree branch")
    log("transfers restricted to time-overlapping branches: %s" % ("yes" if dated else "no (undated)"))
    collapse = getattr(a, "collapse", 0.0)
    if collapse > 0:
        log("gene-tree branches with support < %g contracted, re-resolved along the species tree "
            "(families whose ML tree carries supports)" % collapse)
    split = getattr(a, "split_paralogs", 0)
    if split > 0:
        parts = _pool_map(_split_work, tasks, a.threads,
                          (sp.to_newick(), sp.names, costs, dated, a.bt, a.seed, {}, collapse, split), None)
        n_split = sum(1 for p in parts if len(p) > 1)
        tasks = []
        for p in parts:
            for name, trees, gsp in p:
                for g in gsp:
                    gene_by_id[g].family = name
                if len(gsp) >= 2 and trees:
                    tasks.append((name, trees, gsp))
        log("ancient duplications (species overlap >= %d): %d families split into %d subfamilies; "
            "%d families with trees" % (split, n_split, sum(len(p) for p in parts if len(p) > 1), len(tasks)))
    gi = GenomeIndex(genes, mes, domains, a.domain, a.window)
    if getattr(a, "composition", None):
        comp = load_composition(a.composition)
        gi.set_composition(comp)
        log("composition (GC, GC3) for %d genes" % len(comp))

    bg = background_distances(tasks, sp, seed=a.seed)
    log("background distances for %d species pairs" % len(bg))
    init_args = (sp.to_newick(), sp.names, costs, dated, a.bt, a.seed, bg, collapse, split)
    results = _pool_map(_work, tasks, a.threads, init_args, "reconciled")

    events = build_events(results, a.min_freq)
    n_tree = len(events)
    if a.clock_z > 0 and bg:
        clock_events, stats = build_clock_events(results, sp, a.clock_z, events)
        events += clock_events
        log("molecular-clock scan: %d speciation-node tests, %d candidates (z <= -%.1f)"
            % (sum(v[2] for v in stats.values()), len(clock_events), a.clock_z))
    n_clock = len(events) - n_tree
    if getattr(a, "gain", True):
        gains = build_gain_events(gi, sp, costs[1], costs[2])
        events += gains
        log("gain candidates (family confined to a non-root clade, unsampled donor): %d" % len(gains))
    log("candidate events: %d from reconciliation, %d from clock, %d gains; adding synteny / ME / "
        "composition evidence" % (n_tree, n_clock, len(events) - n_tree - n_clock))
    annotate(events, sp, gi)
    events = merge_clock_pairs(events)
    model = load_model(a.model)
    P = predict(model, [dict({f: ev[f] for f in ALL_FEATURES}, source=ev["source"]) for ev in events])
    for ev, p in zip(events, P):
        ev["P_HGT"] = float(p)
        ev["tier"] = tier(float(p))
        ev["flags"] = flags_for(ev)
    lines = write_outputs(events, sp, gi, a.out, model)
    for ln in lines[:4]:
        log(ln)
    log("done in %.1fs -> %s" % (time.time() - t0, a.out))
