"""Single-genome donor screen: which genes of one genome look like they came from mammals, from
prokaryotes, or from some other distant lineage?

Works on similarity alone (no gene trees), so it gives candidates, not proof. For every gene:

  outside hits   hits covering >= MIN_QCOV of the gene, outside the genome's own genus ("self")
  repeat filter  hits that lie mostly on repeat-like regions and are < 40% identical are dropped;
                 genes whose top outside hits are mostly such hits are "repeat"
  donor group    the group of the closest relatives outside the home group: top-10 clean
                 outside-home hits, weighted by bit score (17 groups)
  home group     the genome's own neighbourhood: its phylum for prokaryotes; Metazoa, Fungi or
                 Viridiplantae for eukaryotes, else the eukaryotic supergroup (Amoebozoa, Sar ...)
  Alien Index    AI = ln(best E inside home) - ln(best E outside home)   (Gladyshev et al. 2008);
                 a call needs AI >= 15 (near-ties of conserved genes are not evidence); AI >= 45: strong
  category       mammal / prokaryote / eukaryote / virus by the donor group when AI >= 15
                 (for a mammal donor inside the home group, e.g. a nematode, the AI is taken
                 against the non-mammal part of the home group);
                 native when AI < 15; orphan without outside hits; repeat as above
  support        distinct species among the top-10 clean hits of the donor group: a single
                 eukaryote species is a typical sign of contamination in its genome assembly
"""
from __future__ import annotations

import math
from collections import Counter

from .repeats import REPEAT_DOMINANCE, is_repeat_hit

MIN_QCOV = 0.4
MIN_AI = 15.0        # a call needs the outside hit to be clearly better (E-value ~3e6-fold): near-ties of
                     # conserved genes in sparsely sampled lineages are not evidence
STRONG_AI = 45.0
TOP = 10

BACTERIA, ARCHAEA, EUKARYOTA, VIRUSES = 2, 2157, 2759, 10239
MAMMALIA, METAZOA, FUNGI, VIRIDIPLANTAE = 40674, 33208, 4751, 33090

CATEGORY_LABELS = {
    "mammal": "possibly from mammals",
    "prokaryote": "possibly from prokaryotes",
    "eukaryote": "possibly from other eukaryotes",
    "virus": "possibly from viruses / phages",
    "native": "closest relatives in its own lineage",
    "orphan": "no outside homolog",
    "repeat": "repeat-driven matches only (no call)",
}


def group_class(tax, t):
    lin = set(tax.lineage(t))
    if MAMMALIA in lin:
        return "mammal"
    if EUKARYOTA in lin:
        return "eukaryote"
    if BACTERIA in lin or ARCHAEA in lin:
        return "prokaryote"
    if VIRUSES in lin:
        return "virus"
    return "unknown"


HOME_RANKS = ("auto", "genus", "family", "order", "class", "phylum", "kingdom", "domain")


def home_group(tax, self_taxid, rank="auto"):
    """(taxid, name) of the genome's home neighbourhood, used as the Alien Index in-group.
    auto: the phylum for prokaryotes; Metazoa / Fungi / Viridiplantae for those eukaryotes, the
    eukaryotic supergroup (Amoebozoa, Sar, Discoba ...) for protists.
    A narrower rank (e.g. family) also calls transfers from within the phylum."""
    lin = tax.lineage(self_taxid)
    s = set(lin)
    if rank == "domain":
        for g in (BACTERIA, ARCHAEA, EUKARYOTA, VIRUSES):
            if g in s:
                return g, tax.name.get(g, str(g))
    elif rank == "kingdom":
        rank = "auto" if EUKARYOTA in s else "domain"
        return home_group(tax, self_taxid, rank)
    elif rank not in ("auto", "", None):
        t = tax.at_rank(self_taxid, rank)
        if t:
            return t, tax.name.get(t, str(t))
    if EUKARYOTA in s:
        for g in (METAZOA, FUNGI, VIRIDIPLANTAE):
            if g in s:
                return g, tax.name.get(g, str(g))
        # protists: the eukaryotic supergroup (the lineage node right below Eukaryota, e.g. Amoebozoa,
        # Sar, Discoba), so that genes from animals, fungi or plants still count as foreign
        for x in lin:
            if tax.parent.get(x) == EUKARYOTA:
                return x, tax.name.get(x, str(x))
        return EUKARYOTA, "Eukaryota"
    ph = tax.at_rank(self_taxid, "phylum")
    if ph:
        return ph, tax.name.get(ph, str(ph))
    for g in (BACTERIA, ARCHAEA, VIRUSES):
        if g in s:
            return g, tax.name.get(g, str(g))
    return (lin[-1] if lin else 1), "root"


def infer_self(hits, tax, min_ident=95.0, min_qcov=0.8):
    """Guess the genome's own taxon from near-identical best hits: the most common genus (or family,
    or species) among them. Returns (taxid, rank, n_genes_supporting) or (None, None, 0)."""
    best = {}
    for h in hits:
        if h["pident"] >= min_ident and h["qcov"] >= min_qcov:
            if h["query"] not in best or h["bits"] > best[h["query"]]["bits"]:
                best[h["query"]] = h
    if not best:
        return None, None, 0
    for rank, need in (("genus", 0.3), ("family", 0.3), ("species", 0.2)):
        c = Counter(tax.at_rank(h["taxid"], rank) for h in best.values())
        c.pop(None, None)
        if c:
            t, n = c.most_common(1)[0]
            if n >= need * len(best):
                return t, rank, n
    return None, None, 0


def _ln(e):
    return math.log(max(e, 1e-200))


def classify_gene(gene, length, hits, tax, self_taxid, home, qreg=None, treg=None):
    """hits: dicts with query, target, taxid, pident (0-100), evalue, bits, qstart, qend, tstart, tend,
    qcov (fraction). Returns one result row (dict)."""
    self_lin = set(tax.lineage(self_taxid)) if self_taxid else set()
    out = []
    for h in sorted(hits, key=lambda h: -h["bits"]):
        if h["qcov"] < MIN_QCOV:
            continue
        lin = tax.lineage(h["taxid"])
        if self_taxid and self_taxid in lin:
            continue
        rep = is_repeat_hit(qreg or [], (treg or {}).get(h["target"], []), h.get("qstart"), h.get("qend"),
                            h.get("tstart"), h.get("tend"), h["pident"] / 100.0)
        out.append((h, set(lin), rep))
    row = dict(gene_id=gene, length=length, n_outside=len(out), category="orphan", strength="",
               donor_group="", donor_share="", support=0, AI="", best_hit="", best_taxon="", best_group="",
               best_pident="", best_evalue="", best_home_taxon="", best_home_pident="", mammal_hit="",
               repeat_hits=sum(1 for _, _, r in out if r))
    if not out:
        return row
    top = out[:TOP]
    tb = sum(h["bits"] for h, _, _ in top)
    rb = sum(h["bits"] for h, _, r in top if r)
    if rb >= REPEAT_DOMINANCE * tb:
        row.update(category="repeat")
        return row
    clean = [(h, lin) for h, lin, r in out if not r]
    b = clean[0][0]
    home_t, _ = home
    in_home = [(h, lin) for h, lin in clean if home_t in lin]
    away = [(h, lin) for h, lin in clean if home_t not in lin]
    # donor group: where the closest outside-home relatives are (top 10 by bit score); support: their species
    w, species = Counter(), {}
    for h, lin in (away or clean)[:TOP]:
        g = tax.group(h["taxid"])
        w[g] += h["bits"]
        species.setdefault(g, set()).add(tax.at_rank(h["taxid"], "species") or h["taxid"])
    grp, gb = w.most_common(1)[0]
    if (group_class(tax, b["taxid"]) == "mammal" and MAMMALIA not in self_lin
            and home_t in tax.lineage(MAMMALIA)):
        # mammal donor inside the home group (e.g. a nematode): compare with the non-mammal part
        cls = "mammal"
        e_in = min((h["evalue"] for h, lin in in_home if MAMMALIA not in lin), default=1.0)
        e_out = min((h["evalue"] for h, lin in in_home if MAMMALIA in lin), default=1.0)
    else:
        cls = group_class(tax, away[0][0]["taxid"]) if away else "native"
        e_in = min((h["evalue"] for h, lin in in_home), default=1.0)
        e_out = min((h["evalue"] for h, lin in away), default=1.0)
    ai = _ln(e_in) - _ln(e_out)
    if ai >= MIN_AI:
        cat = cls if cls in ("mammal", "eukaryote", "prokaryote", "virus") else "native"
        if cat == "mammal" and MAMMALIA in self_lin:
            cat = "native"
    else:
        cat = "native"
    bh = in_home[0][0] if in_home else None
    mam = next((h for h, lin in clean if MAMMALIA in lin), None)
    row.update(category=cat, strength=("strong" if ai >= STRONG_AI else "weak") if cat != "native" else "",
               donor_group=grp, donor_share="%.2f" % (gb / sum(w.values())), support=len(species.get(grp, ())),
               AI="%.1f" % ai, best_hit=b["target"], best_taxon=tax.name.get(b["taxid"], str(b["taxid"])),
               best_group=tax.group(b["taxid"]), best_pident="%.1f" % b["pident"], best_evalue="%.2g" % b["evalue"],
               best_home_taxon=tax.name.get(bh["taxid"], "") if bh else "",
               best_home_pident="%.1f" % bh["pident"] if bh else "",
               mammal_hit=("%s (%.0f%% id, E %.1g)" % (tax.name.get(mam["taxid"], ""), mam["pident"], mam["evalue"])
                           if mam else ""))
    return row


S_MIN_PROB, S_MAX_EVALUE, S_MIN_QCOV, S_MIN_TM = 0.5, 1e-3, 0.4, 0.5   # Foldseek: TM >= 0.5 = same fold


def classify_structure(hits, tax, self_taxid, qreg=None, treg=None):
    """Foldseek hits (dicts with target, taxid, name, evalue, bits, prob, tm, fident, qcov, coordinates)
    of one predicted structure -> structural group of the closest same-fold matches outside self."""
    good = []
    for h in sorted(hits, key=lambda h: -h["bits"]):
        if (h["evalue"] > S_MAX_EVALUE or h["prob"] < S_MIN_PROB or h["qcov"] < S_MIN_QCOV or h["tm"] < S_MIN_TM
                or not h["taxid"] or (self_taxid and self_taxid in tax.lineage(h["taxid"]))):
            continue
        rep = is_repeat_hit(qreg or [], (treg or {}).get(h["target"], []), h.get("qstart"), h.get("qend"),
                            h.get("tstart"), h.get("tend"), h["fident"])
        good.append((h, rep))
    res = dict(struct_group="no structural match" if not good else "", struct_share="", struct_tm="",
               struct_hit="", struct_name="", struct_taxon="", struct_mammal="")
    if not good:
        return res
    tb = sum(h["bits"] for h, _ in good[:TOP])
    if sum(h["bits"] for h, r in good[:TOP] if r) >= REPEAT_DOMINANCE * tb:
        res.update(struct_group="repeat")
        return res
    clean = [h for h, r in good if not r]
    w = Counter()
    for h in clean[:TOP]:
        w[tax.group(h["taxid"])] += h["bits"]
    grp, gb = w.most_common(1)[0]
    b = clean[0]
    mam = next((h for h in clean if MAMMALIA in tax.lineage(h["taxid"])), None)
    res.update(struct_group=grp, struct_share="%.2f" % (gb / sum(w.values())), struct_tm="%.2f" % b["tm"],
               struct_hit=b["target"], struct_name=b.get("name", ""),
               struct_taxon=tax.name.get(b["taxid"], str(b["taxid"])),
               struct_mammal=("%s (%s, TM %.2f)" % (tax.name.get(mam["taxid"], ""), mam.get("name", "")[:40], mam["tm"])
                              if mam else ""))
    return res


def summarize(rows):
    c = Counter(r["category"] for r in rows)
    strong = Counter(r["category"] for r in rows if r["strength"] == "strong")
    groups = Counter(r["donor_group"] for r in rows if r["category"] not in ("native", "orphan", "repeat"))
    return dict(n_genes=len(rows), categories=dict(c), strong=dict(strong), donor_groups=dict(groups.most_common()),
                single_species=sum(1 for r in rows if r["category"] in ("mammal", "eukaryote") and r["support"] == 1))
