"""Stage family_a (pipeline family_a): benchmark v2, family A re-cut on the development chromosomes (rules
bench/hard/V2_FAMILY_A.md, thresholds configs/benchmark_v2/family_a.yaml). Label code only: it reads genome records,
panel and outgroup genomes, GTDB and the NCBI taxdump, and never a score, a reference library or a library search.
It never opens bench/hard/results/sealed.tsv or v2_sealed_candidates.tsv (the parts come from v2_reference_set.tsv).

  python -m ci.stages.family_a RUN_DIR taxa    GTDB R232 (md5 against the release's MD5SUM.txt, stored in the CAS), the
                                              chromosomes' GTDB placement, the panel re-cut, the outgroups
  python -m ci.stages.family_a RUN_DIR fetch   the panel and outgroup genomes not in the store (ENA, CAS), their DNA
                                              checked for N-runs; an outgroup genome that fails is replaced in salt order

Configuration (`famA:`): store (the island store), spec (default configs/benchmark_v2/family_a.yaml), threads.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys

import yaml

from ci.hashing import sha256_file
from ci.stages.common import Stage, main_args, write_json

SPEC = "configs/benchmark_v2/family_a.yaml"
RANK = {"d": "domain", "p": "phylum", "c": "class", "o": "order", "f": "family", "g": "genus", "s": "species"}


def _bench_panel(repo):
    """bench/hard/panel.py (the ENA queries of the old panels), importable by its directory."""
    d = repo("bench/hard")
    if d not in sys.path:
        sys.path.insert(0, d)
    import panel
    return panel


def order_key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def gtdb_key(acc):
    """The digits of a GCA/GCF accession (a paired GCA and GCF map to one key), or None."""
    m = re.search(r"GC[AF]_(\d{9})", acc or "")
    return m.group(1) if m else None


def gtdb_taxonomy(path):
    """{key: {rank: name}} of the GTDB taxonomy file (gz), names without their rank prefix."""
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            gid, _, lin = line.rstrip("\n").partition("\t")
            k = gtdb_key(gid)
            if not k:
                continue
            out[k] = {RANK[p[0]]: p[3:] for p in lin.split(";") if len(p) > 3 and p[1:3] == "__" and p[0] in RANK}
    return out


def read_tree(text):
    """An iterative newick parser: parent, children, branch length, leaf name and the taxa named in internal labels
    (GTDB writes them as '<support>:<rank>__<name>; <rank>__<name>')."""
    parent, children, length, name, taxa = [-1], [[]], [0.0], [None], [[]]
    cur, i, n = 0, 0, len(text)

    def new(p):
        parent.append(p)
        children.append([])
        length.append(0.0)
        name.append(None)
        taxa.append([])
        children[p].append(len(parent) - 1)
        return len(parent) - 1
    while i < n:
        c = text[i]
        if c == "(":
            cur = new(cur)
            i += 1
        elif c == ",":
            cur = new(parent[cur])
            i += 1
        elif c == ")":
            cur = parent[cur]
            i += 1
        elif c == ":":
            j = i + 1
            while j < n and text[j] not in ",();":
                j += 1
            length[cur] = float(text[i + 1:j] or 0)
            i = j
        elif c == ";":
            break
        elif c in " \n\r\t":
            i += 1
        else:
            if c == "'":
                j = text.index("'", i + 1)
                label = text[i + 1:j]
                i = j + 1
            else:
                j = i
                while j < n and text[j] not in ",():;":
                    j += 1
                label = text[i:j]
                i = j
            found = re.findall(r"\b([dpcofgs])__([^;:']+)", label)
            if children[cur]:
                taxa[cur] = [(RANK[r], v.strip()) for r, v in found]
            else:
                name[cur] = label.strip()
    return dict(parent=parent, children=children, length=length, name=name, taxa=taxa)


def leaves_under(tree, node):
    out, stack = [], [node]
    while stack:
        x = stack.pop()
        if tree["children"][x]:
            stack.extend(tree["children"][x])
        else:
            out.append(x)
    return out


def mrca(tree, leaves):
    """The most recent common ancestor of the given leaves."""
    paths = []
    for leaf in leaves:
        p, x = [], leaf
        while x != -1:
            p.append(x)
            x = tree["parent"][x]
        paths.append(p[::-1])
    k = 0
    while all(len(p) > k for p in paths) and len({p[k] for p in paths}) == 1:
        k += 1
    return paths[0][k - 1]


def sister_clades(tree, node):
    """[(clade roots, patristic distance from `node` to the clade's attachment)] walking towards the root."""
    out, dist, x = [], 0.0, node
    while tree["parent"][x] != -1:
        p = tree["parent"][x]
        dist += tree["length"][x]
        sib = [c for c in tree["children"][p] if c != x]
        if sib:                                      # the parser's virtual root has the tree's root as its only child
            out.append((sib, dist))
        x = p
    return out


def dev_chromosomes(repo):
    """The development chromosomes: v2_reference_set.tsv part development, intersected with dev_genomes.tsv."""
    with open(repo("bench/hard/results/v2_reference_set.tsv")) as f:
        dev = {r["accession"]: r for r in csv.DictReader(f, delimiter="\t") if r["part"] == "development"}
    with open(repo("bench/islands/results/dev_genomes.tsv")) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["accession"] in dev]
    return rows


def forbidden(repo, tax):
    """NCBI genus and family taxids no panel or outgroup genome may come from: EXCLUDE.tsv genera, the genera of every
    non-development chromosome of the register (sealed_v1, secondary, candidates, demoted) and their families, and the
    island-set genera (all 104 chromosomes)."""
    names = set()
    with open(repo("bench/hard/EXCLUDE.tsv")) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] == "genus":
                names.add(p[2])
    with open(repo("bench/hard/results/v2_reference_set.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    island_genera = {r["organism"].split()[0] for r in rows if r["organism"]}
    names |= {r["organism"].split()[0] for r in rows if r["part"] != "development" and r["organism"]}
    genus_ids, family_ids = set(), set()
    for nm in names:
        for t in tax.by_name.get(nm, []):
            if tax.rank.get(t) == "genus":
                genus_ids.add(t)
                fam = tax.at_rank(t, "family")
                if fam:
                    family_ids.add(fam)
    island_ids = {t for nm in island_genera for t in tax.by_name.get(nm, []) if tax.rank.get(t) == "genus"}
    return dict(genera=genus_ids, families=family_ids, island_genera=island_ids, names=sorted(names))


def mapped(rows, gt):
    """ENA candidate rows with their GTDB lineage; unmapped rows dropped (counted)."""
    out, lost = [], 0
    for r in rows:
        k = gtdb_key(r["assembly"])
        if k in gt:
            out.append(dict(r, gtdb=gt[k]))
        else:
            lost += 1
    return out, lost


def recut_panel(sp, genus, pool, same_n, other_n, salt, species=None):
    """The same and other rows of one chromosome (V2_FAMILY_A.md, Panels): other species are those of its GTDB genus
    (part 1) or, for a chromosome walked to a larger clade (the amendment of 2026-11-21), the given `species`."""
    key = lambda r: order_key("%s|%s" % (r["assembly"], salt))
    same = sorted((r for r in pool if r["gtdb"].get("species") == sp), key=key)[:same_n]
    ok = (lambda r: r["gtdb"].get("genus") == genus) if species is None else (lambda r: r["gtdb"].get("species") in species)
    by_sp = {}
    for r in sorted((r for r in pool if r["gtdb"].get("species") != sp and ok(r)), key=key):
        by_sp.setdefault(r["gtdb"]["species"], r)
    other = sorted(by_sp.values(), key=key)[:other_n]
    return same, other


def representative(tree, gt, sp):
    """The tree leaf (index) of a GTDB species' representative genome."""
    for i, nm in enumerate(tree["name"]):
        if nm and not tree["children"][i] and gt.get(gtdb_key(nm), {}).get("species") == sp:
            return i
    raise SystemExit("family_a: no representative of s__%s in the GTDB tree" % sp)


def genus_mrca(tree, gt, genus):
    reps = [i for i, nm in enumerate(tree["name"]) if nm and not tree["children"][i] and gt.get(gtdb_key(nm), {}).get("genus") == genus]
    if not reps:
        raise SystemExit("family_a: no representative of g__%s in the GTDB tree" % genus)
    return mrca(tree, reps)


def ncbi_species(tax, taxid):
    """The NCBI species taxid of a taxid, 0 when the pinned taxdump does not know it."""
    from ci.taxonomy import UnknownTaxid
    try:
        return tax.at_rank(int(taxid), "species")
    except (UnknownTaxid, ValueError):
        return 0


def walk_panel(tree, gt, sp, genus, pool, tax, other_min, majority=0.5, banned=(), anchor_rows=()):
    """The amendment of 2026-11-21 for a chromosome whose GTDB genus offers fewer than other_min other species. Anchor:
    the pool row of its species' representative gives an NCBI species; when more than `majority` of the pool's
    assemblies of that NCBI species sit in one other GTDB genus G', start at G''s MRCA, else at its own genus's. Walk
    towards the root to the first node under which at least other_min GTDB species other than its own have pool
    assemblies (never a species of a GTDB genus named after a forbidden genus). The representative's row is looked up in
    the pool, else in anchor_rows (the unit's candidates before the library and island-set filters); without it, or
    without an NCBI species for it, the walk stops with an error rather than start from a misplaced genus. Returns
    (node, species, record)."""
    rep = representative(tree, gt, sp)
    key = gtdb_key(tree["name"][rep])
    row = next((r for r in pool if gtdb_key(r["assembly"]) == key), None)
    where = "pool"
    if row is None:
        row, where = next((r for r in anchor_rows if gtdb_key(r["assembly"]) == key), None), "candidates"
    if row is None:
        raise SystemExit("family_a: the representative %s of s__%s is not among the unit's ENA candidates" % (tree["name"][rep], sp))
    ncbi_sp = ncbi_species(tax, row["tax_id"])
    if not ncbi_sp:
        raise SystemExit("family_a: the representative %s of s__%s has no NCBI species in the taxdump" % (tree["name"][rep], sp))
    start, rec = genus, dict(representative=tree["name"][rep], representative_from=where)
    by = {}
    for r in pool:
        if ncbi_species(tax, r["tax_id"]) == ncbi_sp:
            by[r["gtdb"]["genus"]] = by.get(r["gtdb"]["genus"], 0) + 1
    if not by:
        raise SystemExit("family_a: no pool assembly of the NCBI species of s__%s's representative" % sp)
    top = max(sorted(by), key=lambda g: by[g])
    share = by[top] / float(sum(by.values()))
    rec.update(ncbi_species=ncbi_sp, ncbi_species_genera=by, majority_genus=top, majority_share=round(share, 4))
    if top != genus and share > majority:
        start = top
    n_sp = {}
    for r in pool:
        n_sp[r["gtdb"]["species"]] = n_sp.get(r["gtdb"]["species"], 0) + 1
    node, steps = genus_mrca(tree, gt, start), 0
    while True:
        under = {gt.get(gtdb_key(tree["name"][x] or ""), {}).get("species") for x in leaves_under(tree, node)}
        cand = {x for x in under if x and x != sp and n_sp.get(x) and re.sub(r"_[A-Z]+$", "", x.split()[0]) not in banned}
        if len(cand) >= other_min:
            break
        if tree["parent"][node] == -1:
            raise SystemExit("family_a: no clade with %d other species above s__%s" % (other_min, sp))
        node, steps = tree["parent"][node], steps + 1
    rec.update(start_genus=start, steps=steps, species=sorted(cand))
    return node, cand, rec


def eligible(rows, tax, forb, order_id, lib, island_taxa):
    """Rows of one candidate genus that may serve as outgroup genomes."""
    out = []
    for r in rows:
        t = int(r["tax_id"])
        try:
            g, fam, o = tax.at_rank(t, "genus"), tax.at_rank(t, "family"), tax.at_rank(t, "order")
        except Exception:                           # noqa: BLE001 - an unknown taxid is not eligible
            continue
        if o != order_id or g in forb["genera"] or g in forb["island_genera"] or fam in forb["families"]:
            continue
        if r["assembly"].split(".")[0].replace("GCF_", "GCA_") in lib or t in island_taxa:
            continue
        out.append(r)
    return out


def median_bases(rows):
    """The median ENA base count of assembly rows (0 when none has one)."""
    xs = sorted(int(r["base_count"]) for r in rows if str(r.get("base_count") or "").isdigit())
    if not xs:
        return 0
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0


def choose_outgroups(tree, gt, focal_genus, candidates_of, tax, forb, order_id, lib, island_taxa, k, n, min_asm, salt, min_bases=0, start=None):
    """The outgroup genera and genomes of one unit (V2_FAMILY_A.md, Outgroups, with the amendment of 2026-11-21: a genus
    whose eligible assemblies have a median length below min_bases is not eligible). candidates_of(ncbi genus taxid)
    gives the ENA rows of an NCBI genus; start: the node to walk from (default the focal genus's MRCA; a walked unit's
    node). Returns [dict(genus, distance, genomes, considered, median_bases)]."""
    out = []
    for clade, dist in sister_clades(tree, genus_mrca(tree, gt, focal_genus) if start is None else start):
        genera = {}
        for root in clade:
            for leaf in leaves_under(tree, root):
                g = gt.get(gtdb_key(tree["name"][leaf] or ""), {}).get("genus")
                if g and g != focal_genus:
                    genera[g] = genera.get(g, 0) + 1
        best = None
        for g in sorted(genera, key=lambda x: (-genera[x], order_key(x))):
            ncbi = [t for t in tax.by_name.get(re.sub(r"_[A-Z]+$", "", g), []) if tax.rank.get(t) == "genus"]
            rows = []
            for t in ncbi:
                rows += candidates_of(t)
            rows, _ = mapped(rows, gt)
            rows = [r for r in eligible(rows, tax, forb, order_id, lib, island_taxa) if r["gtdb"].get("genus") == g]
            if len(rows) >= min_asm and median_bases(rows) >= min_bases and (best is None or len(rows) > len(best[1])):
                best = (g, rows)
            if best:                                   # genera are tried in descending order of representatives
                break
        if best is None:
            continue
        g, rows = best
        key = lambda r: order_key("%s|%s" % (r["assembly"], salt))
        rows = sorted(rows, key=key)
        picked, species = [], set()
        for r in rows:                                 # different GTDB species first
            if r["gtdb"]["species"] not in species:
                picked.append(r)
                species.add(r["gtdb"]["species"])
        picked += [r for r in rows if r not in picked]
        out.append(dict(genus=g, distance=round(dist, 4), genomes=picked, considered=len(genera), median_bases=median_bases(rows)))
        if len(out) == k:
            break
    return out


def gtdb_files(cfg_gtdb, cas, get, store, known=None):
    """The GTDB files of the release, each checked against MD5SUM.txt and stored in the CAS: {name: {path, sha256, md5}}
    and the MD5SUM.txt record. known: {name: sha256} pinned by the profile; such a file already in the CAS is used, not
    downloaded, when its content has that sha256 and the release's md5."""
    base = cfg_gtdb["base"].rstrip("/")
    md5txt = get(base + "/MD5SUM.txt")
    sums = {}
    for line in md5txt.decode(errors="replace").splitlines():
        p = line.split()
        if len(p) == 2:
            sums[p[1].lstrip("./").lstrip("*")] = p[0]
    mpath, msha = store(cas, md5txt, ".txt")
    out = {}
    for name in cfg_gtdb["files"]:
        if name not in sums:
            raise SystemExit("family_a: %s is not listed in the release's MD5SUM.txt" % name)
        suffix = "." + name.split(".", 1)[1]
        pin = (known or {}).get(name)
        cached = os.path.join(cas, pin[:2], pin + suffix) if pin else None
        data, from_cas = None, False
        if cached and os.path.exists(cached):
            with open(cached, "rb") as f:
                data = f.read()
            from_cas = True
            if hashlib.sha256(data).hexdigest() != pin:
                raise SystemExit("family_a: the CAS copy of %s does not match its pinned sha256" % name)
        if data is None:
            data = get(base + "/" + name)
        if hashlib.md5(data).hexdigest() != sums[name]:
            raise SystemExit("family_a: %s does not match its md5" % name)
        path, sha = store(cas, data, suffix)
        out[name] = dict(path=path, sha256=sha, md5=sums[name], bytes=len(data), cached=from_cas)
    return out, dict(path=mpath, sha256=msha)


def dev_placement(rows, store, gt, tax, candidates_of):
    """GTDB species and genus of every development chromosome: its record's DBLINK assembly, else the GTDB species most
    ENA assemblies of its NCBI species fall into."""
    out = {}
    for r in rows:
        with open(os.path.join(store, "dev_genomes", r["accession"] + ".gb")) as f:
            head = f.read(20000)
        m = re.search(r"Assembly:\s*(GC[AF]_\d{9}\.\d+)", head)
        k = gtdb_key(m.group(1)) if m else None
        if k and k in gt:
            out[r["accession"]] = dict(gt[k], method="dblink", assembly=m.group(1))
            continue
        t = re.search(r'/db_xref="taxon:(\d+)"', head)
        spid = tax.at_rank(int(t.group(1)), "species") if t else 0
        rows_sp, _ = mapped(candidates_of(spid), gt) if spid else ([], 0)
        votes = {}
        for x in rows_sp:
            key = (x["gtdb"].get("genus"), x["gtdb"].get("species"))
            votes[key] = votes.get(key, 0) + 1
        if not votes:
            raise SystemExit("family_a: no GTDB placement for %s" % r["accession"])
        (g, s), v = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
        out[r["accession"]] = dict(genus=g, species=s, method="ncbi_species_majority", votes=v, of=sum(votes.values()), ncbi_species=spid)
    return out


def do_taxa(run_dir):
    with Stage(run_dir, "famA_taxa", requires=("pre",)) as st:
        cfg = st.config
        fc = cfg["famA"]
        with open(st.repo(fc.get("spec", SPEC))) as f:
            spec = yaml.safe_load(f)
        from ci.stages import fetch as FE
        from ci.taxonomy import Taxonomy
        cas = os.path.join(cfg["data_root"], "cas")
        files, md5rec = gtdb_files(spec["gtdb"], cas, lambda u: FE.get_ranged(u, parts=8, min_bytes=1_000_000), FE.store, fc.get("gtdb_known"))
        gt = gtdb_taxonomy(files["bac120_taxonomy_r232.tsv.gz"]["path"])
        with gzip.open(files["bac120_r232.tree.gz"]["path"], "rt") as f:
            tree = read_tree(f.read())
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        panel = _bench_panel(st.repo)
        cache = {}

        def candidates_of(taxid):
            if taxid not in cache:
                cache[taxid] = panel.candidates(int(taxid))
            return cache[taxid]
        store = fc["store"]
        rows = dev_chromosomes(st.repo)
        if len(rows) != 24:
            raise SystemExit("family_a: %d development chromosomes, expected 24" % len(rows))
        place = dev_placement(rows, store, gt, tax, candidates_of)
        lib = panel.library_assemblies(os.path.join(store, "panel", "library_assemblies.json"))
        with open(os.path.join(store, "panel", "island_set_taxa.json")) as f:
            island_taxa = {int(v) for v in json.load(f).values()}
        forb = forbidden(st.repo, tax)
        units = {}
        for r in rows:
            units.setdefault(place[r["accession"]]["genus"], []).append(r["accession"])
        pc, oc = spec["panel"], spec["outgroups"]
        out, panels = dict(gtdb=dict(files={k: {x: v[x] for x in ("sha256", "md5", "bytes", "cached")} for k, v in files.items()}, md5sum=md5rec["sha256"]),
                           placement=place, units={}, forbidden=forb["names"]), {}
        for genus, accs in sorted(units.items()):
            ncbi_genera = sorted({tax.at_rank(int(re.search(r'/db_xref="taxon:(\d+)"', open(os.path.join(store, "dev_genomes", a + ".gb")).read(20000)).group(1)), "genus")
                                  for a in accs})
            pool = []
            for t in ncbi_genera:
                pool += candidates_of(t)
            pool, lost = mapped(pool, gt)
            unfiltered = pool
            pool = [r for r in pool if r["assembly"].split(".")[0].replace("GCF_", "GCA_") not in lib and int(r["tax_id"]) not in island_taxa]
            order_id = tax.at_rank(ncbi_genera[0], "order")
            pool_bases = median_bases(pool)
            cut, walks = {}, {}
            for acc in accs:                            # part 1; a chromosome short of other species is walked
                sp = place[acc]["species"]
                same, other = recut_panel(sp, genus, pool, int(pc["same"]), int(pc["other"]), pc["salt"])
                if len(other) < int(pc["other_min"]):
                    node, species, rec = walk_panel(tree, gt, sp, genus, pool, tax, int(pc["other_min"]), float(pc["anchor_majority"]), set(forb["names"]),
                                                     unfiltered)
                    same, other = recut_panel(sp, genus, pool, int(pc["same"]), int(pc["other"]), pc["salt"], species)
                    walks[acc] = (node, rec)
                cut[acc] = (same, other)
            nodes = {node for node, _ in walks.values()}
            if walks and (len(nodes) > 1 or len(walks) < len(accs)):
                raise SystemExit("family_a: unit %s has chromosomes walked to different clades or not walked; its outgroups are not defined" % genus)
            ogs = choose_outgroups(tree, gt, genus, candidates_of, tax, forb, order_id, lib, island_taxa, int(oc["genera"]), int(oc["genomes"]),
                                   int(oc["min_assemblies"]), oc["salt"], float(oc["min_size_share"]) * pool_bases, nodes.pop() if walks else None)
            out["units"][genus] = dict(chromosomes=accs, ncbi_genera=ncbi_genera, pool=len(pool), unmapped=lost, pool_median_bases=pool_bases,
                                       walked=bool(walks), outgroups=[dict(genus=o["genus"], distance=o["distance"], considered=o["considered"],
                                                                           median_bases=o["median_bases"], genomes=[g["assembly"] for g in o["genomes"]])
                                                                      for o in ogs])
            for acc in accs:
                same, other = cut[acc]
                panels[acc] = dict(same=same, other=other, outgroups=ogs)
                out["units"][genus].setdefault("panels", {})[acc] = dict(same=len(same), other=len(other), walk=walks[acc][1] if acc in walks else None)
        base = os.path.join(run_dir, "data", "famA")
        os.makedirs(os.path.join(base, "panel"), exist_ok=True)
        cols = ("assembly", "role", "rank", "gtdb_species", "gtdb_genus", "ncbi_taxid", "level", "name", "strain")
        for acc, p in panels.items():
            with open(os.path.join(base, "panel", acc + ".tsv"), "w") as f:
                f.write("\t".join(cols) + "\n")
                for role, rws in (("same", p["same"]), ("other", p["other"])):
                    for i, r in enumerate(rws):
                        f.write("\t".join(str(x) for x in (r["assembly"], role, i, r["gtdb"].get("species", ""), r["gtdb"].get("genus", ""), r["tax_id"],
                                                           r.get("assembly_level", ""), r.get("scientific_name", ""), r.get("strain", ""))) + "\n")
                for k_, o in enumerate(p["outgroups"]):
                    for i, r in enumerate(o["genomes"][:int(oc["genomes"]) + 3]):          # the chosen genomes and 3 reserves
                        f.write("\t".join(str(x) for x in (r["assembly"], "outgroup%d" % (k_ + 1), i, r["gtdb"].get("species", ""), r["gtdb"].get("genus", ""),
                                                           r["tax_id"], r.get("assembly_level", ""), r.get("scientific_name", ""), r.get("strain", ""))) + "\n")
        tp = os.path.join(base, "taxa.json")
        write_json(tp, out)
        st.output("famA_taxa", tp)
        st.extra.update(units={g: len(u["chromosomes"]) for g, u in out["units"].items()},
                        outgroups={g: [o["genus"] for o in u["outgroups"]] for g, u in out["units"].items()})


def do_fetch(run_dir):
    with Stage(run_dir, "famA_fetch", requires=("pre",)) as st:
        cfg = st.config
        fc = cfg["famA"]
        with open(st.repo(fc.get("spec", SPEC))) as f:
            spec = yaml.safe_load(f)
        from concurrent.futures import ThreadPoolExecutor
        from ci.stages import fetch as FE
        base = os.path.join(run_dir, "data", "famA")
        store = fc["store"]
        cas = os.path.join(cfg["data_root"], "cas")
        index_path = os.path.join(cas, "fetch_index.json")
        index = json.load(open(index_path)) if os.path.exists(index_path) else {}
        _bench_panel(st.repo)                                  # fetch_source imports bench/hard/panel.py
        need = int(spec["outgroups"]["genomes"])
        rows, reserves = {}, {}
        for fn in sorted(os.listdir(os.path.join(base, "panel"))):
            with open(os.path.join(base, "panel", fn)) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r["role"].startswith("outgroup") and int(r["rank"]) >= need:
                        reserves.setdefault((r["role"], r["gtdb_genus"]), {}).setdefault(int(r["rank"]), r)
                    else:
                        rows.setdefault(r["assembly"], r)
        stored, got = {}, {}
        adir = os.path.join(store, "panel", "assemblies")
        def one(a):
            try:
                files, prov, hit = FE.fetch_source({"ena_assembly": a, "id": a}, cas, index)
                return a, dict(files=[dict(f, path=os.path.relpath(f["path"], cas)) for f in files], provenance=prov, reused=hit)
            except Exception as err:                            # noqa: BLE001 - an unavailable genome is recorded
                return a, dict(error="%s: %s" % (type(err).__name__, err))
        limit = int(spec["outgroups"]["n_run_max"])

        def locate(a):
            if a in stored:
                return stored[a], "store"
            if "files" in got.get(a, {}):
                return os.path.join(cas, got[a]["files"][0]["path"]), "cas"
            return None, None
        for a in rows:
            for ext in (".embl.gz", ".gb.gz"):
                if os.path.exists(os.path.join(adir, a + ext)):
                    stored[a] = os.path.join(adir, a + ext)
        todo = sorted(a for a in rows if a not in stored)
        with ThreadPoolExecutor(int(fc.get("threads", 8))) as pool:
            got = dict(pool.map(one, todo))
        # an outgroup genome that is unavailable or has a long N-run is replaced by the next reserve of its genus
        replaced = []
        for a in sorted(rows):
            r = rows[a]
            if not r["role"].startswith("outgroup"):
                continue
            path, _ = locate(a)
            if path is not None and longest_n_run(path) < limit:
                continue
            for rank in sorted(reserves.get((r["role"], r["gtdb_genus"]), {})):
                alt = reserves[(r["role"], r["gtdb_genus"])].pop(rank)
                if alt["assembly"] in rows:
                    continue
                if alt["assembly"] not in stored:
                    got.update(dict([one(alt["assembly"])]))
                path, _ = locate(alt["assembly"])
                if path is not None and longest_n_run(path) < limit:
                    rows[alt["assembly"]] = alt
                    rows[a] = dict(r, role=r["role"] + "_replaced")
                    replaced.append(dict(replaced=a, by=alt["assembly"]))
                    break
                todo.append(alt["assembly"])
        new_index = {}
        for a, g in got.items():
            if "files" in g and not g["reused"]:
                new_index[FE.sha256_json(FE.describe({"ena_assembly": a}))] = dict(files=g["files"], provenance=g["provenance"])
        if new_index:
            import fcntl
            with open(os.path.join(cas, "fetch_index.lock"), "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                cur = json.load(open(index_path)) if os.path.exists(index_path) else {}
                cur.update(new_index)
                with open(index_path + ".part", "w") as f:
                    json.dump(cur, f, indent=1, sort_keys=True)
                os.replace(index_path + ".part", index_path)
        genomes = {}
        for a, r in sorted(rows.items()):
            path, src = locate(a)
            if path is None:
                genomes[a] = dict(role=r["role"], gtdb_genus=r["gtdb_genus"], error=got.get(a, {}).get("error", "not fetched"))
                continue
            longest = longest_n_run(path)
            genomes[a] = dict(role=r["role"], gtdb_genus=r["gtdb_genus"], gtdb_species=r["gtdb_species"], source=src, path=path,
                              sha256=sha256_file(path, use_sidecar=False), longest_n_run=longest, n_run_ok=longest < limit)
        bad = sorted(a for a, g in genomes.items() if "error" in g and not g["role"].endswith("_replaced"))
        fp = os.path.join(base, "fetch.json")
        write_json(fp, dict(genomes=genomes, fetched=len(todo), unavailable=bad, replaced=replaced, n_run_max=limit))
        st.output("famA_fetch", fp)
        st.extra.update(genomes=len(genomes), fetched=len(todo), unavailable=len(bad),
                        n_run_failures=sorted(a for a, g in genomes.items() if g.get("n_run_ok") is False))


# --- increment 2: presence on genome DNA, and the labels ------------------------------------------------------------

CALL = dict(ortho=1, hom=2, og=4, pos=8)


def record_genes(gb_path):
    """The genes of a development record as the island stage names them (seeddrift.inputs.from_genbank's ids and skip
    rule, seeddrift.pipeline.clean_proteins' names), in record order, with their places and the record QC: partial
    (fuzzy start or end), frame (length not a multiple of 3, or a stop before the last codon in table 11), n_run (a
    non-ACGT base in the coding sequence), origin (a CDS joined over the origin of a circular record: its place is its
    largest part). Returns names, ids (the record's), proteins, coords [(replicon, start, end, strand)], flags
    {name: [..]}, closed, length (bp), records."""
    import io as _io
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import AfterPosition, BeforePosition
    from seeddrift.inputs import translate
    from seeddrift.pipeline import clean_proteins
    with open(gb_path) as f:
        text = f.read()
    prots, info, closed, kept, length, records = {}, {}, False, 0, 0, 0
    for rec in SeqIO.parse(_io.StringIO(text), "genbank"):
        records += 1
        closed = closed or rec.annotations.get("topology") == "circular"
        length += len(rec.seq) if rec.seq is not None else 0
        dna = str(rec.seq) if rec.seq is not None else ""
        for ft in rec.features:
            if ft.type != "CDS" or "pseudo" in ft.qualifiers:
                continue
            q = ft.qualifiers
            gid = (q.get("locus_tag") or q.get("protein_id") or q.get("gene") or ["%s_%d" % (rec.id, kept + 1)])[0]
            try:
                nt = str(ft.location.extract(rec.seq)) if dna else ""
            except Exception:                                   # noqa: BLE001 - as from_genbank
                nt = ""
            prot = q.get("translation", [""])[0] or (translate(nt, int(q.get("transl_table", ["11"])[0])) if nt else "")
            if not prot:
                continue
            if gid not in prots:
                kept += 1
            prots[gid] = prot
            fz = isinstance(ft.location.start, BeforePosition) or isinstance(ft.location.end, AfterPosition)
            full = str(Seq(nt[:len(nt) // 3 * 3]).translate(table=11)) if nt else ""
            frame = bool(nt) and (len(nt) % 3 != 0 or "*" in full[:-1])
            parts = list(ft.location.parts)
            a, b = int(ft.location.start) + 1, int(ft.location.end)
            origin = len(parts) > 1 and bool(nt) and (b - a + 1) > 3 * len(nt)        # a join over the origin spans the record
            if origin:
                big = max(parts, key=lambda x: int(x.end) - int(x.start))
                a, b = int(big.start) + 1, int(big.end)
            info[gid] = dict(coords=(rec.id, a, b, "+" if ft.location.strand != -1 else "-"),
                             flags=[x for x, on in (("partial", fz), ("frame", frame), ("n_run", bool(re.search(r"[^ACGTacgt]", nt))), ("origin", origin)) if on])
    clean, ren = clean_proteins(prots)
    names = list(ren.values())
    back = {v: k for k, v in ren.items()}
    return dict(names=names, ids=[back[n] for n in names], proteins=[clean[n] for n in names], coords=[info[back[n]]["coords"] for n in names],
                flags={n: info[back[n]]["flags"] for n in names}, closed=closed, length=length, records=records)


def genome_dna(path):
    """{replicon: (sequence, circular)} of a stored EMBL or GenBank file (gz) or a FASTA (topology unknown: a replicon
    of 1 Mb or more is taken as circular)."""
    from Bio import SeqIO
    opener = gzip.open if path.endswith(".gz") else open
    out = {}
    with opener(path, "rt") as f:
        head = f.read(1)
        f.seek(0)
        if head == ">":
            for rec in SeqIO.parse(f, "fasta"):
                out[rec.id.split("|")[-1]] = (str(rec.seq).upper(), len(rec.seq) >= 1_000_000)
        else:
            fmt = "embl" if head == "I" else "genbank"
            for rec in SeqIO.parse(f, fmt):
                out[rec.id] = (str(rec.seq).upper(), rec.annotations.get("topology") == "circular")
    return out


def chunks(dna, size=200_000, overlap=30_000):
    """[(name, replicon, offset, sequence)] windows of every replicon."""
    out = []
    for rep, (seq, _) in dna.items():
        step = size - overlap
        for off in range(0, max(len(seq) - overlap, 1), step):
            out.append(("%s|%d" % (rep, off), rep, off, seq[off:off + size]))
    return out


def parse_hits(lines, offsets):
    """Hit rows of `diamond blastx --outfmt 6 qseqid sseqid pident length qstart qend sstart send slen bitscore` on
    chunks, in replicon coordinates; a hit seen twice in an overlap of two chunks is kept once."""
    out, seen = [], set()
    for line in lines:
        p = line.rstrip("\n").split("\t")
        if len(p) < 10:
            continue
        rep, off = offsets[p[0]]
        qs, qe = int(p[4]), int(p[5])
        strand = "+" if qs <= qe else "-"
        a, b = off + min(qs, qe), off + max(qs, qe)
        key = (p[1], rep, strand, a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(target=p[1], rep=rep, strand=strand, start=a, end=b, pident=float(p[2]), length=int(p[3]),
                        sstart=int(p[6]), send=int(p[7]), slen=int(p[8]), bits=float(p[9])))
    return out


def keep_iv(hits):
    return [(min(h["sstart"], h["send"]), max(h["sstart"], h["send"])) for h in hits]


def merge_loci(hits):
    """Loci: the hits of one target on one replicon and strand merged when within 3 x the gene length + 3 kb. Of a
    locus's hits, those that do not overlap a better one on the target by more than 10 % count (alternative alignments
    and the truncated copy of a hit at a window edge do not): coverage of the target (union of their aligned residues
    / its length), length-weighted identity, summed bit score, extent."""
    by = {}
    for h in hits:
        by.setdefault((h["target"], h["rep"], h["strand"]), []).append(h)
    out = []
    for (t, rep, strand), hs in by.items():
        hs.sort(key=lambda h: h["start"])
        groups, cur = [], [hs[0]]
        for h in hs[1:]:
            gap = h["start"] - max(x["end"] for x in cur)
            if gap <= 3 * 3 * h["slen"] + 3000:
                cur.append(h)
            else:
                groups.append(cur)
                cur = [h]
        groups.append(cur)
        for g in groups:
            keep = []                                     # best first; an HSP overlapping a kept one on the protein by >10 % is an alternative
            for h in sorted(g, key=lambda h: -h["bits"]):
                a, b = min(h["sstart"], h["send"]), max(h["sstart"], h["send"])
                if all(min(b, y) - max(a, x) + 1 <= 0.1 * min(b - a + 1, y - x + 1) for x, y in keep_iv(keep)):
                    keep.append(h)
            g = sorted(keep, key=lambda h: h["start"])
            iv = sorted((min(h["sstart"], h["send"]), max(h["sstart"], h["send"])) for h in g)
            cov, lo, hi = 0, None, None
            for a, b in iv:
                if lo is None or a > hi + 1:
                    if lo is not None:
                        cov += hi - lo + 1
                    lo, hi = a, b
                else:
                    hi = max(hi, b)
            cov += hi - lo + 1
            w = sum(h["length"] for h in g)
            out.append(dict(target=t, rep=rep, strand=strand, start=min(h["start"] for h in g), end=max(h["end"] for h in g),
                            cov=min(cov / g[0]["slen"], 1.0), ident=sum(h["pident"] * h["length"] for h in g) / max(w, 1),
                            bits=sum(h["bits"] for h in g)))
    return out


def owned(loci, members):
    """The loci owned by the focal chromosome's genes (`members`: target names of one chromosome): loci of different
    targets overlapping by 50 % or more reciprocally (either strand) are one place, owned by the highest bit score (ties:
    first name)."""
    mine = sorted((x for x in loci if x["target"] in members), key=lambda x: (-x["bits"], x["target"]))
    taken, out = [], []
    for x in mine:
        clash = False
        for y in taken:
            if y["rep"] == x["rep"]:
                ov = min(x["end"], y["end"]) - max(x["start"], y["start"]) + 1
                if ov > 0 and ov >= 0.5 * (x["end"] - x["start"] + 1) and ov >= 0.5 * (y["end"] - y["start"] + 1):
                    clash = True
                    break
        if not clash:
            taken.append(x)
            out.append(x)
    return out


def locus_gap(x, y, rep_info=None):
    """The gap (bp) between two loci on one replicon, the shorter way round on a circular one; None on different
    replicons. rep_info: {replicon: (length, circular)}."""
    if x["rep"] != y["rep"]:
        return None
    (a0, a1), (b0, b1) = sorted([(x["start"], x["end"]), (y["start"], y["end"])])
    gap = max(b0 - a1 - 1, 0)
    L, circ = (rep_info or {}).get(x["rep"], (0, False))
    if circ and L:
        gap = min(gap, max(L - (max(a1, b1) - a0 + 1), 0))
    return gap


def calls_for(names, loci_all, loci_owned, closed, order, near_bp=10_000, th=None, best_at="ortho", rep_info=None):
    """Per gene of one focal chromosome in one genome: the call bits (CALL) and its best owned locus at the level
    best_at ("ortho" in panel genomes, "og" in outgroups). pos: ortho, with an ortho locus of a neighbour (one of the next
    two genes in focal order) within near_bp of the gene's best ortho locus on each side (around the origin of a
    circular target replicon). th: the call thresholds {ortho: (id, cov), hom: (id, cov), og: (id, cov)}."""
    import numpy as np
    th = th or dict(ortho=(40.0, 0.7), hom=(40.0, 0.5), og=(30.0, 0.5))
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    bits = np.zeros(n, np.uint8)
    best, bo, orth = [None] * n, [None] * n, [[] for _ in range(n)]
    for x in loci_all:
        i = idx.get(x["target"])
        if i is not None and x["ident"] >= th["hom"][0] and x["cov"] >= th["hom"][1]:
            bits[i] |= CALL["hom"]
    for x in loci_owned:
        i = idx.get(x["target"])
        if i is None:
            continue
        for lvl in ("og", "ortho"):
            if x["ident"] >= th[lvl][0] and x["cov"] >= th[lvl][1]:
                bits[i] |= CALL[lvl]
                if lvl == best_at and (best[i] is None or x["bits"] > best[i]["bits"]):
                    best[i] = x
                if lvl == "ortho":
                    orth[i].append(x)
                    if bo[i] is None or x["bits"] > bo[i]["bits"]:
                        bo[i] = x
    pos = np.asarray(order, int)                         # pos[k]: the gene at place k of the genome order
    rank = np.empty(n, int)
    rank[pos] = np.arange(n)                             # rank[i]: the place of gene i
    for i in range(n):
        if bo[i] is None:
            continue
        ok = []
        for side in (-1, 1):
            hit = False
            for d in (1, 2):
                k = rank[i] + side * d
                if not closed and not (0 <= k < n):
                    continue
                for y in orth[pos[k % n]]:
                    g_ = locus_gap(bo[i], y, rep_info)
                    if g_ is not None and g_ <= near_bp:
                        hit = True
                        break
                if hit:
                    break
            ok.append(hit)
        if all(ok):
            bits[i] |= CALL["pos"]
    return bits, best


def arc(genes, start, end, length):
    """(left, right, bp) of a run of genes in genome order: from the smallest start to the largest end of its genes, or,
    for a run over the origin of a circular chromosome, from the smallest start of the genes before the origin to the
    largest end of those after it."""
    cut = next((k for k in range(1, len(genes)) if start[genes[k]] < start[genes[k - 1]]), None)
    if cut is None:
        a, b = int(min(start[i] for i in genes)), int(max(end[i] for i in genes))
        return a, b, b - a + 1
    a, b = int(min(start[i] for i in genes[:cut])), int(max(end[i] for i in genes[cut:]))
    return a, b, length - a + 1 + b


def arc_distance(s_, e_, arcs, length, closed):
    """The distance (bp) from the gene [s_, e_] to the nearest of the arcs [(left, right, bp)], 0 when it overlaps one;
    around the origin only on a closed chromosome."""
    best = float("inf")
    for a, b, bp in arcs:
        if closed:
            if (s_ - a) % length < bp or (e_ - a) % length < bp or (a - s_) % length <= e_ - s_:
                return 0.0
            best = min(best, (a - e_) % length, (s_ - b) % length)
        else:
            if e_ >= a and s_ <= b:
                return 0.0
            best = min(best, a - e_ if a > e_ else s_ - b)
    return float(best)


def runs(absent, order, start, end, closed, min_bp=4000, min_genes=3, gap=1, length=None):
    """Maximal runs of consecutive absent genes (genome order; single-gene gaps bridged; wrapping on a closed
    chromosome), at least min_bp long and min_genes genes: [[gene indices in order]]."""
    length = length or (max(int(x) for x in end) if len(end) else 0)
    seq = [int(i) for i in order]
    a = [bool(absent[i]) for i in seq]
    n = len(seq)
    if not any(a):
        return []
    br = a[:]
    for k in range(n):                                    # bridge gaps of `gap` genes
        if not a[k]:
            left = any(a[(k - d) % n] if closed else (k - d >= 0 and a[k - d]) for d in range(1, gap + 1))
            right = any(a[(k + d) % n] if closed else (k + d < n and a[k + d]) for d in range(1, gap + 1))
            br[k] = left and right
    start_k = 0
    if closed and all(br):
        groups = [list(range(n))]
    else:
        if closed:
            start_k = next(k for k in range(n) if not br[k])
        groups, cur = [], []
        for t in range(n):
            k = (start_k + t) % n if closed else t
            if br[k]:
                cur.append(k)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    out = []
    for g in groups:
        while g and not a[g[0]]:                          # a run starts and ends with an absent gene
            g = g[1:]
        while g and not a[g[-1]]:
            g = g[:-1]
        genes = [seq[k] for k in g]
        if len(genes) >= min_genes and arc(genes, start, end, length)[2] >= min_bp:
            out.append(genes)
    return out


def calibrate_n(lit_edges, own_edges, match_bp=20_000, q=0.9, floor_kb=1, cap_kb=10, default_kb=5, min_edges=10):
    """N (kb) from the literature islands' edges against the own runs' edges of the same side: the q-quantile of the
    distances of matched edges (within match_bp), rounded up to kb and clipped; default_kb when fewer than min_edges
    edges match. lit_edges and own_edges: [(chromosome, side, position)]."""
    import math
    d = []
    for c, side, x in lit_edges:
        cand = [abs(x - y) for c2, s2, y in own_edges if c2 == c and s2 == side]
        if cand and min(cand) <= match_bp:
            d.append(min(cand))
    if len(d) < min_edges:
        return dict(n_kb=default_kb, calibrated=False, matched=len(d))
    d.sort()
    v = d[min(len(d) - 1, int(math.ceil(q * len(d))) - 1)]
    return dict(n_kb=int(min(max(math.ceil(v / 1000.0), floor_kb), cap_kb)), calibrated=True, matched=len(d))


def site_state(L, R, focal_strand_l, focal_strand_r, rep_len, circular, empty_bp=5000, max_bp=100_000):
    """The state of a segment's site in one genome from its innermost mapped flank loci L and R (dicts with rep, start,
    end, strand, or None): empty, replaced, broken, inverted or flank_lost."""
    if L is None or R is None:
        return "flank_lost", None
    if L["rep"] != R["rep"]:
        return "broken", None
    if (L["strand"] == R["strand"]) != (focal_strand_l == focal_strand_r):
        return "inverted", None
    a, b = sorted([(L["start"], L["end"]), (R["start"], R["end"])])
    gap = max(b[0] - a[1] - 1, 0)
    if circular and rep_len:
        gap = min(gap, max(rep_len - (b[1] - a[0] + 1), 0))
    if gap <= empty_bp:
        return "empty", gap
    if gap <= max_bp:
        return "replaced", gap
    return "broken", gap


def unit_genomes(taxa_units, panels_dir, fetched, unit):
    """The genomes of one unit: every chromosome's same and other rows and the unit's chosen outgroup genomes (fetch.json:
    not replaced, no long N-run), with roles: {assembly: dict(role, outgroup_rank, path)} and per chromosome its rows."""
    per, genomes = {}, {}
    for acc in taxa_units[unit]["chromosomes"]:
        with open(os.path.join(panels_dir, acc + ".tsv")) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        per[acc] = dict(same=[r["assembly"] for r in rows if r["role"] == "same"], other=[r["assembly"] for r in rows if r["role"] == "other"])
    og_order = [o["genus"] for o in taxa_units[unit]["outgroups"]]
    for a, g in fetched["genomes"].items():
        if "error" in g or g["role"].endswith("_replaced"):
            continue
        if g["role"].startswith("outgroup") and g.get("gtdb_genus") in og_order and g.get("n_run_ok"):
            genomes[a] = dict(role="outgroup", outgroup_rank=og_order.index(g["gtdb_genus"]), genus=g["gtdb_genus"], path=g["path"])
    for acc, r in per.items():
        for role in ("same", "other"):
            for a in r[role]:
                g = fetched["genomes"].get(a)
                if g and "error" not in g:
                    genomes.setdefault(a, dict(role=role, path=g["path"]))
    return genomes, per, og_order


def do_presence(run_dir):
    with Stage(run_dir, "famA_presence", requires=("pre", "famA_taxa")) as st:
        import subprocess
        import numpy as np
        cfg = st.config
        fc = cfg["famA"]
        with open(st.repo(fc.get("spec", SPEC))) as f:
            spec = yaml.safe_load(f)
        pc = spec["presence"]
        th = dict(ortho=tuple(pc["ortho"]), hom=tuple(pc["hom"]), og=tuple(pc["og"]))
        base = os.path.join(run_dir, "data", "famA")
        with open(os.path.join(base, "taxa.json")) as f:
            taxa = json.load(f)
        with open(os.path.join(base, "fetch.json")) as f:
            fetched = json.load(f)
        exe, threads = cfg["tools"]["diamond"], str(int(fc.get("threads", 8)))
        store = fc["store"]
        out_dir = os.path.join(base, "presence")
        work = os.path.join(run_dir, "tmp", "famA")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(work, exist_ok=True)
        index, timings = {}, {}
        for unit in sorted(taxa["units"]):
            genomes, per, og_order = unit_genomes(taxa["units"], os.path.join(base, "panel"), fetched, unit)
            genes = {acc: record_genes(os.path.join(store, "dev_genomes", acc + ".gb")) for acc in per}
            faa = os.path.join(work, "unit_%s.faa" % unit)
            with open(faa, "w") as f:
                for acc, g in genes.items():
                    for n_, p_ in zip(g["names"], g["proteins"]):
                        f.write(">%s__%s\n%s\n" % (acc, n_, p_))
            db = faa[:-4]
            subprocess.run([exe, "makedb", "--in", faa, "-d", db, "--threads", threads], check=True, capture_output=True)
            # multi-copy genes: self hits at 40 % identity over 80 % of both
            selfp = os.path.join(work, "self_%s.tsv" % unit)
            subprocess.run([exe, "blastp", "-q", faa, "-d", db, "-o", selfp, "--sensitive", "--id", "40", "--query-cover", "80", "--subject-cover", "80",
                            "-k", "0", "--threads", threads, "--outfmt", "6", "qseqid", "sseqid"], check=True, capture_output=True)
            multi = set()
            with open(selfp) as f:
                for line in f:
                    q, t = line.split()[:2]
                    if q != t and q.split("__")[0] == t.split("__")[0]:
                        multi.add(q)
            members = {acc: {"%s__%s" % (acc, n_) for n_ in g["names"]} for acc, g in genes.items()}
            calls = {acc: {} for acc in genes}
            loci_best = {acc: {} for acc in genes}
            for a in sorted(genomes):
                t0 = __import__("time").time()
                dna = genome_dna(genomes[a]["path"])
                ch = chunks(dna)
                qf = os.path.join(work, "q.fna")
                with open(qf, "w") as f:
                    for name, _, _, seq in ch:
                        f.write(">%s\n%s\n" % (name, seq))
                hp = os.path.join(work, "hits.tsv")
                subprocess.run([exe, "blastx", "-q", qf, "-d", db, "-o", hp, "--" + pc["mode"], "-F", str(pc["frameshift"]), "--id", str(pc["min_id"]),
                                "--subject-cover", str(pc["min_subject_cover"]), "-k", "0", "--max-hsps", "0", "-e", str(pc["evalue"]), "--query-gencode", "11",
                                "--threads", threads, "--outfmt", "6", "qseqid", "sseqid", "pident", "length", "qstart", "qend", "sstart", "send",
                                "slen", "bitscore"], check=True, capture_output=True)
                with open(hp) as f:
                    hits = parse_hits(f, {name: (rep, off) for name, rep, off, _ in ch})
                loci = merge_loci(hits) if hits else []
                rep_len = {rep: len(seq) for rep, (seq, _) in dna.items()}
                rep_info = {rep: (len(seq), bool(circ)) for rep, (seq, circ) in dna.items()}
                for acc, g in genes.items():
                    mine_all = [dict(x, target=x["target"].split("__", 1)[1]) for x in loci if x["target"] in members[acc]]
                    own = [dict(x, target=x["target"].split("__", 1)[1]) for x in owned(loci, members[acc])]
                    order = sorted(range(len(g["names"])), key=lambda i: (g["coords"][i][0], g["coords"][i][1]))
                    b, best = calls_for(g["names"], mine_all, own, g["closed"], order, int(pc["near_bp"]), th,
                                        "og" if genomes[a]["role"] == "outgroup" else "ortho", rep_info)
                    calls[acc][a] = b
                    loci_best[acc][a] = [None if x is None else (x["rep"], x["start"], x["end"], x["strand"], rep_len.get(x["rep"], 0),
                                                                 bool(dna.get(x["rep"], ("", False))[1])) for x in best]
                timings[a] = round(__import__("time").time() - t0, 1)
            for acc, g in genes.items():
                gl = sorted(calls[acc])
                mat = np.stack([calls[acc][a] for a in gl], 1) if gl else np.zeros((len(g["names"]), 0), np.uint8)
                np.savez_compressed(os.path.join(out_dir, acc + ".npz"), calls=mat, genomes=np.array(gl), names=np.array(g["names"]),
                                    multi=np.array(["%s__%s" % (acc, n_) in multi for n_ in g["names"]]))
                with open(os.path.join(out_dir, acc + ".loci.json"), "w") as f:
                    json.dump({a: loci_best[acc][a] for a in gl}, f)
                index[acc] = dict(unit=unit, genomes={a: dict(role=genomes[a]["role"], outgroup_rank=genomes[a].get("outgroup_rank"),
                                                             genus=genomes[a].get("genus")) for a in gl}, same=per[acc]["same"], other=per[acc]["other"],
                                  outgroups=og_order, genes=len(g["names"]), multi=int(sum(1 for n_ in g["names"] if "%s__%s" % (acc, n_) in multi)))
        ip = os.path.join(out_dir, "index.json")
        write_json(ip, dict(chromosomes=index, timings_s=timings, thresholds={k: list(v) for k, v in th.items()}))
        st.output("famA_presence", ip)
        st.extra.update(chromosomes=len(index), genomes=len(timings), seconds=round(sum(timings.values()), 1))


def published_regions(store, table, accs):
    """{accession: [(start, end, id)]} of one island table of the store for the given accessions only."""
    out = {}
    txt = open(os.path.join(store, table), newline="").read().replace("\r\n", "\n").replace("\r", "\n")
    for r in csv.DictReader(io.StringIO(txt), delimiter="\t"):
        a = r["Accession"].strip()
        if a in accs:
            out.setdefault(a, []).append((int(r["Start"]), int(r["End"]), r.get("GI_ID") or r.get("Lit_ID") or ""))
    return out


def flank_genes(rank, order, n, closed, lo, hi, side, usable, k):
    """Up to k genes on one side of the places lo..hi (genome order), nearest first, that pass `usable`."""
    out, step = [], 1
    while len(out) < k and step < n:
        kk = (lo - step) if side < 0 else (hi + step)
        if not closed and not (0 <= kk < n):
            break
        j = order[kk % n]
        if usable(j):
            out.append(j)
        step += 1
    return out


def classify(g, calls, genomes, other, same, og_genomes, multi, spec, n_bp, loci):
    """The family A classes of one development chromosome from its own evidence only (bench/hard/V2_FAMILY_A.md,
    part 2): per gene its class and columns, and the S1 segments with their site states. calls: genes x genomes call
    bits; og_genomes: [(assembly, outgroup rank)]; loci: {genome: [per gene (rep, start, end, strand, rep_len,
    circular) or None]}."""
    import numpy as np
    lc = spec["labels"]
    n = len(g["names"])
    if len({c[0] for c in g["coords"]}) > 1 or g.get("records", 1) > 1:
        raise SystemExit("family_a: a development record with more than one replicon; the label geometry is per replicon")
    col = {a: i for i, a in enumerate(genomes)}
    oth = [col[a] for a in other if a in col]
    sam = [col[a] for a in same if a in col]
    ogs = dict((a, rk) for a, rk in og_genomes if a in col)

    def share(bit, idx):
        return (calls[:, idx] & bit > 0).mean(1) if idx else np.full(n, np.nan)
    f_ortho, f_hom, f_pos, f_same = share(CALL["ortho"], oth), share(CALL["hom"], oth), share(CALL["pos"], oth), share(CALL["ortho"], sam)
    n_ortho = (calls[:, oth] & CALL["ortho"] > 0).sum(1) if oth else np.full(n, -1)
    start = np.array([c[1] for c in g["coords"]])
    end = np.array([c[2] for c in g["coords"]])
    order = sorted(range(n), key=lambda i: (g["coords"][i][0], g["coords"][i][1]))
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    closed = g["closed"]
    qc = np.array([bool(g["flags"][nm]) for nm in g["names"]])
    multi = np.array(multi, bool)
    enough = len(oth) >= lc["other_min"]
    absent1 = (n_ortho == 0) if enough else np.zeros(n, bool)                     # S1: no ortho call in any other species
    absent2 = (f_ortho <= lc["band_share"]) if enough else np.zeros(n, bool)      # S2: at most band_share of them
    length = int(g.get("length") or max(end))
    r1 = runs(absent1, order, start, end, closed, lc["min_bp"], lc["min_genes"], lc["gap"], length)
    r2 = runs(absent2, order, start, end, closed, lc["min_bp"], lc["min_genes"], lc["gap"], length)
    in1 = np.full(n, -1)
    for k, r in enumerate(r1):
        in1[r] = k
    in2 = np.zeros(n, bool)
    for r in r2:
        in2[r] = True
    band = np.zeros(n, bool)
    for r in r2:                                          # S2-run genes outside an S1 run that the S2 run contains
        if any(in1[i] >= 0 for i in r):
            band[[i for i in r if in1[i] < 0]] = True
    inU = (in1 >= 0) | in2
    # polarity: no og call in both genomes of the nearest outgroup genus with genomes and of at least one more genus
    by = {}
    for a, rk in ogs.items():
        by.setdefault(rk, []).append(col[a])
    ranks = sorted(by)
    og_ok = np.zeros(n, bool)
    if len(ranks) >= lc["outgroup_genera_min"]:
        gone = {rk: (calls[:, by[rk]] & CALL["og"] == 0).all(1) for rk in ranks}
        og_ok = gone[ranks[0]] & np.any(np.stack([gone[rk] for rk in ranks[1:]], 0), 0)
    # the site of every S1 segment in the genomes that lack it
    single = ~multi & ~qc & ~inU
    segs, seg_pass = [], {}
    for k, r in enumerate(r1):
        lo, hi = rank[r[0]], rank[r[-1]]                   # the run's first and last places (it may wrap)
        left = flank_genes(rank, order, n, closed, lo, hi, -1, lambda j: bool(single[j]), lc["flank_genes"])
        right = flank_genes(rank, order, n, closed, lo, hi, 1, lambda j: bool(single[j]), lc["flank_genes"])
        states = {}
        for a in [x for x in other if x in col] + sorted(ogs):
            ci = col[a]
            bit = CALL["og"] if a in ogs else CALL["ortho"]
            if (calls[r, ci] & bit == 0).mean() < lc["lacking"]:
                continue
            lp = loci.get(a) or [None] * n
            jl = next((j for j in left if calls[j, ci] & bit and lp[j]), None)
            jr = next((j for j in right if calls[j, ci] & bit and lp[j]), None)
            L = None if jl is None else dict(rep=lp[jl][0], start=lp[jl][1], end=lp[jl][2], strand=lp[jl][3])
            R = None if jr is None else dict(rep=lp[jr][0], start=lp[jr][1], end=lp[jr][2], strand=lp[jr][3])
            ref = lp[jl] if jl is not None else (lp[jr] if jr is not None else None)
            stt, gap = site_state(L, R, g["coords"][jl][3] if jl is not None else "+", g["coords"][jr][3] if jr is not None else "+",
                                  ref[4] if ref else 0, bool(ref[5]) if ref else False, lc["empty_bp"], lc["max_bp"])
            states[a] = dict(state=stt, gap=gap, outgroup=a in ogs)
        empty = sum(1 for v in states.values() if v["state"] == "empty")
        repl = sum(1 for v in states.values() if v["state"] == "replaced")
        seg_pass[k] = empty >= lc["flank_empty"] and empty + repl >= lc["flank_empty_or_replaced"]
        a_, b_, bp = arc(r, start, end, length)
        segs.append(dict(segment=k, genes=len(r), start=a_, end=b_, bp=bp, passes=bool(seg_pass[k]), empty=empty, replaced=repl, states=states))
    # distances: to the nearer edge of the gene's own S1 segment; to any U run (bp, along the chromosome)
    arcs1 = [arc(r, start, end, length) for r in r1]
    arcsU = arcs1 + [arc(r, start, end, length) for r in r2]
    d_edge = np.full(n, np.inf)
    for i in np.where(in1 >= 0)[0]:                      # from the gene's ends to its run's nearer edge
        a_, b_, _ = arcs1[in1[i]]
        d_edge[i] = min((start[i] - a_) % length, (b_ - end[i]) % length) if closed else min(start[i] - a_, b_ - end[i])
    d_u = np.array([arc_distance(int(start[i]), int(end[i]), arcsU, length, closed) for i in range(n)], float)
    near_u = np.zeros(n, bool)
    for i in np.where(inU)[0]:
        for d in range(-lc["margin_genes"], lc["margin_genes"] + 1):
            kk = rank[i] + d
            if closed or 0 <= kk < n:
                near_u[order[kk % n]] = True
    cls = np.array(["unresolved"] * n, dtype=object)
    for i in range(n):
        if qc[i]:
            cls[i] = "qc_fail"
        elif not enough:
            cls[i] = "no_panel"
        elif in1[i] >= 0:
            if not absent1[i]:
                cls[i] = "bridged"
            elif d_edge[i] <= n_bp:
                cls[i] = "edge_in"
            elif not og_ok[i]:
                cls[i] = "no_polarity"
            elif not seg_pass[in1[i]]:
                cls[i] = "no_flank"
            else:
                cls[i] = "insertion"
        elif band[i]:
            cls[i] = "boundary"
        elif inU[i]:
            cls[i] = "unresolved"
        elif absent1[i]:
            cls[i] = "accessory"
        elif not multi[i] and f_pos[i] >= lc["core"]:
            cls[i] = "core" if (not near_u[i] and d_u[i] > n_bp) else "edge"
    cols = dict(class_=cls, f_ortho=f_ortho, f_hom=f_hom, f_pos=f_pos, f_same=f_same, run=in1, band=band, og_absent=og_ok, multi=multi,
                qc=np.array([",".join(g["flags"][nm]) for nm in g["names"]], dtype=object), dist_edge=d_edge, in_u=inU, near_u=near_u, dist_u=d_u)
    return cols, segs


def do_labels(run_dir):
    with Stage(run_dir, "famA_labels", requires=("pre", "famA_taxa")) as st:
        import numpy as np
        cfg = st.config
        fc = cfg["famA"]
        with open(st.repo(fc.get("spec", SPEC))) as f:
            spec = yaml.safe_load(f)
        base = os.path.join(run_dir, "data", "famA")
        store = fc["store"]
        with open(os.path.join(base, "presence", "index.json")) as f:
            pidx = json.load(f)
        accs = sorted(pidx["chromosomes"])
        lit = published_regions(store, "SupTable3.txt", set(accs))
        lc = spec["labels"]
        data = {}
        for acc in accs:
            g = record_genes(os.path.join(store, "dev_genomes", acc + ".gb"))
            z = np.load(os.path.join(base, "presence", acc + ".npz"))
            if list(z["names"]) != g["names"]:
                raise SystemExit("family_a: the presence genes of %s are not the record's" % acc)
            with open(os.path.join(base, "presence", acc + ".loci.json")) as f:
                loci = json.load(f)
            with open(os.path.join(store, "labels", acc + ".tsv")) as f:
                plab = {r["gene_id"]: r["label"] for r in csv.DictReader(f, delimiter="\t")}
            info = pidx["chromosomes"][acc]
            args = (g, z["calls"], list(z["genomes"]), info["other"], info["same"],
                    [(a, d["outgroup_rank"]) for a, d in info["genomes"].items() if d["role"] == "outgroup"], list(z["multi"]), spec)
            # the store's labels are keyed by the island stage's (cleaned) names
            data[acc] = (g, args, loci, [plab.get(nm, plab.get(i, "absent")) for nm, i in zip(g["names"], g["ids"])])
        # N: the literature islands' edges (SupTable3) against the S1 segments' edges of the same side (segments do not
        # depend on N)
        own_edges, lit_edges = [], []
        for acc in accs:
            g, args, loci, _ = data[acc]
            _, segs = classify(*args, n_bp=lc["n_default_kb"] * 1000, loci=loci)
            own_edges += [(acc, side, x) for sg in segs for side, x in (("L", sg["start"]), ("R", sg["end"]))]
            lit_edges += [(acc, side, x) for a_, b_, _ in lit.get(acc, []) for side, x in (("L", a_), ("R", b_))]
        cal = calibrate_n(lit_edges, own_edges, lc["match_bp"], lc["n_quantile"], lc["n_floor_kb"], lc["n_cap_kb"], lc["n_default_kb"], lc["n_min_edges"])
        n_bp = cal["n_kb"] * 1000
        out = os.path.join(base, "labels")
        os.makedirs(os.path.join(out, "segments"), exist_ok=True)
        counts = {}
        keys = ["class_", "f_ortho", "f_hom", "f_pos", "f_same", "run", "band", "og_absent", "multi", "qc", "dist_edge", "in_u", "near_u", "dist_u"]
        for acc in accs:
            g, args, loci, pub = data[acc]
            cols, segs = classify(*args, n_bp=n_bp, loci=loci)
            with open(os.path.join(out, acc + ".tsv"), "w") as f:
                f.write("gene_id\trecord_id\treplicon\tstart\tend\tstrand\t" + "\t".join(k.rstrip("_") for k in keys) + "\tpublished\n")
                for i, nm in enumerate(g["names"]):
                    c = g["coords"][i]
                    vals = []
                    for k in keys:
                        v = cols[k][i]
                        if isinstance(v, (float, np.floating)):
                            vals.append("%.4f" % v if np.isfinite(v) else "")
                        else:
                            vals.append(str(v))
                    f.write("%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\n" % (nm, g["ids"][i], c[0], c[1], c[2], c[3], "\t".join(vals), pub[i]))
            with open(os.path.join(out, "segments", acc + ".json"), "w") as f:
                json.dump(segs, f)
            cc = {}
            for v in cols["class_"]:
                cc[v] = cc.get(v, 0) + 1
            agree = {}
            for v, p_ in zip(cols["class_"], pub):                  # the published labels as an agreement stratum only
                agree.setdefault(v, {}).setdefault(p_, 0)
                agree[v][p_] += 1
            counts[acc] = dict(genes=len(g["names"]), classes=cc, by_published=agree, segments=len(segs),
                               segments_passing=sum(1 for x in segs if x["passes"]),
                               other=sum(1 for a in args[3] if a in args[2]), outgroup_genomes=len(args[5]))
        ip = os.path.join(out, "index.json")
        write_json(ip, dict(calibration=cal, counts=counts, unit=dict((a, pidx["chromosomes"][a]["unit"]) for a in accs)))
        st.output("famA_labels", ip)
        st.extra.update(n_kb=cal["n_kb"], calibrated=cal["calibrated"], insertions=sum(c["classes"].get("insertion", 0) for c in counts.values()),
                        core=sum(c["classes"].get("core", 0) for c in counts.values()))


def longest_n_run(path):
    """The longest run of non-ACGT bases in the sequences of a FASTA, EMBL or GenBank file (gz allowed)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as f:
        text = f.read()
    if text.startswith(">"):
        seqs = re.findall(r">[^\n]*\n([^>]*)", text)
        seq = "".join(s.replace("\n", "") for s in seqs)
    else:
        parts = re.findall(r"\n(?:ORIGIN|SQ)[^\n]*\n(.*?)\n//", text, re.S)
        seq = re.sub(r"[^A-Za-z]", "", "".join(parts))
    runs = re.findall(r"[^ACGTacgt]+", seq)
    return max((len(x) for x in runs), default=0)


def main():
    run_dir = main_args()
    {"taxa": do_taxa, "fetch": do_fetch, "presence": do_presence, "labels": do_labels}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
