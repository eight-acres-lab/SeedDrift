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


def recut_panel(sp, genus, pool, same_n, other_n, salt):
    """The same and other rows of one chromosome (V2_FAMILY_A.md, Panels)."""
    key = lambda r: order_key("%s|%s" % (r["assembly"], salt))
    same = sorted((r for r in pool if r["gtdb"].get("species") == sp), key=key)[:same_n]
    by_sp = {}
    for r in sorted((r for r in pool if r["gtdb"].get("genus") == genus and r["gtdb"].get("species") != sp), key=key):
        by_sp.setdefault(r["gtdb"]["species"], r)
    other = sorted(by_sp.values(), key=key)[:other_n]
    return same, other


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


def choose_outgroups(tree, gt, focal_genus, candidates_of, tax, forb, order_id, lib, island_taxa, k, n, min_asm, salt):
    """The outgroup genera and genomes of one unit (V2_FAMILY_A.md, Outgroups). candidates_of(ncbi genus taxid) gives the
    ENA rows of an NCBI genus. Returns [dict(genus, distance, genomes, considered)]."""
    reps = [i for i, nm in enumerate(tree["name"]) if nm and gt.get(gtdb_key(nm), {}).get("genus") == focal_genus]
    if not reps:
        raise SystemExit("family_a: no representative of g__%s in the GTDB tree" % focal_genus)
    out = []
    for clade, dist in sister_clades(tree, mrca(tree, reps)):
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
            if len(rows) >= min_asm and (best is None or len(rows) > len(best[1])):
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
        out.append(dict(genus=g, distance=round(dist, 4), genomes=picked, considered=len(genera)))
        if len(out) == k:
            break
    return out


def gtdb_files(cfg_gtdb, cas, get, store):
    """The GTDB files of the release, each checked against MD5SUM.txt and stored in the CAS: {name: {path, sha256, md5}}
    and the MD5SUM.txt record."""
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
        data = get(base + "/" + name)
        if hashlib.md5(data).hexdigest() != sums[name]:
            raise SystemExit("family_a: %s does not match its md5" % name)
        path, sha = store(cas, data, "." + name.split(".", 1)[1])
        out[name] = dict(path=path, sha256=sha, md5=sums[name], bytes=len(data))
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
        files, md5rec = gtdb_files(spec["gtdb"], cas, FE.get, FE.store)
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
        out, panels = dict(gtdb=dict(files={k: {x: v[x] for x in ("sha256", "md5", "bytes")} for k, v in files.items()}, md5sum=md5rec["sha256"]),
                           placement=place, units={}, forbidden=forb["names"]), {}
        for genus, accs in sorted(units.items()):
            ncbi_genera = sorted({tax.at_rank(int(re.search(r'/db_xref="taxon:(\d+)"', open(os.path.join(store, "dev_genomes", a + ".gb")).read(20000)).group(1)), "genus")
                                  for a in accs})
            pool = []
            for t in ncbi_genera:
                pool += candidates_of(t)
            pool, lost = mapped(pool, gt)
            pool = [r for r in pool if r["assembly"].split(".")[0].replace("GCF_", "GCA_") not in lib and int(r["tax_id"]) not in island_taxa]
            order_id = tax.at_rank(ncbi_genera[0], "order")
            ogs = choose_outgroups(tree, gt, genus, candidates_of, tax, forb, order_id, lib, island_taxa, int(oc["genera"]), int(oc["genomes"]),
                                   int(oc["min_assemblies"]), oc["salt"])
            out["units"][genus] = dict(chromosomes=accs, ncbi_genera=ncbi_genera, pool=len(pool), unmapped=lost,
                                       outgroups=[dict(genus=o["genus"], distance=o["distance"], considered=o["considered"],
                                                       genomes=[g["assembly"] for g in o["genomes"]]) for o in ogs])
            for acc in accs:
                same, other = recut_panel(place[acc]["species"], genus, pool, int(pc["same"]), int(pc["other"]), pc["salt"])
                panels[acc] = dict(same=same, other=other, outgroups=ogs)
                out["units"][genus].setdefault("panels", {})[acc] = dict(same=len(same), other=len(other))
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
    {"taxa": do_taxa, "fetch": do_fetch}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
