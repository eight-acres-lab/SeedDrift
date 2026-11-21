"""Stage gates: `python -m ci.stages.gates RUN_DIR pre|corpus|train|eval|report|select|islands|selfsup|codon|register|audit|famA_taxa|famA_labels
|famA_eval [--check-only]`.

pre      before any data step: G6 clean committed code (against the commit the launcher asked for), G7 environment
         lock, prefix (not moved) and resources, G8 CI version, G1 on the spec (tier must equal the profile's), G2 static,
         G3 static (no sealed accession in the configuration, the corpus spec or the evaluation tables)
corpus   on the built corpus: G1 group floors from taxids, G2 exclusions by full lineage on every genome
train    on the exact id list the training loader will read: non-empty, no duplicates, a subset of the corpus manifest
         and of the same corpus dataset; then G1 and G2 again on exactly those genomes
eval     G2_holdout: no evaluation gene (protein or coding DNA) occurs in the training corpus; G3: no evaluation genome
         under a sealed genus and no sealed accession in any manifest of the run
report   G4 on the evaluation report: six rows per scenario on one gene set, the released screen recomputed from its tag
islands  (pipeline selfsup) G3_islands: the island chromosomes are development chromosomes, no sealed accession anywhere
selfsup  (pipeline selfsup) G4_selfsup on data/selfsup/report.json: every variant diagnosed, every island gene scored
codon    (pipeline codon) G3_codon: no panel or manifold genome shares an island chromosome's genus (merged as in the
         evaluation) or a taxon its view hides, all of them from the train and held-out splits; G4_codon on
         data/codon/report.json: every admitted chromosome, every column, the baseline scoring every labelled gene,
         the reproductions of codon.expect
register (pipeline register) G3_register on data/register/: the parts as the lists in the repository define them, the
         rule of bench/hard/V2_REGISTER.md applied (no candidate outside the rest part or at L2 or above; NC_009504.1
         demoted; every candidate in a sealed_v1 genus), the outputs as reported
audit    (pipeline audit) G3_audit: development chromosomes only, and no accession of the register's other parts (the
         committed bench/hard/results/v2_reference_set.tsv) in any audit output; G4_audit: the admitted chromosomes,
         column coverage, the negative control back at 0.5, the fused baselines and the detector present
famA_taxa (pipeline family_a) G3_famA: development chromosomes only; no panel or outgroup genome of an island-set taxon,
         a library assembly, or a forbidden genus or family (EXCLUDE.tsv, the register's other parts); outgroups in the
         focal NCBI order; the label code imports no score or search module; no accession of the register's other parts in
         the outputs. G4_famA_taxa: at least 3 other species per chromosome, at least 2 outgroup genera (3 sought) with 2
         genomes each, every chosen genome fetched without a long N-run
famA_eval (pipeline family_a_eval) G3_famA_eval: development chromosomes only, the label code free of score imports, no
         other part's accession in the outputs; G4_famA_eval: every arm and score present, the main arm balanced, each
         matched covariate's own AUROC interval containing 0.5, the shuffle control at 0.5, the v1 arm reproducing the
         audit, lk_og present
famA_labels (pipeline family_a) G4_famA_labels: every development chromosome labelled, its genes those of its record and
         of its presence calls; every class one of part 2's and every insertion and core gene meeting its rule as written
         in the label table; insertions and core genes present; the label code imports no score or search module; no
         accession of the register's other parts in the presence or label outputs
select   (pipeline select) G1_select, G2_select, G3_select on the selected corpus table; the pre phase of that pipeline
         checks the select spec (G1_select_spec) instead of a corpus spec
Each gate writes gates/<gate>.json. A failure exits non-zero, so no rule that takes the sentinel gates/<phase>.ok as
input can run. The sentinel holds the hash of the phase's inputs and is not rewritten when its content would not
change. --check-only decides the gates without touching the sentinel (used by a resume before Snakemake starts).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import yaml

from ci import ROOT
from ci.gates import GateFailure, GateResult, find_waiver, require, run
from ci.gates import g1_kingdom, g2_exclusion, g3_sealed, g4_report, g6_clean, g7_env, g8_version
from ci.hashing import sha256_file, sha256_json
from ci.kingdom import KingdomMap
from ci.record import atomic_write_json
from ci.stages.common import Stage, load_select_spec, read_fasta
from ci.taxonomy import Taxonomy


def _manifest(run_dir, name):
    with open(os.path.join(run_dir, "data", name, "manifest.json")) as f:
        return json.load(f)


def _seq_hashes(folder, names=("proteins.faa", "cds.fna")):
    out = set()
    for n in names:
        for s in read_fasta(os.path.join(folder, n)).values():
            out.add((n, hashlib.sha256(s.upper().rstrip("*").encode()).hexdigest()))
    return out


def phase_pre(st, cfg, run_dir):
    require(g6_clean.check(ROOT, cfg.get("expect_sha") or cfg["git"]["sha"], ci_root=cfg.get("ci_root", ""), require_readonly=cfg.get("require_readonly", False)), run_dir)
    if cfg.get("env_export"):
        require(g7_env.lock_matches(open(cfg["env_export"]).read()), run_dir)
    elif cfg.get("require_lock", True):
        require(GateResult("G7", "fail", dict(error="the environment could not be exported (micromamba not found), so the lock cannot be verified")), run_dir)
    else:
        run(GateResult("G7", "waived", dict(note="profile sets require_lock: false")), run_dir)
    require(g7_env.relocation(), run_dir)
    res = g7_env.resources(cfg["data_root"], float(cfg.get("disk_gb", 1)), float(cfg.get("disk_reserve_gb", 30)), gpu_mem_min_mb=int(cfg.get("gpu_mem_min_mb", 0)))
    res.gate = "G7_resources"
    require(res, run_dir)
    require(g8_version.check(ROOT), run_dir)
    if pipeline_of(cfg) == "select":
        spec = load_select_spec(cfg["select_spec"], st.repo)
        require(g1_kingdom.select_static(spec, cfg.get("tier")), run_dir)
    else:
        with open(st.repo(cfg["corpus_spec"])) as f:
            spec = yaml.safe_load(f)
        g1 = g1_kingdom.static(spec, cfg.get("tier"))
        g1.gate = "G1_spec"
        require(g1, run_dir)
    g2 = g2_exclusion.static(ROOT)
    g2.gate = "G2_static"
    require(g2, run_dir)
    require(g3_sealed.static(_g3_files(st, cfg, run_dir), cfg.get("tier")), run_dir)
    return sha256_json([cfg.get("expect_sha") or cfg["git"]["sha"], cfg["config_sha256"]])


def pipeline_of(cfg):
    p = cfg.get("pipeline", "train")
    if p not in ("train", "select", "search", "selfsup", "codon", "register", "audit", "family_a", "family_a_eval"):
        raise SystemExit("unknown pipeline %r" % p)
    return p


def phase_islands(st, cfg, run_dir):
    """G3_islands (pipeline selfsup): the island chromosomes are development chromosomes only (listed in
    bench/islands/results/dev_genomes.tsv, none in bench/hard/results/sealed.tsv), no sealed accession occurs (with or
    without version, any case, inside composite names: the scanner of gate G3) in the island index, manifest or gene
    tables, no corpus genome shares an island chromosome's genus (merged as in the evaluation: Shigella=Escherichia)
    or belongs to a taxon the chromosome's view hides (the island chromosomes are new to the model at genus level), and
    every chromosome has a genus. Proteins identical between an island chromosome and the corpus are counted (a real
    shared gene is not a leak; the count is reported)."""
    import csv
    from ci.gates import g3_sealed
    with open(os.path.join(run_dir, "data", "islands", "index.json")) as f:
        idx = json.load(f)
    sealed = g3_sealed.load_sealed(st.repo("bench/hard/results/sealed.tsv"))
    with open(st.repo("bench/islands/results/dev_genomes.tsv")) as f:
        dev = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
    accs = [g["id"] for g in idx["genomes"]]
    sealed_acc = {a for a, _ in sealed}
    problems = [dict(accession=a, error="sealed" if a in sealed_acc else "not a development chromosome") for a in accs if a in sealed_acc or a not in dev]
    files = [os.path.join(run_dir, "data", "islands", n) for n in ("index.json", "manifest.json")] + \
        [os.path.join(run_dir, g["folder"], "genes.tsv") for g in idx["genomes"]]
    found = g3_sealed._scan(files, g3_sealed._pattern(sealed))
    problems += [dict(file=k, sealed_accessions=v) for k, v in sorted(found.items())]
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    corpus = _manifest(run_dir, "corpus")["entries"]
    merge = dict(x.split("=", 1) for x in (cfg.get("islands") or {}).get("merge", ["Shigella=Escherichia"]))
    unit = lambda name: merge.get(name, name)
    by_unit = {}
    for e in corpus:
        gt = tax.at_rank(int(e["taxid"]), "genus")
        if gt:
            by_unit.setdefault(unit(tax.name(gt)), []).append(e["id"])
    island_prot = set()
    for g in idx["genomes"]:
        if not g["ranks"][4]:
            problems.append(dict(accession=g["id"], error="no genus in the taxonomy: the genus hold-out cannot be checked"))
        same = by_unit.get(unit(g["genus"]), [])
        if same:
            problems.append(dict(accession=g["id"], error="corpus genomes of the same genus (merged as in the evaluation)", genomes=same[:10]))
        hid = set(g.get("hidden_taxids") or []) | {int(g["taxid"])}
        close = [e["id"] for e in corpus if int(e["taxid"]) in hid]
        if close:
            problems.append(dict(accession=g["id"], error="corpus genomes of a taxon the chromosome's view hides", genomes=close[:10]))
        island_prot |= {h for n, h in _seq_hashes(os.path.join(run_dir, g["folder"]), ("proteins.faa",))}
    shared = 0
    for e in corpus:                                       # one corpus genome at a time against the island proteins
        shared += len({h for n, h in _seq_hashes(os.path.join(run_dir, "data", "corpus", e["id"]), ("proteins.faa",))} & island_prot)
    require(GateResult("G3_islands", "fail" if problems else "pass", dict(chromosomes=accs, identical_proteins_with_corpus=shared, problems=problems[:50])), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False) for p in files])


def phase_selfsup(st, cfg, run_dir):
    """G4_selfsup on data/selfsup/report.json: every variant has its held-out diagnosis with finite numbers; with
    islands, every gene of every island chromosome of the index is scored by every variant (a gene without visible
    hits may lack B2 only), only development chromosomes were evaluated, and the contrasts exist."""
    import math
    p = os.path.join(run_dir, "data", "selfsup", "report.json")
    with open(p) as f:
        rep = json.load(f)
    if not cfg.get("islands"):                    # a run without island chromosomes (tiny) has none to check
        require(GateResult("G3_islands", "pass", dict(note="no island chromosomes in this run")), run_dir)
    problems = []
    if not rep.get("variants"):
        problems.append(dict(error="no variants"))
    for name, v in (rep.get("variants") or {}).items():
        h = v.get("heldout") or {}
        nums = [x for k, x in h.items() if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if not h or any(not math.isfinite(x) for x in nums):
            problems.append(dict(variant=name, error="held-out diagnosis missing or not finite"))
        elif cfg.get("tier") != "tiny" and not h.get("genes"):          # the fixture of tier tiny may hold out nothing
            problems.append(dict(variant=name, error="no held-out gene was diagnosed"))
    if cfg.get("islands"):
        isl = rep.get("islands") or {}
        if not isl:
            problems.append(dict(error="islands configured but not in the report"))
        else:
            if isl.get("genes") != isl.get("index_genes"):
                problems.append(dict(error="scored genes %s differ from the genes of the island index %s" % (isl.get("genes"), isl.get("index_genes"))))
            with open(st.repo("bench/hard/results/sealed.tsv")) as f:
                import csv
                sealed = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
            if set(isl.get("evaluated") or []) & sealed or set(isl.get("chromosomes") or []) & sealed:
                problems.append(dict(error="a sealed chromosome was scored or evaluated"))
            if not isl.get("contrasts"):
                problems.append(dict(error="no contrasts on the island chromosomes"))
            # every gene scored by A; B1 for at least 90% of a chromosome's genes (a gene of unknown or synonym-free
            # codons only has none); B2, C and the combination for at least one gene (B2 needs visible hits, C a partner)
            for name, v in (rep.get("variants") or {}).items():
                want = ["%s_%s" % (name, k) for k in list(v.get("objectives") or []) + ["comb"]]
                d = os.path.join(run_dir, "data", "islands", "scores", name)
                for acc in isl.get("chromosomes") or []:
                    sp = os.path.join(d, "scores_%s.tsv" % acc)
                    if not os.path.exists(sp):
                        problems.append(dict(variant=name, accession=acc, error="no score table"))
                        continue
                    with open(sp) as f:
                        head = f.readline().rstrip("\n").split("\t")
                        rows = [line.rstrip("\n").split("\t") for line in f]
                    ro = [h for h in head[1:] if h.startswith(name + "_R_")]          # train.readouts columns (round 1 of
                    base_cols = [h for h in head[1:] if h not in ro]                  # the objectives of 2026-11-20)
                    if sorted(base_cols) != sorted(want):
                        problems.append(dict(variant=name, accession=acc, error="columns %s, the objectives give %s" % (base_cols, want)))
                        continue
                    if ro and name not in ((cfg.get("train") or {}).get("readouts") or {}).get("variants", []):
                        problems.append(dict(variant=name, accession=acc, error="readout columns for a variant without readouts"))
                        continue
                    for k, col in enumerate(head[1:], 1):
                        if col in ro:
                            continue
                        share = sum(1 for r in rows if r[k] != "") / max(len(rows), 1)
                        need = 1.0 if col.endswith("_A") else 0.9 if col.endswith("_B1") else 1e-9
                        if share < need:
                            problems.append(dict(variant=name, accession=acc, column=col, scored_share=round(share, 4), error="too few genes scored"))
    require(GateResult("G4_selfsup", "fail" if problems else "pass", dict(problems=problems[:50])), run_dir)
    return sha256_file(p, use_sidecar=False)


def phase_codon(st, cfg, run_dir):
    """G3_codon and G4_codon (pipeline codon, ci/stages/codon.py)."""
    import math
    cc = cfg["codon"]
    with open(os.path.join(run_dir, "data", "codon", "study.json")) as f:
        study = json.load(f)
    with open(os.path.join(cc["islands_from"], "data", "islands", "index.json")) as f:
        chroms = json.load(f)["genomes"]
    meta = {}                                     # per corpus: a genome's split can differ between corpora
    for name, src in cc["corpora"].items():
        with open(os.path.join(src, "data", "train_data", "index.json")) as f:
            meta[name] = {m["id"]: m for m in json.load(f)["genomes"]}
    used = {}                                     # genome -> the corpora it was drawn from
    for p in study["panels"].values():
        for g in set(p["base"]).union(*[set(v) for v in (p.get("per_chromosome") or {}).values()]):
            used.setdefault(g, set()).add(p["corpus"])
    mcorp = {m["name"]: m["corpus"] for m in cc.get("manifolds") or []}
    for name, m in study["manifolds"].items():
        for g in m.get("members") or []:
            used.setdefault(g, set()).add(mcorp.get(name))
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    merge = dict(x.split("=", 1) for x in (cfg.get("islands") or {}).get("merge", ["Shigella=Escherichia"]))
    unit = lambda name: merge.get(name, name)
    genus, taxid = {}, {}
    problems = []
    for g in sorted(used):
        for c in sorted(used[g], key=str):
            m = (meta.get(c) or {}).get(g)
            if m is None or m["split"] not in ("train", "heldout"):
                problems.append(dict(genome=g, corpus=c, error="not a train or held-out genome of the corpus it was drawn from"))
                continue
            taxid.setdefault(g, set()).add(int(m["taxid"]))
        # the sealed discipline across corpora: a genome in the test part (or the evaluation inputs) of any configured
        # corpus is not used, even when another corpus has it in a training split
        for c, mm in sorted(meta.items()):
            if g in mm and mm[g]["split"] not in ("train", "heldout"):
                problems.append(dict(genome=g, corpus=c, split=mm[g]["split"], error="in the test part or the evaluation inputs of a configured corpus"))
        genus[g] = set()
        for t in taxid.get(g, ()):
            gt = tax.at_rank(t, "genus")
            genus[g].add(unit(tax.name(gt)) if gt else "")
    for ch in chroms:
        hid = set(ch.get("hidden_taxids") or []) | {int(ch["taxid"])}
        same = [g for g in sorted(used) if unit(ch["genus"]) in genus.get(g, set())]
        close = [g for g in sorted(used) if taxid.get(g, set()) & hid]
        if same:
            problems.append(dict(accession=ch["id"], error="panel or manifold genomes of the same genus", genomes=same[:10]))
        if close:
            problems.append(dict(accession=ch["id"], error="panel or manifold genomes of a taxon the chromosome's view hides", genomes=close[:10]))
    require(GateResult("G3_codon", "fail" if problems else "pass", dict(genomes=len(used), chromosomes=len(chroms), problems=problems[:50])), run_dir)
    p = os.path.join(run_dir, "data", "codon", "report.json")
    with open(p) as f:
        rep = json.load(f)
    problems = []
    with open(os.path.join(cc["islands_from"], "data", "islands", "eval", "dev.json")) as f:
        res = json.load(f)
    from ci.stages.islands import admitted
    want = sorted(admitted(res))
    if rep.get("chromosomes") != want:
        problems.append(dict(error="evaluated chromosomes %s differ from the admitted %s" % (rep.get("chromosomes"), want)))
    cols = [c for c in study.get("columns") or [] if not c.startswith("cd_cu_rc_")]
    missing = [c for c in cols if c not in (rep.get("scores") or {})]
    if missing:
        problems.append(dict(error="columns not evaluated", columns=missing[:20]))
    b = (rep.get("scores") or {}).get(rep.get("baseline")) or {}
    if b.get("coverage_island") != 1.0 or b.get("coverage_native") != 1.0:
        problems.append(dict(error="the baseline %s does not score every labelled gene" % rep.get("baseline")))
    for c, a in (cc.get("expect") or {}).items():
        got = ((rep.get("scores") or {}).get(c) or {}).get("auroc")
        if got is None or not math.isfinite(got) or abs(got - float(a)) > 0.001:
            problems.append(dict(column=c, expected=a, got=got, error="reproduction"))
    if not rep.get("intervals"):
        problems.append(dict(error="no paired intervals"))
    require(GateResult("G4_codon", "fail" if problems else "pass", dict(problems=problems[:50], baseline=rep.get("baseline"),
                                                                        reproduced={c: ((rep.get("scores") or {}).get(c) or {}).get("auroc") for c in cc.get("expect") or {}})), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False), sha256_file(os.path.join(run_dir, "data", "codon", "study.json"), use_sidecar=False)])


def phase_register(st, cfg, run_dir):
    """G3_register (pipeline register, ci/stages/register.py)."""
    import csv
    from ci.stages import register as RG
    base = os.path.join(run_dir, "data", "register")
    with open(os.path.join(base, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(base, "reference_set.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    with open(os.path.join(base, "candidates.tsv")) as f:
        cands = [r["accession"] for r in csv.DictReader(f, delimiter="\t")]
    with open(st.repo("bench/islands/results/dev_genomes.tsv")) as f:
        dev = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
    with open(st.repo("bench/hard/results/sealed.tsv")) as f:
        sealed = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
    rc = cfg.get("register") or {}
    problems = []
    if len(rows) != int(rc.get("expect", 104)) or len({r["accession"] for r in rows}) != len(rows):
        problems.append(dict(error="%d rows (expected %s, distinct)" % (len(rows), rc.get("expect", 104))))
    by = {r["accession"]: r for r in rows}
    if {a for a, r in by.items() if r["part"] == "development"} != dev:
        problems.append(dict(error="the development part is not bench/islands/results/dev_genomes.tsv"))
    if {a for a, r in by.items() if r["part"] in ("sealed_v1", "sealed_secondary")} != sealed:
        problems.append(dict(error="the sealed parts are not bench/hard/results/sealed.tsv"))
    if by.get(RG.SECONDARY, {}).get("part") != "sealed_secondary":
        problems.append(dict(error="%s is not the sealed_secondary chromosome" % RG.SECONDARY))
    if by.get("NC_009504.1", {}).get("status") not in ("demoted", None):
        problems.append(dict(error="NC_009504.1 is not demoted (bench/hard/V2_REGISTER.md)"))
    genus = lambda r: r["organism"].split()[0] if r["organism"] else ""
    sealed_genera = {genus(r) for r in rows if r["part"] == "sealed_v1"}
    for r in rows:
        cand = r["status"] == "candidate"
        if (a := r["accession"]) and cand != (a in cands):
            problems.append(dict(accession=a, error="status and candidates.tsv disagree"))
        if cand and (r["part"] != "rest" or RG.LEVELS.index(r["level"]) >= RG.LEVELS.index(RG.DEMOTE_FROM) or a in dev or a in sealed):
            problems.append(dict(accession=a, error="a candidate outside the rest part, at %s or above, or listed" % RG.DEMOTE_FROM))
        if cand and genus(r) not in sealed_genera:
            problems.append(dict(accession=a, error="a candidate outside the sealed_v1 genera"))
        if r["part"] == "rest" and not cand and r["status"] != "demoted":
            problems.append(dict(accession=a, error="a rest chromosome neither candidate nor demoted"))
    for name, sha in (rep.get("outputs") or {}).items():
        if sha256_file(os.path.join(base, name), use_sidecar=False) != sha:
            problems.append(dict(file=name, error="differs from the report"))
    rule = st.repo(RG.RULE)
    if (rep.get("inputs") or {}).get(RG.RULE) != sha256_file(rule, use_sidecar=False):
        problems.append(dict(error="the rule file is not the one the stage read"))
    sp = os.path.join(base, "sealed_registry.json")              # the registry of everything sealed, when configured
    if rc.get("sealed") and not os.path.exists(sp):
        problems.append(dict(error="register.sealed is configured but the sealed registry is missing"))
    if os.path.exists(sp):
        from ci.stages import sealed_registry as SR
        with open(sp) as f:
            reg = json.load(f)
        want = sealed | set(cands)
        have = {r["accession"] for r in reg.get("chromosomes") or []}
        if have != want:
            problems.append(dict(error="the registry's chromosomes are not sealed.tsv and the candidates", missing=sorted(want - have)[:10], extra=sorted(have - want)[:10]))
        for p_ in SR.SOURCES:
            if (reg.get("sources") or {}).get(p_) != sha256_file(st.repo(p_), use_sidecar=False):
                problems.append(dict(file=p_, error="the registry hashes another version of this source"))
        if any(not isinstance(r.get("taxid"), int) or not r.get("aliases") for r in reg.get("chromosomes") or []):
            problems.append(dict(error="a registry chromosome lacks its taxid or aliases"))
        runs = [os.path.basename(r) for r in ((rc.get("sealed") or {}).get("test_split_runs") or [])]
        if sorted(s_["run"] for s_ in reg.get("corpus_test_splits") or []) != sorted(runs):
            problems.append(dict(error="the registry's test splits are not those of register.sealed.test_split_runs"))
        for s_ in reg.get("corpus_test_splits") or []:
            if not s_.get("genomes") or any(not g.get("aliases") for g in s_["genomes"]) or not s_.get("split_code_sha256"):
                problems.append(dict(run=s_.get("run"), error="a test split without genomes, aliases or its split code"))
    require(GateResult("G3_register", "fail" if problems else "pass", dict(problems=problems[:50], parts=rep.get("parts"), status=rep.get("status"),
                                                                           candidates_sha256=(rep.get("outputs") or {}).get("candidates.tsv"))), run_dir)
    return sha256_file(os.path.join(base, "report.json"), use_sidecar=False)


def phase_audit(st, cfg, run_dir):
    """G3_audit and G4_audit (pipeline audit, ci/stages/audit.py)."""
    import csv
    from ci.stages.islands import admitted
    base = os.path.join(run_dir, "data", "audit")
    with open(os.path.join(base, "report.json")) as f:
        rep = json.load(f)
    with open(os.path.join(base, "columns.json")) as f:
        cols = json.load(f)
    with open(st.repo("bench/islands/results/dev_genomes.tsv")) as f:
        dev = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
    with open(st.repo("bench/hard/results/v2_reference_set.tsv")) as f:
        others = [(r["accession"], r["genus"]) for r in csv.DictReader(f, delimiter="\t") if r["part"] != "development"]
    problems = []
    chroms = set(rep.get("chromosomes") or []) | set(cols.get("coverage") or {})
    if chroms - dev:
        problems.append(dict(error="chromosomes outside the development list", accessions=sorted(chroms - dev)))
    files = [os.path.join(dp, fn) for dp, _, fns in os.walk(base) for fn in fns]
    found = g3_sealed._scan(files, g3_sealed._pattern(others))
    if found:
        problems.append(dict(error="accessions of the register's other parts in audit outputs", found=found))
    require(GateResult("G3_audit", "fail" if problems else "pass", dict(problems=problems[:50], files=len(files), others=len(others))), run_dir)
    problems = []
    ac = cfg["audit"]
    with open(os.path.join(ac["islands_from"], "data", "islands", "eval", "dev.json")) as f:
        want = sorted(admitted(json.load(f)))
    if rep.get("chromosomes") != want:
        problems.append(dict(error="evaluated chromosomes differ from the admitted ones"))
    if set(cols.get("coverage") or {}) != set(want) or set(cols.get("records_sha256") or {}) != set(want):
        problems.append(dict(error="the columns' coverage or record checksums do not cover exactly the admitted chromosomes"))
    missing = sorted(set(rep.get("scores") or {}) - set(rep.get("negative_control") or {}))
    if missing or not rep.get("scores"):
        problems.append(dict(error="the label-shuffle control is missing for scores", scores=missing[:20]))
    for acc, cv in (cols.get("coverage") or {}).items():
        for k, share in cv.items():
            if k == "ribosomal_reference_genes" or (k == "tv_cai" and cv.get("ribosomal_reference_genes", 0) < 20):
                continue
            if share < 0.9:
                problems.append(dict(accession=acc, column=k, coverage=share, error="column covers under 90% of the genes"))
    tol = float((ac.get("controls") or {}).get("tolerance", 0.01))
    for k, v in (rep.get("negative_control") or {}).items():
        if abs(v - 0.5) > tol:
            problems.append(dict(score=k, shuffled=v, error="the label-shuffle control is not back at 0.5"))
    for k in ("fz_l1", "fz_gbm"):
        if ((rep.get("scores") or {}).get(k) or {}).get("auroc") is None:
            problems.append(dict(score=k, error="fused baseline missing"))
    for k in ("core", "annotation"):
        if ((rep.get("detector") or {}).get(k) or {}).get("auroc_within") is None:
            problems.append(dict(detector=k, error="the shortcut detector did not run"))
    require(GateResult("G4_audit", "fail" if problems else "pass", dict(problems=problems[:50], detector={k: v.get("auroc_within") for k, v in (rep.get("detector") or {}).items()})), run_dir)
    return sha256_file(os.path.join(base, "report.json"), use_sidecar=False)


FAMA_BANNED = ("islands", "audit", "protocol", "search", "eval", "codon", "train_data")


def phase_famA_taxa(st, cfg, run_dir):
    """G3_famA and G4_famA_taxa (pipeline family_a, ci/stages/family_a.py: taxa and fetch)."""
    import csv
    import re as _re
    from ci.stages import family_a as FA
    base = os.path.join(run_dir, "data", "famA")
    with open(os.path.join(base, "taxa.json")) as f:
        taxa = json.load(f)
    with open(os.path.join(base, "fetch.json")) as f:
        fetched = json.load(f)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    dev = {r["accession"] for r in FA.dev_chromosomes(st.repo)}
    problems = []
    if set(taxa["placement"]) != dev:
        problems.append(dict(error="the placed chromosomes are not the development chromosomes"))
    forb = FA.forbidden(st.repo, tax)
    store = cfg["famA"]["store"]
    with open(os.path.join(store, "panel", "island_set_taxa.json")) as f:
        island_taxa = {int(v) for v in json.load(f).values()}
    rows = []
    for fn in sorted(os.listdir(os.path.join(base, "panel"))):
        with open(os.path.join(base, "panel", fn)) as f:
            rows += [dict(r, chromosome=fn[:-4]) for r in csv.DictReader(f, delimiter="\t")]
    chosen = {a for a, g in fetched["genomes"].items() if not g["role"].endswith("_replaced")}
    for r in rows:
        if r["assembly"] not in chosen:
            continue
        t = int(r["ncbi_taxid"])
        g, fam = tax.at_rank(t, "genus"), tax.at_rank(t, "family")
        if t in island_taxa or g in forb["genera"] or fam in forb["families"]:
            problems.append(dict(assembly=r["assembly"], role=r["role"], error="an island-set taxon or a forbidden genus or family"))
        if r["role"].startswith("outgroup") and g in forb["island_genera"]:
            problems.append(dict(assembly=r["assembly"], error="an outgroup of an island-set genus"))
    for genus, u in taxa["units"].items():
        orders = {tax.at_rank(int(n), "order") for n in u["ncbi_genera"]}
        for r in rows:
            if r["assembly"] in chosen and r["role"].startswith("outgroup") and r["chromosome"] in u["chromosomes"] and tax.at_rank(int(r["ncbi_taxid"]), "order") not in orders:
                problems.append(dict(assembly=r["assembly"], error="an outgroup outside the focal NCBI order"))
    src = open(st.repo("ci/stages/family_a.py")).read()
    for m in FAMA_BANNED:
        if _re.search(r"from ci\.stages(\.%s| import [^\n]*\b%s\b)" % (m, m), src):
            problems.append(dict(module=m, error="the label code imports a score or search module"))
    with open(st.repo("bench/hard/results/v2_reference_set.tsv")) as f:
        others = [(r["accession"], r["genus"]) for r in csv.DictReader(f, delimiter="\t") if r["part"] != "development"]
    texts = [os.path.join(dp, fn) for dp, _, fns in os.walk(base) for fn in fns if fn.endswith((".json", ".tsv"))]
    found = g3_sealed._scan(texts, g3_sealed._pattern(others))
    if found:
        problems.append(dict(error="accessions of the register's other parts in the outputs", found=found))
    require(GateResult("G3_famA", "fail" if problems else "pass", dict(problems=problems[:50], genomes=len(chosen))), run_dir)
    problems = []
    with open(st.repo((cfg["famA"].get("spec")) or FA.SPEC)) as f:
        spec = yaml.safe_load(f)
    need_other, need_og, need_g = int(spec["panel"]["other_min"]), int(spec["outgroups"]["genera_min"]), int(spec["outgroups"]["genomes"])
    for genus, u in taxa["units"].items():
        for acc, c in (u.get("panels") or {}).items():
            if c["other"] < need_other:
                problems.append(dict(chromosome=acc, other=c["other"], error="fewer other-species genomes than required"))
        if len(u["outgroups"]) < need_og:
            problems.append(dict(unit=genus, outgroups=len(u["outgroups"]), error="fewer outgroup genera than required"))
        for o in u["outgroups"]:
            ok = [a for a, g in fetched["genomes"].items() if g.get("gtdb_genus") == o["genus"] and g["role"].startswith("outgroup")
                  and not g["role"].endswith("_replaced") and g.get("n_run_ok")]
            if len(ok) < need_g:
                problems.append(dict(unit=genus, outgroup=o["genus"], usable=len(ok), error="fewer usable outgroup genomes than required"))
    if fetched.get("unavailable"):
        problems.append(dict(error="chosen genomes not fetched", genomes=fetched["unavailable"][:20]))
    require(GateResult("G4_famA_taxa", "fail" if problems else "pass", dict(problems=problems[:50], units={g: dict(chromosomes=len(u["chromosomes"]),
                        outgroups=[o["genus"] for o in u["outgroups"]]) for g, u in taxa["units"].items()})), run_dir)
    return sha256_file(os.path.join(base, "fetch.json"), use_sidecar=False)


FAMA_CLASSES = ("qc_fail", "no_panel", "bridged", "edge_in", "no_polarity", "no_flank", "insertion", "boundary", "unresolved", "accessory",
                "core", "edge")


def phase_famA_labels(st, cfg, run_dir):
    """G4_famA_labels (pipeline family_a, ci/stages/family_a.py: presence and labels; bench/hard/V2_FAMILY_A.md part 2)."""
    import csv
    import re as _re
    import numpy as np
    from ci.stages import family_a as FA
    base = os.path.join(run_dir, "data", "famA")
    with open(os.path.join(base, "taxa.json")) as f:
        taxa = json.load(f)
    with open(os.path.join(base, "labels", "index.json")) as f:
        idx = json.load(f)
    with open(st.repo((cfg["famA"].get("spec")) or FA.SPEC)) as f:
        lc = yaml.safe_load(f)["labels"]
    n_bp = idx["calibration"]["n_kb"] * 1000
    problems, totals = [], dict(insertion=0, core=0)
    if set(idx["counts"]) != set(taxa["placement"]):
        problems.append(dict(error="the labelled chromosomes are not the placed development chromosomes",
                             missing=sorted(set(taxa["placement"]) - set(idx["counts"]))))
    for acc in sorted(idx["counts"]):
        with open(os.path.join(base, "labels", acc + ".tsv")) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        with open(os.path.join(base, "labels", "segments", acc + ".json")) as f:
            segs = {s_["segment"]: s_ for s_ in json.load(f)}
        z = np.load(os.path.join(base, "presence", acc + ".npz"))
        if [r["gene_id"] for r in rows] != list(z["names"]):
            problems.append(dict(chromosome=acc, error="the label genes are not the presence genes"))
        bad = []
        for r in rows:
            c = r["class"]
            if c not in FAMA_CLASSES:
                bad.append((r["gene_id"], c, "unknown class"))
            elif c == "insertion":
                run_ = int(r["run"])
                ok = (r["f_ortho"] == "0.0000" and r["og_absent"] == "True" and run_ >= 0 and segs.get(run_, {}).get("passes") is True
                      and not r["qc"] and float(r["dist_edge"]) > n_bp)
                if not ok:
                    bad.append((r["gene_id"], c, "an insertion outside its rule"))
            elif c == "core":
                ok = (float(r["f_pos"]) >= lc["core"] and r["multi"] == "False" and not r["qc"] and int(r["run"]) < 0 and r["band"] == "False"
                      and r["in_u"] == "False" and r["near_u"] == "False" and (r["dist_u"] == "" or float(r["dist_u"]) > n_bp))
                if not ok:
                    bad.append((r["gene_id"], c, "a core gene outside its rule"))
            if c in totals:
                totals[c] += 1
        if bad:
            problems.append(dict(chromosome=acc, error="genes against their class rule", genes=bad[:10], n=len(bad)))
    if not totals["insertion"] or not totals["core"]:
        problems.append(dict(error="no insertion or no core gene at all", totals=totals))
    src = open(st.repo("ci/stages/family_a.py")).read()
    for m in FAMA_BANNED:
        if _re.search(r"from ci\.stages(\.%s| import [^\n]*\b%s\b)" % (m, m), src):
            problems.append(dict(module=m, error="the label code imports a score or search module"))
    with open(st.repo("bench/hard/results/v2_reference_set.tsv")) as f:
        others = [(r["accession"], r["genus"]) for r in csv.DictReader(f, delimiter="\t") if r["part"] != "development"]
    texts = [os.path.join(dp, fn) for d in ("presence", "labels") for dp, _, fns in os.walk(os.path.join(base, d)) for fn in fns
             if fn.endswith((".json", ".tsv"))]
    found = g3_sealed._scan(texts, g3_sealed._pattern(others))
    if found:
        problems.append(dict(error="accessions of the register's other parts in the outputs", found=found))
    require(GateResult("G4_famA_labels", "fail" if problems else "pass", dict(problems=problems[:50], n_kb=idx["calibration"]["n_kb"],
                       calibrated=idx["calibration"]["calibrated"], totals=totals)), run_dir)
    return sha256_file(os.path.join(base, "labels", "index.json"), use_sidecar=False)


def phase_famA_eval(st, cfg, run_dir):
    """G3_famA_eval and G4_famA_eval (pipeline family_a_eval, ci/stages/family_a_eval.py; bench/hard/V2_FAMILY_A.md part 3)."""
    import csv
    import re as _re
    from ci.stages import family_a as FA
    from ci.stages import family_a_eval as FE
    base = os.path.join(run_dir, "data", "famA")
    with open(os.path.join(base, "eval", "report.json")) as f:
        rep = json.load(f)
    with open(st.repo(cfg["famAe"].get("spec") or FA.SPEC)) as f:
        ec = yaml.safe_load(f)["eval"]
    # G3: development chromosomes only, the label code still free of score imports, no other part's accession anywhere
    problems = []
    dev = {r["accession"] for r in FA.dev_chromosomes(st.repo)}
    if not set(rep["chromosomes"]) <= dev:
        problems.append(dict(error="chromosomes outside the development part", found=sorted(set(rep["chromosomes"]) - dev)))
    src = open(st.repo("ci/stages/family_a.py")).read()
    for m in FAMA_BANNED:
        if _re.search(r"from ci\.stages(\.%s| import [^\n]*\b%s\b)" % (m, m), src):
            problems.append(dict(module=m, error="the label code imports a score or search module"))
    with open(st.repo("bench/hard/results/v2_reference_set.tsv")) as f:
        others = [(r["accession"], r["genus"]) for r in csv.DictReader(f, delimiter="\t") if r["part"] != "development"]
    texts = [os.path.join(dp, fn) for dp, _, fns in os.walk(base) for fn in fns if fn.endswith((".json", ".tsv"))]
    found = g3_sealed._scan(texts, g3_sealed._pattern(others))
    if found:
        problems.append(dict(error="accessions of the register's other parts in the outputs", found=found))
    require(GateResult("G3_famA_eval", "fail" if problems else "pass", dict(problems=problems[:50])), run_dir)
    # G4: every arm and score present; balance; own covariates; shuffle; reproduction; lk_og
    problems = []
    for arm in FE.ARMS:
        if arm not in rep["arms"]:
            problems.append(dict(arm=arm, error="arm missing"))
    for arm in ("main", "core", "noise"):
        miss = [s for s in rep["scores"] if (rep["arms"].get(arm) or {}).get(s, {}).get("auroc") is None]
        if miss:
            problems.append(dict(arm=arm, error="scores without an AUROC", scores=miss[:20]))
    bad = [b for b in rep["balance"] if b["smd_after"] is None or abs(b["smd_after"]) > float(ec["smd_max"])]
    if bad or {b["unit"] for b in rep["balance"]} != set(rep["evaluated_units"]):
        problems.append(dict(error="main arm out of balance, or a unit without matched pairs", rows=bad))
    for c in FE.COVS:
        iv = (rep["intervals"].get("main") or {}).get(c)
        if not iv or iv["lo"] is None or iv.get("n_boot", 0) < 0.9 * int(ec["boot"]) or not (iv["lo"] <= 0.5 <= iv["hi"]):
            problems.append(dict(covariate=c, interval=iv, error="a matched covariate's own AUROC interval excludes 0.5"))
    off = {s: v for s, v in rep["shuffle"].items() if abs(v - 0.5) > float(ec["shuffle_tolerance"])}
    if off or not rep["shuffle"]:
        problems.append(dict(error="shuffle control away from 0.5", scores=off))
    v1 = rep["arms"].get("v1") or {}
    unrep = {s: r for s, r in v1.items() if r.get("audit") is not None and (r["diff"] is None or r["diff"] > float(ec["reproduce"]))}
    if unrep or not any(r.get("audit") is not None for r in v1.values()):
        problems.append(dict(error="the v1 arm does not reproduce the audit", scores=unrep))
    if rep.get("lk_og", {}).get("auroc_main") is None:
        problems.append(dict(error="lk_og missing"))
    require(GateResult("G4_famA_eval", "fail" if problems else "pass", dict(problems=problems[:50], units=rep["evaluated_units"],
                       lk_og=rep.get("lk_og"), kept=rep["ladder"]["len+hom+ident"])), run_dir)
    return sha256_file(os.path.join(base, "eval", "report.json"), use_sidecar=False)


def _g3_files(st, cfg, run_dir):
    if pipeline_of(cfg) == "select":
        read = []
        spec = load_select_spec(cfg["select_spec"], st.repo, read)     # the variant and every spec it is based on
        lock = (cfg.get("catalog") or {}).get("lock")
        return [os.path.join(run_dir, "config.resolved.yaml"), st.repo(spec["assumptions"])] + [st.repo(f) for f in read] + \
            ([st.repo(lock)] if lock and os.path.exists(st.repo(lock)) else [])
    sc = cfg.get("scenario") or {}
    isl = [st.repo((cfg.get("islands") or {}).get("genomes", "bench/islands/results/dev_genomes.tsv"))] if cfg.get("islands") else []
    return [os.path.join(run_dir, "config.resolved.yaml"), st.repo(cfg["corpus_spec"])] + [st.repo(sc[k]) for k in ("table", "truth") if sc.get(k)] + isl


def phase_select(st, cfg, run_dir):
    """G1_select, G2_select and G3_select on the selected table; all three are decided and recorded before a failure stops
    the run (the report of the selection is written by the budget stage whatever they say)."""
    import csv
    spec = load_select_spec(cfg["select_spec"], st.repo)
    d = os.path.join(run_dir, "data", "select")
    with open(os.path.join(d, "selected.tsv")) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    results = [run(g1_kingdom.selection(spec, rows, tax, cfg.get("tier")), run_dir)]
    def hosts(r):
        return sorted({int(h) for h in (r.get("hosts") or "").split(";") if h} | ({int(r["host_taxid"])} if r.get("host_taxid") not in ("", "0", None) else set()))
    ent = []
    for r in rows:                                     # one entry per host, so every host of a virus is checked by lineage
        base = dict(id=r["id"], taxid=int(r["taxid"]), accession=r["accession"], paired_accession=r["paired_accession"])
        ent += [dict(base, host_taxid=h) for h in hosts(r)] or [base]
    unchecked = sum(1 for r in rows if r["group"].startswith("viruses.") and not hosts(r))   # relaxed variant: no host to check
    variant = (spec.get("rules") or {}).get("viruses", {}).get("unknown_host", "exclude")
    g2 = g2_exclusion.entries(ent, tax, extra_paths=[st.repo(x) for x in cfg.get("extra_exclusions") or []])
    g2.gate = "G2_select"
    g2.evidence.update(unknown_host=variant, viruses_without_host_checked=unchecked)
    results.append(run(g2, run_dir))
    files = [os.path.join(d, n) for n in ("selected.tsv", "by_group.json", "manifest.json")]
    g3 = g3_sealed.selection(files, tax, {r["id"]: [int(r["taxid"])] + hosts(r) for r in rows}, cfg.get("tier"))
    g3.evidence.update(unknown_host=variant, viruses_without_host_checked=unchecked)
    results.append(run(g3, run_dir))
    bad = [r.gate for r in results if r.status == "fail"]
    if bad:
        raise GateFailure("gates failed: %s" % ", ".join(bad))
    return sha256_file(os.path.join(d, "selected.tsv"), use_sidecar=False)


def phase_built(st, cfg, run_dir, phase):
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    man = _manifest(run_dir, "corpus")
    entries = man["entries"]
    inputs = [man["sha256"]]
    if phase == "train":
        p = os.path.join(run_dir, "data", "train", "ids.json")
        with open(p) as f:
            doc = json.load(f)
        ids = doc.get("genomes") or []
        known = {e["id"] for e in entries}
        problems = []
        if not ids:
            problems.append("the training id list is empty")
        if len(ids) != len(set(ids)):
            problems.append("the training id list has duplicates")
        if set(ids) - known:
            problems.append("training ids not in the corpus manifest: %s" % sorted(set(ids) - known)[:5])
        if doc.get("corpus_dataset") != man["dataset_id"]:
            problems.append("the id list names corpus %s, the manifest is %s" % (doc.get("corpus_dataset"), man["dataset_id"]))
        require(GateResult("G2_train_ids", "fail" if problems else "pass", dict(problems=problems, n=len(ids))), run_dir)
        entries = [e for e in entries if e["id"] in set(ids)]
        inputs.append(sha256_file(p, use_sidecar=False))
    with open(st.repo(cfg["corpus_spec"])) as f:
        spec = yaml.safe_load(f)
    g1 = g1_kingdom.built(spec, entries, tax, cfg.get("tier"), KingdomMap())
    g1.gate = "G1_" + phase
    require(g1, run_dir)
    g2 = g2_exclusion.entries(entries, tax, extra_paths=[st.repo(x) for x in cfg.get("extra_exclusions") or []])
    g2.gate = "G2_" + phase
    require(g2, run_dir)
    if find_waiver("G5", "*"):
        st.extra["g5"] = "waived: no public-event catalogue yet"
    return sha256_json(inputs)


def phase_eval(st, cfg, run_dir):
    corpus = _manifest(run_dir, "corpus")
    evals = _manifest(run_dir, "eval_inputs")
    train = set()
    for e in corpus["entries"]:
        train |= _seq_hashes(os.path.join(run_dir, "data", "corpus", e["id"]))
    shared = {}
    for e in evals["entries"]:
        n = len(_seq_hashes(os.path.join(run_dir, "data", "eval_inputs", e["id"])) & train)
        if n:
            shared[e["id"]] = n
    ids = {e["id"] for e in corpus["entries"]} & {e["id"] for e in evals["entries"]}
    problems = [dict(error="evaluation genome is a corpus genome", ids=sorted(ids))] if ids else []
    if shared:
        problems.append(dict(error="evaluation sequences found in the training corpus", per_genome=shared))
    require(GateResult("G2_holdout", "fail" if problems else "pass", dict(problems=problems, eval_genomes=len(evals["entries"]))), run_dir)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    eval_taxids = {}
    truth = []
    sc = cfg.get("scenario") or {}
    if sc.get("truth"):
        import csv
        with open(st.repo(sc["truth"])) as f:
            truth = list(csv.DictReader(f, delimiter="\t"))
    for e in evals["entries"]:
        eval_taxids[e["id"]] = [int(e["host_taxid"])] + [int(t["donor_taxid"]) for t in truth if t.get("scenario") == e["id"] and t.get("donor_taxid")]
    require(g3_sealed.built(run_dir, tax, eval_taxids, cfg.get("tier"), extra_files=_g3_files(st, cfg, run_dir)), run_dir)
    return sha256_json([corpus["sha256"], evals["sha256"]])


def phase_report(st, cfg, run_dir):
    import subprocess
    p = os.path.join(run_dir, "data", "eval", "eval_report.json")
    with open(p) as f:
        report = json.load(f)
    r = subprocess.run(["git", "rev-parse", g4_report.TAG + "^{commit}"], cwd=ROOT, capture_output=True, text=True)
    tag_commit = r.stdout.strip() if r.returncode == 0 else ""
    skips = [k for k, v in (cfg.get("comparators") or {}).items() if v == "skip"]
    require(g4_report.check(report, cfg.get("tier"), skips, tag_commit), run_dir)
    return sha256_file(p, use_sidecar=False)


def main():
    run_dir, phase = sys.argv[1], sys.argv[2]
    check_only = "--check-only" in sys.argv[3:]
    with Stage(run_dir, "gates_" + phase + ("_recheck" if check_only else "")) as st:
        cfg = st.config
        try:
            if phase == "pre":
                key = phase_pre(st, cfg, run_dir)
            elif phase in ("corpus", "train"):
                key = phase_built(st, cfg, run_dir, phase)
            elif phase == "eval":
                key = phase_eval(st, cfg, run_dir)
            elif phase == "report":
                key = phase_report(st, cfg, run_dir)
            elif phase == "select":
                key = phase_select(st, cfg, run_dir)
            elif phase == "islands":
                key = phase_islands(st, cfg, run_dir)
            elif phase == "selfsup":
                key = phase_selfsup(st, cfg, run_dir)
            elif phase == "codon":
                key = phase_codon(st, cfg, run_dir)
            elif phase == "register":
                key = phase_register(st, cfg, run_dir)
            elif phase == "audit":
                key = phase_audit(st, cfg, run_dir)
            elif phase == "famA_taxa":
                key = phase_famA_taxa(st, cfg, run_dir)
            elif phase == "famA_labels":
                key = phase_famA_labels(st, cfg, run_dir)
            elif phase == "famA_eval":
                key = phase_famA_eval(st, cfg, run_dir)
            else:
                raise SystemExit("unknown gate phase %s" % phase)
        except GateFailure as err:
            st.extra["gate_failure"] = str(err)[:2000]
            raise SystemExit(str(err)[:2000])
        except Exception as err:  # noqa: BLE001 - a gate that cannot run has failed (fail closed)
            run(GateResult("gates_%s_error" % phase, "fail", dict(error="%s: %s" % (type(err).__name__, err))), run_dir)
            st.extra["gate_failure"] = "%s: %s" % (type(err).__name__, err)
            raise
        if check_only:
            return
        sentinel = os.path.join(run_dir, "gates", "%s.ok" % phase)
        content = {"phase": phase, "passed": True, "inputs": key}
        old = None
        if os.path.exists(sentinel):
            with open(sentinel) as f:
                old = json.load(f)
        if old != content:
            atomic_write_json(sentinel, content)
        st.output("sentinel", sentinel)


if __name__ == "__main__":
    main()
