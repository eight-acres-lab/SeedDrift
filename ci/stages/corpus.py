"""Stage corpus: build the training corpus of a spec (configs/corpus/<name>.yaml) into data/corpus/.

Source kinds:
  raw_prokaryote, raw_virus, plasmid   DNA only; genes called by pyrodigal as the app does (single-genome mode from
                                       100 kb of sequence, metagenome mode below), with the source's genetic code
  annotated                            DNA plus a gene table (gene_id, replicon, start, end, strand) and proteins; the
                                       coding DNA is cut from the DNA and must translate to the given protein
Every genome gets data/corpus/<id>/{genes.tsv, proteins.faa, cds.fna}; the manifest (data/corpus/manifest.json,
content-addressed copy in manifests/) lists each genome with taxid, source kind, protein count and file hashes.
"""
from __future__ import annotations

import csv
import os

import yaml

from ci import manifest as M
from ci.stages.common import Stage, main_args, read_fasta, write_fasta, write_json

COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(s):
    return s.translate(COMP)[::-1]


def sources_of(spec, repo):
    rows = [dict(r, files=[repo(x) for x in r["files"]]) for r in (spec.get("sources") or [])]
    if spec.get("sources_table"):
        base = os.path.dirname(repo(spec["sources_table"]))
        with open(repo(spec["sources_table"])) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows.append(dict(id=r["id"], source_kind=r["source_kind"], taxid=int(r["taxid"]), table=int(r.get("table") or 11),
                                 accession=r.get("accession", ""), paired_accession=r.get("paired_accession", ""), synthetic=r.get("synthetic", "no"),
                                 files=[os.path.join(base, x) for x in r["files"].split(";")]))
    for r in rows:
        if r["source_kind"] != "annotated" and r.get("synthetic", "no") == "no" and not str(r.get("accession", "")).startswith(("GCA_", "GCF_")):
            raise SystemExit("source %s comes from an assembly but has no assembly accession (needed by gate G2)" % r["id"])
    return rows


def call_genes(contigs, table, force_meta=False):
    import pyrodigal
    total = sum(len(s) for s in contigs.values())
    meta = force_meta or total < 100_000
    gf = pyrodigal.GeneFinder(meta=meta)
    if not meta:
        gf.train(*[s.upper().encode() for s in contigs.values()], translation_table=table)
    genes = []
    for cid, seq in contigs.items():
        up = seq.upper()
        for k, g in enumerate(gf.find_genes(up.encode()), 1):
            nt = up[g.begin - 1:g.end]
            genes.append(dict(gene_id="%s_%d" % (cid, k), replicon=cid, start=g.begin, end=g.end, strand="+" if g.strand == 1 else "-",
                              partial="%d%d" % (int(g.partial_begin), int(g.partial_end)), cds=revcomp(nt) if g.strand == -1 else nt,
                              protein=g.translate(translation_table=None if meta else table).rstrip("*")))
    return genes, ("meta" if meta else "single")


def annotated(files, table):
    from Bio.Seq import Seq
    fna = next(f for f in files if f.endswith(".fna"))
    tsv = next(f for f in files if f.endswith(".tsv"))
    faa = next(f for f in files if f.endswith(".faa"))
    contigs, prots = read_fasta(fna), read_fasta(faa)
    genes = []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s, e = int(r["start"]), int(r["end"])
            nt = contigs[r["replicon"]][s - 1:e].upper()
            cds = revcomp(nt) if r["strand"] == "-" else nt
            p = str(Seq(cds).translate(table=table)).rstrip("*")
            if p != prots[r["gene_id"]].rstrip("*"):
                raise SystemExit("annotated gene %s does not translate to its protein" % r["gene_id"])
            genes.append(dict(gene_id=r["gene_id"], replicon=r["replicon"], start=s, end=e, strand=r["strand"], partial="00", cds=cds, protein=p))
    return genes, "annotated"


def build_one(src, out_dir):
    kind = src["source_kind"]
    table = int(src.get("table", 11))
    if kind == "annotated":
        genes, mode = annotated(src["files"], table)
    elif kind in ("raw_prokaryote", "raw_virus", "plasmid"):
        contigs = {}
        for f in src["files"]:
            contigs.update(read_fasta(f))
        genes, mode = call_genes(contigs, table, force_meta=kind in ("raw_virus", "plasmid"))
    else:
        raise SystemExit("unknown source kind %s" % kind)
    os.makedirs(out_dir, exist_ok=True)
    gp = os.path.join(out_dir, "genes.tsv")
    with open(gp + ".part", "w") as f:
        f.write("gene_id\treplicon\tstart\tend\tstrand\tpartial\n")
        for g in genes:
            f.write("%(gene_id)s\t%(replicon)s\t%(start)d\t%(end)d\t%(strand)s\t%(partial)s\n" % g)
    os.replace(gp + ".part", gp)
    write_fasta(os.path.join(out_dir, "proteins.faa"), [("%s__%s" % (src["id"], g["gene_id"]), g["protein"]) for g in genes])
    write_fasta(os.path.join(out_dir, "cds.fna"), [("%s__%s" % (src["id"], g["gene_id"]), g["cds"]) for g in genes])
    return genes, mode


def main():
    run_dir = main_args()
    with Stage(run_dir, "corpus", requires=("pre",)) as st:
        with open(st.repo(st.config["corpus_spec"])) as f:
            spec = yaml.safe_load(f)
        entries = []
        for src in sources_of(spec, st.repo):
            out_dir = st.path("corpus", src["id"], "x")[:-2]
            genes, mode = build_one(src, out_dir)
            files = [M.file_entry(os.path.join(out_dir, n), run_dir) for n in ("genes.tsv", "proteins.faa", "cds.fna")]
            entries.append(dict(id=src["id"], taxid=int(src["taxid"]), source_kind=src["source_kind"], proteins=len(genes), gene_calling=mode, files=files,
                                accession=src.get("accession", ""), paired_accession=src.get("paired_accession", "")))
        man = M.build(spec["name"], "corpus", entries, meta=dict(spec=st.config["corpus_spec"]))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("corpus", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("corpus_manifest", mp)
        st.extra.update(dataset_id=M.dataset_id(man), genomes=len(entries), proteins=sum(e["proteins"] for e in entries))


if __name__ == "__main__":
    main()
