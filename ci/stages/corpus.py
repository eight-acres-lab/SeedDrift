"""Stage corpus: build the training corpus of a spec (configs/corpus/<name>.yaml) into data/corpus/.

Source kinds:
  raw_prokaryote, raw_virus, plasmid   DNA only; genes called by pyrodigal as the app does (single-genome mode from
                                       100 kb of sequence, metagenome mode below), with the source's genetic code;
                                       `exclude_regions` cuts held-out regions (an evaluation region) out first;
                                       `max_genes` keeps the first genes in genome order
  annotated                            DNA plus a gene table (gene_id, replicon, start, end, strand) and proteins; the
                                       coding DNA is cut from the DNA and must translate to the given protein
  annotated_cds                        Ensembl coding sequences (cds.all): protein-coding genes only, the longest
                                       coding sequence of each gene, coordinates from the FASTA headers, organelle
                                       replicons left out (a separate class), optionally only some `replicons`
                                       and at most `max_genes` genes in genome order
Remote sources (ena_assembly, ena_sequences, ensembl) take their files from the fetch stage (data/fetch/manifest.json).
Every genome gets data/corpus/<id>/{genes.tsv, proteins.faa, cds.fna}; the manifest (data/corpus/manifest.json,
content-addressed copy in manifests/) lists each genome with taxid, source kind, protein count and file hashes.
"""
from __future__ import annotations

import csv
import json
import os

import yaml

from ci import manifest as M
from ci.stages.common import Stage, main_args, read_fasta, write_fasta, write_json

COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(s):
    return s.translate(COMP)[::-1]


REMOTE = ("ena_assembly", "ena_sequences", "ensembl")
ORGANELLE = {"MT", "Mt", "mt", "MtDNA", "mitochondrion", "Mito", "Pt", "chloroplast", "plastid", "apicoplast", "M"}


def sources_of(spec, repo, fetched=None, unavailable=()):
    """The corpus sources with their files; remote sources the fetch stage found no record of (`unavailable`, listed
    in its manifest under fetch.allow_missing) are left out."""
    rows = []
    for r in spec.get("sources") or []:
        if any(k in r for k in REMOTE):
            if r["id"] in unavailable:
                continue
            if fetched is None or r["id"] not in fetched:
                raise SystemExit("source %s needs the fetch stage" % r["id"])
            rows.append(dict(r, files=fetched[r["id"]]))
        else:
            rows.append(dict(r, files=[repo(x) for x in r["files"]]))
    if spec.get("sources_table"):
        base = os.path.dirname(repo(spec["sources_table"]))
        with open(repo(spec["sources_table"])) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                rows.append(dict(id=r["id"], source_kind=r["source_kind"], taxid=int(r["taxid"]), table=int(r.get("table") or 11),
                                 accession=r.get("accession", ""), paired_accession=r.get("paired_accession", ""), synthetic=r.get("synthetic", "no"),
                                 files=[os.path.join(base, x) for x in r["files"].split(";")]))
    for r in rows:
        if r["source_kind"] in ("raw_prokaryote", "plasmid") and r.get("synthetic", "no") == "no" and not str(r.get("accession", "")).startswith(("GCA_", "GCF_")):
            raise SystemExit("source %s comes from an assembly but has no assembly accession (needed by gate G2)" % r["id"])
    return rows


def clean_name(name):
    """ENA FASTA names (ENA|AE016828|AE016828.3) become the sequence accession (AE016828.3)."""
    return name.split("|")[-1] if name.startswith("ENA|") else name


def cut_regions(contigs, regions):
    """Remove regions ({replicon, start, end}, 1-based inclusive) from the sequences: a replicon with a region cut out
    becomes its pieces <replicon>_p1, <replicon>_p2, ... (an evaluation region held out of the training corpus)."""
    out = dict(contigs)
    by_rep = {}
    for r in regions:
        if r["replicon"] not in contigs:
            raise SystemExit("exclude_regions: no replicon %s (have %s)" % (r["replicon"], sorted(contigs)[:5]))
        a, b = int(r["start"]), int(r["end"])
        if not 1 <= a <= b <= len(contigs[r["replicon"]]):
            raise SystemExit("exclude_regions: %s %d-%d is not inside the replicon (1-%d)" % (r["replicon"], a, b, len(contigs[r["replicon"]])))
        by_rep.setdefault(r["replicon"], []).append((a, b))
    for rep, cuts in by_rep.items():
        seq, pieces, pos = out.pop(rep), [], 1
        for a, b in sorted(cuts):
            if a > pos:
                pieces.append(seq[pos - 1:a - 1])
            pos = max(pos, b + 1)
        if pos <= len(seq):
            pieces.append(seq[pos - 1:])
        for k, piece in enumerate(pieces, 1):
            out["%s_p%d" % (rep, k)] = piece
    return out


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


def annotated_cds(path, table, replicons=None, max_genes=0):
    """Genes of an Ensembl cds.all FASTA: header '>TRANSCRIPT cds TYPE:ASSEMBLY:NAME:START:END:STRAND gene:GENE ...'."""
    import gzip
    from Bio.Seq import Seq
    best = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        head, buf = None, []

        def take(head, seq):
            fields = head[1:].split()
            tags = dict(x.split(":", 1) for x in fields[2:] if ":" in x)
            if tags.get("gene_biotype") != "protein_coding" or tags.get("transcript_biotype") != "protein_coding":
                return
            loc = fields[2].split(":") if len(fields) > 2 else []
            if len(loc) < 6:
                return
            name, start, end, strand = loc[2], int(loc[3]), int(loc[4]), loc[5]
            if name in ORGANELLE or (replicons and name not in replicons):
                return
            gene = tags.get("gene", fields[0]).split(".")[0]
            if gene not in best or len(seq) > len(best[gene]["cds"]):
                best[gene] = dict(gene_id=gene, replicon=name, start=start, end=end, strand="+" if strand == "1" else "-", partial="00", cds=seq.upper())
        for line in f:
            if line.startswith(">"):
                if head:
                    take(head, "".join(buf))
                head, buf = line.rstrip("\n"), []
            else:
                buf.append(line.strip())
        if head:
            take(head, "".join(buf))
    genes = []
    for g in sorted(best.values(), key=lambda g: (g["replicon"], g["start"], g["gene_id"])):
        if len(g["cds"]) % 3:
            continue
        p = str(Seq(g["cds"]).translate(table=table))
        if "*" in p.rstrip("*") or len(p) < 30:
            continue
        g["protein"] = p.rstrip("*")
        genes.append(g)
        if max_genes and len(genes) >= max_genes:
            break
    return genes, "annotated (Ensembl)"


def build_one(src, out_dir):
    kind = src["source_kind"]
    table = int(src.get("table", 11))
    if src.get("exclude_regions") and kind not in ("raw_prokaryote", "raw_virus", "plasmid"):
        raise SystemExit("source %s: exclude_regions works on DNA sources only, not on %s" % (src["id"], kind))
    if kind == "annotated":
        genes, mode = annotated(src["files"], table)
    elif kind == "annotated_cds":
        genes, mode = annotated_cds(src["files"][0], int(src.get("table", 1)), set(src.get("replicons") or []), int(src.get("max_genes", 0)))
    elif kind in ("raw_prokaryote", "raw_virus", "plasmid"):
        contigs = {}
        for f in src["files"]:
            contigs.update({clean_name(k): v for k, v in read_fasta(f).items()})
        contigs = cut_regions(contigs, src.get("exclude_regions") or [])
        genes, mode = call_genes(contigs, table, force_meta=kind in ("raw_virus", "plasmid"))
        if int(src.get("max_genes", 0)) and len(genes) > int(src["max_genes"]):   # the first genes in genome order
            genes = genes[:int(src["max_genes"])]
            mode += ", first %d genes" % int(src["max_genes"])
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
        fetched, unavailable = None, set()
        fm = os.path.join(run_dir, "data", "fetch", "manifest.json")
        if os.path.exists(fm):
            cas = os.path.join(st.config["data_root"], "cas")
            with open(fm) as f:
                fman = json.load(f)
            fetched = {e["id"]: [os.path.join(cas, x["path"]) for x in e["files"]] for e in fman["entries"]}
            unavailable = {m["id"] for m in (fman.get("meta") or {}).get("unavailable") or []}
            st.extra["left_out_no_record"] = sorted(unavailable)
        entries = []
        for src in sources_of(spec, st.repo, fetched, unavailable):
            out_dir = st.path("corpus", src["id"], "x")[:-2]
            genes, mode = build_one(src, out_dir)
            files = [M.file_entry(os.path.join(out_dir, n), run_dir) for n in ("genes.tsv", "proteins.faa", "cds.fna")]
            entries.append(dict(id=src["id"], taxid=int(src["taxid"]), source_kind=src["source_kind"], proteins=len(genes), gene_calling=mode, files=files,
                                accession=src.get("accession", ""), paired_accession=src.get("paired_accession", ""),
                                **({k: src[k] for k in ("cluster", "relation") if src.get(k)})))
        man = M.build(spec["name"], "corpus", entries, meta=dict(spec=st.config["corpus_spec"]))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("corpus", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("corpus_manifest", mp)
        st.extra.update(dataset_id=M.dataset_id(man), genomes=len(entries), proteins=sum(e["proteins"] for e in entries))


if __name__ == "__main__":
    main()
