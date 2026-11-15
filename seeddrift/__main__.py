"""Command-line interface:  python3 -m seeddrift <command> ..."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="seeddrift", description="SeedDrift: which genes drifted in from elsewhere "
                                 "(horizontal gene transfer: donors, single-genome screen, services)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="detect HGT events genome-wide")
    s.add_argument("--species-tree", required=True, help="rooted species tree (Newick)")
    s.add_argument("--gene-trees", required=True,
                   help="folder with one file per family (<family>.nwk / .treefile / .ufboot); ML tree first")
    s.add_argument("--genes", required=True,
                   help="TSV: gene_id species family contig start end (gene_id = gene-tree leaf name)")
    s.add_argument("--me", help="TSV: species contig start end type (mobile elements)")
    s.add_argument("--species-meta", help="TSV: species domain (prok|euk), overrides --domain per genome")
    s.add_argument("--domain", default="prok", choices=["prok", "euk"], help="default domain")
    s.add_argument("--model", help="calibrated model JSON (default: built-in hand-set weights)")
    s.add_argument("--out", required=True)
    s.add_argument("--samples", type=int, default=11, help="max gene trees per family (ML + bootstraps)")
    s.add_argument("--bt", type=int, default=3, help="random optimal reconciliations per tree")
    s.add_argument("--costs", default="2,3,1", help="duplication,transfer,loss costs")
    s.add_argument("--dated", default="auto", choices=["auto", "on", "off"],
                   help="only allow transfers between branches that coexist in time "
                        "(needs an ultrametric/dated species tree; auto = on if branch lengths present)")
    s.add_argument("--window", type=int, default=4, help="synteny window (genes on each side)")
    s.add_argument("--clock-z", type=float, default=2.5,
                   help="molecular-clock candidates: speciation nodes whose gene divergence is this many "
                        "robust SDs younger than usual (0 = off; needs gene-tree branch lengths)")
    s.add_argument("--collapse", type=float, default=0.0,
                   help="contract gene-tree branches with support below this (e.g. 95 for IQ-TREE UFBoot; "
                        "also ~zero-length branches) and resolve them along the species tree; the ML tree "
                        "then replaces the bootstrap sample. 0 = off")
    s.add_argument("--composition", help="TSV: gene_id gc gc3 (CDS composition; `prepare` writes "
                                          "gene_composition.tsv)")
    s.add_argument("--no-gain", dest="gain", action="store_false",
                   help="do not score families confined to a clade as transfers from unsampled donors")
    s.add_argument("--split-paralogs", type=int, default=2,
                   help="split multi-copy families into orthologous subfamilies at gene-tree nodes whose two "
                        "subtrees share at least this many species (ancient duplications); 0 = off")
    s.add_argument("--min-freq", type=float, default=0.1, help="report events with f_T >= this")
    s.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    s.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prepare", help="build scan inputs from real genomes (Ensembl Genomes)")
    p.add_argument("--genomes", required=True, help="TSV: label ensembl_name role (role=outgroup roots the tree)")
    p.add_argument("--index", help="species_Ensembl<Division>.txt (needed for fetch)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--steps", default="fetch,proteomes,composition,orthogroups,genetrees,speciestree")
    p.add_argument("--tools", default=os.path.expanduser("~/mamba/envs/hgt/bin"),
                   help="bin directory with orthofinder, mafft, iqtree2")
    p.add_argument("--division", default="bacteria", help="Ensembl Genomes division (bacteria, fungi, ...)")
    p.add_argument("--max-genes", type=int, default=150, help="skip larger families in genetrees")
    p.add_argument("--tree-method", default="iqtree", choices=["iqtree", "fasttree"],
                   help="gene trees: IQ-TREE (LG+G4, UFBoot) or FastTree (~100x less CPU, SH-like supports)")
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))

    dn = sub.add_parser("donors", help="donor taxa of candidates from outside the dataset (UniProt reference "
                                       "proteomes, MMseqs2-GPU / DIAMOND, phylogenies)")
    dn.add_argument("--workdir", required=True, help="prepare workdir (genomes.tsv, proteomes/)")
    dn.add_argument("--scan", required=True, help="scan output folder (hgt_genes.tsv, species_tree.labeled.nwk)")
    dn.add_argument("--ref", required=True, help="reference folder (downloaded proteomes, taxonomy, databases)")
    dn.add_argument("--out", required=True)
    dn.add_argument("--index", required=True, help="species_EnsemblBacteria.txt (NCBI taxids of the genomes)")
    dn.add_argument("--steps", default="reference,queries,search,classify,trees")
    dn.add_argument("--threshold", type=float, default=0.6, help="P_HGT of candidate genes")
    dn.add_argument("--sources", default="gain", help="candidate sources, comma-separated (gain,tree,clock)")
    dn.add_argument("--engine", default="auto", choices=["auto", "mmseqs-gpu", "mmseqs", "diamond"])
    dn.add_argument("--mmseqs", help="mmseqs binary (GPU build for mmseqs-gpu)")
    dn.add_argument("--ingroup", type=int, default=91347,
                    help="NCBI taxid of the recipients' wider group for the Alien Index (default Enterobacterales)")
    dn.add_argument("--tree-method", default="fasttree", choices=["fasttree", "iqtree"],
                    help="donor trees: FastTree (fast, approximate ML) or IQ-TREE LG+G4 with UFBoot")
    dn.add_argument("--max-proteins", type=int, default=25_000_000)
    dn.add_argument("--download-threads", type=int, default=16)
    dn.add_argument("--tools", default=os.path.expanduser("~/mamba/envs/hgt/bin"))
    dn.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))

    st = sub.add_parser("structure", help="structural donor search (ESMFold + Foldseek) for candidates "
                                          "without outside sequence homologs")
    st.add_argument("--donors", required=True, help="output folder of `donors`")
    st.add_argument("--out", required=True)
    st.add_argument("--steps", default="queries,predict,pfam,search,classify")
    st.add_argument("--model", help="local folder with the HuggingFace ESMFold model (facebook/esmfold_v1)")
    st.add_argument("--max-len", type=int, default=1000, help="skip longer proteins in ESMFold")
    st.add_argument("--target", help="target structures for Foldseek (PDB/mmCIF folder or tar)")
    st.add_argument("--foldseek", default="foldseek")
    st.add_argument("--mmseqs", default="mmseqs", help="mmseqs binary for the profile step (GPU build)")
    st.add_argument("--cpu", action="store_true", help="profile and Foldseek steps on CPU instead of GPU")
    st.add_argument("--ref", help="donors reference folder (NCBI taxonomy)")
    st.add_argument("--sprot", help="uniprot_sprot.fasta.gz (accession -> taxid of the target structures)")
    st.add_argument("--pfam", help="folder with Pfam-A.hmm.gz and Pfam-A.hmm.dat.gz (repeat filter)")
    st.add_argument("--ncoils", default="ncoils", help="COILS binary for coiled coils (bioconda `coils`)")
    st.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))

    m = sub.add_parser("simulate", help="simulate a test dataset with known transfers")
    m.add_argument("--out", required=True)
    m.add_argument("--domain", default="prok", choices=["prok", "euk", "mixed"])
    m.add_argument("--species", type=int, default=16)
    m.add_argument("--families", type=int, default=600)
    m.add_argument("--samples", type=int, default=10, help="bootstrap-like trees per family")
    m.add_argument("--noise", type=float, default=0.04, help="NNI errors per internal edge")
    m.add_argument("--seed", type=int, default=1)

    e = sub.add_parser("evaluate", help="compare scan output with simulated truth")
    e.add_argument("--events", required=True)
    e.add_argument("--truth", required=True)
    e.add_argument("--species-tree", required=True)
    e.add_argument("--genes")
    e.add_argument("--contaminants")
    e.add_argument("--threshold", type=float, default=0.6)
    e.add_argument("--oracle", help="hgt_events.tsv from a scan of the true gene trees (defines identifiable events)")
    e.add_argument("--json", help="also write metrics as JSON")

    c = sub.add_parser("calibrate", help="fit P(HGT) weights on simulated (or curated) truth")
    c.add_argument("--events", required=True, nargs="+")
    c.add_argument("--truth", required=True, nargs="+")
    c.add_argument("--out", required=True)
    c.add_argument("--l2", type=float, default=1.0)

    r = sub.add_parser("rescore", help="re-apply a model to an existing hgt_events.tsv")
    r.add_argument("--events", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)

    g = sub.add_parser("genome", help="single-genome screen: genes that look like they came from mammals, "
                       "prokaryotes or other distant lineages (uses the tools/ESMFold services or local tools)")
    g.add_argument("input", nargs="+", help="proteins / CDS / GenBank / EMBL / GFF3 (+ its FASTA) / genome contigs; "
                   "gz, bz2, xz ok")
    g.add_argument("--domain", default="auto", choices=["auto", "prokaryote", "eukaryote"],
                   help="for raw contigs: gene caller (pyrodigal / MetaEuk); auto = probe a sample of genes")
    g.add_argument("--home-rank", default="auto", help="the Alien Index in-group: auto (phylum / kingdom), genus, "
                   "family, order, class, phylum, kingdom, domain")
    g.add_argument("--no-deep", dest="deep", action="store_false", help="skip the unmasked ultra-sensitive re-search")
    g.add_argument("--no-trees", dest="trees", action="store_false", help="skip the phylogenies of candidates")
    g.add_argument("--out", required=True, help="output folder")
    g.add_argument("--self-taxon", default="", help="the genome's own taxon (name or NCBI taxid); default: inferred")
    g.add_argument("--structure", action="store_true", help="also ESMFold + Foldseek for genes without outside homolog")
    g.add_argument("--max-structures", type=int, default=400)

    sv = sub.add_parser("serve", help="run one of the services (api / tools / esmfold) with uvicorn")
    sv.add_argument("service", choices=["api", "tools", "esmfold"])
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, help="default: api 8600, tools 8601, esmfold 8602")

    w = sub.add_parser("web", help="serve the web page locally (static files; the API address is set in the page)")
    w.add_argument("--port", type=int, default=8700)
    w.add_argument("--api", default="http://localhost:8600", help="API address the page starts with")
    w.add_argument("--no-open", action="store_true", help="do not open a browser")

    d = sub.add_parser("data", help="the data store: status / fetch / build / link (see data/references.yaml)")
    d.add_argument("action", choices=["status", "fetch", "build", "link"])
    d.add_argument("items", nargs="*", help="dataset names (default: all)")
    d.add_argument("--from", dest="src", help="link: existing folder with the files (see data/references.yaml)")
    d.add_argument("--threads", type=int, default=None)

    a = ap.parse_args(argv)
    if a.cmd == "genome":
        from . import clients, pipeline
        os.makedirs(a.out, exist_ok=True)
        params = dict(self_taxon=a.self_taxon, structure=a.structure, max_structures=a.max_structures,
                      domain=a.domain, home_rank=a.home_rank, deep=a.deep, trees=a.trees)
        summary = pipeline.run(a.out, [open(f, "rb").read() for f in a.input], params, clients.tools(),
                               clients.folder() if a.structure else None,
                               lambda stage, msg, frac: print("[%3.0f%%] %s" % (100 * frac, msg), file=sys.stderr))
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return
    if a.cmd == "serve":
        import uvicorn
        from .config import HOME
        sys.path.insert(0, HOME)
        port = a.port or {"api": 8600, "tools": 8601, "esmfold": 8602}[a.service]
        uvicorn.run("services.%s.app:app" % a.service, host=a.host, port=port, log_level="info")
        return
    if a.cmd == "web":
        from .webserve import serve
        serve(a.port, a.api, not a.no_open)
        return
    if a.cmd == "data":
        from . import data
        data.main(a)
        return
    if a.cmd == "scan":
        from .scan import run_scan
        run_scan(a)
    elif a.cmd == "prepare":
        from .prepare import run_prepare
        run_prepare(a)
    elif a.cmd == "donors":
        from .donors import run_donors
        run_donors(a)
    elif a.cmd == "structure":
        from .structure import run_structure
        run_structure(a)
    elif a.cmd == "simulate":
        from .simulate import simulate
        info = simulate(a.out, a.domain, a.species, a.families, a.seed, a.samples, a.noise)
        print(json.dumps(info, indent=2))
    elif a.cmd == "evaluate":
        from .evaluate import evaluate, print_report
        res = evaluate(a.events, a.truth, a.species_tree, a.genes, a.contaminants, a.threshold, a.oracle)
        print_report(res)
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(res, fh, indent=2)
    elif a.cmd == "calibrate":
        from .evaluate import calibrate
        if len(a.events) != len(a.truth):
            sys.exit("--events and --truth need the same number of files")
        calibrate(a.events, a.truth, a.out, a.l2)
    elif a.cmd == "rescore":
        from .evaluate import rescore
        rescore(a.events, a.model, a.out)


if __name__ == "__main__":
    main()
