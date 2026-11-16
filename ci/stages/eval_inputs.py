"""Stage eval_inputs: the genomes the evaluation scores (never part of the training corpus), with their truth.

For ci-tiny and smoke this is the scenario of the fixture (a real chromosome segment with donor genes spliced in;
scenario_truth.tsv gives their coordinates). Genes are called the same way as for raw prokaryotes in the corpus; a
called gene is a positive when at least half of it lies in a truth interval. Later profiles add the public-event
catalogue here (gate G5).
"""
from __future__ import annotations

import csv
import os

from ci import manifest as M
from ci.stages.common import Stage, main_args, read_fasta, write_json
from ci.stages.corpus import call_genes
from ci.stages.common import write_fasta


def main():
    run_dir = main_args()
    with Stage(run_dir, "eval_inputs") as st:
        sc = st.config["scenario"]
        base = os.path.dirname(st.repo(sc["table"]))
        entries = []
        with open(st.repo(sc["table"])) as f:
            scen = list(csv.DictReader(f, delimiter="\t"))
        truth = []
        with open(st.repo(sc["truth"])) as f:
            truth = list(csv.DictReader(f, delimiter="\t"))
        for s in scen:
            contigs = {}
            for p in s["files"].split(";"):
                contigs.update(read_fasta(os.path.join(base, p)))
            genes, mode = call_genes(contigs, 11)
            out = st.path("eval_inputs", s["id"], "x")[:-2]
            iv = [(t["replicon"], int(t["start"]), int(t["end"]), t["class"]) for t in truth]
            with open(os.path.join(out, "genes.tsv"), "w") as f:
                f.write("gene_id\treplicon\tstart\tend\tstrand\tpartial\ttruth\n")
                for g in genes:
                    label = "native"
                    for rep, a, b, c in iv:
                        ov = min(b, g["end"]) - max(a, g["start"]) + 1
                        if rep == g["replicon"] and ov >= 0.5 * (g["end"] - g["start"] + 1):
                            label = c
                    f.write("%s\t%s\t%d\t%d\t%s\t%s\t%s\n" % (g["gene_id"], g["replicon"], g["start"], g["end"], g["strand"], g["partial"], label))
            write_fasta(os.path.join(out, "proteins.faa"), [("%s__%s" % (s["id"], g["gene_id"]), g["protein"]) for g in genes])
            write_fasta(os.path.join(out, "cds.fna"), [("%s__%s" % (s["id"], g["gene_id"]), g["cds"]) for g in genes])
            files = [M.file_entry(os.path.join(out, n), run_dir) for n in ("genes.tsv", "proteins.faa", "cds.fna")]
            entries.append(dict(id=s["id"], host=s["host"], host_taxid=int(s["host_taxid"]), proteins=len(genes), gene_calling=mode, files=files))
        man = M.build("eval_inputs", "eval_inputs", entries)
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("eval_inputs", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("eval_inputs_manifest", mp)


if __name__ == "__main__":
    main()
