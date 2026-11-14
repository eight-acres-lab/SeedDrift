"""Read whatever genome a user has: proteins, CDS, GenBank / EMBL, GFF3 (+ FASTA), or raw contigs.

Returns a Genome: proteins plus, when known, coordinates, exon counts and coding DNA (for composition).
Raw contigs need gene calling: pyrodigal for prokaryotes; for eukaryotes the caller passes a
`euk_caller` (MetaEuk through the tools service) because introns rule out a prokaryotic gene finder.
"""
from __future__ import annotations

import gzip
import io
import re
from dataclasses import dataclass, field

COMP = str.maketrans("ACGTRYKMBDHVNacgtrykmbdhvn", "TGCAYRMKVHDBNtgcayrmkvhdbn")


@dataclass
class Genome:
    kind: str                                    # protein / cds / genbank / gff / prokaryote-dna / eukaryote-dna
    proteins: dict = field(default_factory=dict)   # gene -> protein
    coords: dict = field(default_factory=dict)     # gene -> (contig, start, end, strand, n_exons)
    cds: dict = field(default_factory=dict)        # gene -> coding DNA
    contigs: dict = field(default_factory=dict)    # contig -> length
    notes: list = field(default_factory=list)


def revcomp(s):
    return s.translate(COMP)[::-1]


def decompress(data):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif data[:3] == b"BZh":
        import bz2
        data = bz2.decompress(data)
    elif data[:6] == b"\xfd7zXZ\x00":
        import lzma
        data = lzma.decompress(data)
    return data.decode(errors="replace")


def read_fasta_text(text):
    seqs, name, buf = {}, None, []
    for line in io.StringIO(text):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            toks = line[1:].split()
            name, buf = (toks[0] if toks else "seq%d" % (len(seqs) + 1)), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def detect(text):
    head = text.lstrip()[:5000]
    if head.startswith("LOCUS"):
        return "genbank"
    if head.startswith("ID   "):
        return "embl"
    if head.startswith("##gff-version") or re.search(r"^[^#>\n][^\t\n]*\t[^\t\n]*\t[^\t\n]*\t\d+\t\d+\t", head, re.M):
        return "gff"
    if head.startswith(">"):
        return "fasta"
    return "unknown"


def is_nucleotide(seqs):
    s = "".join(list(seqs.values())[:200])[:300000].upper()
    return bool(s) and sum(s.count(c) for c in "ACGTUN") >= 0.9 * len(s)


def translate(dna, table=11):
    from Bio.Seq import Seq
    dna = dna[: len(dna) // 3 * 3]
    return str(Seq(dna).translate(table=table)).rstrip("*")


def looks_like_cds(seqs):
    """Most sequences are single ORFs (length multiple of 3, start codon, no internal stop)."""
    sample = list(seqs.values())[:300]
    ok = 0
    for s in sample:
        s = s.upper()
        if len(s) >= 90 and len(s) % 3 == 0 and s[:3] in ("ATG", "GTG", "TTG", "CTG"):
            p = translate(s)
            ok += "*" not in p
    return sample and ok >= 0.8 * len(sample)


def gc_stats(dna):
    """(GC, GC3) of a coding sequence."""
    d = dna.upper()
    n = sum(d.count(c) for c in "ACGT")
    gc = (d.count("G") + d.count("C")) / n if n else 0.0
    t = d[2::3]
    n3 = sum(t.count(c) for c in "ACGT")
    gc3 = (t.count("G") + t.count("C")) / n3 if n3 else 0.0
    return gc, gc3


# -- annotated formats --------------------------------------------------------------------------------

def from_genbank(text, fmt="genbank"):
    from Bio import SeqIO
    g = Genome(kind="genbank")
    for rec in SeqIO.parse(io.StringIO(text), fmt):
        g.contigs[rec.id] = len(rec.seq)
        dna = str(rec.seq) if rec.seq is not None else ""
        for f in rec.features:
            if f.type != "CDS" or "pseudo" in f.qualifiers:
                continue
            q = f.qualifiers
            gid = (q.get("locus_tag") or q.get("protein_id") or q.get("gene") or ["%s_%d" % (rec.id, len(g.proteins) + 1)])[0]
            try:
                nt = str(f.location.extract(rec.seq)) if dna else ""
            except Exception:                   # noqa: BLE001 - undefined sequence in the record
                nt = ""
            prot = q.get("translation", [""])[0] or (translate(nt, int(q.get("transl_table", ["11"])[0])) if nt else "")
            if not prot:
                continue
            g.proteins[gid] = prot
            g.coords[gid] = (rec.id, int(f.location.start) + 1, int(f.location.end),
                             "+" if f.location.strand != -1 else "-", len(getattr(f.location, "parts", [f.location])))
            if nt:
                g.cds[gid] = nt
    return g


def _gff_attrs(s):
    out = {}
    for kv in s.strip().split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def from_gff(text, fasta_text=None):
    """GFF3 with CDS features; genome sequence from a ##FASTA section or a separate FASTA."""
    g = Genome(kind="gff")
    gff, _, embedded = text.partition("##FASTA")
    contigs = read_fasta_text(embedded or fasta_text or "")
    if not contigs:
        raise ValueError("GFF3 needs the genome sequence: a ##FASTA section or a second (FASTA) file")
    g.contigs = {k: len(v) for k, v in contigs.items()}
    parent_of, segs = {}, {}
    for line in gff.splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        a = _gff_attrs(f[8])
        if f[2] in ("mRNA", "transcript") and "ID" in a:
            parent_of[a["ID"]] = a.get("Parent", a["ID"]).split(",")[0]
        if f[2] != "CDS":
            continue
        key = (a.get("Parent") or a.get("ID") or "%s:%s" % (f[0], f[3])).split(",")[0]
        segs.setdefault(key, []).append((f[0], int(f[3]), int(f[4]), f[6], int(f[7]) if f[7].isdigit() else 0,
                                         a.get("protein_id") or a.get("Name") or a.get("ID") or key))
    best = {}
    multi = 0
    for tx, ss in segs.items():
        contig, strand = ss[0][0], ss[0][3]
        if contig not in contigs:
            continue
        ss.sort(key=lambda x: x[1], reverse=(strand == "-"))
        parts = [contigs[contig][s - 1:e] for _, s, e, _, _, _ in ss]
        nt = "".join(revcomp(p) if strand == "-" else p for p in parts)[ss[0][4]:]
        multi += len(ss) > 1
        gene = parent_of.get(tx, tx)
        prot = None
        if gene not in best or len(nt) > len(best[gene][1]):
            prot = (ss, nt)
        if prot:
            best[gene] = prot
    table = 1 if multi > 0.05 * max(1, len(best)) else 11
    for gene, (ss, nt) in best.items():
        prot = translate(nt, table)
        if len(prot) < 10:
            continue
        gid = re.sub(r"^(cds|gene|rna)[-:]", "", gene, flags=re.I)
        g.proteins[gid] = prot.replace("*", "X")
        g.cds[gid] = nt
        g.coords[gid] = (ss[0][0], min(s for _, s, _, _, _, _ in ss), max(e for _, _, e, _, _, _ in ss), ss[0][3], len(ss))
    if not g.proteins:
        raise ValueError("no CDS features found in the GFF3")
    return g


# -- raw contigs --------------------------------------------------------------------------------------

def call_prokaryote(contigs):
    import pyrodigal
    total = sum(len(s) for s in contigs.values())
    meta = total < 100_000
    gf = pyrodigal.GeneFinder(meta=meta)
    if not meta:
        gf.train(*[s.upper().encode() for s in contigs.values()])
    g = Genome(kind="prokaryote-dna", contigs={k: len(v) for k, v in contigs.items()})
    for cid, seq in contigs.items():
        up = seq.upper()
        for k, gene in enumerate(gf.find_genes(up.encode()), 1):
            gid = "%s_%d" % (cid, k)
            g.proteins[gid] = gene.translate().rstrip("*")
            nt = up[gene.begin - 1:gene.end]
            g.cds[gid] = revcomp(nt) if gene.strand == -1 else nt
            g.coords[gid] = (cid, gene.begin, gene.end, "+" if gene.strand == 1 else "-", 1)
    g.notes.append("基因由 pyrodigal 预测（%s 模式）" % ("宏基因组" if meta else "单基因组"))
    return g


def parse_metaeuk(faa_text, contigs):
    """MetaEuk easy-predict proteins: header TargetID|ContigID|strand|bits|evalue|n_exons|low|high|exons...
    exon field: low[taken_low]:high[taken_high]:len[taken_len]. Keeps the best prediction per locus."""
    g = Genome(kind="eukaryote-dna", contigs={k: len(v) for k, v in contigs.items()})
    best = {}
    for name, prot in read_fasta_text(faa_text).items():
        f = name.split("|")
        if len(f) < 8:
            continue
        contig, strand, bits, nex, low, high = f[1], f[2], float(f[3]), int(f[5]), int(f[6]), int(f[7])
        key = (contig, strand, low // 500)
        if key in best and best[key][0] >= bits:
            continue
        exons = []
        for e in f[8:]:
            m = re.match(r"(\d+)\[(\d+)\]:(\d+)\[(\d+)\]", e)
            if m:
                a, b = int(m.group(2)), int(m.group(4))
                exons.append((min(a, b), max(a, b)))
        best[key] = (bits, contig, strand, nex, low, high, exons, prot)
    for k, (bits, contig, strand, nex, low, high, exons, prot) in enumerate(
            sorted(best.values(), key=lambda x: (x[1], x[4])), 1):
        gid = "%s_g%d" % (contig, k)
        g.proteins[gid] = prot.replace("*", "")
        g.coords[gid] = (contig, low + 1, high + 1, strand, nex)
        seq = contigs.get(contig, "")
        if seq and exons:
            parts = [seq[a:b + 1] for a, b in sorted(exons)]
            nt = "".join(parts).upper()
            g.cds[gid] = revcomp(nt) if strand == "-" else nt
    g.notes.append("真核基因由 MetaEuk 按与 Swiss-Prot 的同源性预测：没有任何已知同源蛋白的基因不会被预测出来")
    return g


def load(files, domain="auto", euk_caller=None, domain_probe=None):
    """files: list of raw bytes (the main file first). domain: auto / prokaryote / eukaryote.
    euk_caller(contigs) -> MetaEuk protein FASTA text; domain_probe(proteins) -> 'eukaryote' / 'prokaryote'."""
    texts = [decompress(b) for b in files if b]
    if not texts:
        raise ValueError("no input")
    main, kind = texts[0], detect(texts[0])
    other = texts[1] if len(texts) > 1 else None
    if kind == "fasta" and other and detect(other) == "gff":     # FASTA + GFF in either order
        main, other, kind = other, main, "gff"
    if kind in ("genbank", "embl"):
        return from_genbank(main, kind)
    if kind == "gff":
        return from_gff(main, other)
    if kind != "fasta":
        raise ValueError("无法识别的格式：需要 FASTA、GenBank、EMBL 或 GFF3")
    seqs = read_fasta_text(main)
    if not seqs:
        raise ValueError("输入里没有序列")
    if not is_nucleotide(seqs):
        return Genome(kind="protein", proteins=seqs)
    if looks_like_cds(seqs):
        g = Genome(kind="cds")
        for k, v in seqs.items():
            g.proteins[k] = translate(v.upper())
            g.cds[k] = v.upper()
        return g
    if domain == "auto":
        g = call_prokaryote(seqs)
        domain = domain_probe(g.proteins) if domain_probe else "prokaryote"
        if domain != "eukaryote":
            g.notes.append("按抽样比对判断为原核基因组")
            return g
        g_note = "按抽样比对判断为真核基因组"
    else:
        g_note = None
    if domain == "eukaryote":
        if euk_caller is None:
            raise ValueError("真核基因组需要基因预测（MetaEuk），但没有可用的工具服务")
        g = parse_metaeuk(euk_caller(seqs), seqs)
        if g_note:
            g.notes.insert(0, g_note)
        return g
    return call_prokaryote(seqs)
