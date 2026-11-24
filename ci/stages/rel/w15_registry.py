"""Stage rel_w15_registry (package W15; pipeline rel_events): the event registry checked on the server, without any
score. `python -m ci.stages.rel.w15_registry RUN_DIR`.

Up to seven things, all from the registry draft the profile names (bench/rel/registry/events_*.tsv) and none from any model:
1. the draft's own checks (bench/rel/registry/registry.py, the rules of RULES.md);
2. every exclusion taxon looked up in NCBI taxonomy (the run's taxdump): its taxid, rank and lineage, and the names of
   the draft's lineage that NCBI does not have;
3. whether genomes inside each exclusion radius are in the medium corpus. Only the train and held-out splits of the
   source's training-data index are read; the sealed test split is never looked at (its genomes are not opened, and
   their taxa are not counted per radius). Overlap with the sealed island genera (and their families) of
   bench/hard/EXCLUDE.tsv is reported from that list alone;
4. with `fetch`, the papers: Europe PMC metadata of every DOI in the draft; for an article in PMC's open-access
   dataset (PMC Article Datasets on AWS Open Data, the sanctioned route for automated retrieval) its full text and,
   for the DOIs the profile lists, its supplementary files (figures left out) from there; otherwise the open full
   text and supplements from Europe PMC. Each file with its URL, size and sha256. Papers are stored under the run's
   data folder, never in the repository. Articles outside the open subset are not retrieved automatically.
5. with `annotation` (a WGS master accession), which identifiers the drafts cite as ASEM gene IDs are locus tags of
   that assembly: E-utilities link the WGS record to its assembly, NCBI Datasets gives the annotation report page by
   page (gene records only, never sequence), every page stored with its sha256.
6. with `exposure_roots`, the exposure scan of bench/hard/V2_REGISTER.md, names only, never file contents: every file
   and folder under the roots is matched by its own name (as register.scan_names) against the assembly numbers and the
   species of the corpus genomes inside each radius (train and held-out splits), the island-set chromosomes inside it
   (with `exposure_islands`; never the sealed ones or the sealing candidates), the nucleotide accessions the record
   cites, and the record's exclusion taxon (genus, or binomials below a genus) and recipient binomials. A hit is L3, L4
   under a score folder, L5 under eval/ (register.path_level). Paths
   naming a genome outside the train and held-out splits (by assembly number, id or species words) are skipped and
   counted in total only; walk errors and examples are checked the same way; this run's own folder is not scanned.
7. with `pdf_text` (from_run, files), the text of PDFs an earlier registry run stored: that run must have passed with
   its registry gate passed, its report must be the output its stage recorded, and each file's bytes must match the
   sha256 that report records (a stored supplement archive is checked against its own sha256 before its members are
   listed); bench/rel/registry/pdf_text.py gives the readable pages' text and every other page's reason only. Texts go
   to data/rel_w15/registry/pdf_text/, never to the repository.

Profile block rel_w15_registry: registry (repository path of the draft, or a list of drafts read as one registry), source (the run directory of a passed selfsup
run, for its training-data index; optional), fetch (bool), supplements (DOIs whose supplements are fetched),
max_supplement_mb (default 50), pause_s (default 0.5), annotation (wgs_master, max_mb; optional), exposure_roots
(absolute folders, or {path, prune: [folders relative to it]}; needs source), exposure_islands (the island store, for
the island set's taxid cache; optional), exposure_max_paths (default 20 million) and
pdf_text (from_run, files: names or fnmatch patterns of stored PDFs).
Output data/rel_w15/registry/report.json and report.md; papers in data/rel_w15/registry/papers/.
Gate G4_rel_w15_registry: the draft breaks no rule, every role is unassigned, nothing sits at bench/catalogue/events.tsv,
the report's sha256 of the drafts matches the drafts, and with fetch on at least half the paper lookups returned a
parsed record (a wholesale failure fails the run instead of passing it with nothing fetched), and the exposure scan
found every root, walked each to the end with no error and found at least one path under each, and every PDF named and
every stored archive was there, unchanged and readable as a file (an unreadable PDF is a result, not a failure).
"""
from __future__ import annotations

import csv
import errno
import fnmatch
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict

from bench.rel.registry import registry as reg
from ci.gates import GateResult, require
from ci.hashing import sha256_file
from ci.stages.common import main_args, write_json
from ci.stages.rel import common

STAGE = "rel_w15_registry"
OUT = os.path.join("data", "rel_w15", "registry")
KEYS = {"registry", "source", "fetch", "supplements", "max_supplement_mb", "pause_s", "annotation", "exposure_roots",
        "exposure_max_paths", "exposure_islands", "pdf_text"}
ACC_RE = re.compile(r"(?<![a-z0-9])gc[af]_?(\d{9})(?!\d)", re.I)
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
ASEM_RE = re.compile(r"\bASEM_(?:C4|AUS1)_[0-9A-Za-z]+")
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
USER_AGENT = "literature-registry-fetch/0.2 (python urllib)"
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;\[\]]+")
RETRIES = (10, 30)           # seconds to wait before the second and third try of a lookup or a supplement
# what a retry may cure: connection errors and resets, a body cut short (http.client.IncompleteRead, run
# 20261122T175251), an error body; HTTPError is an URLError and is handled apart (5xx retried, 4xx not)
TRANSIENT = (urllib.error.URLError, http.client.HTTPException, OSError, ValueError)
CATALOGUE = os.path.join("bench", "catalogue", "events.tsv")
PRETRAIN_SPLITS = ("train", "heldout")
FETCH_MIN_SHARE = 0.5        # gate: with fetch on, at least half the lookups must return a parsed record


def settings(cfg) -> dict:
    c = dict(cfg.get(STAGE) or {})
    unknown = set(c) - KEYS
    if unknown or not c.get("registry"):
        raise SystemExit("%s: the profile names registry (and only %s): %s" % (STAGE, sorted(KEYS), sorted(unknown)))
    paths = [str(x) for x in (c["registry"] if isinstance(c["registry"], list) else [c["registry"]])]
    if any(os.path.normpath(x) == CATALOGUE for x in paths):
        raise SystemExit("%s: the draft never lives at %s (gate G2 reads that file)" % (STAGE, CATALOGUE))
    ann = c.get("annotation")
    if ann is not None and (not isinstance(ann, dict) or not ann.get("wgs_master")):
        raise SystemExit("%s: annotation needs wgs_master (and optionally max_mb)" % STAGE)
    pdf = c.get("pdf_text")
    if pdf is not None and (not isinstance(pdf, dict) or set(pdf) - {"from_run", "files"} or not isinstance(pdf.get("from_run"), str)
                            or not os.path.isabs(pdf["from_run"]) or not isinstance(pdf.get("files"), list) or not pdf["files"]
                            or not all(isinstance(x, str) and x for x in pdf["files"])):
        raise SystemExit("%s: pdf_text needs from_run (an absolute run directory) and files (names or patterns)" % STAGE)
    islands = c.get("exposure_islands")
    if islands is not None and not (isinstance(islands, str) and os.path.isabs(islands)):
        raise SystemExit("%s: exposure_islands is the island store's absolute path" % STAGE)
    roots = c.get("exposure_roots") or []
    ok = isinstance(roots, list)
    for x in roots if ok else []:
        spec = x if isinstance(x, dict) else dict(path=x)
        prune = spec.get("prune") or []
        ok = ok and set(spec) <= {"path", "prune"} and isinstance(spec.get("path"), str) and os.path.isabs(spec["path"]) \
            and isinstance(prune, list) and all(isinstance(q, str) and q and not os.path.isabs(q) and ".." not in q.split("/") for q in prune)
    if not ok:
        raise SystemExit("%s: exposure_roots is a list of absolute paths, or of {path, prune: [paths relative to it]}" % STAGE)
    return dict(registry=paths, source=c.get("source"), fetch=bool(c.get("fetch", False)), annotation=ann, exposure_roots=roots, exposure_islands=islands, pdf_text=pdf,
                exposure_max_paths=int(c.get("exposure_max_paths", 20_000_000)),
                supplements=[d.lower() for d in c.get("supplements") or []],
                max_supplement_mb=float(c.get("max_supplement_mb", 50)), pause_s=float(c.get("pause_s", 0.5)))


def dois(rows) -> list:
    """Every DOI named in the draft's papers and later columns, lower case, trailing punctuation removed."""
    out = []
    for r in rows:
        for col in ("papers", "later", "exclusion_basis", "notes"):
            for m in DOI_RE.finditer(r.get(col) or ""):
                d = m.group(0).rstrip(".,:")
                while d.endswith(")") and d.count(")") > d.count("("):   # a DOI inside brackets; keep 10.1016/S1473-3099(15)...
                    d = d[:-1].rstrip(".,:")
                d = d.lower()
                if d not in out:
                    out.append(d)
    return out


def _rank(r) -> str:
    m = re.search(r"\(([a-z ]+)\)\s*$", r["exclusion_taxon"])
    return m.group(1) if m else ""


def taxa(rows, tax) -> dict:
    """event_id -> the NCBI reading of the exclusion taxon: status ok / rank_differs / ambiguous / unresolved."""
    out = {}
    for r in rows:
        name, rank = reg._taxon(r), _rank(r)
        cands = list(tax.by_name.get(name, []))
        if rank and rank != "no rank":
            same = [t for t in cands if tax.rank.get(t) == rank]
        else:
            same = cands
        wanted = set(reg._lineage(r))
        stated = (r.get("recipient_exclusion_taxid") or "").strip()
        if stated and int(stated) in same:                       # a stated taxid settles homonyms (environmental samples)
            same = [int(stated)]
        scored = sorted(((len(wanted & {tax.name(x) for x in tax.lineage(t)}), t) for t in same), reverse=True)
        rec = dict(name=name, rank=rank, candidates=len(cands))
        if not cands:
            rec["status"] = "unresolved"
        elif not same:
            rec.update(status="rank_differs", ncbi_ranks=sorted({tax.rank.get(t, "?") for t in cands}))
            byline = sorted(((len(wanted & {tax.name(x) for x in tax.lineage(t)}), t) for t in cands), reverse=True)
            if len(byline) == 1 or byline[0][0] > byline[1][0]:     # one candidate fits the lineage: keep its taxid
                t = byline[0][1]
                rec.update(taxid=t, ncbi_rank=tax.rank.get(t), ncbi_lineage=[tax.name(x) for x in reversed(tax.lineage(t))])
        elif len(scored) > 1 and scored[0][0] == scored[1][0]:
            rec.update(status="ambiguous", taxids=[t for _, t in scored])
        else:
            t = scored[0][1]
            lin = [tax.name(x) for x in reversed(tax.lineage(t))]
            rec.update(status="ok", taxid=t, ncbi_rank=tax.rank.get(t), ncbi_lineage=lin,
                       names_not_in_ncbi=[x for x in reg._lineage(r) if x not in lin])
            if stated and int(stated) != t:
                rec["stated_taxid_differs"] = int(stated)
        out[r["event_id"]] = rec
    return out


def pretraining(found, index, tax) -> dict:
    """event_id -> genomes inside the radius per split of the training-data index, train and held-out splits only."""
    genomes = [(m["split"], int(m["taxid"])) for m in index["genomes"] if m.get("split") in PRETRAIN_SPLITS]
    lineages = []
    for split, t in genomes:
        try:
            lineages.append((split, set(tax.lineage(t))))
        except Exception:  # noqa: BLE001 - a genome whose taxid does not resolve is counted apart
            lineages.append((split, None))
    out = {}
    for eid, rec in found.items():
        if not rec.get("taxid"):
            continue
        counts = {s: 0 for s in PRETRAIN_SPLITS}
        for split, lin in lineages:
            if lin and rec["taxid"] in lin:
                counts[split] += 1
        out[eid] = dict(counts, in_pretraining="y" if counts["train"] else "n",
                        unresolved_genomes=sum(1 for _, lin in lineages if lin is None))
    return out


def _acc_keys(text) -> set:
    """Assembly numbers named in text: GCA_000151295.1, GCF_000151295, gca_000151295 and gca000764305v4rs all give
    000151295 / 000764305 (a GCA and its GCF copy share the number, so either spelling counts)."""
    return {m.group(1) for m in ACC_RE.finditer(text)}


def _words(text) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


NOT_EPITHET = {"sp", "spp", "and", "or", "of", "the", "with", "group", "x", "cf", "aff", "nov", "complex", "clade", "strain",
               "strains", "isolate", "isolates", "species", "genomes", "genome", "lines", "line", "lineage", "lineages",
               "relatives", "others", "several", "many", "five"}


LATIN_PAIR = re.compile(r"\b([A-Z][a-z]+)\s+([a-z]{3,})\b")
UNNAMED = re.compile(r"\b[A-Z][a-z]+\s+spp?\.")


def _latin(text, single=False) -> list:
    """The Latin names in a free-text name, lower case and joined by '_' (the reviewers' rule, 2026-11-24): every pair
    'Genus epithet' ([A-Z][a-z]+ [a-z]{3,}) in every clause, inside brackets too, stop words aside; 'Candidatus' (or
    'Ca.') is a prefix and the two words after it are the name, never 'candidatus' alone. A species-level name without
    an epithet ('Pseudomonas sp.', 'Chaetothyriales sp. CBS 135597') gives nothing, never its genus or order (the
    assembly numbers cover those genomes). A lone word only when single is true: the name is a genus in NCBI."""
    text = re.sub(r"\b(?:Candidatus|Ca\.)\s+", "", text or "")
    out = [("%s_%s" % (m.group(1).lower(), m.group(2))) for m in LATIN_PAIR.finditer(text) if m.group(2) not in NOT_EPITHET]
    if not out and single and not UNNAMED.search(text):
        m = re.fullmatch(r"\s*([A-Z][a-z]+)\s*", text)
        if m:
            out.append(m.group(1).lower())
    return list(dict.fromkeys(out))


# nucleotide accessions (GenBank, RefSeq, WGS) as the drafts cite them; matched as register.scan_names does
NUC_RE = re.compile(r"(?<![A-Za-z0-9])((?:AC|NC|NG|NT|NW|NZ)_(?:[A-Z]{2})?\d{6,9}|[A-Z]\d{5}|[A-Z]{2}\d{6}(?:\d{2})?"
                    r"|[A-Z]{4}\d{8,10}|[A-Z]{6}\d{9,11})(?:\.\d+)?(?![0-9])")
REFERENCE_SET = os.path.join("bench", "hard", "results", "v2_reference_set.tsv")


def _nucleotide(text) -> set:
    return {m.group(1) for m in NUC_RE.finditer(text or "")}


def island_taxa(ref_path, store) -> dict:
    """accession -> taxid of the island-set chromosomes that may be tokens: the rows of bench/hard/results/
    v2_reference_set.tsv that are neither sealed (part sealed_*) nor sealing candidates (status candidate), with taxids
    from the store's taxid cache (panel/island_set_taxa.json). Sealed and candidate rows are left out unused: their
    exposure is the V2 register's (bench/hard/V2_REGISTER.md)."""
    with open(ref_path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    keep = [r["accession"] for r in rows if not r["part"].startswith("sealed") and r["status"] not in ("candidate",)
            and not r["status"].startswith("sealed")]
    taxa_path = os.path.join(store, "panel", "island_set_taxa.json")
    with open(taxa_path) as f:
        cache = json.load(f)
    taxids = {a: int(cache[a]) for a in keep if a in cache}
    return dict(taxids=taxids, used=len(taxids), without_taxid=sorted(a for a in keep if a not in cache),
                left_out=len(rows) - len(keep), reference_sha256=sha256_file(ref_path, use_sidecar=False),
                taxa_sha256=sha256_file(taxa_path, use_sidecar=False))


def radius_genomes(found, index, tax) -> dict:
    """event_id -> [(id, taxid)] of the train and held-out genomes inside the radius (the sealed test split is never read)."""
    lin = {}
    for m in index["genomes"]:
        if m.get("split") in PRETRAIN_SPLITS:
            try:
                lin[m["id"]] = (int(m["taxid"]), set(tax.lineage(int(m["taxid"]))))
            except Exception:  # noqa: BLE001 - unresolved genomes cannot be placed
                continue
    return {eid: sorted((g, t) for g, (t, l) in lin.items() if rec["taxid"] in l) for eid, rec in found.items() if rec.get("taxid")}


EUK_PREFIXES = ("plants_algae", "other_vertebrates", "invertebrates", "mammals", "fungi", "protists")


def _species_words(gid, group=None) -> tuple:
    """The species words of a corpus id not named by its assembly (Ensembl genomes): the id without its group prefix,
    first two words (plants_algae_nicotiana_attenuata -> nicotiana attenuata; fungi_allomyces_macrogynus_atcc_38327_gca_
    000151295 -> allomyces macrogynus). Paths in the store, the old workspace or Ensembl drop the prefix and the strain."""
    low = gid.lower()
    prefixes = ([str(group).lower().split(".")[-1]] if group else []) + list(EUK_PREFIXES)
    for p in prefixes:
        if low.startswith(p + "_"):
            w = _words(low[len(p) + 1:])
            return tuple(w[:2]) if len(w) >= 2 else ()
    return ()


def unseen_genomes(index) -> dict:
    """The genomes of the index outside the train and held-out splits, by name only: their ids, the assembly numbers in
    those ids and, for ids not named by assembly, their species words (prefix and strain dropped). No taxon of theirs is
    looked up and nothing is counted per radius; a path naming one is skipped by the scan."""
    ms = [m for m in index["genomes"] if m.get("split") not in PRETRAIN_SPLITS]
    by_name = [m for m in ms if not re.search(r"GC[AF]_\d{9}\.\d", m["id"])]        # the others: by assembly number
    seqs = {tuple(_words(m["id"])) for m in by_name} | {_species_words(m["id"], m.get("group")) for m in by_name}
    return dict(ids={m["id"].lower() for m in ms}, accs=set().union(*[_acc_keys(m["id"]) for m in ms]) if ms else set(),
                seqs={s for s in seqs if s})


class Unseen:
    """Does a text name a genome outside the train and held-out splits (an assembly number, or the word sequence of an
    id or of its species, anywhere)?"""

    def __init__(self, unseen):
        unseen = unseen or {}
        self.accs = set(unseen.get("accs") or ())
        self.by_first = defaultdict(set)
        for seq in unseen.get("seqs") or ():
            self.by_first[seq[0]].add(seq)

    def __call__(self, text) -> bool:
        if self.accs and _acc_keys(text) & self.accs:
            return True
        if self.by_first:
            w = _words(text)
            for k, x in enumerate(w):
                for seq in self.by_first.get(x, ()):
                    if tuple(w[k:k + len(seq)]) == seq:
                        return True
        return False


def exposure_tokens(rows, found, inside, tax, islands=None) -> dict:
    """event_id -> dict(genomes, accessions, nucleotide, names): what the scan looks for.
    - of every corpus genome inside the radius (train and held-out): its assembly number, and its NCBI species when the
      corpus does not name it by assembly (Ensembl genomes);
    - of the island set (island_taxa: neither sealed nor candidates): the chromosomes inside the radius;
    - of the record: the nucleotide accessions it cites in any column (EV012's JN900499), the exclusion taxon's name when
      NCBI ranks it genus (one word allowed) or any rank below a genus (binomials only), and the binomials of the
      recipient column."""
    lineage = {}

    def lin(t):
        if t not in lineage:
            try:
                lineage[t] = set(tax.lineage(t))
            except Exception:  # noqa: BLE001 - an island taxid NCBI does not resolve is left out
                lineage[t] = set()
        return lineage[t]
    out = {}
    for r in rows:
        eid = r["event_id"]
        accs, names, nuc = set(), set(), set()
        genomes = inside.get(eid, [])
        for gid, t in genomes:
            accs |= _acc_keys(gid)
            sp = tax.at_rank(t, "species")
            if sp and not re.search(r"GC[AF]_\d{9}\.\d", gid):
                names.update(_latin(tax.name(sp)))
        rec = found.get(eid) or {}
        t = rec.get("taxid")
        if t:
            genus = tax.at_rank(t, "genus")
            if rec.get("ncbi_rank") == "genus":
                names.update(_latin(reg._taxon(r), single=True))
            elif genus and genus != t:                   # subgenus, species group, species, varietas, isolate, ...
                names.update(_latin(reg._taxon(r)))
            for acc, it in (islands or {}).items():
                if t in lin(it):
                    nuc.add(acc.split(".")[0])
        names.update(_latin(r.get("recipient") or ""))
        for col, value in r.items():
            if col != "event_id" and value:
                nuc |= _nucleotide(value)
        out[eid] = dict(genomes=len(genomes), accessions=sorted(accs), nucleotide=sorted(nuc), names=sorted(n for n in names if n))
    return out


def _root_spec(x) -> dict:
    return dict(path=x, prune=[]) if isinstance(x, str) else dict(path=x["path"], prune=list(x.get("prune") or []))


def exposure_scan(tokens, roots, unseen=None, skip=(), max_paths=20_000_000) -> dict:
    """bench/hard/V2_REGISTER.md, Scans: names only, never file contents. Every file and folder under each root is
    matched by its own name, as register.scan_names does, against each event's assembly numbers and names (one word or
    two consecutive words of the name, lower case); its level comes from its path below the root (register.path_level:
    L3, L4 under a score folder, L5 under eval/). Left out: the folders a root prunes (paths relative to it), the paths
    in skip, and every path naming a genome outside the train and held-out splits (folders of such a genome are not
    entered; counted in total only). Walk errors are kept as errno names and relative paths, those naming such a genome
    counted only; every example and error is checked once more and dropped (counted) if it names one."""
    from ci.stages.register import path_level
    names_unseen = Unseen(unseen)
    acc_ev, name_ev, nuc_ev = defaultdict(set), defaultdict(set), defaultdict(set)
    for eid, t in tokens.items():
        for a in t["accessions"]:
            acc_ev[a].add(eid)
        for n in t["names"]:
            name_ev[n].add(eid)
        for a in t.get("nucleotide") or ():
            nuc_ev[a.split(".")[0]].add(eid)
    nuc_rx = re.compile(r"(?<![A-Za-z0-9])(%s)(?:\.\d+)?(?![0-9])" % "|".join(sorted(map(re.escape, nuc_ev)))) if nuc_ev else None
    hits, per_root, errors = {}, [], []
    walked = skipped = errors_unseen = dropped = 0
    stopped = False
    specs = [_root_spec(x) for x in roots]
    missing = [r["path"] for r in specs if not os.path.isdir(r["path"])]
    for spec in specs:
        root = spec["path"]
        rec = dict(path=root, walked=0, pruned=[], skipped_unseen=0, errors=0)
        per_root.append(rec)
        if root in missing:
            continue
        rec["prune_missing"] = [p for p in spec["prune"] if not os.path.exists(os.path.join(root, p))]
        prune = {os.path.normpath(os.path.join(root, p)) for p in spec["prune"]}
        digest = hashlib.sha256()

        def onerror(err, root=root, rec=rec):
            nonlocal errors_unseen
            rel = os.path.relpath(err.filename, root) if getattr(err, "filename", None) else "?"
            rec["errors"] += 1
            if names_unseen(rel):
                errors_unseen += 1
            else:
                errors.append(dict(root=root, path=rel, errno=errno.errorcode.get(err.errno, type(err).__name__)))
        for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
            keep = []
            for d in sorted(dirnames):
                full = os.path.join(dirpath, d)
                if os.path.normpath(full) in prune:
                    rec["pruned"].append(os.path.relpath(full, root))
                elif full not in skip:
                    keep.append(d)
            dirnames[:] = keep
            for name in dirnames + sorted(filenames):
                walked += 1
                rec["walked"] += 1
                if walked > max_paths:
                    stopped = True
                    break
                rel = os.path.relpath(os.path.join(dirpath, name), root)
                digest.update(rel.encode("utf-8", "surrogateescape") + b"\n")
                if names_unseen(rel):
                    skipped += 1
                    rec["skipped_unseen"] += 1
                    if name in dirnames:
                        dirnames.remove(name)                  # a test genome's folder is not entered
                    continue
                low = name.lower()
                matched = defaultdict(set)
                for a in _acc_keys(low):
                    for eid in acc_ev.get(a, ()):
                        matched[eid].add(a)
                for m in nuc_rx.finditer(name) if nuc_rx else ():
                    for eid in nuc_ev[m.group(1)]:
                        matched[eid].add(m.group(1))
                words = re.findall(r"[a-z]+", low)
                for i, w in enumerate(words):
                    for key in (w, w + "_" + words[i + 1] if i + 1 < len(words) else None):
                        for eid in name_ev.get(key, ()):
                            matched[eid].add(key)
                if not matched:
                    continue
                level = path_level(rel)
                for eid, toks in matched.items():
                    h = hits.setdefault(eid, dict(per_level=Counter(), examples={}, matched=set()))
                    h["per_level"][level] += 1
                    h["matched"] |= toks
                    ex = h["examples"].setdefault(level, [])
                    if len(ex) < 3:
                        ex.append(os.path.join(os.path.basename(root), rel))
            if stopped:
                break
        rec["digest"] = digest.hexdigest()
        if stopped:
            break
    for h in hits.values():                              # the last check: no output names a test genome
        for level, ex in h["examples"].items():
            kept = [p for p in ex if not names_unseen(p)]
            dropped += len(ex) - len(kept)
            h["examples"][level] = kept
    kept_errors = [e for e in errors if not names_unseen(e["path"])]
    dropped += len(errors) - len(kept_errors)
    events = {}
    for eid, t in sorted(tokens.items()):
        h = hits.get(eid)
        if not (t["accessions"] or t["names"] or t.get("nucleotide")):
            events[eid] = dict(level="no_tokens")
        elif not h:
            events[eid] = dict(level="no_hit")
        else:
            events[eid] = dict(level=max(h["per_level"]), paths=dict(sorted(h["per_level"].items())), examples=h["examples"],
                               matched=sorted(h["matched"])[:20])
    return dict(roots=per_root, missing_roots=missing, paths_walked=walked, stopped_at_cap=stopped, skipped_unseen=skipped,
                walk_errors=kept_errors[:50], walk_error_count=sum(r["errors"] for r in per_root), walk_errors_unseen=errors_unseen,
                dropped_outputs=dropped, tokens=tokens, events=events)


LEVEL_ORDER = ("L0", "L1", "L2", "L3", "L4", "L5")


def exposure_levels(rows, scan) -> dict:
    """event_id -> the draft's declared level, the scan's level (L3-L5, no_hit or no_tokens) and the higher of the two;
    a declared value that is no level is kept as it stands and flagged, never raised."""
    out = {}
    for r in rows:
        declared = (r.get("exposure") or "?").strip()
        found = ((scan or {}).get("events") or {}).get(r["event_id"], {}).get("level", "no_tokens")
        rec = dict(declared=declared, scan=found, level=declared)
        known = declared in LEVEL_ORDER
        if not known and declared != "?":
            rec["declared_unknown"] = True
        if found in LEVEL_ORDER:
            if not known:
                rec["level"] = found
            elif LEVEL_ORDER.index(found) > LEVEL_ORDER.index(declared):
                rec.update(level=found, above_declared=True)
        out[r["event_id"]] = rec
    return out


def sealed_genera(tax, exclude_path) -> dict:
    """Taxid -> reason, for the sealed island genera of EXCLUDE.tsv and their families."""
    out = {}
    with open(exclude_path) as f:
        rows = csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t")
        for r in rows:
            if r["kind"] != "genus" or not r["reason"].startswith("sealed genus"):
                continue
            for g in tax.genus_by_name(r["name"]):
                out[g] = "sealed genus %s" % r["name"]
                fam = tax.at_rank(g, "family")
                if fam:
                    out.setdefault(fam, "family of sealed genus %s" % r["name"])
    return out


def sealed_overlap(found, sealed, tax) -> dict:
    """event_id -> y when the radius contains a sealed genus or family, or lies inside one."""
    out = {}
    for eid, rec in found.items():
        if not rec.get("taxid"):
            continue
        t = rec["taxid"]
        hits = [why for s, why in sealed.items() if s in set(tax.lineage(t)) or t in set(tax.lineage(s))]
        out[eid] = dict(sealed_overlap="y" if hits else "n", reasons=hits)
    return out


def _get(url, opener, max_bytes=None):
    """(status, body). read() with no argument for the whole body: read(-1) on a chunked response returns the raw
    chunk framing (run 20261122T151028 stored 79 metadata files that way)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(req, timeout=60) as resp:
        data = resp.read(max_bytes + 1) if max_bytes else resp.read()
        return resp.status, data


def _store(dest, name, data):
    os.makedirs(dest, exist_ok=True)
    p = os.path.join(dest, name)
    with open(p, "wb") as f:
        f.write(data)
    return dict(file=name, bytes=len(data), sha256=sha256_file(p, use_sidecar=False))


def _tries(fn, waits, sleep=time.sleep):
    """fn() with retries after transient failures (connection errors, 5xx, an error body); the last error is raised."""
    for i, wait in enumerate((0,) + tuple(waits)):
        if wait:
            sleep(wait)
        try:
            return fn()
        except urllib.error.HTTPError as err:
            if err.code < 500 or i == len(waits):
                raise
        except TRANSIENT:
            if i == len(waits):
                raise


def _failure(err, url=None) -> dict:
    """A part that could not be fetched: the error's type, the last HTTP status if there was one, and its message."""
    out = dict(error_type=type(err).__name__, status=getattr(err, "code", None), error=str(err)[:200])
    if url:
        out["url"] = url
    return out


def failures(papers) -> dict:
    """Per part (meta, fulltext, supplements, paper): counts of the error types of what could not be fetched."""
    out = {}
    for p in papers:
        for part in ("meta", "fulltext", "supplements"):
            x = p.get(part)
            if isinstance(x, dict) and x.get("error_type"):
                key = x["error_type"] + ("" if x.get("status") is None else " %s" % x["status"])
                out.setdefault(part, {}).setdefault(key, 0)
                out[part][key] += 1
        if p.get("error_type"):
            out.setdefault("paper", {}).setdefault(p["error_type"], 0)
            out["paper"][p["error_type"]] += 1
    return out


def _supplement(url, opener, max_bytes):
    status, data = _get(url, opener, max_bytes)
    if not data.startswith(b"PK"):                                 # Europe PMC answers 200 with an XML errorBean when its back end fails
        raise ValueError("not a zip archive: %r" % data[:160])
    return status, data


def _fulltext(url, opener):
    status, data = _get(url, opener)
    if b"errorBean" in data[:400] or not data.lstrip().startswith(b"<"):
        raise ValueError("not an article: %r" % data[:160])
    return status, data


S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"     # PMC Article Datasets on AWS Open Data: the open-access subset
FIGURE_EXT = (".jpg", ".jpeg", ".gif", ".png", ".tif", ".tiff")


def s3_listing(pmcid, opener, waits=RETRIES, sleep=time.sleep) -> list:
    """(key, size) of every file the open-access dataset holds for this article, all versions; empty when none."""
    url = S3 + "?" + urllib.parse.urlencode({"list-type": "2", "prefix": pmcid + "."})
    status, data = _tries(lambda: _get(url, opener), waits, sleep)
    root = ET.fromstring(data)
    ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda tag: "s:" + tag) if ns else (lambda tag: tag)
    out = []
    for c in root.findall(q("Contents"), ns):
        out.append((c.findtext(q("Key"), "", ns), int(c.findtext(q("Size"), "0", ns) or 0)))
    return out


def s3_plan(pmcid, keys) -> dict:
    """The files of the latest version: the article (xml, txt, json) and its supplementary files (figures left out)."""
    versions = {}
    for key, size in keys:
        head = key.split("/", 1)[0]
        m = re.fullmatch(re.escape(pmcid) + r"\.(\d+)", head)
        if m:
            versions.setdefault(int(m.group(1)), []).append((key, size))
    if not versions:
        return {}
    v = max(versions)
    stem = "%s.%d/%s.%d" % (pmcid, v, pmcid, v)
    files = versions[v]
    article = {ext: (k, n) for k, n in files for ext in (".xml", ".txt", ".json") if k == stem + ext}
    supp = [(k, n) for k, n in files if not k.startswith(stem + ".") and not k.lower().endswith(FIGURE_EXT)]
    return dict(version=v, article=article, supplements=supp)


def _s3_file(key, dest, opener, waits, sleep, cap_bytes=None) -> dict:
    url = S3 + urllib.parse.quote(key)
    try:
        status, data = _tries(lambda: _get(url, opener, cap_bytes), waits, sleep)
        if cap_bytes and len(data) > cap_bytes:
            return dict(url=url, error_type="TooLarge", status=status, error="larger than the cap, not stored")
        return dict(_store(dest, os.path.basename(key), data), url=url, status=status)
    except TRANSIENT as err:
        return _failure(err, url)


def fetch(doi, dest, opener=urllib.request.urlopen, supplements=False, max_supplement_mb=50.0, pause_s=0.5,
          waits=RETRIES, sleep=time.sleep) -> dict:
    """Europe PMC metadata for one DOI. For an article in the open-access dataset (PMC Article Datasets on AWS Open
    Data, the sanctioned route for automated retrieval), its full text and, when asked, its supplementary files from
    there; otherwise the open full text and supplements from Europe PMC."""
    rec = dict(doi=doi, when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    url = EPMC + "search?" + urllib.parse.urlencode(dict(query='DOI:"%s"' % doi, resultType="core", format="json"))
    try:
        status, data = _tries(lambda: _get(url, opener), waits, sleep)
        rec["meta"] = dict(_store(dest, "meta.json", data), url=url, status=status)
        hits = (json.loads(data, strict=False).get("resultList") or {}).get("result") or []
        hit = hits[0] if hits else {}
        rec.update(found=bool(hits), pmcid=hit.get("pmcid"), open_access=hit.get("isOpenAccess"), pmid=hit.get("pmid"),
                   title=hit.get("title"), year=hit.get("pubYear"))
    except TRANSIENT as err:
        rec["meta"] = _failure(err, url)
        return rec
    if not rec.get("pmcid"):
        return rec
    cap = int(max_supplement_mb * 1e6)
    plan = {}
    try:
        sleep(pause_s)
        plan = s3_plan(rec["pmcid"], s3_listing(rec["pmcid"], opener, waits, sleep))
    except (TRANSIENT + (ET.ParseError,)) as err:
        rec["pmc_oa"] = _failure(err, S3)
    if plan:
        rec["pmc_oa"] = dict(version=plan["version"], files=len(plan["article"]) + len(plan["supplements"]))
        oa = os.path.join(dest, "pmc_oa")
        for ext, (key, _) in sorted(plan["article"].items()):
            got = _s3_file(key, oa, opener, waits, sleep)
            if ext == ".xml":
                rec["fulltext"] = dict(got, source="pmc-oa-opendata")
            else:
                rec.setdefault("article_files", []).append(got)
        if supplements and plan["supplements"]:
            files = [_s3_file(k, oa, opener, waits, sleep, cap) for k, n in plan["supplements"]]
            stored = [f for f in files if f.get("sha256")]
            rec["supplements"] = dict(source="pmc-oa-opendata", files=files, bytes=sum(f["bytes"] for f in stored))
            if stored:
                rec["supplements"]["sha256"] = hashlib.sha256("".join(sorted(f["sha256"] for f in stored)).encode()).hexdigest()
            else:
                rec["supplements"].update(error_type="NoneStored", status=None, error="no supplementary file stored")
        elif supplements:                      # the dataset holds no supplementary file (figures aside): try Europe PMC
            rec["pmc_oa"]["supplements"] = "none in the open-access dataset"
        if (rec.get("fulltext") or {}).get("sha256") and (not supplements or (rec.get("supplements") or {}).get("sha256")):
            return rec
    for what, path, cap_mb in (("fulltext", "fullTextXML", None), ("supplements", "supplementaryFiles", max_supplement_mb)):
        if (what == "supplements" and not supplements) or (rec.get(what) or {}).get("sha256"):
            continue
        sleep(pause_s)
        url = EPMC + "%s/%s" % (rec["pmcid"], path)
        max_bytes = int(cap_mb * 1e6) if cap_mb else None
        try:
            if what == "supplements":
                status, data = _tries(lambda: _supplement(url, opener, max_bytes), waits, sleep)
            else:
                status, data = _tries(lambda: _fulltext(url, opener), waits, sleep)
            if max_bytes and len(data) > max_bytes:
                rec[what] = dict(url=url, error_type="TooLarge", status=status, error="larger than %.0f MB, not stored" % cap_mb)
            else:
                rec[what] = dict(_store(dest, "fulltext.xml" if what == "fulltext" else "supplements.zip", data), url=url,
                                 status=status, source="europepmc")
        except urllib.error.HTTPError as err:
            rec[what] = _failure(err, url)
            if what == "fulltext" and err.code == 500:
                rec[what]["note"] = ("HTTP 500 after retries, and the article is not in the open-access dataset: Europe PMC's "
                                     "back end failed, or the article is outside the open subset (PMC's terms bar automated "
                                     "bulk download from its own site)")
        except TRANSIENT as err:
            rec[what] = _failure(err, url)
    return rec


def fetch_all(doi_list, base, s, fetch_one=None, sleep=time.sleep) -> list:
    """fetch() for every DOI; one paper's unexpected failure is recorded and never stops the stage (run 20261122T175251
    stopped on an IncompleteRead in one supplement)."""
    fetch_one = fetch_one or fetch
    out = []
    for d in doi_list:
        try:
            out.append(fetch_one(d, os.path.join(base, _folder(d)), supplements=d in s["supplements"],
                                 max_supplement_mb=s["max_supplement_mb"], pause_s=s["pause_s"]))
        except Exception as err:  # noqa: BLE001 - recorded as not fetched; the other papers go on
            out.append(dict(_failure(err), doi=d, fetched=False))
        sleep(s["pause_s"])
    return out


def registry_ids(rows) -> list:
    """The ASEM gene identifiers the drafts cite (region, claim, notes), as written."""
    out = set()
    for r in rows:
        for col in ("region", "claim", "notes"):
            out.update(ASEM_RE.findall(r.get(col) or ""))
    return sorted(out)


def _json_get(url, opener, waits, sleep):
    status, data = _tries(lambda: _get(url, opener), waits, sleep)
    return json.loads(data, strict=False), data


def _walk(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                yield v
            else:
                yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def annotation_lookup(spec, rows, dest, opener=urllib.request.urlopen, waits=RETRIES, sleep=time.sleep, pause_s=0.5) -> dict:
    """Which ASEM identifiers of the drafts are locus tags of the assembly behind a WGS master record: E-utilities find
    the assembly (nuccore -> assembly link -> summary), NCBI Datasets gives its annotation report page by page (gene
    records only, no sequence). Every page is stored with its sha256."""
    master = spec["wgs_master"]
    cap = int(float(spec.get("max_mb", 150)) * 1e6)
    rec = dict(wgs_master=master)
    q = lambda base, **kw: base + "?" + urllib.parse.urlencode(dict(kw, retmode="json", tool="literature-registry-fetch"))
    try:
        es, _ = _json_get(q(EUTILS + "esearch.fcgi", db="nuccore", term=master), opener, waits, sleep)
        nuc = (es.get("esearchresult") or {}).get("idlist") or []
        sleep(pause_s)
        el, _ = _json_get(q(EUTILS + "elink.fcgi", dbfrom="nuccore", db="assembly", id=",".join(nuc[:1])), opener, waits, sleep) if nuc else ({}, b"")
        links = [l for ls in el.get("linksets") or [] for db in ls.get("linksetdbs") or [] if db.get("dbto") == "assembly" for l in db.get("links") or []]
        sleep(pause_s)
        su, _ = _json_get(q(EUTILS + "esummary.fcgi", db="assembly", id=",".join(map(str, links[:1]))), opener, waits, sleep) if links else ({}, b"")
        res = su.get("result") or {}
        doc = res.get(str(links[0])) if links else None
        rec["assembly"] = dict(nuccore_uid=nuc[:1], assembly_uid=links[:1], accession=(doc or {}).get("assemblyaccession"),
                               name=(doc or {}).get("assemblyname"), properties=(doc or {}).get("propertylist"))
    except TRANSIENT as err:
        rec.update(status="error", error=_failure(err))
        return rec
    acc = rec["assembly"]["accession"]
    if not acc:
        rec["status"] = "no assembly found for the WGS master record"
        return rec
    tags, genes, pages, total, token = set(), 0, [], 0, None
    while True:
        url = DATASETS + "%s/annotation_report?%s" % (acc, urllib.parse.urlencode(dict(page_size=1000, **({"page_token": token} if token else {}))))
        try:
            sleep(pause_s)
            page, data = _json_get(url, opener, waits, sleep)
        except TRANSIENT as err:
            rec.update(status="error", error=_failure(err, url), pages=pages)
            return rec
        pages.append(dict(_store(dest, "annotation_%03d.json" % len(pages), data), url=url))
        total += len(data)
        reports = page.get("reports") or []
        genes += len(reports)
        tags.update(_walk(reports, "locus_tag"))
        token = page.get("next_page_token")
        if not token or not reports or total > cap:
            break
    ids = registry_ids(rows)
    prefixes = Counter(t.rsplit("_", 1)[0] for t in tags)
    rec.update(status="annotated" if tags else ("no locus tags in the annotation report" if genes else "no annotation report"),
               genes=genes, locus_tags=len(tags), locus_tag_prefixes=prefixes.most_common(10), pages=pages,
               truncated=bool(token) and total > cap, ids=len(ids), matched=sorted(i for i in ids if i in tags),
               unmatched=sorted(i for i in ids if i not in tags))
    return rec


def _folder(doi):
    return re.sub(r"[^a-z0-9._-]+", "_", doi.lower())


SOURCE_GATE = "G4_rel_w15_registry"
ZIP_ERRORS = (OSError, KeyError, ValueError, EOFError, RuntimeError, NotImplementedError, zipfile.BadZipFile, zlib.error)


def stored_files(run):
    """(name -> where a passed registry run stored it, problems). The run must have passed with its registry gate
    passed (record.json), and its report must be the output its stage recorded. Files are the supplement and article
    files of the report with the sha256 it records; a stored supplement archive is first checked against its own
    sha256 and then gives its members. An archive missing, changed or not opened is a problem, never skipped."""
    from ci.stages.rel import w2_v0
    rec_path = os.path.join(run, "record.json")
    if not os.path.exists(rec_path):
        raise SystemExit("%s: %s holds no run record" % (STAGE, run))
    with open(rec_path) as f:
        rec = json.load(f)
    gate_ = (rec.get("gates") or {}).get(SOURCE_GATE)
    if rec.get("status") != "passed" or gate_ != "pass":
        raise SystemExit("%s: the source run %s is %s with %s %s; only a passed run whose registry gate passed is read" % (
            STAGE, os.path.basename(run), rec.get("status"), SOURCE_GATE, gate_))
    rp = os.path.join(run, OUT, "report.json")
    w2_v0._output(run, STAGE, "report", rp)
    with open(rp) as f:
        doc = json.load(f)
    out, problems = {}, []
    for p in doc.get("papers") or []:
        folder = os.path.join(run, OUT, "papers", _folder(p["doi"]))
        supp = p.get("supplements") or {}
        if supp.get("source") == "pmc-oa-opendata":
            files = [f for f in supp.get("files") or [] if f.get("sha256")]
            for f in files + [f for f in p.get("article_files") or [] if f.get("sha256")]:
                out.setdefault(f["file"], []).append(dict(doi=p["doi"], path=os.path.join(folder, "pmc_oa", f["file"]), sha256=f["sha256"]))
        elif supp.get("sha256") and supp.get("file"):
            path = os.path.join(folder, supp["file"])
            row = dict(file=supp["file"], doi=p["doi"], archive=True)
            try:
                if sha256_file(path, use_sidecar=False) != supp["sha256"]:
                    problems.append(dict(row, status="checksum_differs"))
                    continue
                with zipfile.ZipFile(path) as z:
                    members = [m for m in z.namelist() if not m.endswith("/")]
            except ZIP_ERRORS as err:
                problems.append(dict(row, status="not_read", error="%s: %s" % (type(err).__name__, str(err)[:200])))
                continue
            for m in members:
                out.setdefault(os.path.basename(m), []).append(dict(doi=p["doi"], path=path, sha256=supp["sha256"], member=m))
    return out, problems


def pdf_texts(spec, dest) -> list:
    """The text of the PDFs a passed registry run stored (spec: from_run, files as names or fnmatch patterns), page by
    page through bench/rel/registry/pdf_text.py: readable pages give text, the others their reason only. A file whose
    bytes do not match the recorded sha256 is not read; a name the run did not store and a stored archive that is
    missing, changed or not opened are reported as rows of their own."""
    from bench.rel.registry import pdf_text as PT
    have, problems = stored_files(spec["from_run"])
    out = list(problems)
    names = []
    for pat in spec["files"]:
        hit = sorted(n for n in have if fnmatch.fnmatchcase(n, pat) and n.lower().endswith(".pdf"))
        names += [n for n in hit if n not in names] or ([pat] if pat not in names else [])
    os.makedirs(dest, exist_ok=True)
    for name in names:
        recs = have.get(name)
        if not recs:
            out.append(dict(file=name, status="not_stored"))
            continue
        if len(recs) > 1:
            out.append(dict(file=name, status="ambiguous", dois=sorted({r["doi"] for r in recs})))
            continue
        rec = recs[0]
        try:
            if sha256_file(rec["path"], use_sidecar=False) != rec["sha256"]:
                out.append(dict(file=name, doi=rec["doi"], status="checksum_differs"))
                continue
            if rec.get("member"):
                with zipfile.ZipFile(rec["path"]) as z:
                    data = z.read(rec["member"])
            else:
                with open(rec["path"], "rb") as f:
                    data = f.read()
        except ZIP_ERRORS as err:
            out.append(dict(file=name, doi=rec["doi"], status="not_read", error="%s: %s" % (type(err).__name__, str(err)[:200])))
            continue
        doc = PT.extract(data)
        text = os.path.join(dest, name + ".txt")
        with open(text, "w", encoding="utf-8") as f:
            f.write(PT.render(name, doc))
        pages = doc["pages"]
        out.append(dict(file=name, doi=rec["doi"], member=rec.get("member"), source_sha256=rec["sha256"], pdf_sha256=doc["sha256"],
                        status=doc["status"], reason=doc.get("reason"), pages=len(pages),
                        per_status=dict(Counter(p_["status"] for p_ in pages)), fonts=doc.get("fonts", {}),
                        text=dict(file=os.path.basename(text), sha256=sha256_file(text, use_sidecar=False))))
    return out


def build(st, cfg, run_dir):
    s = settings(cfg)
    t0 = time.time()
    paths = [st.repo(x) for x in s["registry"]]
    rows = [r for x in paths for r in reg.read(x)]
    errors, warnings = reg.check(rows)
    summary_ = reg.summarise(rows) if not errors else {}
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    found = taxa(rows, tax)
    pre, exposure = {}, None
    if s["source"]:
        from ci.stages.rel import w2_v0
        src = w2_v0.source(s["source"])
        pre = pretraining(found, src["index"], tax)
        st.extra["source"] = src["provenance"]
        if s["exposure_roots"]:
            inside = radius_genomes(found, src["index"], tax)
            islands = island_taxa(st.repo(REFERENCE_SET), s["exposure_islands"]) if s["exposure_islands"] else None
            exposure = exposure_scan(exposure_tokens(rows, found, inside, tax, (islands or {}).get("taxids")), s["exposure_roots"],
                                     unseen_genomes(src["index"]),
                                     skip={os.path.abspath(run_dir), run_dir}, max_paths=s["exposure_max_paths"])
            exposure["islands"] = {k: v for k, v in (islands or {}).items() if k != "taxids"} or None
            exposure["levels"] = exposure_levels(rows, exposure)
            exposure["above_declared"] = sorted(e for e, v in exposure["levels"].items() if v.get("above_declared"))
    elif s["exposure_roots"]:
        raise SystemExit("%s: the exposure scan needs source (the corpus genomes inside each radius)" % STAGE)
    overlap = sealed_overlap(found, sealed_genera(tax, st.repo(os.path.join("bench", "hard", "EXCLUDE.tsv"))), tax)
    papers = []
    if s["fetch"]:
        base = os.path.dirname(st.path("rel_w15", "registry", "papers", "x"))
        papers = fetch_all(dois(rows), base, s)
    annotation = None
    if s["annotation"]:
        annotation = annotation_lookup(s["annotation"], rows, st.path("rel_w15", "registry", "annotation", "x").rsplit("/", 1)[0])
    pdfs = None
    if s["pdf_text"]:
        pdfs = dict(from_run=s["pdf_text"]["from_run"], files=pdf_texts(s["pdf_text"], os.path.dirname(st.path("rel_w15", "registry", "pdf_text", "x"))))
    doc = dict(format="rel-w15-registry", rules_version=reg.RULES_VERSION, registry=s["registry"], annotation=annotation, exposure=exposure,
               pdf_text=pdfs,
               registry_sha256={x: sha256_file(st.repo(x), use_sidecar=False) for x in s["registry"]}, records=len(rows), errors=errors, warnings=warnings,
               summary=summary_, taxa=found, pretraining=pre, sealed_overlap=overlap, papers=papers,
               fetch_failures=failures(papers),
               note="no model output and no score is read; the sealed test split of the medium corpus is not looked at",
               seconds=round(time.time() - t0, 1))
    out = st.path("rel_w15", "registry", "report.json")
    write_json(out, doc)
    st.output("report", out)
    md = st.path("rel_w15", "registry", "report.md")
    with open(md, "w") as f:
        f.write(reg.markdown(rows, summary_, errors, warnings) if summary_ else "# Event registry report\n\nThe draft breaks rules.\n")
        f.write(markdown_server(doc))
    st.output("report_md", md)
    st.extra.update(records=len(rows), errors=len(errors), papers=len(papers),
                    taxa={k: sum(1 for v in found.values() if v["status"] == k) for k in ("ok", "rank_differs", "ambiguous", "unresolved")})
    return doc


def markdown_server(doc) -> str:
    out = ["", "## Server checks", "", "| record | NCBI | taxid | names not in NCBI | train | held-out | sealed genera |",
           "|---|---|---|---|---|---|---|"]
    for eid, rec in sorted(doc["taxa"].items()):
        pre = doc["pretraining"].get(eid, {})
        ov = doc["sealed_overlap"].get(eid, {})
        out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            eid, rec["status"], rec.get("taxid", ""), "; ".join(rec.get("names_not_in_ncbi", [])),
            pre.get("train", ""), pre.get("heldout", ""), ov.get("sealed_overlap", "")))
    ex = doc.get("exposure")
    if ex:
        out += ["", "### Exposure scan", "",
                "Names only (bench/hard/V2_REGISTER.md): %d paths walked%s%s; %d paths skipped because they name a genome outside "
                "the train and held-out splits; %d walk errors (%d of them on such paths, counted only); %d outputs dropped by "
                "the last check." % (
                    ex["paths_walked"], "; stopped at the cap" if ex["stopped_at_cap"] else "",
                    "; missing roots: " + ", ".join(ex["missing_roots"]) if ex["missing_roots"] else "",
                    ex["skipped_unseen"], ex["walk_error_count"], ex["walk_errors_unseen"], ex["dropped_outputs"]),
                "", "| root | paths walked | pruned | skipped | errors | digest |", "|---|---|---|---|---|---|"]
        for r in ex["roots"]:
            out.append("| %s | %d | %s | %d | %d | %s |" % (r["path"], r["walked"], ", ".join(r["pruned"]) or "-", r["skipped_unseen"],
                                                          r["errors"], (r.get("digest") or "-")[:16]))
        out += ["", "| record | genomes in radius | tokens | declared | scan | level | paths per level | matched | example |",
                "|---|---|---|---|---|---|---|---|---|"]
        for eid, lv in sorted((ex.get("levels") or {}).items()):
            v = ex["events"].get(eid, {})
            t = (ex.get("tokens") or {}).get(eid, {})
            top = ((v.get("examples") or {}).get(v.get("level")) or [""])[0]
            out.append("| %s | %s | %d assemblies, %d nucleotide, %s | %s%s | %s | %s%s | %s | %s | %s |" % (
                eid, t.get("genomes", 0), len(t.get("accessions") or []), len(t.get("nucleotide") or []),
                ", ".join(t.get("names") or []) or "no names",
                lv["declared"], " (not a level)" if lv.get("declared_unknown") else "", lv["scan"], lv["level"],
                " (above the declared level)" if lv.get("above_declared") else "",
                ", ".join("%s %d" % kv for kv in (v.get("paths") or {}).items()), ", ".join(v.get("matched") or []), top))
    pt = doc.get("pdf_text")
    if pt:
        out += ["", "### PDF text", "", "Files stored by run %s, read by bench/rel/registry/pdf_text.py; readable pages only give text." % (
            os.path.basename(pt["from_run"])), "", "| file | status | pages | per page | reason |", "|---|---|---|---|---|"]
        for f in pt["files"]:
            out.append("| %s | %s | %s | %s | %s |" % (f["file"], f["status"], f.get("pages", ""),
                                                      ", ".join("%s %d" % kv for kv in sorted((f.get("per_status") or {}).items())),
                                                      f.get("reason") or f.get("error") or ""))
    a = doc.get("annotation")
    if a:
        out += ["", "Annotation of %s (%s): %s; %s genes, %s locus tags, prefixes %s; %s of the drafts' %s ASEM identifiers are locus tags." % (
            a.get("wgs_master"), (a.get("assembly") or {}).get("accession"), a.get("status"), a.get("genes", 0), a.get("locus_tags", 0),
            a.get("locus_tag_prefixes", []), len(a.get("matched") or []), a.get("ids", 0))]
    if doc["papers"]:
        ok = sum(1 for p in doc["papers"] if (p.get("fulltext") or {}).get("sha256"))
        out += ["", "%d DOIs looked up (%d with a parsed record); %d with an open full text stored; %d with supplements stored." % (
            len(doc["papers"]), sum(1 for p in doc["papers"] if "found" in p), ok,
            sum(1 for p in doc["papers"] if (p.get("supplements") or {}).get("sha256")))]
        for part, counts in sorted((doc.get("fetch_failures") or {}).items()):
            out.append("Not fetched, %s: %s." % (part, ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    return "\n".join(out) + "\n"


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        doc = json.load(f)
    bad = [dict(error=e) for e in doc.get("errors") or []]
    for rel, sha in (doc.get("registry_sha256") or {}).items():
        path = rel if os.path.isabs(rel) else os.path.join(common.ROOT, rel)
        if sha256_file(path, use_sidecar=False) != sha:
            bad.append(dict(error="%s is not the file the report was made from" % rel))
    if os.path.exists(os.path.join(common.ROOT, CATALOGUE)):
        bad.append(dict(error="%s exists: the draft must not live there" % CATALOGUE))
    for f in (doc.get("pdf_text") or {}).get("files") or []:
        if f["status"] in ("not_stored", "ambiguous", "checksum_differs", "not_read"):
            bad.append(dict(error="PDF %s: %s" % (f["file"], f["status"])))
    ex = doc.get("exposure") or {}
    if ex.get("missing_roots"):
        bad.append(dict(error="exposure scan roots missing: %s" % ex["missing_roots"]))
    if ex.get("stopped_at_cap"):
        bad.append(dict(error="the exposure scan stopped at %d paths: raise exposure_max_paths or narrow the roots" % ex["paths_walked"]))
    if ex.get("walk_error_count"):
        bad.append(dict(error="the exposure scan met %d walk errors: the scan is incomplete" % ex["walk_error_count"]))
    empty = [r["path"] for r in ex.get("roots") or [] if r["path"] not in (ex.get("missing_roots") or []) and not r["walked"]]
    if empty:
        bad.append(dict(error="exposure scan roots with no path walked: %s" % empty))
    if (doc.get("annotation") or {}).get("status") == "error":
        bad.append(dict(error="the annotation lookup failed: %s" % ((doc["annotation"].get("error") or {}).get("error"),)))
    papers = doc.get("papers") or []
    looked_up = sum(1 for x in papers if "found" in x)
    if papers and looked_up < FETCH_MIN_SHARE * len(papers):
        bad.append(dict(error="only %d of %d paper lookups returned a parsed record: a systematic fetch failure" % (looked_up, len(papers))))
    require(GateResult("G4_rel_w15_registry", "fail" if bad else "pass",
                       dict(problems=bad[:50], records=doc.get("records"), warnings=len(doc.get("warnings") or []),
                            taxa={k: sum(1 for v in doc["taxa"].values() if v["status"] == k) for k in ("ok", "rank_differs", "ambiguous", "unresolved")})),
            run_dir)
    return sha256_file(p, use_sidecar=False)


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        doc = json.load(f)
    s = doc.get("summary") or {}
    md = ["| label | records |", "|---|---|"] + ["| %s | %d |" % kv for kv in (s.get("per_label") or {}).items()]
    return dict(json=dict(records=doc["records"], errors=len(doc["errors"]), papers=len(doc["papers"]),
                          rules_version=doc["rules_version"]), md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
