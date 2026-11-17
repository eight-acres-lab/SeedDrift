"""Write the catalogue snapshots of the select-tiny profile (tests/ci/fixtures/select/*.jsonl) from the tiny taxdump.

Run from the repository root: python tests/ci/fixtures/make_select.py. The records are synthetic (made-up accessions
and counts) but their taxids are real nodes of tests/ci/fixtures/tiny/taxdump, chosen in a fixed order, and they
include the cases the selection must handle: an excluded order (Enterobacterales), a sealed genus (Pseudomonas), two
assemblies of one species, too few genes, plasmid-bearing assemblies, a phage of an excluded host, a tailed phage
with a eukaryote host (a group conflict), a virus without host records, a red alga in the protist list (placed by its
taxid), and an Ensembl genome without an NCBI annotation (group median).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from ci.taxonomy import Taxonomy  # noqa: E402

OUT = os.path.join(ROOT, "tests", "ci", "fixtures", "select")
SEALED = ["Acinetobacter", "Bacillus", "Bradyrhizobium", "Burkholderia", "Corynebacterium", "Haemophilus", "Mycobacterium", "Prochlorococcus", "Pseudomonas",
          "Rhizobium", "Rhodopseudomonas", "Rickettsia", "Shewanella", "Sinorhizobium", "Staphylococcus", "Streptococcus", "Vibrio"]


def species_under(tax, root, exclude=()):
    out = []
    for t in sorted(tax.rank):
        if tax.rank[t] != "species":
            continue
        lin = set(tax.lineage(t))
        if root in lin and not lin & set(exclude):
            out.append(t)
    return out


def genome(acc, taxid, pc, chromosomes=1, components=1, level="Complete Genome", ref="reference genome", paired=""):
    return dict(accession=acc, paired_accession=paired or acc.replace("GCF_", "GCA_"), taxid=taxid, organism="", assembly_level=level, assembly_status="current",
                refseq_category=ref, length=pc * 1000, chromosomes=chromosomes, components=components, protein_coding=pc, checkm_completeness=99.0,
                checkm_contamination=0.5, source_database="SOURCE_DATABASE_REFSEQ")


def canonical(records, key):
    """The canonical form of ci/stages/catalog.py (the catalog stage refuses a fixture file that is not in it)."""
    import json
    by = {r[key]: r for r in records}
    assert len(by) == len(records), "duplicate keys"
    return "".join(json.dumps(by[k], sort_keys=True, separators=(",", ":")) + "\n" for k in sorted(by)).encode(), len(by)


def write(name, recs, key):
    data, _ = canonical(recs, key)
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)


def main():
    tax = Taxonomy(os.path.join(ROOT, "tests", "ci", "fixtures", "tiny", "taxdump"))
    os.makedirs(OUT, exist_ok=True)
    entero = tax.taxa_by_name("Enterobacterales", "order")
    sealed_fams = set()
    for g in SEALED:
        for t in tax.genus_by_name(g):
            sealed_fams.add(tax.at_rank(t, "family"))
    bact, seen_orders = [], set()
    for t in species_under(tax, 2, exclude=entero + sorted(sealed_fams)):
        o = tax.at_rank(t, "order")
        if o and tax.at_rank(t, "genus") and o not in seen_orders:
            seen_orders.add(o)
            bact.append(t)
        if len(bact) == 10:
            break
    ent = species_under(tax, entero[0])[0]
    pseud = tax.genus_by_name("Pseudomonas")[0]
    pse = [t for t in species_under(tax, pseud)][0]
    recs = []
    for i, t in enumerate(bact):
        comps = 1 + (i % 3 == 0) + (i % 5 == 0)                       # some assemblies carry one or two plasmids
        recs.append(genome("GCF_%09d.1" % (100 + i), t, 2000 + 100 * i, components=comps))
    recs.append(genome("GCF_%09d.1" % 200, bact[0], 2500, ref="", components=2))   # a second assembly of the first species
    recs.append(genome("GCF_%09d.1" % 201, bact[1], 300))            # too few protein-coding genes (but its species has another)
    recs.append(genome("GCF_%09d.1" % 202, ent, 4500, components=3))  # excluded order
    recs.append(genome("GCF_%09d.1" % 203, pse, 5000))               # sealed genus
    write("bacteria.jsonl", recs, "accession")
    arch = species_under(tax, 2157)[:3]
    write("archaea.jsonl", [genome("GCF_%09d.1" % (300 + i), t, 1800 + 50 * i) for i, t in enumerate(arch)], "accession")

    phage = species_under(tax, 2731619)
    giant = species_under(tax, 2732007)
    euk_virus = [t for t in species_under(tax, 10239) if t not in set(phage) | set(giant)]
    human = 9606
    vir, hosts = [], []
    for i, t in enumerate(phage[:4]):
        vir.append(genome("GCF_%09d.1" % (400 + i), t, 60 + 10 * i))
        hosts.append(dict(accession="NC_%06d.1" % (400 + i), virus_taxid=t, host_taxid=bact[i], completeness="COMPLETE", protein_count=60, length=40000,
                          source_database="RefSeq"))
    vir.append(genome("GCF_%09d.1" % 410, phage[4], 70))                 # a tailed phage recorded on a eukaryote host: conflict
    hosts.append(dict(accession="NC_%06d.1" % 410, virus_taxid=phage[4], host_taxid=human, completeness="COMPLETE", protein_count=70, length=40000,
                      source_database="RefSeq"))
    vir.append(genome("GCF_%09d.1" % 411, phage[5], 80))                 # a phage of an excluded host
    hosts.append(dict(accession="NC_%06d.1" % 411, virus_taxid=phage[5], host_taxid=ent, completeness="COMPLETE", protein_count=80, length=40000,
                      source_database="RefSeq"))
    for i, t in enumerate(giant[:3]):
        vir.append(genome("GCF_%09d.1" % (420 + i), t, 200 + 10 * i))
        hosts.append(dict(accession="NC_%06d.1" % (420 + i), virus_taxid=t, host_taxid=human, completeness="COMPLETE", protein_count=200, length=200000,
                          source_database="RefSeq"))
    for i, t in enumerate(euk_virus[:4]):
        vir.append(genome("GCF_%09d.1" % (430 + i), t, 8 + i, chromosomes=2, components=2))   # segmented: one assembly
        if i < 3:                                                    # the fourth has no host record
            hosts.append(dict(accession="NC_%06d.1" % (430 + i), virus_taxid=t, host_taxid=human, completeness="COMPLETE", protein_count=8, length=10000,
                              source_database="RefSeq"))
    write("viruses.jsonl", vir, "accession")
    write("virus_hosts.jsonl", hosts, "accession")

    def ens(name, t, site, division, release, acc):
        return dict(name=name, display_name=tax.name(t), division="Ensembl" + division.capitalize(), taxonomy_id=t, assembly="asm%d" % t, assembly_accession=acc,
                    genebuild="2026-01", core_db="%s_core_%d_1" % (name, release), site=site, site_division=division, release=release)
    groups = [("vertebrates", "ensembl", 116, species_under(tax, 40674)[:3] + species_under(tax, 7742, exclude=[40674])[:2]),
              ("plants", "ensemblgenomes", 63, species_under(tax, 33090)[:3]),
              ("fungi", "ensemblgenomes", 63, species_under(tax, 4751)[:3]),
              ("metazoa", "ensemblgenomes", 63, species_under(tax, 33208, exclude=[7742])[:3]),
              ("protists", "ensemblgenomes", 63, species_under(tax, 2759, exclude=[33208, 33090, 2763, 38254, 4751])[:2] + species_under(tax, 2763)[:1])]
    ann, k = [], 0
    for division, site, release, ts in groups:
        rows = []
        for t in ts:
            k += 1
            acc = "GCA_%09d.1" % (500 + k)
            nm = tax.name(t).lower().replace(" ", "_").replace(".", "")
            rows.append(ens(nm, t, site, division, release, acc))
            if k % 7 != 0:                                           # every seventh genome has no NCBI annotation
                ann.append(genome(acc, t, 9000 + 500 * k, ref="", paired=""))
        write("ens_%s.jsonl" % division, rows, "name")
    for a in ann:
        a["paired_accession"] = ""
    write("ens_annotation.jsonl", ann, "accession")


if __name__ == "__main__":
    main()
