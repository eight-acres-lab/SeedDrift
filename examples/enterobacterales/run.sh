#!/usr/bin/env bash
# Reproduce the Enterobacterales real-data run (18 genomes from Ensembl Bacteria).
#
# One-time tool setup (macOS / Linux):
#   brew install micromamba            # or any conda
#   MAMBA_ROOT_PREFIX=~/mamba micromamba create -y -n hgt -c conda-forge -c bioconda \
#       orthofinder mafft iqtree diamond
set -euo pipefail
cd "$(dirname "$0")/../.."
W=real/enterobacterales
THREADS=${THREADS:-9}

# genome index (6.5 MB) - needed once for `fetch`
mkdir -p real/ensembl
[ -s real/ensembl/species_EnsemblBacteria.txt ] || curl -sS -o real/ensembl/species_EnsemblBacteria.txt \
  https://ftp.ensemblgenomes.ebi.ac.uk/pub/bacteria/current/species_EnsemblBacteria.txt

# download -> proteomes -> OrthoFinder -> gene trees (~1 h) -> species tree; finished steps are skipped
python3 -m seeddrift prepare --genomes $W/genomes.tsv --index real/ensembl/species_EnsemblBacteria.txt \
  --workdir $W --threads "$THREADS"

# scan (UFBoot < 95 branches collapsed) with the weights calibrated on simulated data (demo.sh)
# and with the built-in defaults -> scan_calibrated/, scan_default/
bash $W/scan.sh

# compare with documented islands (SPI-1, SPI-2, LEE)
for M in calibrated default; do
  python3 $W/check_known_islands.py $W $W/scan_$M > $W/islands_$M.txt
done
cat $W/islands_default.txt
