#!/usr/bin/env bash
# Scan the prepared Enterobacterales data (UFBoot < 95 branches collapsed) with calibrated (demo.sh) and built-in default weights.
set -euo pipefail
cd "$(dirname "$0")/../.."
W=real/enterobacterales
THREADS=${THREADS:-9}
for M in calibrated default; do
  OPT=()
  [ "$M" = calibrated ] && OPT=(--model demo/model.calibrated.json)
  python3 -m seeddrift scan --species-tree $W/species_tree.nwk --gene-trees $W/gene_trees \
    --genes $W/genes.tsv --me $W/mobile_elements.tsv --species-meta $W/species.tsv --composition $W/gene_composition.tsv \
    ${OPT[@]+"${OPT[@]}"} --collapse 95 --out $W/scan_$M --threads "$THREADS" > $W/scan_$M.log 2>&1
  echo "scan_$M done"
done
