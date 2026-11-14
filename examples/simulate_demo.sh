#!/usr/bin/env bash
# End-to-end demo: simulate prok / euk / mixed datasets (train + test), scan,
# calibrate on the training sets, evaluate on held-out test sets.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
OUT=${OUT:-demo}
THREADS=${THREADS:-8}

run_scan () {  # $1 = dataset dir, $2 = gene-tree folder name, $3 = output folder name
  local D=$1
  $PY -m seeddrift scan --species-tree "$D/species_tree.nwk" --gene-trees "$D/${2:-gene_trees}" \
      --genes "$D/genes.tsv" --me "$D/mobile_elements.tsv" --species-meta "$D/species.tsv" \
      --composition "$D/gene_composition.tsv" \
      --out "$D/${3:-scan}" --threads "$THREADS" 2>&1 | grep -E "high=|done"
}

evaluate () {  # $1 = dataset dir, $2 = events file
  local D=$1
  $PY -m seeddrift evaluate --events "$2" --truth "$D/truth_transfers.tsv" \
      --species-tree "$D/species_tree.nwk" --genes "$D/genes.tsv" \
      --contaminants "$D/truth_contaminants.tsv" --oracle "$D/scan_true_trees/hgt_events.tsv" \
      --json "${2%.tsv}.metrics.json"
}

seed=1
for dom in prok euk mixed; do
  for split in train test; do
    D="$OUT/${dom}_${split}"
    echo "== simulate + scan $D"
    $PY -m seeddrift simulate --out "$D" --domain "$dom" --seed "$seed" > /dev/null
    seed=$((seed + 1))
    run_scan "$D"
    if [ "$split" = test ]; then   # oracle: noise-free gene trees define which transfers are identifiable
      run_scan "$D" gene_trees_true scan_true_trees > /dev/null
    fi
  done
done

echo "== calibrate on the three training sets"
$PY -m seeddrift calibrate \
  --events "$OUT/prok_train/scan/hgt_events.tsv" "$OUT/euk_train/scan/hgt_events.tsv" "$OUT/mixed_train/scan/hgt_events.tsv" \
  --truth  "$OUT/prok_train/truth_transfers.tsv" "$OUT/euk_train/truth_transfers.tsv" "$OUT/mixed_train/truth_transfers.tsv" \
  --out "$OUT/model.calibrated.json"

for dom in prok euk mixed; do
  D="$OUT/${dom}_test"
  $PY -m seeddrift rescore --events "$D/scan/hgt_events.tsv" --model "$OUT/model.calibrated.json" \
      --out "$D/scan/hgt_events.calibrated.tsv" > /dev/null
  echo
  echo "################ $D : default weights ################"
  evaluate "$D" "$D/scan/hgt_events.tsv"
  echo
  echo "################ $D : calibrated weights ################"
  evaluate "$D" "$D/scan/hgt_events.calibrated.tsv"
done
