#!/bin/bash
# Sample reference proteins and search them against the whole library with a deep hit list (training data of
# the placement model). Needs $SEEDDRIFT_DATA (data store with reference/ref.dmnd) and diamond on the PATH.
#   usage: SEEDDRIFT_DATA=... research/placement/search.sh
# The benchmark genomes are searched the same way (same options, their proteins as queries) for score.py.
set -eu
D=$SEEDDRIFT_DATA/placement
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/tmp"
python "$HERE/sample.py" 44 100
for part in a b; do
  diamond blastp -q "$D/queries_$part.faa" -d "$SEEDDRIFT_DATA/reference/ref.dmnd" -o "$D/hits_$part.tsv" --sensitive -b 4 -c 1 \
    --max-target-seqs 1000 -e 1e-5 --threads "${SEEDDRIFT_THREADS:-16}" --tmpdir "$D/tmp" \
    --outfmt 6 qseqid sseqid pident length evalue bitscore qstart qend sstart send qlen slen
done
