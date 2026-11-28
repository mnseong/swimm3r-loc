#!/usr/bin/env bash
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
OUT="${1:-out_pamir}"
SC="${PAMIR_SCENE:-data/pamir_a}"
[ -f "$SC/ref.json" ] && [ -f "$SC/train_files.txt" ] && [ -f "$SC/split.json" ] || { echo "$SC is not a complete scene (ref.json, train_files.txt, split.json)"; exit 1; }
mkdir -p "$OUT"
N="$(basename "$SC")"
for M in v18e10 splatt3r; do
  echo "[pamir] $M start $(date +%T)"
  SCENE="$SC" HOLD=0 EVERY=0 FIXED_SPLIT=1 HOLD_MAP=moved REF="$SC/ref.json" WIN="${WIN:-10}" TRT="${TRT:-1}" \
    bash run.sh "$M" "$OUT" --gscpe --gscpe_no_ablation "${@:2}" > "$OUT/log_${M}_${N}.txt" 2>&1
  rc=$?
  echo "[pamir] $M exit $rc $(date +%T) | $(grep -a -E 'one denominator|reference check on held out|\[gscpe\] queries' "$OUT/log_${M}_${N}.txt" | cut -c1-200 | tr '\n' ' ')"
  [ $rc -eq 0 ] || echo "[pamir] see $OUT/log_${M}_${N}.txt: $(grep -a -E 'Error|Traceback' "$OUT/log_${M}_${N}.txt" | tail -n 1 | cut -c1-200)"
done
python tools/split_scores.py "$OUT/v18e10_${N}_fixed_moved/loc/scores.json" "$OUT/splatt3r_${N}_fixed_moved/loc/scores.json" | tee "$OUT/pamir_split_scores.txt"
