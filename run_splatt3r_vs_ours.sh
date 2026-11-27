#!/usr/bin/env bash
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
OUT="${1:-out}"
mkdir -p "$OUT"
for V in ${VIDEOS:-video1 video2 video3 video4}; do
  for M in v18e10 splatt3r; do
    echo "[cmp] $M $V start $(date +%T)"
    SCENE="data/$V" HOLD=12 BLOCK=3 REF_MATCHER=loftr TRT="${TRT:-1}" bash run.sh "$M" "$OUT" --gscpe --gscpe_no_ablation "${@:2}" > "$OUT/log_${M}_${V}.txt" 2>&1
    rc=$?
    echo "[cmp] $M $V exit $rc $(date +%T) | $(grep -a -E '^ In short| fast \+ check, TensorRT|coarse: render, match, PnP' "$OUT/log_${M}_${V}.txt" | tail -n 2 | tr -s ' ' | tr '\n' ';' | cut -c1-220)"
    [ $rc -eq 0 ] || echo "[cmp] see $OUT/log_${M}_${V}.txt: $(grep -a -E 'Error|Traceback' "$OUT/log_${M}_${V}.txt" | tail -n 1 | cut -c1-200)"
  done
done
python tools/compare_models.py "$OUT"
