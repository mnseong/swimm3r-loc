#!/usr/bin/env bash
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
OUT="${1:-out}"
RMODE="${REF_MODE:-loftr}"
case "$RMODE" in loftr|common) ;; *) echo "REF_MODE must be loftr or common"; exit 1 ;; esac
mkdir -p "$OUT"
for V in ${VIDEOS:-video1 video2 video3 video4}; do
  for M in v18e10 splatt3r; do
    echo "[cmp] $M $V start $(date +%T)"
    if [ "$RMODE" = common ]; then
      R="data/$V/ref.json"
      [ -f "$R" ] || { echo "[cmp] no $R, $V skipped"; continue; }
      LG="$OUT/log_${M}_${V}_moved$([ "${WIN:-5}" = 5 ] || echo "_w$WIN").txt"
      SCENE="data/$V" HOLD=12 BLOCK=3 HOLD_MAP=moved REF="$R" REF_LOCAL="${REF_LOCAL:-10}" TRT="${TRT:-1}" bash run.sh "$M" "$OUT" --gscpe --gscpe_no_ablation "${@:2}" > "$LG" 2>&1
    else
      LG="$OUT/log_${M}_${V}.txt"
      SCENE="data/$V" HOLD=12 BLOCK=3 REF_MATCHER=loftr TRT="${TRT:-1}" bash run.sh "$M" "$OUT" --gscpe --gscpe_no_ablation "${@:2}" > "$LG" 2>&1
    fi
    rc=$?
    echo "[cmp] $M $V exit $rc $(date +%T) | $(grep -a -E '^ In short| fast \+ check, TensorRT|coarse: render, match, PnP' "$LG" | tail -n 2 | tr -s ' ' | tr '\n' ';' | cut -c1-220)"
    [ $rc -eq 0 ] || echo "[cmp] see $LG: $(grep -a -E 'Error|Traceback' "$LG" | tail -n 1 | cut -c1-200)"
  done
done
if [ "$RMODE" = common ]; then python tools/compare_models.py "$OUT" moved; else python tools/compare_models.py "$OUT"; fi
