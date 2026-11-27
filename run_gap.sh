#!/usr/bin/env bash
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v13c}"
OUT="${2:-out}"
case "$M" in
  v13c) ;;
  *) echo "usage: [GAP_BLOCKS='3 6 9'] [GAP_PRIORS='2:0 5:0 10:0 20:0 0:2 0:5 0:10'] [GAP_SEED=0] [GAP_QUERIES=0] run_gap.sh [v13c] [out_dir] [more options of tools/stage1_splatt3r.py]"
     echo "HOLD=12 with BLOCK=3, 6 and 9 consecutive queries of every 12 frames: every split gets its own Stage 1 of its map frames alone (run.sh, HOLD_MAP=register),"
     echo "its own map and output folder, the GS-CPE benchmark (run_gscpe.sh) and the gap experiment (tools/bench_gap.py, gap.txt and gap.json next to gscpe.json)"; exit 1 ;;
esac
HOLD=12
WIN="${WIN:-5}"
TAG0="${TAG:-}"
TG="${TAG0:+_$TAG0}"
PRIORS="${GAP_PRIORS:-2:0 5:0 10:0 20:0 0:2 0:5 0:10}"
read -r -a PR <<< "$PRIORS"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=()
DONE=()
for B in ${GAP_BLOCKS:-3 6 9}; do
  [ "$B" -ge 1 ] 2>/dev/null && [ "$B" -lt "$HOLD" ] || { echo "BLOCK $B must be 1 or more and below $HOLD, skipped"; FAILED+=("$B"); continue; }
  HB="$HOLD"
  [ "$B" = 1 ] || HB="${HOLD}b${B}"
  SD="$OUT/map_${M}${TG}_hold${HB}w${WIN}"
  O="$OUT/${M}${TG}_hold${HB}w${WIN}_loftr"
  echo "gap split HOLD=$HOLD BLOCK=$B: map $SD (Stage 1 of its map frames alone, $( [ -f "$SD/mapping_seconds" ] && echo "already built, reused" || echo "built now")), results $O/loc"
  if ls "$O/loc"/*.json >/dev/null 2>&1; then
    K="$O/loc/before_gap_$STAMP"
    mkdir -p "$K" && cp -p "$O/loc"/*.json "$K"/ && { cp -p "$O/loc"/*.txt "$K"/ 2>/dev/null || true; } && echo "earlier results of this split kept in $K"
  fi
  if HOLD="$HOLD" BLOCK="$B" WIN="$WIN" HOLD_MAP=register REF_MATCHER=loftr bash "$D/run_gscpe.sh" "$M" "$OUT" --gap_bench --gap_priors "${PR[@]}" \
      --gap_seed "${GAP_SEED:-0}" --gap_queries "${GAP_QUERIES:-0}" "${@:3}"; then
    DONE+=("$O/loc")
  else
    echo "gap split BLOCK=$B failed, the next split runs"
    FAILED+=("$B")
  fi
done
echo "gap runs done: ${#DONE[@]} splits"
for o in ${DONE[@]+"${DONE[@]}"}; do
  case "$o" in /*) p="$o" ;; *) p="$D/$o" ;; esac
  echo "  $p/gap.txt $( [ -f "$p/gap.txt" ] && echo written || echo missing)"
done
[ ${#FAILED[@]} = 0 ] || { echo "failed splits (BLOCK): ${FAILED[*]}"; exit 1; }
