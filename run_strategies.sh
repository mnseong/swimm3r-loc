#!/usr/bin/env bash
set -uo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v13c}"
OUT="${2:-out}"
case "$M" in
  v13c) ;;
  *) echo "usage: [TAG=name] [WIN=5] [GAP_BLOCKS='6 9'] [SEALOC=data/sealoc_s1] [SEALOC_DEPTH_M=2.05] run_strategies.sh [v13c] [out_dir] [more options of tools/stage1_splatt3r.py for every step]"
     echo "three steps one after another with HF_HUB_OFFLINE=1, a step that fails does not stop the next"
     echo "  1 video1, HOLD=12 BLOCK=3 (run_gscpe.sh), with pose verification, the viewpoint gap experiment and the medium adapted to the query"
     echo "  2 run_gap.sh on the sparser maps rebuilt from fewer map frames (BLOCK 6 and 9 of every 12)"
     echo "  3 the SEALOC revisit scene, its own split, its reference moved into the map by a similarity on the map frames (HOLD_MAP=moved), with the three experiments"
     exit 1 ;;
esac
export HF_HUB_OFFLINE=1
WIN="${WIN:-5}"
TG="${TAG:+_$TAG}"
SL="${SEALOC:-data/sealoc_s1}"
SL="${SL%/}"
BLOCKS="${GAP_BLOCKS:-6 9}"
STRAT=(--verify_bench --gap_bench --medium_adapt)
LABEL=()
STATUS=()
WHERE=()

abs() { case "$1" in /*) echo "$1" ;; *) echo "$D/$1" ;; esac; }

finish() {
  local name="$1" ok="$2"
  shift 2
  for p in "$@"; do
    LABEL+=("$name")
    STATUS+=("$ok")
    WHERE+=("$(abs "$p")")
    echo "[strategies] $name $ok, its gscpe.txt is $(abs "$p")/gscpe.txt $( [ -f "$p/gscpe.txt" ] && echo "(written)" || echo "(missing)")"
  done
}

P1="$OUT/${M}${TG}_hold12b3w${WIN}_loftr/loc"
echo "[strategies] step 1 of 3, video1 HOLD=12 BLOCK=3 with the three experiments, results in $(abs "$P1")"
if SCENE=data/video1 FIXED_SPLIT=0 HOLD=12 BLOCK=3 HOLD_MAP=register REF= REF_MATCHER=loftr WIN="$WIN" bash "$D/run_gscpe.sh" "$M" "$OUT" "${STRAT[@]}" "${@:3}"; then
  finish "step 1" done "$P1"
else
  finish "step 1" failed "$P1"
fi

P2=()
for B in $BLOCKS; do
  HB=12
  [ "$B" = 1 ] || HB="12b$B"
  P2+=("$OUT/${M}${TG}_hold${HB}w${WIN}_loftr/loc")
done
echo "[strategies] step 2 of 3, run_gap.sh on the maps rebuilt with BLOCK $BLOCKS, results in ${P2[*]+${P2[*]}}"
if SCENE=data/video1 FIXED_SPLIT=0 REF= GAP_BLOCKS="$BLOCKS" WIN="$WIN" bash "$D/run_gap.sh" "$M" "$OUT" "${@:3}"; then
  finish "step 2" done ${P2[@]+"${P2[@]}"}
else
  finish "step 2" failed ${P2[@]+"${P2[@]}"}
fi

P3="$OUT/${M}_$(basename "$SL")${TG}_fixed_moved/loc"
echo "[strategies] step 3 of 3, SEALOC $SL with its own split and reference, the three experiments, results in $(abs "$P3")"
if [ ! -f "$SL/reference_poses.json" ] || [ ! -f "$SL/split.json" ]; then
  echo "[strategies] step 3 needs $SL/reference_poses.json and $SL/split.json (tools/make_sealoc_scene.py builds them), skipped"
  finish "step 3" failed "$P3"
elif SCENE="$SL" FIXED_SPLIT=1 HOLD=0 EVERY=0 HOLD_MAP=moved REF="$SL/reference_poses.json" TRT="${TRT:-1}" WIN="$WIN" bash "$D/run.sh" "$M" "$OUT" \
    --gscpe "${STRAT[@]}" --gscpe_ref_metric --gscpe_depth_m "${SEALOC_DEPTH_M:-2.05}" "${@:3}"; then
  finish "step 3" done "$P3"
else
  finish "step 3" failed "$P3"
fi

echo
echo "======================================================================================================================================"
echo " strategies done, ${#LABEL[@]} results"
for i in "${!LABEL[@]}"; do
  echo "   ${LABEL[$i]} ${STATUS[$i]} | ${WHERE[$i]}/gscpe.txt"
done
for i in "${!LABEL[@]}"; do
  if [ -f "${WHERE[$i]}/short.txt" ]; then
    echo
    echo " ${LABEL[$i]}, ${WHERE[$i]}"
    cat "${WHERE[$i]}/short.txt"
  fi
done
for s in "${STATUS[@]}"; do
  [ "$s" = done ] || exit 1
done
exit 0
