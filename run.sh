#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v9c}"
OUT="${2:-out}"
W="${WEIGHTS:-ckpt/joint_s3r_v8/weights}"
LORA="${LORA:-$W/0019_lora.pth}"
PHYS="${PHYS:-$W/0019_physics.pth}"
H="${HEAD:-ckpt/head_s3r_v9c/final_splatt3r.pth}"
TAG0="${TAG:-}"
TAG="${TAG:+_$TAG}"
LOC=(--anchor_px 6 --min_conf_thr 0.2 --fps 5 --medium_raster preshade --binf_veil)
case "$M" in
  v9c) ;;
  v9c_best) LOC+=(--time_bucket 8 --time_fill --far_layer) ;;
  *) echo "usage: [TAG=name LORA=path PHYS=path HEAD=path EVERY=n HOLD=n REF=poses] run.sh [v9c|v9c_best] [out_dir] [benchmark options, --proto for the localization protocol, --gscpe for the GS-CPE protocol]"; echo "EVERY=1 maps all frames as the reference, EVERY=4 maps every 4th of all frames and localizes the others"; echo "HOLD=4 holds out every 4th frame as a query and maps the other three (test 1 : train 3), REF is a reconstruction of all frames used as the reference in place of the mapping of all frames"; exit 1 ;;
esac
export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}"
STRIDE="${STRIDE:-1}"
WIN="${WIN:-5}"
EVERY="${EVERY:-0}"
HOLD="${HOLD:-0}"
REF="${REF:-}"
EXTRA=()
if [ "$HOLD" != 0 ]; then
  [ "$HOLD" -ge 2 ] || { echo "HOLD must be 2 or more"; exit 1; }
  MP="$OUT/map${TAG}_hold${HOLD}w${WIN}"; O="$OUT/${M}${TAG}_hold${HOLD}w${WIN}"
  mkdir -p "$MP" "$O"
  ls data/video1/images | grep -E '\.(jpg|jpeg|png)$' | sort > "$MP/all.txt"
  awk -v s="$HOLD" '{a[NR]=$0} END{for(i=1;i<=NR;i++) if (i%s!=0 || i==NR) print a[i]}' "$MP/all.txt" > "$MP/frames.txt"
  python -c "import json,sys; d=sys.argv[1]; a=[l.strip() for l in open(d+'/all.txt') if l.strip()]; m=set(l.strip() for l in open(d+'/frames.txt')); json.dump({'test_files':[x for x in a if x not in m],'centre_files':[]}, open(d+'/split.json','w'))" "$MP"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < "$MP/all.txt") (every ${HOLD}th frame is held out, test 1 : train $((HOLD - 1))), the other $(( $(wc -l < "$MP/all.txt") - $(wc -l < "$MP/frames.txt") )) are queries, pair window $WIN"
  EXTRA=(--split_json "$MP/split.json")
  if [ -n "$REF" ]; then
    [ -e "$REF" ] || { echo "no reference reconstruction at $REF, stopped"; exit 1; }
    EXTRA+=(--proto_ref_poses "$REF" --proto_ref_frames 2)
  else
    if [ ! -f "$OUT/map${TAG}_all/scene.pkl" ]; then
      echo "the mapping of all frames, the reference of the queries, is missing: it is made first (EVERY=1)"
      HOLD=0 REF= TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
    fi
    [ -f "$OUT/map${TAG}_all/scene.pkl" ] || { echo "no reference mapping at $OUT/map${TAG}_all, stopped"; exit 1; }
    EXTRA+=(--proto_ref_scene "$OUT/map${TAG}_all" --proto_ref_frames 2)
  fi
elif [ "$EVERY" != 0 ]; then
  if [ "$EVERY" = 1 ]; then MP="$OUT/map${TAG}_all"; O="$OUT/${M}${TAG}_all"; else MP="$OUT/map${TAG}_every${EVERY}w${WIN}"; O="$OUT/${M}${TAG}_every${EVERY}w${WIN}"; fi
  mkdir -p "$MP" "$O"
  ls data/video1/images | grep -E '\.(jpg|jpeg|png)$' | sort > "$MP/all.txt"
  awk -v s="$EVERY" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' "$MP/all.txt" > "$MP/frames.txt"
  python -c "import json,sys; d=sys.argv[1]; a=[l.strip() for l in open(d+'/all.txt') if l.strip()]; m=set(l.strip() for l in open(d+'/frames.txt')); json.dump({'test_files':[x for x in a if x not in m],'centre_files':[]}, open(d+'/split.json','w'))" "$MP"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < "$MP/all.txt") (every ${EVERY} of all frames), the other $(( $(wc -l < "$MP/all.txt") - $(wc -l < "$MP/frames.txt") )) are queries, pair window $WIN"
  EXTRA=(--split_json "$MP/split.json")
  if [ "$EVERY" != 1 ]; then
    if [ ! -f "$OUT/map${TAG}_all/scene.pkl" ]; then
      echo "the mapping of all frames, the reference of the sparse maps, is missing: it is made first (EVERY=1)"
      TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
    fi
    [ -f "$OUT/map${TAG}_all/scene.pkl" ] || { echo "no reference mapping at $OUT/map${TAG}_all, stopped"; exit 1; }
    EXTRA+=(--proto_ref_scene "$OUT/map${TAG}_all" --proto_ref_frames 2)
  fi
else
  if [ "$STRIDE" = 1 ] && [ "$WIN" = 5 ]; then MP="$OUT/map$TAG"; O="$OUT/$M$TAG"; else MP="$OUT/map${TAG}_s${STRIDE}w${WIN}"; O="$OUT/${M}${TAG}_s${STRIDE}w${WIN}"; fi
  mkdir -p "$MP" "$O"
  awk -v s="$STRIDE" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' data/video1/train_files.txt > "$MP/frames.txt"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < data/video1/train_files.txt) (every ${STRIDE}), pair window $WIN"
fi
if [ ! -f "$MP/mapping_seconds" ]; then
  T0=$(date +%s)
  python swimm3r_colmap.py --imagedir data/video1/images --outdir "$MP" --finetuned_model "$LORA" --physics_model "$PHYS" \
    --train_test_split none --phys_seathru_betaB_le_betaD --save_medium_maps --subset_list "$MP/frames.txt" --winsize "$WIN" --phys_beta_D_scale 0.6 1.0 1.0
  echo $(( $(date +%s) - T0 )) > "$MP/mapping_seconds"
fi
if [ "$EVERY" = 1 ]; then echo "mapping of all frames done in $(cat "$MP/mapping_seconds") s, the reference of the EVERY=N runs: $MP"; exit 0; fi
python tools/stage1_splatt3r.py --scene_dir "$MP" --data_dir data/video1 --lora "$LORA" --physics "$PHYS" \
  --head "$H" "${LOC[@]}" --medium field --out_dir "$O/loc" --bench \
  --bench_mapping_seconds "$(cat "$MP/mapping_seconds")" ${EXTRA[@]+"${EXTRA[@]}"} "${@:3}"
