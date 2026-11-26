#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v13c}"
OUT="${2:-out}"
case "$M" in
  v13c) W0=ckpt/joint_s3r_v12/weights; H0=ckpt/head_s3r_v13c/final_splatt3r.pth ;;
  *) W0=""; H0="" ;;
esac
W="${WEIGHTS:-$W0}"
LORA="${LORA:-$W/0019_lora.pth}"
PHYS="${PHYS:-$W/0019_physics.pth}"
H="${HEAD:-$H0}"
TAG0="${TAG:-}"
TAG="${TAG:+_$TAG}"
LOC=(--anchor_px 6 --min_conf_thr 0.5 --fps 5 --medium_raster s2 --binf_veil)
case "$M" in
  v13c) ;;
  *) echo "usage: [TAG=name LORA=path PHYS=path HEAD=path EVERY=n HOLD=n REF=poses] run.sh [v13c] [out_dir] [benchmark options, --proto for the localization protocol, --gscpe for the GS-CPE protocol]"; echo "EVERY=1 maps all frames as the reference, EVERY=4 maps every 4th of all frames and localizes the others"; echo "HOLD=12 BLOCK=3 holds out three consecutive frames in the middle of every 12 as queries (HOLD=4 every 4th frame). HOLD_MAP=register (default): Stage 1 maps the other frames alone and every query is registered into that fixed map for its reference pose; moved: the reference comes from another reconstruction of all frames (REF, or the mapping of all frames) through a similarity; same: one reconstruction of all frames, the queries left out of the Gaussian map only"; exit 1 ;;
esac
export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}"
mkdir -p "$D/.cache/torch" "$D/.cache/tmp" "$D/.cache/mpl"
export TORCH_HOME="$D/.cache/torch" TMPDIR="$D/.cache/tmp" MPLCONFIGDIR="$D/.cache/mpl"
if ! python -c "import sys; sys.path.insert(0, '$D/tools/csrc'); import torch, water_gn_ext as w; assert w.has_cuda() or not torch.cuda.is_available()" 2>/dev/null; then
  command -v nvcc >/dev/null || export PATH="/usr/local/cuda/bin:$PATH"
  echo "building the C++ steps of the water aligned localization (tools/csrc), once"
  (cd "$D/tools/csrc" && rm -rf build && python setup.py build_ext --inplace > "$D/.cache/water_gn_build.log" 2>&1) \
    && echo "built: $(python -c "import sys; sys.path.insert(0, '$D/tools/csrc'); import torch; import water_gn_ext as w; print('with CUDA' if w.has_cuda() else 'for the CPU only')")" \
    || echo "the C++ steps did not build (see .cache/water_gn_build.log), the PyTorch steps are used"
fi
STRIDE="${STRIDE:-1}"
WIN="${WIN:-5}"
EVERY="${EVERY:-0}"
HOLD="${HOLD:-0}"
REF="${REF:-}"
EXTRA=()
if [ "$HOLD" != 0 ]; then
  [ "$HOLD" -ge 2 ] || { echo "HOLD must be 2 or more"; exit 1; }
  HM="${HOLD_MAP:-register}"
  case "$HM" in register|moved|same) ;; *) echo "HOLD_MAP must be register, moved or same"; exit 1 ;; esac
  BLOCK="${BLOCK:-1}"
  [ "$BLOCK" -ge 1 ] && [ "$BLOCK" -lt "$HOLD" ] || { echo "BLOCK must be 1 or more and below HOLD"; exit 1; }
  HB="$HOLD"
  [ "$BLOCK" = 1 ] || HB="${HOLD}b${BLOCK}"
  SD="$OUT/map_${M}${TAG}_hold${HB}w${WIN}"
  RS=""
  [ "$HM" = register ] && [ "${REF_MATCHER:-ours}" = loftr ] && RS="_loftr"
  case "$HM" in register) O="$OUT/${M}${TAG}_hold${HB}w${WIN}${RS}" ;; moved) O="$OUT/${M}${TAG}_hold${HB}w${WIN}_moved" ;; same) SD="$OUT/split_${M}${TAG}_hold${HB}"; O="$OUT/${M}${TAG}_hold${HB}_same" ;; esac
  mkdir -p "$SD" "$O"
  ls data/video1/images | grep -E '\.(jpg|jpeg|png)$' | sort > "$SD/all.txt"
  awk -v s="$HOLD" -v b="$BLOCK" '{a[NR]=$0} END{c=int((s-b)/2); for(i=1;i<=NR;i++){r=(i-1)%s; h=(b==1)?(i%s==0):(r>=c && r<c+b); if (i==1 || i==NR || (b>1 && i-r+s-1>NR)) h=0; if (!h) print a[i]}}' "$SD/all.txt" > "$SD/frames.txt"
  python -c "import json,sys; d=sys.argv[1]; a=[l.strip() for l in open(d+'/all.txt') if l.strip()]; m=set(l.strip() for l in open(d+'/frames.txt')); json.dump({'test_files':[x for x in a if x not in m],'centre_files':[]}, open(d+'/split.json','w'))" "$SD"
  NQ=$(( $(wc -l < "$SD/all.txt") - $(wc -l < "$SD/frames.txt") ))
  SPLIT="$BLOCK consecutive frames of every $HOLD are queries, test $BLOCK : train $((HOLD - BLOCK))"
  if [ "$HM" = same ]; then
    echo "map frames: $(wc -l < "$SD/frames.txt") of $(wc -l < "$SD/all.txt") ($SPLIT), $NQ queries; one Stage 1 reconstruction of all frames gives every pose, the queries are left out of the Gaussian map, the medium field and the retrieval store, and their poses there are the reference (the queries do take part in that reconstruction)"
    if [ ! -f "$OUT/map_${M}${TAG}_all/scene.pkl" ]; then
      echo "the Stage 1 reconstruction of all frames is missing: it is made first (EVERY=1)"
      HOLD=0 REF= TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
    fi
    [ -f "$OUT/map_${M}${TAG}_all/scene.pkl" ] || { echo "no reconstruction of all frames at $OUT/map_${M}${TAG}_all, stopped"; exit 1; }
    MP="$OUT/map_${M}${TAG}_all"
    EXTRA=(--split_json "$SD/split.json" --map_holdout "$SD/split.json" --proto_ref_frames 2)
  elif [ "$HM" = register ]; then
    MP="$SD"
    echo "mapping frames: $(wc -l < "$SD/frames.txt") of $(wc -l < "$SD/all.txt") ($SPLIT), $NQ queries, pair window $WIN; Stage 1 sees the map frames alone, every query is then registered into that fixed map for its reference pose"
    RM="${REF_MATCHER:-ours}"
    case "$RM" in ours|loftr) ;; *) echo "REF_MATCHER must be ours or loftr"; exit 1 ;; esac
    if [ "$RM" = loftr ] && ! python -c "import kornia.feature" 2>/dev/null; then
      echo "REF_MATCHER=loftr registers the queries with LoFTR of kornia, which this environment lacks: pip install kornia (then run again, the map is kept)"; exit 1
    fi
    echo "reference of the queries: registration into the fixed map, matcher $RM"
    EXTRA=(--split_json "$SD/split.json" --ref_register "${REF_K:-6}" --ref_matcher "$RM" --proto_ref_frames 2)
  else
    MP="$SD"
    echo "mapping frames: $(wc -l < "$SD/frames.txt") of $(wc -l < "$SD/all.txt") ($SPLIT), $NQ queries, pair window $WIN; the reference is moved in from another reconstruction of all frames"
    EXTRA=(--split_json "$SD/split.json")
    if [ -n "$REF" ]; then
      [ -e "$REF" ] || { echo "no reference reconstruction at $REF, stopped"; exit 1; }
      EXTRA+=(--proto_ref_poses "$REF" --proto_ref_frames 2)
    else
      if [ ! -f "$OUT/map_${M}${TAG}_all/scene.pkl" ]; then
        echo "the mapping of all frames, the reference of the queries, is missing: it is made first (EVERY=1)"
        HOLD=0 REF= TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
      fi
      [ -f "$OUT/map_${M}${TAG}_all/scene.pkl" ] || { echo "no reference mapping at $OUT/map_${M}${TAG}_all, stopped"; exit 1; }
      EXTRA+=(--proto_ref_scene "$OUT/map_${M}${TAG}_all" --proto_ref_frames 2)
    fi
  fi
elif [ "$EVERY" != 0 ]; then
  if [ "$EVERY" = 1 ]; then MP="$OUT/map_${M}${TAG}_all"; O="$OUT/${M}${TAG}_all"; else MP="$OUT/map_${M}${TAG}_every${EVERY}w${WIN}"; O="$OUT/${M}${TAG}_every${EVERY}w${WIN}"; fi
  mkdir -p "$MP" "$O"
  ls data/video1/images | grep -E '\.(jpg|jpeg|png)$' | sort > "$MP/all.txt"
  awk -v s="$EVERY" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' "$MP/all.txt" > "$MP/frames.txt"
  python -c "import json,sys; d=sys.argv[1]; a=[l.strip() for l in open(d+'/all.txt') if l.strip()]; m=set(l.strip() for l in open(d+'/frames.txt')); json.dump({'test_files':[x for x in a if x not in m],'centre_files':[]}, open(d+'/split.json','w'))" "$MP"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < "$MP/all.txt") (every ${EVERY} of all frames), the other $(( $(wc -l < "$MP/all.txt") - $(wc -l < "$MP/frames.txt") )) are queries, pair window $WIN"
  EXTRA=(--split_json "$MP/split.json")
  if [ "$EVERY" != 1 ]; then
    if [ ! -f "$OUT/map_${M}${TAG}_all/scene.pkl" ]; then
      echo "the mapping of all frames, the reference of the sparse maps, is missing: it is made first (EVERY=1)"
      TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
    fi
    [ -f "$OUT/map_${M}${TAG}_all/scene.pkl" ] || { echo "no reference mapping at $OUT/map_${M}${TAG}_all, stopped"; exit 1; }
    EXTRA+=(--proto_ref_scene "$OUT/map_${M}${TAG}_all" --proto_ref_frames 2)
  fi
else
  if [ "$STRIDE" = 1 ] && [ "$WIN" = 5 ]; then MP="$OUT/map_${M}${TAG}"; O="$OUT/$M$TAG"; else MP="$OUT/map_${M}${TAG}_s${STRIDE}w${WIN}"; O="$OUT/${M}${TAG}_s${STRIDE}w${WIN}"; fi
  mkdir -p "$MP" "$O"
  awk -v s="$STRIDE" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' data/video1/train_files.txt > "$MP/frames.txt"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < data/video1/train_files.txt) (every ${STRIDE}), pair window $WIN"
fi
for f in "$LORA" "$PHYS" "$H" "$(dirname "$(dirname "$LORA")")/train_encoder_args.json"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
if [ ! -f "$MP/mapping_seconds" ]; then
  echo "mapping: Stage 1 of the map frames with the backbone $LORA, no descatter (the Gaussian head reads the observed colour and the water of the physics head)"
  T0=$(date +%s)
  python swimm3r_colmap.py --imagedir data/video1/images --outdir "$MP" --finetuned_model "$LORA" --physics_model "" \
    --train_test_split none --subset_list "$MP/frames.txt" --winsize "$WIN"
  echo $(( $(date +%s) - T0 )) > "$MP/mapping_seconds"
fi
if [ "$EVERY" = 1 ]; then echo "mapping of all frames done in $(cat "$MP/mapping_seconds") s, the reference of the EVERY=N runs: $MP"; exit 0; fi
python tools/stage1_splatt3r.py --scene_dir "$MP" --data_dir data/video1 --lora "$LORA" --physics "$PHYS" \
  --head "$H" "${LOC[@]}" --medium field --out_dir "$O/loc" --bench \
  --bench_mapping_seconds "$(cat "$MP/mapping_seconds")" ${EXTRA[@]+"${EXTRA[@]}"} "${@:3}"
