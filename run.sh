#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v17}"
OUT="${2:-out}"
case "$M" in
  v17) W0=ckpt/joint_s3r_v17/weights; H0=ckpt/head_s3r_v17/final_splatt3r.pth; E0=0015 ;;
  v18e10) W0=ckpt/joint_s3r_v18e10/weights; H0=ckpt/head_s3r_v18e10/final_splatt3r.pth; E0=0009 ;;
  splatt3r) W0=""; H0=""; E0="" ;;
  v13c) W0=ckpt/joint_s3r_v12/weights; H0=ckpt/head_s3r_v13c/final_splatt3r.pth; E0=0019 ;;
  *) W0=""; H0=""; E0=0019 ;;
esac
W="${WEIGHTS:-$W0}"
if [ "$M" = splatt3r ]; then
  LORA=""; PHYS=""
else
  LORA="${LORA:-$W/${E0}_lora.pth}"
  PHYS="${PHYS:-$W/${E0}_physics.pth}"
fi
H="${HEAD:-$H0}"
TAG0="${TAG:-}"
TAG="${TAG:+_$TAG}"
SCENE="${SCENE:-data/video1}"
SCENE="${SCENE%/}"
LOC=(--anchor_px 6 --min_conf_thr 0.5 --fps 5 --medium_raster s2 --binf_veil)
case "$M" in
  v17|v18e10|v13c|splatt3r) ;;
  *) echo "usage: [TAG=name LORA=path PHYS=path HEAD=path EVERY=n HOLD=n REF=poses SCENE=folder FIXED_SPLIT=1] run.sh [v17|v13c] [out_dir] [benchmark options, --proto for the localization protocol, --gscpe for the GS-CPE protocol]"; echo "EVERY=1 maps all frames as the reference, EVERY=4 maps every 4th of all frames and localizes the others"; echo "HOLD=12 BLOCK=3 holds out three consecutive frames in the middle of every 12 as queries (HOLD=4 every 4th frame). HOLD_MAP=register (default): Stage 1 maps the other frames alone and every query is registered into that fixed map for its reference pose; moved: the reference comes from another reconstruction of all frames (REF, or the mapping of all frames) through a similarity; same: one reconstruction of all frames, the queries left out of the Gaussian map only"; echo "SCENE=data/sealoc_s1 (default data/video1) takes the images and splits of another scene folder, its name is added to the output folders. FIXED_SPLIT=1 maps the train_files.txt of the scene and localizes the test_files of its split.json, the reference REF of all frames is moved into the map through a similarity on the map frames (HOLD_MAP=moved)"; exit 1 ;;
esac
if [ "$(cd "$SCENE" 2>/dev/null && pwd -P)" != "$(cd "$D/data/video1" 2>/dev/null && pwd -P)" ]; then
  [ -d "$SCENE/images" ] || { echo "no images at $SCENE/images (SCENE=$SCENE)"; exit 1; }
  TAG="_$(basename "$SCENE")$TAG"
fi
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
  ls "$SCENE/images" | grep -E '\.(jpg|jpeg|png)$' | sort > "$SD/all.txt"
  awk -v s="$HOLD" -v b="$BLOCK" '{a[NR]=$0} END{c=int((s-b)/2); for(i=1;i<=NR;i++){r=(i-1)%s; h=(b==1)?(i%s==0):(r>=c && r<c+b); if (i==1 || i==NR || (b>1 && i-r+s-1>NR)) h=0; if (!h) print a[i]}}' "$SD/all.txt" > "$SD/frames.txt"
  python -c "import json,sys; d=sys.argv[1]; a=[l.strip() for l in open(d+'/all.txt') if l.strip()]; m=set(l.strip() for l in open(d+'/frames.txt')); json.dump({'test_files':[x for x in a if x not in m],'centre_files':[]}, open(d+'/split.json','w'))" "$SD"
  NQ=$(( $(wc -l < "$SD/all.txt") - $(wc -l < "$SD/frames.txt") ))
  SPLIT="$BLOCK consecutive frames of every $HOLD are queries, test $BLOCK : train $((HOLD - BLOCK))"
  if [ "$HM" = same ]; then
    echo "map frames: $(wc -l < "$SD/frames.txt") of $(wc -l < "$SD/all.txt") ($SPLIT), $NQ queries; one Stage 1 reconstruction of all frames gives every pose, the queries are left out of the Gaussian map, the medium field and the retrieval store, and their poses there are the reference (the queries do take part in that reconstruction)"
    if [ ! -f "$OUT/map_${M}${TAG}_all/scene.pkl" ]; then
      echo "the Stage 1 reconstruction of all frames is missing: it is made first (EVERY=1)"
      HOLD=0 REF= SCENE="$SCENE" TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
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
    [ "${TRT:-0}" = 1 ] && EXTRA+=(--trt_encoder)
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
        HOLD=0 REF= SCENE="$SCENE" TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
      fi
      [ -f "$OUT/map_${M}${TAG}_all/scene.pkl" ] || { echo "no reference mapping at $OUT/map_${M}${TAG}_all, stopped"; exit 1; }
      EXTRA+=(--proto_ref_scene "$OUT/map_${M}${TAG}_all" --proto_ref_frames 2)
    fi
  fi
elif [ "${FIXED_SPLIT:-0}" = 1 ]; then
  [ -f "$SCENE/train_files.txt" ] && [ -f "$SCENE/split.json" ] || { echo "FIXED_SPLIT=1 needs $SCENE/train_files.txt and $SCENE/split.json"; exit 1; }
  [ "${HOLD_MAP:-moved}" = moved ] || { echo "FIXED_SPLIT=1 takes HOLD_MAP=moved only, registration by frame number does not hold across visits"; exit 1; }
  [ -n "$REF" ] && [ -e "$REF" ] || { echo "FIXED_SPLIT=1 needs REF, the reference poses of the map frames and the queries (REF=${REF:-unset})"; exit 1; }
  MP="$OUT/map_${M}${TAG}_fixed"
  O="$OUT/${M}${TAG}_fixed_moved"
  mkdir -p "$MP" "$O"
  cp "$SCENE/train_files.txt" "$MP/frames.txt"
  cp "$SCENE/split.json" "$MP/split.json"
  echo "mapping frames $(wc -l < "$MP/frames.txt") (train_files.txt of $SCENE), queries are the test_files of $SCENE/split.json, pair window $WIN, the reference $REF is moved into the map by a similarity fitted on the map frames"
  EXTRA=(--split_json "$MP/split.json" --proto_ref_poses "$REF" --proto_ref_frames 2)
  if [ "${TRT:-0}" = 1 ]; then EXTRA+=(--trt_encoder); fi
elif [ "$EVERY" != 0 ]; then
  if [ "$EVERY" = 1 ]; then MP="$OUT/map_${M}${TAG}_all"; O="$OUT/${M}${TAG}_all"; else MP="$OUT/map_${M}${TAG}_every${EVERY}w${WIN}"; O="$OUT/${M}${TAG}_every${EVERY}w${WIN}"; fi
  mkdir -p "$MP" "$O"
  ls "$SCENE/images" | grep -E '\.(jpg|jpeg|png)$' | sort > "$MP/all.txt"
  awk -v s="$EVERY" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' "$MP/all.txt" > "$MP/frames.txt"
  python -c "import json,sys; d=sys.argv[1]; a=[l.strip() for l in open(d+'/all.txt') if l.strip()]; m=set(l.strip() for l in open(d+'/frames.txt')); json.dump({'test_files':[x for x in a if x not in m],'centre_files':[]}, open(d+'/split.json','w'))" "$MP"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < "$MP/all.txt") (every ${EVERY} of all frames), the other $(( $(wc -l < "$MP/all.txt") - $(wc -l < "$MP/frames.txt") )) are queries, pair window $WIN"
  EXTRA=(--split_json "$MP/split.json")
  if [ "$EVERY" != 1 ]; then
    if [ ! -f "$OUT/map_${M}${TAG}_all/scene.pkl" ]; then
      echo "the mapping of all frames, the reference of the sparse maps, is missing: it is made first (EVERY=1)"
      SCENE="$SCENE" TAG="$TAG0" WIN=5 EVERY=1 bash "$D/run.sh" "$M" "$OUT"
    fi
    [ -f "$OUT/map_${M}${TAG}_all/scene.pkl" ] || { echo "no reference mapping at $OUT/map_${M}${TAG}_all, stopped"; exit 1; }
    EXTRA+=(--proto_ref_scene "$OUT/map_${M}${TAG}_all" --proto_ref_frames 2)
  fi
else
  if [ "$STRIDE" = 1 ] && [ "$WIN" = 5 ]; then MP="$OUT/map_${M}${TAG}"; O="$OUT/$M$TAG"; else MP="$OUT/map_${M}${TAG}_s${STRIDE}w${WIN}"; O="$OUT/${M}${TAG}_s${STRIDE}w${WIN}"; fi
  mkdir -p "$MP" "$O"
  awk -v s="$STRIDE" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' "$SCENE/train_files.txt" > "$MP/frames.txt"
  echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < "$SCENE/train_files.txt") (every ${STRIDE}), pair window $WIN"
fi
if [ "$M" = splatt3r ]; then
  SH="${SPLATT3R_HEAD:-ckpt/splatt3r/gaussian_head.pth}"
  [ -f "$SH" ] || { echo "missing $SH"; exit 1; }
  MAPARGS=(--pure_mast3r)
  HEADARGS=(--lora "" --physics "" --splatt3r_ckpt "$SH" --no_water)
  LOC=(--min_conf_thr 0.5 --fps 5)
  echo "model: public Splatt3R (public MASt3R backbone, public Splatt3R Gaussian head, no water model), drawn by the same rasteriser as our map"
else
  for f in "$LORA" "$PHYS" "$H" "$(dirname "$(dirname "$LORA")")/train_encoder_args.json"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
  MAPARGS=(--finetuned_model "$LORA")
  HEADARGS=(--lora "$LORA" --physics "$PHYS" --head "$H")
fi
if [ ! -f "$MP/mapping_seconds" ]; then
  echo "mapping: Stage 1 of the map frames with the backbone $LORA, no descatter (the Gaussian head reads the observed colour and the water of the physics head)"
  T0=$(date +%s)
  python swimm3r_colmap.py --imagedir "$SCENE/images" --outdir "$MP" "${MAPARGS[@]}" --physics_model "" \
    --train_test_split none --subset_list "$MP/frames.txt" --winsize "$WIN"
  echo $(( $(date +%s) - T0 )) > "$MP/mapping_seconds"
fi
if [ "$EVERY" = 1 ]; then echo "mapping of all frames done in $(cat "$MP/mapping_seconds") s, the reference of the EVERY=N runs: $MP"; exit 0; fi
MED=(--medium field)
[ "$M" = splatt3r ] && MED=()
python tools/stage1_splatt3r.py --scene_dir "$MP" --data_dir "$SCENE" "${HEADARGS[@]}" "${LOC[@]}" ${MED[@]+"${MED[@]}"} --out_dir "$O/loc" --bench \
  --bench_mapping_seconds "$(cat "$MP/mapping_seconds")" ${EXTRA[@]+"${EXTRA[@]}"} "${@:3}"
