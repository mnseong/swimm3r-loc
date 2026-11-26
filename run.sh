#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v9c}"
OUT="${2:-out}"
W=ckpt/joint_s3r_v8/weights
H=ckpt/head_s3r_v9c/final_splatt3r.pth
LOC=(--anchor_px 6 --min_conf_thr 0.2 --fps 5 --medium_raster preshade --binf_veil)
case "$M" in
  v9c) ;;
  v9c_best) LOC+=(--time_bucket 8 --time_fill --far_layer) ;;
  *) echo "usage: run.sh [v9c|v9c_best] [out_dir] [benchmark options]"; exit 1 ;;
esac
export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}"
STRIDE="${STRIDE:-1}"
WIN="${WIN:-5}"
if [ "$STRIDE" = 1 ] && [ "$WIN" = 5 ]; then MP="$OUT/map"; O="$OUT/$M"; else MP="$OUT/map_s${STRIDE}w${WIN}"; O="$OUT/${M}_s${STRIDE}w${WIN}"; fi
mkdir -p "$MP" "$O"
awk -v s="$STRIDE" '{a[NR]=$0} END{for(i=1;i<=NR;i+=s) print a[i]; if ((NR-1)%s!=0) print a[NR]}' data/video1/train_files.txt > "$MP/frames.txt"
echo "mapping frames: $(wc -l < "$MP/frames.txt") of $(wc -l < data/video1/train_files.txt) (every ${STRIDE}), pair window $WIN"
if [ ! -f "$MP/mapping_seconds" ]; then
  T0=$(date +%s)
  python swimm3r_colmap.py --imagedir data/video1/images --outdir "$MP" --finetuned_model $W/0019_lora.pth --physics_model $W/0019_physics.pth \
    --train_test_split none --phys_seathru_betaB_le_betaD --save_medium_maps --subset_list "$MP/frames.txt" --winsize "$WIN" --phys_beta_D_scale 0.6 1.0 1.0
  echo $(( $(date +%s) - T0 )) > "$MP/mapping_seconds"
fi
python tools/stage1_splatt3r.py --scene_dir "$MP" --data_dir data/video1 --lora $W/0019_lora.pth --physics $W/0019_physics.pth \
  --head "$H" "${LOC[@]}" --medium field --out_dir "$O/loc" --bench \
  --bench_mapping_seconds "$(cat "$MP/mapping_seconds")" "${@:3}"
