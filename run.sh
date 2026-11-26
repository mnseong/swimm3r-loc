#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
M="${1:-v3}"
O="${2:-out}/$M"
case "$M" in
  v3) W=ckpt/joint_s3r_v3/weights; MAP=(); LOC=(--min_conf_thr 0.2) ;;
  v8) W=ckpt/joint_s3r_v8/weights; MAP=(--phys_beta_D_scale 0.6 1.0 1.0); LOC=(--anchor_px 8 --min_conf_thr 0.2 --fps 5)
      export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}" ;;
  v9c) W=ckpt/joint_s3r_v8/weights; H=ckpt/head_s3r_v9c/final_splatt3r.pth; MAP=(--phys_beta_D_scale 0.6 1.0 1.0)
      LOC=(--anchor_px 6 --min_conf_thr 0.2 --fps 5 --medium_raster preshade --binf_veil)
      export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}" ;;
  v9c_best) W=ckpt/joint_s3r_v8/weights; H=ckpt/head_s3r_v9c/final_splatt3r.pth; MAP=(--phys_beta_D_scale 0.6 1.0 1.0)
      LOC=(--anchor_px 6 --min_conf_thr 0.2 --fps 5 --medium_raster preshade --binf_veil --time_bucket 8 --time_fill --far_layer)
      export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}" ;;
  *) echo "usage: run.sh [v3|v8|v9c|v9c_best] [out_dir] [benchmark options]"; exit 1 ;;
esac
H="${H:-$W/0019_splatt3r.pth}"
case "$M" in v9c|v9c_best) MD=v8 ;; *) MD="$M" ;; esac
mkdir -p "$O"
MP="${2:-out}/$MD/map"
mkdir -p "$MP"
if [ ! -f "$MP/mapping_seconds" ]; then
  T0=$(date +%s)
  python swimm3r_colmap.py --imagedir data/video1/images --outdir "$MP" --finetuned_model $W/0019_lora.pth --physics_model $W/0019_physics.pth \
    --train_test_split none --phys_seathru_betaB_le_betaD --save_medium_maps --subset_list data/video1/train_files.txt ${MAP[@]+"${MAP[@]}"}
  echo $(( $(date +%s) - T0 )) > "$MP/mapping_seconds"
fi
python tools/stage1_splatt3r.py --scene_dir "$MP" --data_dir data/video1 --lora $W/0019_lora.pth --physics $W/0019_physics.pth \
  --head "$H" "${LOC[@]}" --medium field --out_dir "$O/loc" --bench \
  --bench_mapping_seconds "$(cat "$MP/mapping_seconds")" "${@:3}"
