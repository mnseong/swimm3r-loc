#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
W=ckpt/joint_s3r_v3/weights
O="${1:-out}"
mkdir -p "$O"
if [ ! -f "$O/map/mapping_seconds" ]; then
  T0=$(date +%s)
  python swimm3r_colmap.py --imagedir data/video1/images --outdir "$O/map" --finetuned_model $W/0019_lora.pth --physics_model $W/0019_physics.pth \
    --train_test_split none --phys_seathru_betaB_le_betaD --save_medium_maps --subset_list data/video1/train_files.txt
  echo $(( $(date +%s) - T0 )) > "$O/map/mapping_seconds"
fi
python tools/stage1_splatt3r.py --scene_dir "$O/map" --data_dir data/video1 --lora $W/0019_lora.pth --physics $W/0019_physics.pth \
  --head $W/0019_splatt3r.pth --min_conf_thr 0.2 --medium nearest --out_dir "$O/loc" --bench \
  --bench_mapping_seconds "$(cat "$O/map/mapping_seconds")" "${@:2}"
