#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$D"
export PYTHONPATH="$D/ubs/submodules:${PYTHONPATH:-}"
W=ckpt/joint_s3r_v8/weights
O="${1:-out}"
T0=$(date +%s)
python swimm3r_colmap.py --imagedir data/video1/images --outdir "$O/map" --finetuned_model $W/0019_lora.pth --physics_model $W/0019_physics.pth \
  --train_test_split none --phys_seathru_betaB_le_betaD --phys_beta_D_scale 0.6 1.0 1.0 --save_medium_maps --subset_list data/video1/train_files.txt
T1=$(date +%s)
python tools/stage1_splatt3r.py --scene_dir "$O/map" --data_dir data/video1 --lora $W/0019_lora.pth --physics $W/0019_physics.pth \
  --head $W/0019_splatt3r.pth --anchor_px 8 --min_conf_thr 0.5 --medium nearest --fps 5 --out_dir "$O/loc"
T2=$(date +%s)
echo "mapping $((T1 - T0)) s, map and localization of the held out frames $((T2 - T1)) s, scores in $O/loc/scores.json"
