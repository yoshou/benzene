#!/usr/bin/env bash
# The models' external inputs, pinned, into work/ (benzene/python/benzene/workspace.py). What is already there is
# checked (commit, checksum, Hugging Face revision) and kept:
#   source checkouts   cvg/LightGlue, verlab/accelerated_features (XFeat, with its weights), davnords/LoMa,
#                      facebookresearch/vggt, InternRobotics/AnySplat
#   weights            LightGlue v0.1_arxiv (SuperPoint, LightGlue for SuperPoint and ALIKED), ALIKED-n16, LoMa, DaD
#   Hugging Face hub   facebook/VGGT-1B, lhjiang/anysplat, nvidia/Cosmos3-Edge; the Cosmos 3 Edge benchmarks
#                      nyu-visionx/CV-Bench, lmms-lab/RealWorldQA
#   datasets           TUM RGB-D sequences (CC BY 4.0), Mip-NeRF 360 and its extra scenes (images_8, COLMAP sparse and
#                      poses_bounds.npy)
# The hf command comes from the models' venv (scripts/install-venv.sh).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
WORK="$WS/work"
HF="$VENV/bin/hf"

checkout() {  # repository commit directory
  if [[ ! -d "$3/.git" ]]; then
    git clone -q "$1" "$3"
    git -C "$3" checkout -q "$2"
  fi
  [[ "$(git -C "$3" rev-parse HEAD)" == "$2" ]] || { echo "$3: not at $2" >&2; exit 1; }
  echo "ok $3"
}

weight() {  # url sha256 file
  if [[ ! -f "$3" ]]; then
    mkdir -p "$(dirname "$3")"
    curl -sS -L -f -o "$3.part" "$1"
    mv "$3.part" "$3"
  fi
  echo "$2  $3" | sha256sum -c --quiet
  echo "ok $3"
}

hub() {  # repository type revision directory files...
  local repository=$1 type=$2 revision=$3 directory=$4
  shift 4
  "$HF" download "$repository" "$@" --repo-type "$type" --revision "$revision" --local-dir "$directory" \
    --format quiet >/dev/null
  echo "ok $directory"
}

checkout https://github.com/cvg/LightGlue.git 5a650c444718caffc2669dc0094cde908f42dd73 "$WORK/lightglue/src"
checkout https://github.com/verlab/accelerated_features.git e92685f57f8318b18725c5c8c0bd28c7fe188d9a \
  "$WORK/xfeat_lightglue/src"
checkout https://github.com/davnords/LoMa.git ecc947d44ddc9d50387e887bbedc743d65ffac00 "$WORK/loma/src"
checkout https://github.com/facebookresearch/vggt.git a288dd0f14786c93483e45524328726ab7b1b4ce "$WORK/vggt/src"
checkout https://github.com/InternRobotics/AnySplat.git 5f5e208a7dd57d52e43ea0d553a95eab526e8775 "$WORK/anysplat/src"

L=https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv
weight $L/superpoint_v1.pth 52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e \
  "$WORK/superpoint_lightglue/weights/superpoint_v1.pth"
weight $L/superpoint_lightglue.pth 6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a \
  "$WORK/superpoint_lightglue/weights/superpoint_lightglue.pth"
weight $L/aliked_lightglue.pth d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503 \
  "$WORK/aliked_lightglue/weights/aliked_lightglue.pth"
weight https://github.com/Shiaoming/ALIKED/raw/683d7c65197395c0b3f01ebe76e1084a27e73a65/models/aliked-n16.pth \
  5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2 "$WORK/aliked_lightglue/weights/aliked-n16.pth"
L=https://github.com/davnords/storage/releases/download/loma
weight $L/loma_B.pt 3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758 "$WORK/loma/weights/loma_B.pt"
weight $L/loma_B128.pth b509cb295b543e6eae37dea76185e805c4e87ff7e35e0a84cad301be1b5984fc \
  "$WORK/loma/weights/loma_B128.pth"
weight $L/loma_B64.pth bd977b672e9ae92f7acbc346acb93c6d71a0ca73dad6a29a2680f8bd28cb5cc5 \
  "$WORK/loma/weights/loma_B64.pth"
weight https://github.com/Parskatt/dad/releases/download/v0.1.0/dad.pth \
  7d827844649faf5213ba97e7adb43e16b936dd3d11ab2d0fc0d602c67fae781f "$WORK/loma/weights/dad.pth"

hub facebook/VGGT-1B model 860abec7937da0a4c03c41d3c269c366e82abdf9 "$WORK/vggt/hf/VGGT-1B" \
  README.md config.json model.safetensors
hub lhjiang/anysplat model d2e8c343672646041ad4ea518184968f94362f01 "$WORK/anysplat/hf/anysplat" \
  README.md LICENSE config.json model.safetensors
hub nvidia/Cosmos3-Edge model 6afc1256321c3a8e623d89be3bc5117a1e90f96b "$WORK/cosmos3_edge/hf/Cosmos3-Edge" \
  README.md chat_template.jinja config.json generation_config.json model.safetensors.index.json \
  preprocessor_config.json processor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json \
  video_preprocessor_config.json vision_encoder/model.safetensors transformer/config.json \
  transformer/diffusion_pytorch_model.safetensors.index.json \
  transformer/diffusion_pytorch_model-00001-of-00002.safetensors \
  transformer/diffusion_pytorch_model-00002-of-00002.safetensors
hub nyu-visionx/CV-Bench dataset bc284db50d036958861cb60cdd7b77612052ce0d "$WORK/datasets/cv_bench" \
  README.md test_2d.parquet test_3d.parquet
hub lmms-lab/RealWorldQA dataset 907c4e5228fd1703c710ed937601cb5f89ab8d5c "$WORK/datasets/realworldqa" \
  README.md data/test-00000-of-00002.parquet data/test-00001-of-00002.parquet

mkdir -p "$WORK/datasets/tum_rgbd"
cd "$WORK/datasets/tum_rgbd"
T=https://cvg.cit.tum.de/rgbd/dataset
for s in freiburg1/rgbd_dataset_freiburg1_desk freiburg1/rgbd_dataset_freiburg1_desk2 \
         freiburg1/rgbd_dataset_freiburg1_room freiburg1/rgbd_dataset_freiburg1_plant \
         freiburg1/rgbd_dataset_freiburg1_teddy freiburg1/rgbd_dataset_freiburg1_360 \
         freiburg3/rgbd_dataset_freiburg3_long_office_household freiburg3/rgbd_dataset_freiburg3_structure_texture_far \
         freiburg2/rgbd_dataset_freiburg2_desk; do
  n=$(basename "$s")
  if [[ ! -d $n ]]; then
    curl -sS -L -f -C - -o "$n.tgz" "$T/$s.tgz"
    tar xzf "$n.tgz"
    rm "$n.tgz"
  fi
  echo "ok tum_rgbd/$n"
done
mkdir -p "$WORK/datasets/mipnerf360"
cd "$WORK/datasets/mipnerf360"
M=https://storage.googleapis.com/gresearch/refraw360
for archive in "360_v2 bicycle bonsai counter garden kitchen room stump" "360_extra_scenes flowers treehill"; do
  read -r name scenes <<< "$archive"
  for scene in $scenes; do
    if [[ ! -d $scene ]]; then
      curl -sS -L -f -C - -o "$name.zip.part" "$M/$name.zip"
      unzip -q -o "$name.zip.part" '*/images_8/*' '*/sparse/*' '*/poses_bounds.npy'
      rm "$name.zip.part"
      break
    fi
  done
  for scene in $scenes; do echo "ok mipnerf360/$scene"; done
done
