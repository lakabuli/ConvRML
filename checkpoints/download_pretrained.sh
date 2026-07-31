#!/usr/bin/env bash
# Download pretrained ConvRML checkpoints into this folder from Google Drive.
#
# Source folder: checkpoints folder from ConvRML google drive
#   https://drive.google.com/drive/folders/14SLe_-DuO34XiLWXBi_ExhgyKS0BUK4K?usp=sharing
#
# Usage (from repo root):
#   ./checkpoints/download_pretrained.sh
#
# Requires: pip install gdown
#
# Each model is stored as:
#   checkpoints/<run_name>/best_model.pth
# matching what infer.py loads via checkpoint.save_checkpoint_path + run_name.


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_URL="https://drive.google.com/drive/folders/14SLe_-DuO34XiLWXBi_ExhgyKS0BUK4K"

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown is required. Install with:  pip install gdown"
  exit 1
fi

echo "Downloading pretrained checkpoints into: $SCRIPT_DIR"
gdown --folder "$DRIVE_URL" -O "$SCRIPT_DIR" --remaining-ok

echo "Done. Each model should be at checkpoints/<run_name>/best_model.pth"
echo "Then run:  python infer.py --config <config.yaml>"
