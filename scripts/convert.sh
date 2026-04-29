#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl

LAFAN_ROOT="${LAFAN_ROOT:-./data/LAFAN}"
DATA_DIR="${DATA_DIR:-${LAFAN_ROOT}/g1}"
INPUT_FPS="${INPUT_FPS:-30}"
OUTPUT_FPS="${OUTPUT_FPS:-50}"
RENDER="${RENDER:-False}"

LAFAN_ROOT="${LAFAN_ROOT%/}"
DATA_DIR="${DATA_DIR%/}"

wandb_motion_exists() {
  local name="$1"

  [[ -d ./wandb ]] || return 1

  if find ./wandb -type f -name '*.npz' -path "*${name}*" -print -quit \
    | grep -q .; then
    return 0
  fi

  find ./wandb -type f -path '*/files/output.log' \
    -exec grep -Fq "Motion saved to wandb registry: motions/${name}" {} \; \
    -print -quit | grep -q .
}

converted=0
skipped=0

while IFS= read -r -d '' csv; do
  rel="${csv#${LAFAN_ROOT}/}"
  name="${rel%.csv}"
  name="${name//\//_}"

  if wandb_motion_exists "$name"; then
    echo "[SKIP] ${name} already exists in wandb."
    ((skipped += 1))
    continue
  fi

  echo "[CONVERT] ${csv} -> ${name}"
  uv run -m mjlab.scripts.csv_to_npz \
    --input-file "$csv" \
    --output-name "$name" \
    --input-fps "$INPUT_FPS" \
    --output-fps "$OUTPUT_FPS" \
    --render "$RENDER"
  ((converted += 1))
done < <(find "$DATA_DIR" -type f -name '*.csv' -print0)

echo "Done. Converted: ${converted}. Skipped: ${skipped}."
