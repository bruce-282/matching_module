#!/bin/bash

# Run from this directory (romatch/weights).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$PWD" != "$SCRIPT_DIR" ]]; then
    echo "Please run this script from the 'romatch/weights' directory."
    exit 1
fi

# Default names match core.matchers.models.roma.Roma (roma_outdoor.pth, dinov2_vitl14_pretrain.pth)
ROMA_OUTDOOR="https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth"
DINOV2_VITL14="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"

FILES=(
    "$ROMA_OUTDOOR"
    "$DINOV2_VITL14"
)

for FILE_URL in "${FILES[@]}"; do
    FILE_NAME=$(basename "$FILE_URL")
    if [[ -f "$FILE_NAME" ]]; then
        echo "$FILE_NAME already exists, skipping download."
    else
        echo "Downloading $FILE_NAME..."
        curl -L -O "$FILE_URL"
        if [[ $? -eq 0 ]]; then
            echo "$FILE_NAME downloaded successfully."
        else
            echo "Failed to download $FILE_NAME."
        fi
    fi
done
