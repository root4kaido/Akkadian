#!/bin/bash

# Akkadianコンペ用モデルアップロードスクリプト (Kaggle Models API)
# Usage: ./upload_model.sh <experiment_name> [--models model1,model2,...] [--variation NAME] [--kaggle-name NAME]
# Examples:
#   ./upload_model.sh exp023_full_preprocessing                                            # best_model + last_model
#   ./upload_model.sh exp023_full_preprocessing --models soup_model --variation exp023_soup # soup_modelを別variationで
#   ./upload_model.sh exp023_full_preprocessing --models fold3/last_model --variation exp023_fold3
#   ./upload_model.sh exp023_full_preprocessing --kaggle-name MyModels                     # Kaggleモデル名を変更

set -e

# === Parse arguments ===
EXP_NAME=""
MODEL_DIRS=""
MODEL_NAME="AkkadianModels"
VARIATION_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)
            MODEL_DIRS="$2"
            shift 2
            ;;
        --variation)
            VARIATION_NAME="$2"
            shift 2
            ;;
        --kaggle-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$EXP_NAME" ]; then
                EXP_NAME="$1"
            fi
            shift
            ;;
    esac
done

# Default variation name = experiment name
if [ -z "$VARIATION_NAME" ]; then
    VARIATION_NAME="$EXP_NAME"
fi

if [ -z "$EXP_NAME" ]; then
    echo "Error: Experiment name required."
    echo "Usage: ./upload_model.sh <experiment_name> [--models model1,model2,...] [--kaggle-name NAME]"
    echo ""
    echo "Examples:"
    echo "  ./upload_model.sh exp023_full_preprocessing"
    echo "  ./upload_model.sh exp023_full_preprocessing --models soup_model"
    echo "  ./upload_model.sh exp023_full_preprocessing --models fold3/best_model,soup_model"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP_DIR="${PROJECT_ROOT}/workspace/${EXP_NAME}"
DICT_FILE="${EXP_DIR}/dataset/form_type_dict.json"
SRC_DIR="${PROJECT_ROOT}/workspace/exp007_mbr_postprocess/src"
FRAMEWORK="pyTorch"

# Default: best_model + last_model
if [ -z "$MODEL_DIRS" ]; then
    MODEL_DIRS="best_model,last_model"
fi

# Get Kaggle username
KAGGLE_USERNAME=""
if [ -f ~/.kaggle/kaggle.json ]; then
    KAGGLE_USERNAME=$(cat ~/.kaggle/kaggle.json | grep "username" | cut -d'"' -f4)
fi
if [ -z "$KAGGLE_USERNAME" ]; then
    echo "Enter your Kaggle username:"
    read KAGGLE_USERNAME
fi
echo "Kaggle username: ${KAGGLE_USERNAME}"

# Create temp directory (within project to comply with no-/tmp rule)
TEMP_DIR="${PROJECT_ROOT}/.tmp/kaggle-model-upload-${MODEL_NAME}-${VARIATION_NAME}"
rm -rf "${TEMP_DIR}"
MODEL_META_DIR="${TEMP_DIR}/model"
VARIATION_DIR="${TEMP_DIR}/variation"
UPLOAD_DIR="${VARIATION_DIR}/${VARIATION_NAME}"
mkdir -p "${MODEL_META_DIR}" "${UPLOAD_DIR}"

# Copy model files
copy_model_files() {
    local src_dir=$1
    local dst_dir=$2
    local name=$3
    if [ ! -d "$src_dir" ]; then
        echo "  WARNING: ${name} not found at ${src_dir}, skipping."
        return 1
    fi
    echo "Copying ${name}..."
    mkdir -p "${dst_dir}"
    cp -v "${src_dir}"/config.json "${dst_dir}/"
    cp -v "${src_dir}"/model.safetensors "${dst_dir}/"
    cp -v "${src_dir}"/tokenizer_config.json "${dst_dir}/"
    cp -v "${src_dir}"/special_tokens_map.json "${dst_dir}/"
    cp -v "${src_dir}"/added_tokens.json "${dst_dir}/"
    cp -v "${src_dir}"/generation_config.json "${dst_dir}/"
    return 0
}

COPIED=0
IFS=',' read -ra MODELS <<< "$MODEL_DIRS"
for mdl in "${MODELS[@]}"; do
    mdl=$(echo "$mdl" | xargs)  # trim whitespace
    if copy_model_files "${EXP_DIR}/results/${mdl}" "${UPLOAD_DIR}/${mdl}" "${mdl}"; then
        COPIED=$((COPIED + 1))
    fi
done

if [ "$COPIED" -eq 0 ]; then
    echo "Error: No models found to upload."
    rm -rf "${TEMP_DIR}"
    exit 1
fi

# Copy form_type_dict.json (PN/GN tags)
if [ -f "$DICT_FILE" ]; then
    echo "Copying form_type_dict.json..."
    cp -v "${DICT_FILE}" "${UPLOAD_DIR}/"
fi

# Copy utility scripts (repeat_cleanup etc.)
if [ -d "$SRC_DIR" ]; then
    echo "Copying utility scripts..."
    mkdir -p "${UPLOAD_DIR}/src"
    cp -v "${SRC_DIR}/infer_mbr.py" "${UPLOAD_DIR}/src/" 2>/dev/null || true
fi

echo ""
echo "Files to upload:"
find "${UPLOAD_DIR}" -type f | sort
echo ""
TOTAL_SIZE=$(du -sh "${UPLOAD_DIR}" | cut -f1)
echo "Total size: ${TOTAL_SIZE}"

# Check if model already exists
MODEL_EXISTS=false
if kaggle models list --owner ${KAGGLE_USERNAME} 2>/dev/null | grep -q "${KAGGLE_USERNAME}/${MODEL_NAME}"; then
    MODEL_EXISTS=true
    echo "Model ${KAGGLE_USERNAME}/${MODEL_NAME} already exists."
fi

# Create model if needed
if [ "$MODEL_EXISTS" = false ]; then
    echo "Creating new model: ${KAGGLE_USERNAME}/${MODEL_NAME}"
    cat > "${MODEL_META_DIR}/model-metadata.json" << EOL
{
  "title": "${MODEL_NAME}",
  "id": "${KAGGLE_USERNAME}/${MODEL_NAME}",
  "ownerSlug": "${KAGGLE_USERNAME}",
  "slug": "${MODEL_NAME}",
  "isPrivate": true,
  "licenses": [{"name": "Apache 2.0"}],
  "description": "Akkadian-to-English translation models for Deep Past Challenge",
  "collaborators": []
}
EOL
    kaggle models create -p "${MODEL_META_DIR}"
fi

# Create variation metadata
echo "Preparing variation metadata..."
kaggle models instances init -p "${VARIATION_DIR}" 2>/dev/null || true
cat > "${VARIATION_DIR}/model-instance-metadata.json" << EOL
{
  "id": "${KAGGLE_USERNAME}/${MODEL_NAME}/${FRAMEWORK}/${VARIATION_NAME}",
  "instanceSlug": "${VARIATION_NAME}",
  "modelId": "${KAGGLE_USERNAME}/${MODEL_NAME}",
  "ownerSlug": "${KAGGLE_USERNAME}",
  "modelSlug": "${MODEL_NAME}",
  "framework": "${FRAMEWORK}",
  "slug": "${VARIATION_NAME}",
  "title": "${VARIATION_NAME}",
  "modelVersionName": "1.0",
  "modelVersionDescription": "Model from ${EXP_NAME} (${VARIATION_NAME}) for Akkadian-to-English translation",
  "isPrivate": true,
  "licenseName": "Apache 2.0"
}
EOL

# Upload
echo "Uploading variation ${VARIATION_NAME}..."
if kaggle models instances create -p "${VARIATION_DIR}" --dir-mode zip; then
    echo "Variation created successfully."
else
    echo "Variation exists, updating..."
    kaggle models instances versions create -p "${VARIATION_DIR}" \
        --version-notes "Updated ${VARIATION_NAME}" \
        --dir-mode zip \
        "${KAGGLE_USERNAME}/${MODEL_NAME}/${FRAMEWORK}/${VARIATION_NAME}"
fi

# Cleanup
rm -rf "${TEMP_DIR}"

echo ""
echo "==================================================="
echo "Upload complete!"
echo "  ${KAGGLE_USERNAME}/${MODEL_NAME}/${FRAMEWORK}/${VARIATION_NAME}"
echo "  https://www.kaggle.com/models/${KAGGLE_USERNAME}/${MODEL_NAME}"
echo "==================================================="
