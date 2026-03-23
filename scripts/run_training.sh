#!/bin/bash
# Training script for Handwritten LaTeX OCR

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM-256M-Instruct}"
DATASET_SUBSET="${DATASET_SUBSET:-human_handwrite}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WANDB_PROJECT="${WANDB_PROJECT:-handwritten-latex-ocr}"

echo "=============================================="
echo "Handwritten LaTeX OCR - Training Script"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Dataset subset: $DATASET_SUBSET"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Setup 1: Train with LaTeX_OCR only
echo ""
echo "=== SETUP 1: Training with LaTeX_OCR only ==="
echo ""

python src/train.py \
    --config configs/train_config.yaml \
    --model_name "$MODEL_NAME" \
    --subset "$DATASET_SUBSET" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --run_name "sft_latex_ocr_only" \
    --wandb_project "$WANDB_PROJECT"

# Setup 2: Train with LaTeX_OCR + MathWriting
echo ""
echo "=== SETUP 2: Training with LaTeX_OCR + MathWriting ==="
echo ""

python src/train.py \
    --config configs/train_config.yaml \
    --model_name "$MODEL_NAME" \
    --subset "$DATASET_SUBSET" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --use_secondary \
    --run_name "sft_latex_ocr_mathwriting" \
    --wandb_project "$WANDB_PROJECT"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
