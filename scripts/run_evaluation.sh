#!/bin/bash
# Evaluation script for Handwritten LaTeX OCR

set -e

# Configuration
MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM-256M-Instruct}"
CHECKPOINT_LATEX_OCR="${CHECKPOINT_LATEX_OCR:-./checkpoints/sft_latex_ocr_only/final}"
CHECKPOINT_COMBINED="${CHECKPOINT_COMBINED:-./checkpoints/sft_latex_ocr_mathwriting/final}"
OUTPUT_FILE="${OUTPUT_FILE:-evaluation_results.json}"

echo "=============================================="
echo "Handwritten LaTeX OCR - Evaluation Script"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Checkpoint (LaTeX_OCR): $CHECKPOINT_LATEX_OCR"
echo "Checkpoint (Combined): $CHECKPOINT_COMBINED"
echo "=============================================="

# Run full evaluation
python src/evaluate.py \
    --model_name "$MODEL_NAME" \
    --checkpoint_latex_ocr "$CHECKPOINT_LATEX_OCR" \
    --checkpoint_combined "$CHECKPOINT_COMBINED" \
    --subset "human_handwrite" \
    --num_samples 70 \
    --eval_mode "all" \
    --output "$OUTPUT_FILE"

echo ""
echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=============================================="
