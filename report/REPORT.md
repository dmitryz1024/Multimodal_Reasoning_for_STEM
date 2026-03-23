# Task 1 Technical Report

## 1. Objective

This project solves handwritten formula recognition as an image-plus-text to text task.
The goal is to convert an input image containing a handwritten mathematical formula into LaTeX and to deploy the result in a Streamlit application.

## 2. Model

- Base model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Fine-tuning method: LoRA
- Quantization during training: 4-bit

Reason for choice:
The model is small enough to fine-tune on a local 16 GB GPU while still supporting multimodal instruction-style generation.

## 3. Datasets

- Primary training dataset: `linxy/LaTeX_OCR`, subset `human_handwrite`
- Additional training dataset: `deepcopy/MathWriting-human`
- Evaluation dataset: `linxy/LaTeX_OCR`, test subset, 70 examples

If `MathWriting-human` is subsampled, record the sample size here:

- MathWriting sample size used: `____________`

## 4. Experimental Setup

Hardware:

- GPU: `____________`
- VRAM: `____________`
- RAM: `____________`

Software:

- OS: `____________`
- Python: `____________`
- PyTorch: `____________`
- CUDA: `____________`

Training hyperparameters:

- Epochs: `____________`
- Per-device train batch size: `____________`
- Per-device eval batch size: `____________`
- Gradient accumulation steps: `____________`
- Learning rate: `____________`
- Weight decay: `____________`
- Warmup ratio: `____________`
- Max sequence length: `____________`
- Precision mode: `____________`
- Gradient checkpointing: `____________`

Checkpoint paths:

- SFT LaTeX_OCR only: `____________`
- SFT LaTeX_OCR + MathWriting: `____________`

Public checkpoint links:

- SFT LaTeX_OCR only: `____________`
- SFT LaTeX_OCR + MathWriting: `____________`

## 5. Evaluation Metric

Primary metric: `token_f1`

Why this metric:
`token_f1` measures token-level overlap between predicted and reference LaTeX while balancing precision and recall.
For handwritten formula transcription, this is more informative than exact match alone because small local differences are common, while still reflecting formula quality better than a purely character-level metric.

Additional reported metrics:

- `BLEU`
- `Exact Match`
- `Edit Distance`

## 6. Results

Fill this table using `evaluation_results.json`.

| Setup | BLEU | Exact Match | Edit Distance | Token F1 |
|---|---:|---:|---:|---:|
| Zero-shot |  |  |  |  |
| One-shot |  |  |  |  |
| SFT on `linxy/LaTeX_OCR` |  |  |  |  |
| SFT on `linxy/LaTeX_OCR` + `deepcopy/MathWriting-human` |  |  |  |  |

Short interpretation:

- Best setup by primary metric: `____________`
- Main observed improvement over zero-shot: `____________`
- Did adding MathWriting help: `____________`

## 7. Qualitative Examples

Add 2-4 examples with:

- input image
- reference LaTeX
- model prediction
- short note if the prediction is correct or partially correct

Suggested files to include from the repo:

- `report/images/dataset_examples.png`
- screenshots from notebook qualitative examples
- screenshots from the Streamlit app

## 8. Streamlit Application

The application is implemented in `app/streamlit_app.py`.
For the final demo, it should be launched with a fine-tuned checkpoint, not just the base model.

What to include in the report:

- one screenshot of the uploaded real handwritten photo
- one screenshot of the generated LaTeX text
- one screenshot of the rendered LaTeX output
- one short note indicating which checkpoint was used

Checkpoint used in the demo:

- `____________`

## 9. Reproducibility

Training:

```bash
python -m src.train --config configs/train_config.local_gpu.yaml --train_all
```

Evaluation:

```bash
python -m src.evaluate \
  --model_name "HuggingFaceTB/SmolVLM-256M-Instruct" \
  --checkpoint_latex_ocr ./checkpoints/sft_latex_ocr_only/final \
  --checkpoint_combined ./checkpoints/sft_latex_ocr_mathwriting/final \
  --dataset "linxy/LaTeX_OCR" \
  --subset "human_handwrite" \
  --eval_mode all \
  --output evaluation_results.json
```

Streamlit:

```bash
streamlit run app/streamlit_app.py
```

Checkpoint upload:

```bash
python scripts/upload_to_hub.py \
  --local_path ./checkpoints/sft_latex_ocr_only/final \
  --repo_id YOUR_USERNAME/latex-ocr-smolvlm-latex-only
```

## 10. Conclusion

Write 3-5 sentences here:

- what model was fine-tuned
- which setup performed best
- whether the combined dataset helped
- whether the Streamlit demo worked on a real handwritten formula
