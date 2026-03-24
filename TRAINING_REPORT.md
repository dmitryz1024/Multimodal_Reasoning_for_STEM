# Task 1 Technical Report

## 1. Objective

The goal of this project is to solve handwritten mathematical formula recognition as an image-to-text task: given an image of a handwritten formula, generate the corresponding LaTeX string. The second goal is to deploy the trained model in a Streamlit application that accepts a handwritten formula image and renders the predicted LaTeX.

## 2. Models

Two model families were explored during the project:

- `HuggingFaceTB/SmolVLM-256M-Instruct`
- `Qwen/Qwen2-VL-2B-Instruct`

All supervised fine-tuning runs used:

- LoRA adapters
- 4-bit quantization
- Hugging Face `transformers` + `peft`

## 3. Datasets

The following datasets were used:

- Primary training dataset: `linxy/LaTeX_OCR`, subset `human_handwrite`
- Additional training dataset: `deepcopy/MathWriting-human`
- Evaluation dataset: `linxy/LaTeX_OCR`, test split, 70 examples

The final successful experiments were evaluated on the required `linxy/LaTeX_OCR:test` split.

## 4. Experimental Setup

Hardware:

- GPU: NVIDIA RTX 5060 Ti
- VRAM: 16 GB
- RAM: 128 GB

Software:

- OS: Debian
- Framework stack: PyTorch, Transformers, PEFT, bitsandbytes, Datasets, Streamlit

Primary metric:

- `Token F1`

Reason for selecting `Token F1` as the main metric:

- It captures partial correctness better than exact match.
- It is better aligned with structured LaTeX generation than plain BLEU alone.
- It remains informative even when the model output is close but not character-perfect.

Additional reported metrics:

- BLEU
- Exact Match
- Normalized Edit Distance

## 5. Baseline and Final Results

Final comparison across the four required setups:

| Setup | BLEU | Exact Match | Edit Distance | Token F1 |
|---|---:|---:|---:|---:|
| zero_shot | 0.5210 | 0.1429 | 0.5390 | 0.7507 |
| one_shot | 0.2082 | 0.1000 | 0.3735 | 0.4234 |
| sft_latex_ocr | 0.8989 | 0.6857 | 0.9115 | 0.9685 |
| sft_combined | 0.8492 | 0.6000 | 0.8722 | 0.9527 |

Key takeaways:

- The best setup was `sft_latex_ocr`.
- Zero-shot performance was already reasonably strong.
- One-shot prompting did not help for the tested prompt format.
- Adding `MathWriting-human` improved performance over the baselines, but it did not outperform SFT on `LaTeX_OCR` alone.

## 6. Training Attempts and Hyperparameters

### Attempt A: SmolVLM initial local run

Model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Hyperparameters:

- Epochs: `3`
- Train batch size: `2`
- Eval batch size: `2`
- Gradient accumulation steps: `8`
- Learning rate: `2e-4`
- Max length: `512`
- Gradient checkpointing: `true`
- Precision: `bf16`
- Secondary dataset: disabled

Outcome:

- Fine-tuning degraded the model instead of improving it.
- The learning rate was too aggressive for this small multimodal model.

### Attempt B: SmolVLM conservative retry

Model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Hyperparameters:

- Epochs: `2`
- Train batch size: `1`
- Eval batch size: `1`
- Gradient accumulation steps: `8`
- Learning rate: `5e-5`
- Max length: `384`
- Gradient checkpointing: `true`
- Precision: `bf16`
- Secondary dataset sample size: `3000`

Outcome:

- Training remained unstable.
- The shorter sequence length made multimodal prompt truncation even more likely.

### Attempt C: SmolVLM max-length safety retry

Model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Hyperparameters:

- Epochs: `2`
- Train batch size: `1`
- Eval batch size: `1`
- Gradient accumulation steps: `8`
- Learning rate: `2e-5`
- Max length: `256`
- Gradient checkpointing: `true`
- Precision: `bf16`
- Secondary dataset sample size: `2000`

Outcome:

- This attempt was explicitly designed to stabilize training after fixing collator-side sequence clipping.
- In practice, `256` tokens were insufficient for multimodal prompt plus answer.
- Training signal was severely damaged.

### Attempt D: SmolVLM context-safe retry

Model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Hyperparameters:

- Epochs: `2`
- Train batch size: `1`
- Eval batch size: `1`
- Gradient accumulation steps: `4`
- Learning rate: `2e-5`
- Max length: `1024`
- Gradient checkpointing: `true`
- Precision: `bf16`
- Secondary dataset sample size: `2000`

Outcome:

- Logs still showed that the prompt and image tokens could consume the entire context window.
- For some examples, no assistant tokens remained inside the loss window.
- The model still failed to learn useful target behavior.

### Attempt E: SmolVLM long-context retry

Model:

- `HuggingFaceTB/SmolVLM-256M-Instruct`

Hyperparameters:

- Epochs: `2`
- Train batch size: `1`
- Eval batch size: `1`
- Gradient accumulation steps: `2`
- Learning rate: `1e-5`
- Max length: `2048`
- Gradient checkpointing: `true`
- Precision: `bf16`
- Secondary dataset sample size: `2000`

Observed metrics (last for this model):

- Exact Match: `0.0000`
- BLEU: `0.0104`
- Edit Distance: `0.0113`
- Token F1: `0.0123`

Outcome:

- The larger context window was not enough to recover the model.
- `SmolVLM-256M-Instruct` remained too weak or too brittle for this task under the tested SFT recipe.

### Attempt F: Qwen2-VL successful run

Config file:

- [configs/train_config.qwen2vl_2b.yaml](c:/Users/user/Documents/projects/Multimodal_Reasoning_for_STEM/configs/train_config.qwen2vl_2b.yaml)

Model:

- `Qwen/Qwen2-VL-2B-Instruct`

Hyperparameters:

- Epochs: `2`
- Train batch size: `1`
- Eval batch size: `1`
- Gradient accumulation steps: `4`
- Learning rate: `1e-5`
- Weight decay: `0.01`
- Warmup ratio: `0.05`
- Max length: `2048`
- Gradient checkpointing: `true`
- Precision: `bf16`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Quantization: `4-bit`

Observed metrics for `sft_latex_ocr`:

- Exact Match: `0.6857`
- BLEU: `0.8989`
- Edit Distance: `0.9115`
- Token F1: `0.9685`

Observed metrics for `sft_combined`:

- Exact Match: `0.6000`
- BLEU: `0.8492`
- Edit Distance: `0.8722`
- Token F1: `0.9527`

Outcome:

- This was the first clearly successful supervised fine-tuning setup.
- Qwen2-VL substantially outperformed both zero-shot prompting and all SmolVLM experiments.
- The best final model was the `LaTeX_OCR`-only Qwen2-VL run.

## 7. Interpretation

The experiments show a clear model-capacity effect:

- `SmolVLM-256M-Instruct` provided a usable zero-shot baseline, but repeated SFT attempts degraded it.
- `Qwen2-VL-2B-Instruct` preserved and improved task performance under the same overall training pipeline.
- The additional `MathWriting-human` data was not harmful overall, but it did not improve on the best single-dataset setup.

The most likely explanation is that `SmolVLM-256M-Instruct` is too small for stable multimodal LaTeX generation fine-tuning, whereas `Qwen2-VL-2B-Instruct` has enough capacity to benefit from SFT.

## 8. Streamlit Application

The repository includes a Streamlit application in [app/streamlit_app.py](c:/Users/user/Documents/projects/Multimodal_Reasoning_for_STEM/app/streamlit_app.py). It loads a fine-tuned checkpoint from `checkpoints/` and can be used to test the final model on real handwritten formula photos.

Recommended demo model:

- `Qwen/Qwen2-VL-2B-Instruct` with the `sft_latex_ocr` adapter

Suggested report attachments:

- one screenshot with the input handwritten image
- one screenshot with the generated LaTeX string
- one screenshot with the rendered formula

## 9. Checkpoints

Local checkpoint paths:

- `./checkpoints/qwen2vl_latex_only/final`
- `./checkpoints/qwen2vl_combined/final`

Public checkpoint links:

- [qwen2vl_latex_only](https://huggingface.co/dmitryz1024/qwen2vl-latex-ocr)
- [qwen2vl_combined](https://huggingface.co/dmitryz1024/qwen2vl-latex-ocr-combined)

## 10. Reproducibility

Training the best model:

```bash
python -m src.train --config configs/train_config.qwen2vl_2b.yaml --run_name qwen2vl_latex_only
```

Training the combined setup:

```bash
python -m src.train --config configs/train_config.qwen2vl_2b.yaml --use_secondary --run_name qwen2vl_combined
```

Final evaluation:

```bash
python -m src.evaluate --model_name "Qwen/Qwen2-VL-2B-Instruct" --checkpoint_latex_ocr ./checkpoints/qwen2vl_latex_only/final --checkpoint_combined ./checkpoints/qwen2vl_combined/final --dataset "linxy/LaTeX_OCR" --subset "human_handwrite" --eval_mode all --output evaluation_results.json
```

Launching the demo:

```bash
streamlit run app/streamlit_app.py
```

## 11. Conclusion

The final project successfully solves handwritten formula recognition with a vision-language model and provides a working Streamlit demo path. The strongest result was achieved with `Qwen/Qwen2-VL-2B-Instruct` fine-tuned on `linxy/LaTeX_OCR`, which reached `Token F1 = 0.9685` on the required test split. The combined `LaTeX_OCR + MathWriting-human` setup also performed strongly, but slightly below the best single-dataset run. The earlier SmolVLM experiments were still valuable because they showed that model capacity, not only hyperparameter tuning, was the main limiting factor in this task.
