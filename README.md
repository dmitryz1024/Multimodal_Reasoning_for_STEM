# Handwritten Formula to LaTeX Converter

A vision-language project for converting handwritten mathematical formulas into LaTeX.
The full usage scenario for my project from A to Z will be described below. The technical report on the task can be found in the file TECHNICAL_REPORT.md

## Project Structure

```text
.
|-- Dockerfile
|-- docker-compose.yml
|-- README.md
|-- requirements.txt
|-- TRAINING_REPORT.md
|-- configs/
|   `-- train_config.qwen2vl_2b.yaml
|-- src/
|   |-- __init__.py
|   |-- data_utils.py
|   |-- evaluate.py
|   |-- inference.py
|   |-- metrics.py
|   |-- model_utils.py
|   `-- train.py
|-- app/
|   `-- streamlit_app.py
|-- scripts/
|   `-- upload_to_hub.py
```

## Functions

- Training via `python -m src.train`
- Evaluation via `python -m src.evaluate`
- Streamlit app via `streamlit run app/streamlit_app.py`
- Docker image build and Docker Compose services

## Requirements

- Python 3.10+
- NVIDIA GPU recommended for training
- Hugging Face account/token for downloading models and datasets

## Local Setup

1. Clone the repository:

```bash
git clone https://github.com/dmitryz1024/Multimodal_Reasoning_for_STEM.git
cd Multimodal_Reasoning_for_STEM
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

3. Install PyTorch with CUDA first if you plan to train on GPU, then install project dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`wandb` is optional. The project now trains without it by default. If you want Weights & Biases logging, install it separately with `pip install wandb` and pass `--wandb_project your-project-name`.

4. Optional but recommended: verify that CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

5. Log in to Hugging Face:

```bash
huggingface-cli login
```

## Training

Default training:

```bash
python -m src.train --config configs/train_config.qwen2vl_2b.yaml --run_name qwen2vl_latex_only
```

Combined training:

```bash
python -m src.train --config configs/train_config.qwen2vl_2b.yaml --use_secondary --run_name qwen2vl_combined
```

Explicitly disable wandb from the CLI even if a config enables it:

```bash
python -m src.train --config configs/train_config.qwen2vl_2b.yaml --no_wandb
```

Training with CLI overrides:

```bash
python -m src.train \
  --config configs/train_config.qwen2vl_2b.yaml \
  --model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --dataset "linxy/LaTeX_OCR" \
  --subset "human_handwrite" \
  --epochs 2 \
  --batch_size 1 \
  --run_name custom_run
```

Train both experimental setups:

```bash
python -m src.train --config configs/train_config.qwen2vl_2b.yaml --train_all
```

Notes:

- `--dataset` now overrides the primary Hugging Face dataset used for training.
- If the custom dataset does not use a Hugging Face subset/config, omit `--subset`.
- The dataset still needs to match the expected image/text structure used by the project.
- The secondary dataset is controlled by config via `secondary_dataset` and `use_secondary`.

## Evaluation

Evaluate a single adapter checkpoint:

```bash
python -m src.evaluate \
  --model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --adapter_path ./checkpoints/qwen2vl_latex_only/final \
  --dataset "linxy/LaTeX_OCR" \
  --subset "human_handwrite" \
  --eval_mode sft \
  --output evaluation_results.json
```

Run full evaluation across zero-shot, one-shot, and available fine-tuned checkpoints:

```bash
python -m src.evaluate \
  --model_name "Qwen/Qwen2-VL-2B-Instruct" \
  --checkpoint_latex_ocr ./checkpoints/qwen2vl_latex_only/final \
  --checkpoint_combined ./checkpoints/qwen2vl_combined/final \
  --dataset "linxy/LaTeX_OCR" \
  --subset "human_handwrite" \
  --eval_mode all
```

## Inference

Command line inference:

```bash
python -m src.inference path/to/image.png --model "Qwen/Qwen2-VL-2B-Instruct"
```

With a fine-tuned checkpoint:

```bash
python -m src.inference path/to/image.png \
  --model "Qwen/Qwen2-VL-2B-Instruct" \
  --checkpoint ./checkpoints/qwen2vl_latex_only/final
```

## Streamlit App

Run locally:

```bash
streamlit run app/streamlit_app.py
```

The app now auto-selects a trained checkpoint if one exists in `checkpoints/`.
For the internship demo, keep the checkpoint path pointed at one of your fine-tuned runs.

If you use private/gated Hugging Face assets, set `HF_TOKEN` in your environment before launch.

## Docker

Build the image:

```bash
docker build -t latex-ocr .
```

Run the Streamlit app:

```bash
docker run --gpus all -p 8501:8501 -v $(pwd):/app latex-ocr \
  streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Run training:

```bash
docker run --gpus all -v $(pwd):/app latex-ocr \
  python -m src.train --config configs/train_config.qwen2vl_2b.yaml --run_name qwen2vl_latex_only
```

Docker Compose:

```bash
docker compose up latex-ocr
docker compose up train
```

Depending on your Docker setup, you may also need:

```bash
docker compose run --rm train python -m src.train --config configs/train_config.qwen2vl_2b.yaml --run_name qwen2vl_latex_only
```

## Uploading Trained Checkpoints

If your `.env` already contains `HF_TOKEN`, you can publish checkpoints with:

```bash
python scripts/upload_to_hub.py \
  --local_path ./checkpoints/qwen2vl_latex_only/final \
  --repo_id YOUR_USERNAME/qwen2vl-latex-ocr
```

and

```bash
python scripts/upload_to_hub.py \
  --local_path ./checkpoints/qwen2vl_combined/final \
  --repo_id YOUR_USERNAME/qwen2vl-latex-ocr-combined
```

After upload, put the public model links into this README and into `TRAINING_REPORT.md`.

## Datasets

- `linxy/LaTeX_OCR`
- `deepcopy/MathWriting-human`

## Model Options

Tested/default model:

- `Qwen/Qwen2-VL-2B-Instruct`

Also referenced in the code:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `HuggingFaceTB/SmolVLM-256M-Instruct`

