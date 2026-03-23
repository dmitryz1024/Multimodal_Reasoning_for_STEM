# Handwritten Formula to LaTeX Converter

A Vision-Language Model fine-tuned for converting handwritten mathematical formulas into LaTeX format.

## 📋 Project Structure

```
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
│
├── configs/
│   └── train_config.yaml        # Training configuration
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py            # Dataset loading and preprocessing
│   ├── model_utils.py           # Model loading and configuration
│   ├── metrics.py               # Evaluation metrics
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference utilities
│
├── app/
│   └── streamlit_app.py         # Streamlit application
│
├── notebooks/
│   └── experiments.ipynb        # Jupyter notebook for experiments
│
├── report/
│   ├── REPORT.md                # Technical report
│   └── images/                  # Screenshots for report
│
└── scripts/
    ├── run_training.sh          # Training launch script
    └── run_evaluation.sh        # Evaluation launch script
```

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

#### Prerequisites
- Docker with NVIDIA support (nvidia-docker2)
- NVIDIA GPU (optional but recommended for training)

#### Build and Run

```bash
# Clone the repository
git clone https://github.com/dmitryz1024/Multimodal_Reasoning_for_STEM.git
cd Multimodal_Reasoning_for_STEM

# Build the Docker image
docker build -t latex-ocr .

# Run Streamlit app
docker run --gpus all -p 8501:8501 -v $(pwd):/app latex-ocr streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Or use docker-compose
docker-compose up latex-ocr
```

#### Training with Docker

```bash
# Run training
docker run --gpus all -v $(pwd):/app latex-ocr python src/train.py --config configs/train_config.yaml

# Or with docker-compose
docker-compose up train
```

### Option 2: Local Installation (optional)

#### Prerequisites
- Python 3.10+
- PyTorch с CUDA (если есть GPU). Если нет, модель будет обучаться медленнее на CPU.

```bash
# Clone the repository
git clone https://github.com/dmitryz1024/Multimodal_Reasoning_for_STEM.git
cd Multimodal_Reasoning_for_STEM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Training

```bash
# Run training with default config
python src/train.py --config configs/train_config.yaml

# Or with custom parameters
python src/train.py \
    --model_name "HuggingFaceTB/SmolVLM-256M-Instruct" \
    --dataset "linxy/LaTeX_OCR" \
    --subset "human_handwrite" \
    --epochs 3 \
    --batch_size 4
```

### 3. Evaluation

```bash
# Evaluate on test set
python src/evaluate.py \
    --model_path ./checkpoints/best_model \
    --dataset "linxy/LaTeX_OCR" \
    --subset "human_handwrite"
```

### 4. Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## 📊 Experimental Results

| Setup | BLEU | Exact Match | Edit Distance |
|-------|------|-------------|---------------|
| Zero-shot | TBD | TBD | TBD |
| One-shot | TBD | TBD | TBD |
| SFT (LaTeX_OCR only) | TBD | TBD | TBD |
| SFT (LaTeX_OCR + MathWriting) | TBD | TBD | TBD |

## 🔗 Model Checkpoints

- **Zero-shot model**: [HuggingFace Link]
- **SFT (LaTeX_OCR)**: [HuggingFace Link]
- **SFT (LaTeX_OCR + MathWriting)**: [HuggingFace Link]

## 📝 Technical Report

Все результаты, таблицы и скриншоты сохраняйте в `report/`:
- `report/REPORT.md` — сводный технический отчет (setup, гиперпараметры, метрики, выводы).
- `report/images/` — скриншоты приложения, примеры работы, графики обучения.

Папка `report` монтируется в контейнер при запуске через Docker Compose, поэтому данные сохраняются локально и остаются после остановки контейнера.

## 📦 Datasets Used

- [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR) - Primary dataset
- [deepcopy/MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human) - Additional training data

## 🛠️ Model Options

Supported models:
- `HuggingFaceTB/SmolVLM-256M-Instruct` (recommended for limited resources)
- `Qwen/Qwen2-VL-2B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`

## License

MIT License
