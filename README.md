# Handwritten Formula to LaTeX Converter

A Vision-Language Model fine-tuned for converting handwritten mathematical formulas into LaTeX format.

## 📋 Project Structure

```
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

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/handwritten-latex-ocr.git
cd handwritten-latex-ocr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

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

See [report/REPORT.md](report/REPORT.md) for detailed technical report.

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
