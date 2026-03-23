# Makefile for Handwritten LaTeX OCR project

.PHONY: install train evaluate app clean help

# Default model
MODEL_NAME ?= HuggingFaceTB/SmolVLM-256M-Instruct
DATASET_SUBSET ?= human_handwrite

help:
	@echo "Handwritten LaTeX OCR - Available commands:"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make train       - Run training"
	@echo "  make train-all   - Train all experimental setups"
	@echo "  make evaluate    - Run evaluation"
	@echo "  make app         - Run Streamlit app"
	@echo "  make notebook    - Start Jupyter notebook"
	@echo "  make clean       - Clean up generated files"
	@echo ""
	@echo "Environment variables:"
	@echo "  MODEL_NAME       - Model to use (default: $(MODEL_NAME))"
	@echo "  DATASET_SUBSET   - Dataset subset (default: $(DATASET_SUBSET))"

install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True)"

train:
	python src/train.py \
		--config configs/train_config.yaml \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET)

train-all:
	python src/train.py \
		--config configs/train_config.yaml \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET) \
		--train_all

train-latex-only:
	python src/train.py \
		--config configs/train_config.yaml \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET) \
		--run_name sft_latex_ocr_only

train-combined:
	python src/train.py \
		--config configs/train_config.yaml \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET) \
		--use_secondary \
		--run_name sft_latex_ocr_mathwriting

evaluate:
	python src/evaluate.py \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET) \
		--eval_mode all \
		--output evaluation_results.json

evaluate-zero-shot:
	python src/evaluate.py \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET) \
		--eval_mode zero_shot

evaluate-one-shot:
	python src/evaluate.py \
		--model_name $(MODEL_NAME) \
		--subset $(DATASET_SUBSET) \
		--eval_mode one_shot

app:
	streamlit run app/streamlit_app.py

notebook:
	jupyter notebook notebooks/experiments.ipynb

inspect-data:
	python src/data_utils.py

test-metrics:
	python src/metrics.py

clean:
	rm -rf __pycache__ src/__pycache__ app/__pycache__
	rm -rf .ipynb_checkpoints notebooks/.ipynb_checkpoints
	rm -rf checkpoints/
	rm -rf wandb/
	rm -f evaluation_results.json
	rm -f *.log

lint:
	black src/ app/
	isort src/ app/
