#!/usr/bin/env python3
"""
Script to upload trained model to HuggingFace Hub.
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_folder


def upload_model(
    local_path: str,
    repo_id: str,
    commit_message: str = "Upload model checkpoint",
    private: bool = False
):
    """
    Upload a trained model to HuggingFace Hub.
    
    Args:
        local_path: Path to the model checkpoint
        repo_id: HuggingFace repo ID (e.g., "username/model-name")
        commit_message: Commit message
        private: Whether to make the repo private
    """
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload
    print(f"Uploading from {local_path} to {repo_id}...")
    
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message
    )
    
    print(f"Upload complete!")
    print(f"Model available at: https://huggingface.co/{repo_id}")


def create_model_card(repo_id: str, base_model: str, training_setup: str):
    """Create a model card for the uploaded model."""
    
    model_card = f"""---
language:
- en
license: apache-2.0
tags:
- latex
- ocr
- handwriting
- vision-language
- math
base_model: {base_model}
datasets:
- linxy/LaTeX_OCR
- deepcopy/MathWriting-human
pipeline_tag: image-to-text
---

# Handwritten Formula to LaTeX OCR

This model converts images of handwritten mathematical formulas to LaTeX code.

## Model Description

- **Base Model**: {base_model}
- **Training Setup**: {training_setup}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)

## Training Data

- Primary: [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR) (human_handwrite subset)
{"- Secondary: [deepcopy/MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human)" if "combined" in training_setup.lower() else ""}

## Usage

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Load model
model = AutoModelForVision2Seq.from_pretrained("{repo_id}")
processor = AutoProcessor.from_pretrained("{repo_id}")

# Load image
image = Image.open("formula.png").convert("RGB")

# Process
inputs = processor(images=image, text="Convert this handwritten formula to LaTeX:", return_tensors="pt")

# Generate
outputs = model.generate(**inputs, max_new_tokens=256)
latex = processor.decode(outputs[0], skip_special_tokens=True)
print(latex)
```

## Evaluation Results

| Metric | Score |
|--------|-------|
| BLEU | TBD |
| Exact Match | TBD |
| Edit Distance | TBD |

## Limitations

- Works best on clear handwriting
- May struggle with very complex or multi-line formulas
- Trained primarily on mathematical formulas

## Citation

```bibtex
@misc{{handwritten-latex-ocr,
  author = {{Your Name}},
  title = {{Handwritten Formula to LaTeX OCR}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""
    
    return model_card


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    
    parser.add_argument(
        "--local_path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/model-name)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--training_setup",
        type=str,
        default="SFT with LaTeX_OCR",
        help="Training setup description"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    args = parser.parse_args()
    
    # Create model card
    model_card = create_model_card(
        args.repo_id,
        args.base_model,
        args.training_setup
    )
    
    # Save model card
    readme_path = os.path.join(args.local_path, "README.md")
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print(f"Model card saved to {readme_path}")
    
    # Upload
    upload_model(
        args.local_path,
        args.repo_id,
        commit_message=f"Upload {args.training_setup} checkpoint",
        private=args.private
    )


if __name__ == "__main__":
    main()
