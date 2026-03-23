"""
Evaluation script for Handwritten LaTeX OCR.
Supports zero-shot, one-shot, and fine-tuned model evaluation.
"""

import os
import argparse
import json
from typing import Dict, Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm

from .data_utils import load_latex_ocr_dataset
from .model_utils import VLMForLatexOCR
from .metrics import MetricTracker, format_metrics


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    adapter_path: Optional[str] = None  # Path to fine-tuned adapters
    dataset: str = "linxy/LaTeX_OCR"
    subset: str = "human_handwrite"
    split: str = "test"
    num_samples: Optional[int] = 70  # As specified in task
    batch_size: int = 1
    max_new_tokens: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_zero_shot(
    model: VLMForLatexOCR,
    test_dataset,
    config: EvalConfig,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model in zero-shot setting.
    """
    if verbose:
        print("\n" + "=" * 50)
        print("Zero-shot Evaluation")
        print("=" * 50)
    
    tracker = MetricTracker()
    
    prompt = "Convert this handwritten mathematical formula to LaTeX:"
    
    samples = test_dataset
    if config.num_samples and len(samples) > config.num_samples:
        samples = samples.select(range(config.num_samples))
    
    for example in tqdm(samples, desc="Zero-shot eval", disable=not verbose):
        image = example.get("image")
        reference = example.get("text") or example.get("latex") or example.get("formula", "")
        
        try:
            prediction = model.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        except Exception as e:
            print(f"Error generating: {e}")
            prediction = ""
        
        tracker.add(prediction, reference)
        
        if verbose and len(tracker) <= 3:
            print(f"\nExample {len(tracker)}:")
            print(f"  Reference:  {reference[:80]}...")
            print(f"  Prediction: {prediction[:80]}...")
    
    metrics = tracker.compute()
    
    if verbose:
        print("\n" + format_metrics(metrics))
    
    return metrics


def evaluate_one_shot(
    model: VLMForLatexOCR,
    test_dataset,
    config: EvalConfig,
    one_shot_example: Optional[Dict] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model in one-shot setting.
    """
    if verbose:
        print("\n" + "=" * 50)
        print("One-shot Evaluation")
        print("=" * 50)
    
    tracker = MetricTracker()
    
    # Default one-shot example
    if one_shot_example is None:
        one_shot_example = {
            "description": "x squared plus y squared equals z squared",
            "latex": "x^2 + y^2 = z^2"
        }
    
    # One-shot prompt
    prompt = f"""I will show you an example of converting a handwritten formula to LaTeX.

Example:
Description: {one_shot_example['description']}
LaTeX output: {one_shot_example['latex']}

Now convert this handwritten formula to LaTeX:"""
    
    samples = test_dataset
    if config.num_samples and len(samples) > config.num_samples:
        samples = samples.select(range(config.num_samples))
    
    for example in tqdm(samples, desc="One-shot eval", disable=not verbose):
        image = example.get("image")
        reference = example.get("text") or example.get("latex") or example.get("formula", "")
        
        try:
            prediction = model.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        except Exception as e:
            print(f"Error generating: {e}")
            prediction = ""
        
        tracker.add(prediction, reference)
        
        if verbose and len(tracker) <= 3:
            print(f"\nExample {len(tracker)}:")
            print(f"  Reference:  {reference[:80]}...")
            print(f"  Prediction: {prediction[:80]}...")
    
    metrics = tracker.compute()
    
    if verbose:
        print("\n" + format_metrics(metrics))
    
    return metrics


def evaluate_finetuned(
    model: VLMForLatexOCR,
    test_dataset,
    config: EvalConfig,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate fine-tuned model.
    """
    if verbose:
        print("\n" + "=" * 50)
        print("Fine-tuned Model Evaluation")
        print("=" * 50)
    
    tracker = MetricTracker()
    
    prompt = "Convert this handwritten formula to LaTeX:"
    
    samples = test_dataset
    if config.num_samples and len(samples) > config.num_samples:
        samples = samples.select(range(config.num_samples))
    
    for example in tqdm(samples, desc="SFT eval", disable=not verbose):
        image = example.get("image")
        reference = example.get("text") or example.get("latex") or example.get("formula", "")
        
        try:
            prediction = model.generate(
                image=image,
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=0.1,
                do_sample=False
            )
        except Exception as e:
            print(f"Error generating: {e}")
            prediction = ""
        
        tracker.add(prediction, reference)
        
        if verbose and len(tracker) <= 3:
            print(f"\nExample {len(tracker)}:")
            print(f"  Reference:  {reference[:80]}...")
            print(f"  Prediction: {prediction[:80]}...")
    
    metrics = tracker.compute()
    
    if verbose:
        print("\n" + format_metrics(metrics))
    
    return metrics


def run_full_evaluation(
    base_model_name: str,
    checkpoint_latex_ocr: Optional[str] = None,
    checkpoint_combined: Optional[str] = None,
    config: Optional[EvalConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Run full evaluation across all setups:
    1. Zero-shot
    2. One-shot
    3. SFT (LaTeX_OCR only)
    4. SFT (LaTeX_OCR + MathWriting)
    
    Args:
        base_model_name: Name of base model
        checkpoint_latex_ocr: Path to LaTeX_OCR only checkpoint
        checkpoint_combined: Path to combined training checkpoint
        config: Evaluation configuration
    
    Returns:
        Dictionary mapping setup names to their metrics
    """
    if config is None:
        config = EvalConfig(model_name=base_model_name)
    
    results = {}
    
    # Load test dataset
    print("Loading test dataset...")
    test_ds = load_latex_ocr_dataset(subset=config.subset, split=config.split)
    print(f"Test samples: {len(test_ds)}")
    
    # 1. Zero-shot evaluation
    print("\n" + "=" * 60)
    print("SETUP 1: Zero-shot Inference")
    print("=" * 60)
    
    model_zero = VLMForLatexOCR.from_pretrained(
        base_model_name,
        load_in_4bit=False  # Use full precision for inference
    )
    
    results["zero_shot"] = evaluate_zero_shot(model_zero, test_ds, config)
    
    # Clear memory
    del model_zero
    torch.cuda.empty_cache()
    
    # 2. One-shot evaluation
    print("\n" + "=" * 60)
    print("SETUP 2: One-shot Inference")
    print("=" * 60)
    
    model_one = VLMForLatexOCR.from_pretrained(
        base_model_name,
        load_in_4bit=False
    )
    
    results["one_shot"] = evaluate_one_shot(model_one, test_ds, config)
    
    del model_one
    torch.cuda.empty_cache()
    
    # 3. SFT (LaTeX_OCR only)
    if checkpoint_latex_ocr and os.path.exists(checkpoint_latex_ocr):
        print("\n" + "=" * 60)
        print("SETUP 3: SFT (LaTeX_OCR only)")
        print("=" * 60)
        
        model_sft1 = VLMForLatexOCR.from_pretrained(
            base_model_name,
            adapter_path=checkpoint_latex_ocr
        )
        
        results["sft_latex_ocr"] = evaluate_finetuned(model_sft1, test_ds, config)
        
        del model_sft1
        torch.cuda.empty_cache()
    else:
        print("\nSkipping SFT (LaTeX_OCR only) - checkpoint not provided")
    
    # 4. SFT (LaTeX_OCR + MathWriting)
    if checkpoint_combined and os.path.exists(checkpoint_combined):
        print("\n" + "=" * 60)
        print("SETUP 4: SFT (LaTeX_OCR + MathWriting)")
        print("=" * 60)
        
        model_sft2 = VLMForLatexOCR.from_pretrained(
            base_model_name,
            adapter_path=checkpoint_combined
        )
        
        results["sft_combined"] = evaluate_finetuned(model_sft2, test_ds, config)
        
        del model_sft2
        torch.cuda.empty_cache()
    else:
        print("\nSkipping SFT (Combined) - checkpoint not provided")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\n| Setup | BLEU | Exact Match | Edit Distance | Token F1 |")
    print("|-------|------|-------------|---------------|----------|")
    
    for setup_name, metrics in results.items():
        print(f"| {setup_name} | "
              f"{metrics.get('bleu', 0):.4f} | "
              f"{metrics.get('exact_match', 0):.4f} | "
              f"{metrics.get('edit_distance', 0):.4f} | "
              f"{metrics.get('token_f1', 0):.4f} |")
    
    return results


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Handwritten LaTeX OCR model")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to fine-tuned adapter (for single model eval)"
    )
    parser.add_argument(
        "--checkpoint_latex_ocr",
        type=str,
        default=None,
        help="Path to LaTeX_OCR only checkpoint (for full eval)"
    )
    parser.add_argument(
        "--checkpoint_combined",
        type=str,
        default=None,
        help="Path to combined training checkpoint (for full eval)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="linxy/LaTeX_OCR",
        help="Dataset name"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="human_handwrite",
        help="Dataset subset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=70,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["zero_shot", "one_shot", "sft", "all"],
        default="all",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    config = EvalConfig(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        num_samples=args.num_samples
    )
    
    if args.eval_mode == "all":
        results = run_full_evaluation(
            base_model_name=args.model_name,
            checkpoint_latex_ocr=args.checkpoint_latex_ocr,
            checkpoint_combined=args.checkpoint_combined,
            config=config
        )
    else:
        # Single mode evaluation
        test_ds = load_latex_ocr_dataset(subset=config.subset, split=config.split)
        
        if args.adapter_path:
            model = VLMForLatexOCR.from_pretrained(
                args.model_name,
                adapter_path=args.adapter_path
            )
        else:
            model = VLMForLatexOCR.from_pretrained(args.model_name)
        
        if args.eval_mode == "zero_shot":
            results = {"zero_shot": evaluate_zero_shot(model, test_ds, config)}
        elif args.eval_mode == "one_shot":
            results = {"one_shot": evaluate_one_shot(model, test_ds, config)}
        elif args.eval_mode == "sft":
            results = {"sft": evaluate_finetuned(model, test_ds, config)}
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
