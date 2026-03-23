#!/usr/bin/env python3
"""
Quick test script to verify the installation and basic functionality.
"""

import sys
sys.path.insert(0, 'src')

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"  ✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ Datasets: {e}")
        return False
    
    try:
        import peft
        print(f"  ✓ PEFT {peft.__version__}")
    except ImportError as e:
        print(f"  ✗ PEFT: {e}")
        return False
    
    try:
        import streamlit
        print(f"  ✓ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"  ✗ Streamlit: {e}")
        return False
    
    try:
        from src.data_utils import DataConfig
        from src.model_utils import ModelConfig
        from src.metrics import compute_all_metrics
        print("  ✓ Project modules")
    except ImportError as e:
        print(f"  ✗ Project modules: {e}")
        return False
    
    return True


def test_metrics():
    """Test the metrics module."""
    print("\nTesting metrics...")
    
    from src.metrics import compute_all_metrics, normalize_latex
    
    # Test normalization
    test_cases = [
        ("$x^2$", "x^2"),
        ("x^{2}", "x^2"),
        ("\\frac{a}{b}", "\\frac{a}{b}"),
    ]
    
    for input_latex, expected in test_cases:
        result = normalize_latex(input_latex)
        status = "✓" if expected in result or result in expected else "✗"
        print(f"  {status} normalize_latex('{input_latex}') -> '{result}'")
    
    # Test full metrics
    pred = "x^2 + y^2 = z^2"
    ref = "x^2 + y^2 = z^2"
    metrics = compute_all_metrics(pred, ref)
    
    print(f"\n  Metrics for exact match:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.4f}")
    
    return True


def test_dataset_loading():
    """Test dataset loading (streaming mode to avoid downloading)."""
    print("\nTesting dataset loading (streaming)...")
    
    try:
        from datasets import load_dataset
        
        # Test LaTeX_OCR
        ds = load_dataset("linxy/LaTeX_OCR", "human_handwrite", split="train", streaming=True)
        example = next(iter(ds))
        print(f"  ✓ LaTeX_OCR loaded")
        print(f"    Columns: {list(example.keys())}")
        
        # Test MathWriting
        ds = load_dataset("deepcopy/MathWriting-human", split="train", streaming=True)
        example = next(iter(ds))
        print(f"  ✓ MathWriting-human loaded")
        print(f"    Columns: {list(example.keys())}")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        return False


def main():
    print("=" * 50)
    print("Handwritten LaTeX OCR - Quick Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_metrics()
    all_passed &= test_dataset_loading()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
