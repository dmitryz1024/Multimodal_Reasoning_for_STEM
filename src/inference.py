"""
Inference utilities for Handwritten LaTeX OCR.
"""

from typing import Optional, Union
from pathlib import Path

import torch
from PIL import Image

from .model_utils import VLMForLatexOCR


class LatexOCRInference:
    """
    High-level inference class for LaTeX OCR.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        adapter_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_name: Base model name
            adapter_path: Path to fine-tuned adapters (optional)
            device: Device to use (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model on {self.device}...")
        self.model = VLMForLatexOCR.from_pretrained(
            model_name,
            adapter_path=adapter_path,
            load_in_4bit=False  # Use full precision for better inference
        )
        print("Model loaded!")
    
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        use_one_shot: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.1
    ) -> str:
        """
        Predict LaTeX from an image.
        
        Args:
            image: Image path or PIL Image
            prompt: Custom prompt (optional)
            use_one_shot: Whether to use one-shot prompting
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            LaTeX string
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Default prompt
        if prompt is None:
            prompt = "Convert this handwritten mathematical formula to LaTeX:"
        
        # Use one-shot if requested
        if use_one_shot:
            return self.model.generate_with_one_shot(
                image=image,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        
        return self.model.generate(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0
        )
    
    def predict_batch(
        self,
        images: list,
        **kwargs
    ) -> list:
        """
        Predict LaTeX for multiple images.
        
        Args:
            images: List of image paths or PIL Images
            **kwargs: Additional arguments for predict()
        
        Returns:
            List of LaTeX strings
        """
        return [self.predict(img, **kwargs) for img in images]


def load_inference_model(
    checkpoint_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
) -> LatexOCRInference:
    """
    Convenience function to load inference model.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint
        model_name: Base model name
    
    Returns:
        LatexOCRInference instance
    """
    return LatexOCRInference(
        model_name=model_name,
        adapter_path=checkpoint_path
    )


def quick_inference(
    image_path: str,
    checkpoint_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
) -> str:
    """
    Quick one-off inference.
    
    Args:
        image_path: Path to image
        checkpoint_path: Path to fine-tuned checkpoint
        model_name: Base model name
    
    Returns:
        LaTeX string
    """
    engine = LatexOCRInference(
        model_name=model_name,
        adapter_path=checkpoint_path
    )
    return engine.predict(image_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LaTeX OCR inference")
    parser.add_argument("image", type=str, help="Path to image")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model name"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint"
    )
    parser.add_argument(
        "--one_shot",
        action="store_true",
        help="Use one-shot prompting"
    )
    
    args = parser.parse_args()
    
    engine = LatexOCRInference(
        model_name=args.model,
        adapter_path=args.checkpoint
    )
    
    result = engine.predict(
        args.image,
        use_one_shot=args.one_shot
    )
    
    print(f"\nPredicted LaTeX:\n{result}")
