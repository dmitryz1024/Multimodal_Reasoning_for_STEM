"""
Model utilities for loading and configuring Vision-Language models.
"""

import os
import platform
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Idefics3ForConditionalGeneration,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


# Mapping of model names to their classes
MODEL_CLASSES = {
    "qwen": Qwen2VLForConditionalGeneration,
    "smolvlm": Idefics3ForConditionalGeneration,
    "idefics3": Idefics3ForConditionalGeneration,
    "default": AutoModelForCausalLM,
}


def get_quantization_config(config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """
    Get quantization configuration for model loading.
    """
    if (config.load_in_4bit or config.load_in_8bit) and not torch.cuda.is_available():
        print("CUDA is not available; disabling bitsandbytes quantization.")
        return None

    if config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, config.dtype),
            bnb_4bit_use_double_quant=True,
        )
    elif config.load_in_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def get_model_class(model_name: str):
    """
    Get the appropriate model class based on model name.
    """
    model_name_lower = model_name.lower()

    if "qwen" in model_name_lower:
        return MODEL_CLASSES["qwen"]
    elif "smolvlm" in model_name_lower or "idefics3" in model_name_lower:
        return MODEL_CLASSES["smolvlm"]
    else:
        return MODEL_CLASSES["default"]


def load_model_and_processor(
    config: ModelConfig,
    for_training: bool = True
) -> Tuple[torch.nn.Module, AutoProcessor]:
    """
    Load model and processor with optional quantization and LoRA.
    
    Args:
        config: Model configuration
        for_training: Whether to prepare model for training
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model: {config.name}")
    
    # Get quantization config
    quantization_config = get_quantization_config(config) if for_training else None
    
    # Resolve dtype
    model_dtype = getattr(torch, config.dtype)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code
    )

    # Load model
    model_class = get_model_class(config.name)

    model_kwargs = {
        "pretrained_model_name_or_path": config.name,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = model_dtype

    try:
        model = model_class.from_pretrained(**model_kwargs)
    except Exception as exc:
        if quantization_config is None:
            raise

        is_windows = platform.system().lower() == "windows"
        print(
            f"Quantized loading failed{' on Windows' if is_windows else ''}: {exc}\n"
            "Retrying without bitsandbytes quantization."
        )
        fallback_kwargs = {
            "pretrained_model_name_or_path": config.name,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "low_cpu_mem_usage": True,
            "torch_dtype": model_dtype,
        }
        model = model_class.from_pretrained(**fallback_kwargs)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare for training with LoRA
    if for_training and config.use_lora:
        model = prepare_model_for_lora(model, config)
    
    return model, processor


def prepare_model_for_lora(model: torch.nn.Module, config: ModelConfig) -> torch.nn.Module:
    """
    Prepare model with LoRA adapters for training.
    """
    print("Preparing model for LoRA training...")
    
    # Prepare for k-bit training if using quantization
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} "
          f"({100 * trainable_params / all_params:.2f}%)")
    
    return model


def load_trained_model(
    base_model_name: str,
    adapter_path: str,
    config: Optional[ModelConfig] = None
) -> Tuple[torch.nn.Module, AutoProcessor]:
    """
    Load a fine-tuned model with LoRA adapters.
    
    Args:
        base_model_name: Name of the base model
        adapter_path: Path to the saved LoRA adapters
        config: Optional model configuration
    
    Returns:
        Tuple of (model, processor)
    """
    if config is None:
        config = ModelConfig(name=base_model_name, load_in_4bit=False)
    
    print(f"Loading base model: {base_model_name}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model_name,
        trust_remote_code=config.trust_remote_code
    )
    
    # Load base model
    model_dtype = getattr(torch, config.dtype)
    model_class = get_model_class(base_model_name)

    model = model_class.from_pretrained(
        base_model_name,
        torch_dtype=model_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapters
    print(f"Loading adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge adapters for faster inference
    print("Merging LoRA adapters...")
    model = model.merge_and_unload()
    
    model.eval()
    
    return model, processor


def save_model(
    model: torch.nn.Module,
    processor: AutoProcessor,
    output_dir: str,
    save_full_model: bool = False
):
    """
    Save model and processor.
    
    Args:
        model: The model to save
        processor: The processor to save
        output_dir: Directory to save to
        save_full_model: If True, merge and save full model; otherwise save adapters only
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, 'save_pretrained'):
        if save_full_model and hasattr(model, 'merge_and_unload'):
            print("Merging adapters and saving full model...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            print("Saving model/adapters...")
            model.save_pretrained(output_dir)
    
    print("Saving processor...")
    processor.save_pretrained(output_dir)
    
    print(f"Model saved to: {output_dir}")


class VLMForLatexOCR:
    """
    Wrapper class for Vision-Language Model for LaTeX OCR.
    Provides unified interface for different model architectures.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        processor: AutoProcessor,
        device: Optional[str] = None
    ):
        self.model = model
        self.processor = processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not hasattr(model, 'device') or str(model.device) == 'meta':
            self.model.to(self.device)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> "VLMForLatexOCR":
        """
        Load a pretrained or fine-tuned model.
        """
        config = ModelConfig(name=model_name, **kwargs)
        
        if adapter_path:
            model, processor = load_trained_model(model_name, adapter_path, config)
        else:
            model, processor = load_model_and_processor(config, for_training=False)
        
        return cls(model, processor)
    
    def generate(
        self,
        image,
        prompt: str = "Convert this handwritten formula to LaTeX:",
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate LaTeX from an image.
        
        Args:
            image: PIL Image or path to image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
        
        Returns:
            Generated LaTeX string
        """
        from PIL import Image as PILImage
        
        # Load image if path
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        elif hasattr(image, 'convert'):
            image = image.convert("RGB")
        
        # Create conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode
        generated_text = self.processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_with_one_shot(
        self,
        image,
        example_latex: str = "x^2 + y^2 = z^2",
        prompt: str = "Convert this handwritten formula to LaTeX:",
        **kwargs
    ) -> str:
        """
        Generate LaTeX with one-shot example in the prompt.
        """
        one_shot_prompt = f"""Example:
Input: [Image of x² + y² = z²]
Output: {example_latex}

Now convert this handwritten formula to LaTeX:"""
        
        return self.generate(image, one_shot_prompt, **kwargs)


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    
    config = ModelConfig(
        name="HuggingFaceTB/SmolVLM-256M-Instruct",
        load_in_4bit=True,
        use_lora=True
    )
    
    model, processor = load_model_and_processor(config, for_training=True)
    print("Model loaded successfully!")
