"""
Training script for Handwritten LaTeX OCR.
"""

import os
import argparse
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import yaml
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    EarlyStoppingCallback,
)

from .data_utils import load_latex_ocr_dataset, load_mathwriting_dataset
from .model_utils import ModelConfig, load_model_and_processor, save_model


@dataclass
class TrainConfig:
    """Full training configuration."""
    # Model config
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    
    # Data config
    primary_dataset: str = "linxy/LaTeX_OCR"
    primary_subset: str = "human_handwrite"
    use_secondary: bool = False
    secondary_sample_size: int = 10000
    
    # Training config
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    
    # Misc
    seed: int = 42
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"


def load_config(config_path: str) -> TrainConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config
    flat_config = {}
    for section in ['model', 'data', 'training']:
        if section in config_dict:
            flat_config.update(config_dict[section])
    
    # Map to TrainConfig fields
    return TrainConfig(
        model_name=flat_config.get('name', TrainConfig.model_name),
        use_lora=flat_config.get('use_lora', TrainConfig.use_lora),
        lora_r=flat_config.get('lora_r', TrainConfig.lora_r),
        lora_alpha=flat_config.get('lora_alpha', TrainConfig.lora_alpha),
        lora_dropout=flat_config.get('lora_dropout', TrainConfig.lora_dropout),
        load_in_4bit=flat_config.get('load_in_4bit', TrainConfig.load_in_4bit),
        primary_dataset=flat_config.get('primary_dataset', TrainConfig.primary_dataset),
        primary_subset=flat_config.get('primary_subset', TrainConfig.primary_subset),
        use_secondary=flat_config.get('use_secondary', TrainConfig.use_secondary),
        secondary_sample_size=flat_config.get('secondary_sample_size', TrainConfig.secondary_sample_size),
        output_dir=flat_config.get('output_dir', TrainConfig.output_dir),
        num_epochs=flat_config.get('num_epochs', TrainConfig.num_epochs),
        batch_size=flat_config.get('per_device_train_batch_size', TrainConfig.batch_size),
        gradient_accumulation_steps=flat_config.get('gradient_accumulation_steps', TrainConfig.gradient_accumulation_steps),
        learning_rate=flat_config.get('learning_rate', TrainConfig.learning_rate),
        weight_decay=flat_config.get('weight_decay', TrainConfig.weight_decay),
        warmup_ratio=flat_config.get('warmup_ratio', TrainConfig.warmup_ratio),
        max_grad_norm=flat_config.get('max_grad_norm', TrainConfig.max_grad_norm),
        bf16=flat_config.get('bf16', TrainConfig.bf16),
        fp16=flat_config.get('fp16', TrainConfig.fp16),
        gradient_checkpointing=flat_config.get('gradient_checkpointing', TrainConfig.gradient_checkpointing),
        seed=flat_config.get('seed', TrainConfig.seed),
        logging_steps=flat_config.get('logging_steps', TrainConfig.logging_steps),
        save_strategy=flat_config.get('save_strategy', TrainConfig.save_strategy),
        eval_strategy=flat_config.get('eval_strategy', TrainConfig.eval_strategy),
    )


class LatexOCRDataCollator:
    """
    Data collator for LaTeX OCR training.
    Handles image-text pairs for Vision-Language models.
    """
    
    def __init__(self, processor: AutoProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.system_prompt = "Convert the handwritten mathematical formula to LaTeX code."
    
    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        images = []
        texts = []
        
        for example in batch:
            # Get image
            image = example.get("image")
            if hasattr(image, 'convert'):
                image = image.convert("RGB")
            images.append(image)
            
            # Get LaTeX target
            latex = example.get("text") or example.get("latex") or example.get("formula", "")
            
            # Create conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Convert this handwritten formula to LaTeX:"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": latex
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Process batch
        batch_encoding = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=False,  # Disable truncation to avoid image token mismatch
            max_length=self.max_length,
        )
        
        # Create labels (same as input_ids but with padding tokens masked)
        labels = batch_encoding["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch_encoding["labels"] = labels
        
        return batch_encoding


def create_training_args(config: TrainConfig, run_name: str) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    # Detect device support and avoid unsupported precision mode
    is_cuda = torch.cuda.is_available()
    bf16 = config.bf16
    fp16 = config.fp16
    num_workers = 4 if is_cuda else 0  # Disable multiprocessing on CPU

    if not is_cuda:
        print("WARNING: CUDA is not available, forcing fp16/bf16 off and using CPU.")
        bf16 = False
        fp16 = False
    else:
        if bf16 and not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            print("WARNING: bf16 not supported on this GPU; falling back to fp16 if enabled.")
            bf16 = False
            if not fp16:
                print("WARNING: fp16 disabled as well; training will run in fp32.")

    return TrainingArguments(
        output_dir=os.path.join(config.output_dir, run_name),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if os.environ.get("WANDB_PROJECT") else ["none"],
        run_name=run_name,
        seed=config.seed,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
    )


def train(
    config: TrainConfig,
    run_name: str = "sft_latex_ocr",
    wandb_project: Optional[str] = None,
):
    """
    Main training function.
    
    Args:
        config: Training configuration
        run_name: Name for this training run
        wandb_project: Optional W&B project name
    """
    print("=" * 60)
    print(f"Starting training: {run_name}")
    print("=" * 60)
    
    # Set up wandb
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
        import wandb
        wandb.init(project=wandb_project, name=run_name)
    
    # Load model and processor
    model_config = ModelConfig(
        name=config.model_name,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        load_in_4bit=config.load_in_4bit,
    )
    
    model, processor = load_model_and_processor(model_config, for_training=True)
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load datasets
    print("\nLoading datasets...")
    primary_ds = load_latex_ocr_dataset(subset=config.primary_subset)
    train_dataset = primary_ds["train"]
    eval_dataset = primary_ds["validation"]
    
    # Optionally add secondary dataset
    if config.use_secondary:
        print(f"\nAdding secondary dataset (sample size: {config.secondary_sample_size})...")
        secondary_ds = load_mathwriting_dataset(
            split="train",
            sample_size=config.secondary_sample_size
        )
        
        # Concatenate datasets
        from datasets import concatenate_datasets
        train_dataset = concatenate_datasets([train_dataset, secondary_ds])
        train_dataset = train_dataset.shuffle(seed=config.seed)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = LatexOCRDataCollator(
        processor=processor,
        max_length=512
    )
    
    # Create training arguments
    training_args = create_training_args(config, run_name)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(config.output_dir, run_name, "final")
    save_model(model, processor, final_output_dir)
    
    print(f"\nTraining complete! Model saved to: {final_output_dir}")
    
    return model, processor


def train_all_setups(config: TrainConfig, wandb_project: Optional[str] = None):
    """
    Train models for all experimental setups:
    1. SFT with LaTeX_OCR only
    2. SFT with LaTeX_OCR + MathWriting
    """
    results = {}
    
    # Setup 1: LaTeX_OCR only
    print("\n" + "=" * 60)
    print("SETUP 1: SFT with LaTeX_OCR only")
    print("=" * 60)
    
    config_setup1 = TrainConfig(
        **{k: v for k, v in config.__dict__.items()},
    )
    config_setup1.use_secondary = False
    
    model1, processor1 = train(
        config_setup1,
        run_name="sft_latex_ocr_only",
        wandb_project=wandb_project
    )
    results["latex_ocr_only"] = (model1, processor1)
    
    # Setup 2: LaTeX_OCR + MathWriting
    print("\n" + "=" * 60)
    print("SETUP 2: SFT with LaTeX_OCR + MathWriting")
    print("=" * 60)
    
    config_setup2 = TrainConfig(
        **{k: v for k, v in config.__dict__.items()},
    )
    config_setup2.use_secondary = True
    
    model2, processor2 = train(
        config_setup2,
        run_name="sft_latex_ocr_mathwriting",
        wandb_project=wandb_project
    )
    results["latex_ocr_mathwriting"] = (model2, processor2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Handwritten LaTeX OCR model")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name from config"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Override dataset subset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--use_secondary",
        action="store_true",
        help="Include MathWriting dataset"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="sft_latex_ocr",
        help="Name for this training run"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name"
    )
    parser.add_argument(
        "--train_all",
        action="store_true",
        help="Train all experimental setups"
    )
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = TrainConfig()
    
    # Override with command line arguments
    if args.model_name:
        config.model_name = args.model_name
    if args.dataset:
        config.primary_dataset = args.dataset
    if args.subset:
        config.primary_subset = args.subset
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.use_secondary:
        config.use_secondary = True
    
    # Run training
    if args.train_all:
        train_all_setups(config, args.wandb_project)
    else:
        train(config, args.run_name, args.wandb_project)


if __name__ == "__main__":
    main()
