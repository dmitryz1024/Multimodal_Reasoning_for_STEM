"""
Training script for Handwritten LaTeX OCR.
"""

import os
import argparse
import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass

import torch
import yaml
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    TrainerCallback,
    EarlyStoppingCallback,
)

# Logging setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj", "k_proj", "o_proj")
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    
    # Data config
    primary_dataset: str = "linxy/LaTeX_OCR"
    primary_subset: Optional[str] = "human_handwrite"
    secondary_dataset: str = "deepcopy/MathWriting-human"
    use_secondary: bool = False
    secondary_sample_size: int = 10000
    max_samples_train: Optional[int] = None
    max_samples_val: Optional[int] = None
    image_size: int = 384
    max_length: int = 512
    
    # Training config
    output_dir: str = "./checkpoints"
    num_epochs: int = 3
    batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Misc
    seed: int = 42
    logging_steps: int = 10
    logging_strategy: str = "steps"
    logging_dir: str = "./logs"
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    use_wandb: bool = False
    wandb_project: Optional[str] = None


class DetailedProgressCallback(TrainerCallback):
    """Callback to print detailed training progress and ETA."""

    def __init__(self):
        self.step_times = []
        self.last_printed_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        logger.info("Training started: %s", args.run_name)

    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        now = time.time()
        if state.global_step > 0:
            self.step_times.append(now - self._last_step_time if hasattr(self, '_last_step_time') else 0)
        self._last_step_time = now

        total = float(state.max_steps)
        completed = float(state.global_step)
        pct = (completed / total) * 100 if total > 0 else 0

        avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        remaining_steps = max(state.max_steps - state.global_step, 0)
        eta = remaining_steps * avg_step

        loss = logs.get('loss', None)
        info = f"Epoch {state.epoch:.2f} | Step {state.global_step}/{state.max_steps} ({pct:.1f}%)"
        if loss is not None:
            info += f" | loss={loss:.4f}"
        if eta >= 0 and avg_step > 0:
            info += f" | ETA {int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"

        # Log every logging_steps or at end of epoch
        if state.global_step % args.logging_steps == 0 or state.global_step == state.max_steps:
            logger.info(info)

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start_time
        logger.info("Epoch %.2f complete. Total elapsed time: %s", state.epoch, format_time(elapsed))

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self.train_start_time
        logger.info("Training finished. Total time: %s", format_time(total))


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_config(config_path: str) -> TrainConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested config
    flat_config = {}
    for section in ['model', 'data', 'training', 'preprocessing', 'logging']:
        if section in config_dict:
            flat_config.update(config_dict[section])
    
    # Map to TrainConfig fields
    return TrainConfig(
        model_name=flat_config.get('name', TrainConfig.model_name),
        use_lora=flat_config.get('use_lora', TrainConfig.use_lora),
        lora_r=flat_config.get('lora_r', TrainConfig.lora_r),
        lora_alpha=flat_config.get('lora_alpha', TrainConfig.lora_alpha),
        lora_dropout=flat_config.get('lora_dropout', TrainConfig.lora_dropout),
        lora_target_modules=tuple(flat_config.get('lora_target_modules', TrainConfig.lora_target_modules)),
        load_in_4bit=flat_config.get('load_in_4bit', TrainConfig.load_in_4bit),
        load_in_8bit=flat_config.get('load_in_8bit', TrainConfig.load_in_8bit),
        primary_dataset=flat_config.get('primary_dataset', TrainConfig.primary_dataset),
        primary_subset=flat_config.get('primary_subset', TrainConfig.primary_subset),
        secondary_dataset=flat_config.get('secondary_dataset', TrainConfig.secondary_dataset),
        use_secondary=flat_config.get('use_secondary', TrainConfig.use_secondary),
        secondary_sample_size=flat_config.get('secondary_sample_size', TrainConfig.secondary_sample_size),
        max_samples_train=flat_config.get('max_samples_train', TrainConfig.max_samples_train),
        max_samples_val=flat_config.get('max_samples_val', TrainConfig.max_samples_val),
        image_size=flat_config.get('image_size', TrainConfig.image_size),
        max_length=flat_config.get('max_length', TrainConfig.max_length),
        output_dir=flat_config.get('output_dir', TrainConfig.output_dir),
        num_epochs=flat_config.get('num_epochs', TrainConfig.num_epochs),
        batch_size=flat_config.get('per_device_train_batch_size', TrainConfig.batch_size),
        eval_batch_size=flat_config.get('per_device_eval_batch_size', TrainConfig.eval_batch_size),
        gradient_accumulation_steps=flat_config.get('gradient_accumulation_steps', TrainConfig.gradient_accumulation_steps),
        learning_rate=flat_config.get('learning_rate', TrainConfig.learning_rate),
        weight_decay=flat_config.get('weight_decay', TrainConfig.weight_decay),
        warmup_ratio=flat_config.get('warmup_ratio', TrainConfig.warmup_ratio),
        max_grad_norm=flat_config.get('max_grad_norm', TrainConfig.max_grad_norm),
        bf16=flat_config.get('bf16', TrainConfig.bf16),
        fp16=flat_config.get('fp16', TrainConfig.fp16),
        gradient_checkpointing=flat_config.get('gradient_checkpointing', TrainConfig.gradient_checkpointing),
        save_total_limit=flat_config.get('save_total_limit', TrainConfig.save_total_limit),
        load_best_model_at_end=flat_config.get('load_best_model_at_end', TrainConfig.load_best_model_at_end),
        metric_for_best_model=flat_config.get('metric_for_best_model', TrainConfig.metric_for_best_model),
        greater_is_better=flat_config.get('greater_is_better', TrainConfig.greater_is_better),
        seed=flat_config.get('seed', TrainConfig.seed),
        logging_steps=flat_config.get('logging_steps', TrainConfig.logging_steps),
        save_strategy=flat_config.get('save_strategy', TrainConfig.save_strategy),
        eval_strategy=flat_config.get('eval_strategy', TrainConfig.eval_strategy),
        use_wandb=flat_config.get('use_wandb', TrainConfig.use_wandb),
        wandb_project=flat_config.get('wandb_project', TrainConfig.wandb_project),
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


def create_training_args(config: TrainConfig, run_name: str, use_wandb: bool = False) -> TrainingArguments:
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
            print("WARNING: bf16 not supported on this GPU; falling back to fp16.")
            bf16 = False
            fp16 = True

    return TrainingArguments(
        output_dir=os.path.join(config.output_dir, run_name),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        logging_strategy=config.logging_strategy,
        logging_dir=os.path.join(config.logging_dir, run_name),
        save_strategy=config.save_strategy,
        eval_strategy=config.eval_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        report_to=["wandb"] if use_wandb else ["none"],
        run_name=run_name,
        seed=config.seed,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        disable_tqdm=False,
    )


def disable_wandb() -> None:
    """Force-disable Weights & Biases integration for this process."""
    os.environ["WANDB_DISABLED"] = "true"
    os.environ.pop("WANDB_PROJECT", None)


def maybe_enable_wandb(config: TrainConfig, run_name: str, wandb_project: Optional[str] = None) -> bool:
    """
    Initialize Weights & Biases if explicitly enabled.

    Returns:
        True if wandb logging is active, otherwise False.
    """
    project_name = wandb_project if wandb_project is not None else config.wandb_project
    if not config.use_wandb or not project_name:
        disable_wandb()
        return False

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_PROJECT"] = project_name

    try:
        import wandb

        wandb.init(project=project_name, name=run_name)
        logger.info("Weights & Biases logging enabled for project: %s", project_name)
        return True
    except Exception as exc:
        logger.warning(
            "Weights & Biases is unavailable (%s). Continuing without wandb logging.",
            exc,
        )
        disable_wandb()
        return False


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
    logger.info("%s", "=" * 60)
    logger.info("Starting training: %s", run_name)
    logger.info("%s", "=" * 60)
    
    use_wandb = maybe_enable_wandb(config, run_name, wandb_project)
    
    # Load model and processor
    model_config = ModelConfig(
        name=config.model_name,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
    )
    
    model, processor = load_model_and_processor(model_config, for_training=True)
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load datasets
    logger.info("Loading datasets...")
    primary_ds = load_latex_ocr_dataset(
        dataset_name=config.primary_dataset,
        subset=config.primary_subset
    )
    train_dataset = primary_ds["train"]
    eval_dataset = primary_ds["validation"]

    if config.max_samples_train and len(train_dataset) > config.max_samples_train:
        train_dataset = train_dataset.shuffle(seed=config.seed).select(range(config.max_samples_train))
    if config.max_samples_val and len(eval_dataset) > config.max_samples_val:
        eval_dataset = eval_dataset.shuffle(seed=config.seed).select(range(config.max_samples_val))
    
    # Optionally add secondary dataset
    if config.use_secondary:
        print(f"\nAdding secondary dataset (sample size: {config.secondary_sample_size})...")
        secondary_ds = load_mathwriting_dataset(
            dataset_name=config.secondary_dataset,
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
        max_length=config.max_length
    )
    
    # Create training arguments
    training_args = create_training_args(config, run_name, use_wandb=use_wandb)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            DetailedProgressCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    final_output_dir = os.path.join(config.output_dir, run_name, "final")
    save_model(model, processor, final_output_dir)
    
    logger.info("Training complete! Model saved to: %s", final_output_dir)
    
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
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases even if enabled in the config"
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
        dataset_changed = config.primary_dataset != args.dataset
        config.primary_dataset = args.dataset
        if dataset_changed and not args.subset:
            config.primary_subset = None
    if args.subset:
        config.primary_subset = args.subset
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.use_secondary:
        config.use_secondary = True
    if args.wandb_project:
        config.use_wandb = True
        config.wandb_project = args.wandb_project
    if args.no_wandb:
        config.use_wandb = False
        config.wandb_project = None

    # Run training
    if args.train_all:
        train_all_setups(config, args.wandb_project)
    else:
        train(config, args.run_name, args.wandb_project)


if __name__ == "__main__":
    main()
