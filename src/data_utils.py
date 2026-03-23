"""
Data utilities for loading and preprocessing datasets.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from PIL import Image
import torch
from torch.utils.data import DataLoader


@dataclass
class DataConfig:
    """Configuration for data loading."""
    primary_dataset: str = "linxy/LaTeX_OCR"
    primary_subset: str = "human_handwrite"
    secondary_dataset: str = "deepcopy/MathWriting-human"
    use_secondary: bool = False
    secondary_sample_size: int = 10000
    max_samples_train: Optional[int] = None
    max_samples_val: Optional[int] = None
    image_size: int = 384
    max_length: int = 512


def load_latex_ocr_dataset(
    subset: str = "human_handwrite",
    split: Optional[str] = None
) -> Union[Dataset, DatasetDict]:
    """
    Load the LaTeX_OCR dataset.
    
    Args:
        subset: One of 'default', 'full', 'human_handwrite', 
                'human_handwrite_print', 'small', 'synthetic_handwrite'
        split: Optional split name ('train', 'test', 'validation')
    
    Returns:
        Dataset or DatasetDict
    """
    print(f"Loading LaTeX_OCR dataset with subset: {subset}")
    
    if split:
        ds = load_dataset("linxy/LaTeX_OCR", subset, split=split)
    else:
        ds = load_dataset("linxy/LaTeX_OCR", subset)
    
    print(f"Loaded dataset: {ds}")
    return ds


def load_mathwriting_dataset(
    split: Optional[str] = None,
    sample_size: Optional[int] = None
) -> Union[Dataset, DatasetDict]:
    """
    Load the MathWriting-human dataset.
    
    Args:
        split: Optional split name ('train', 'test', 'val')
        sample_size: Optional number of samples to take (for large datasets)
    
    Returns:
        Dataset or DatasetDict
    """
    print("Loading MathWriting-human dataset...")
    
    if split:
        ds = load_dataset("deepcopy/MathWriting-human", split=split)
    else:
        ds = load_dataset("deepcopy/MathWriting-human")
    
    if sample_size and isinstance(ds, Dataset) and len(ds) > sample_size:
        ds = ds.shuffle(seed=42).select(range(sample_size))
        print(f"Sampled {sample_size} examples from dataset")
    
    print(f"Loaded dataset: {ds}")
    return ds


def preprocess_latex_ocr_example(example: Dict, processor, config: DataConfig) -> Dict:
    """
    Preprocess a single example from LaTeX_OCR dataset.
    
    Expected format:
    - 'image': PIL Image or path
    - 'text' or 'latex' or 'formula': LaTeX string
    """
    # Get image
    image = example.get("image")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, 'convert'):
        image = image.convert("RGB")
    
    # Get LaTeX text (check different possible column names)
    latex = example.get("text") or example.get("latex") or example.get("formula", "")
    
    return {
        "image": image,
        "latex": latex.strip()
    }


def preprocess_mathwriting_example(example: Dict, processor, config: DataConfig) -> Dict:
    """
    Preprocess a single example from MathWriting dataset.
    
    Expected format may differ - adjust based on actual dataset structure.
    """
    # Get image
    image = example.get("image")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif hasattr(image, 'convert'):
        image = image.convert("RGB")
    
    # Get LaTeX text
    latex = example.get("latex") or example.get("text") or example.get("formula", "")
    
    return {
        "image": image,
        "latex": latex.strip()
    }


class HandwrittenLatexDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for handwritten LaTeX OCR.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        processor,
        config: DataConfig,
        is_mathwriting: bool = False,
        system_prompt: str = "Convert the handwritten formula to LaTeX.",
        user_prompt: str = "Convert this handwritten formula to LaTeX:"
    ):
        self.dataset = dataset
        self.processor = processor
        self.config = config
        self.is_mathwriting = is_mathwriting
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.dataset[idx]
        
        if self.is_mathwriting:
            processed = preprocess_mathwriting_example(example, self.processor, self.config)
        else:
            processed = preprocess_latex_ocr_example(example, self.processor, self.config)
        
        return {
            "image": processed["image"],
            "latex": processed["latex"],
            "prompt": self.user_prompt
        }


def create_chat_messages(
    user_prompt: str,
    latex_output: Optional[str] = None,
    system_prompt: str = "You are a LaTeX OCR assistant. Convert handwritten formulas to LaTeX.",
    one_shot_example: Optional[Dict] = None
) -> List[Dict]:
    """
    Create chat messages for the model.
    
    Args:
        user_prompt: User's request
        latex_output: Expected LaTeX output (for training)
        system_prompt: System instruction
        one_shot_example: Optional one-shot example dict with 'user' and 'assistant' keys
    
    Returns:
        List of message dicts
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add one-shot example if provided
    if one_shot_example:
        messages.append({
            "role": "user",
            "content": one_shot_example.get("user", "Convert: x² + y² = z²")
        })
        messages.append({
            "role": "assistant", 
            "content": one_shot_example.get("assistant", "x^2 + y^2 = z^2")
        })
    
    # Add current query
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    # Add expected output for training
    if latex_output:
        messages.append({
            "role": "assistant",
            "content": latex_output
        })
    
    return messages


def collate_fn(batch: List[Dict], processor, max_length: int = 512) -> Dict:
    """
    Collate function for DataLoader.
    """
    images = [item["image"] for item in batch]
    latex_texts = [item["latex"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    
    # Process images and text together
    # This will vary based on the specific model/processor being used
    
    return {
        "images": images,
        "latex_texts": latex_texts,
        "prompts": prompts
    }


def get_dataloader(
    dataset: Dataset,
    processor,
    config: DataConfig,
    batch_size: int = 4,
    shuffle: bool = True,
    is_mathwriting: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    """
    torch_dataset = HandwrittenLatexDataset(
        dataset=dataset,
        processor=processor,
        config=config,
        is_mathwriting=is_mathwriting
    )
    
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, processor, config.max_length),
        num_workers=4,
        pin_memory=True
    )


def prepare_datasets(config: DataConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Prepare train, validation, and test datasets.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load primary dataset
    primary_ds = load_latex_ocr_dataset(subset=config.primary_subset)
    
    train_ds = primary_ds["train"]
    val_ds = primary_ds["validation"]
    test_ds = primary_ds["test"]
    
    # Optionally limit samples
    if config.max_samples_train and len(train_ds) > config.max_samples_train:
        train_ds = train_ds.shuffle(seed=42).select(range(config.max_samples_train))
    
    if config.max_samples_val and len(val_ds) > config.max_samples_val:
        val_ds = val_ds.shuffle(seed=42).select(range(config.max_samples_val))
    
    # Optionally add secondary dataset
    if config.use_secondary:
        secondary_ds = load_mathwriting_dataset(
            split="train",
            sample_size=config.secondary_sample_size
        )
        
        # Normalize column names before concatenation
        # This may need adjustment based on actual column names
        train_ds = concatenate_datasets([train_ds, secondary_ds])
        train_ds = train_ds.shuffle(seed=42)
    
    print(f"Train size: {len(train_ds)}")
    print(f"Validation size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")
    
    return train_ds, val_ds, test_ds


def inspect_dataset_structure(dataset_name: str, subset: Optional[str] = None):
    """
    Utility function to inspect dataset structure.
    Useful for debugging and understanding data format.
    """
    print(f"\n{'='*50}")
    print(f"Inspecting: {dataset_name}" + (f" ({subset})" if subset else ""))
    print('='*50)
    
    if subset:
        ds = load_dataset(dataset_name, subset, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    
    # Get first example
    example = next(iter(ds))
    
    print("\nColumns/Keys:", list(example.keys()))
    print("\nExample:")
    for key, value in example.items():
        if isinstance(value, Image.Image):
            print(f"  {key}: PIL Image, size={value.size}, mode={value.mode}")
        elif isinstance(value, str):
            print(f"  {key}: '{value[:100]}{'...' if len(value) > 100 else ''}'")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")


if __name__ == "__main__":
    # Test dataset loading
    print("Testing dataset loading...")
    
    # Inspect LaTeX_OCR structure
    inspect_dataset_structure("linxy/LaTeX_OCR", "human_handwrite")
    
    # Inspect MathWriting structure  
    inspect_dataset_structure("deepcopy/MathWriting-human")
