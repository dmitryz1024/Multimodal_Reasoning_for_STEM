"""
Evaluation metrics for LaTeX OCR.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter

import editdistance
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def normalize_latex(latex: str) -> str:
    """
    Normalize LaTeX string for comparison.
    Handles common variations that are mathematically equivalent.
    
    Args:
        latex: Raw LaTeX string
    
    Returns:
        Normalized LaTeX string
    """
    if not latex:
        return ""
    
    # Remove leading/trailing whitespace
    latex = latex.strip()
    
    # Remove surrounding $, $$, \[, \], etc.
    latex = re.sub(r'^[\$]+|[\$]+$', '', latex)
    latex = re.sub(r'^\\\[|\\\]$', '', latex)
    latex = re.sub(r'^\\\(|\\\)$', '', latex)
    
    # Normalize whitespace
    latex = re.sub(r'\s+', ' ', latex)
    
    # Remove spaces around operators and brackets
    latex = re.sub(r'\s*([+\-*/=^_{}()\[\]])\s*', r'\1', latex)
    
    # Normalize fractions: \frac {a} {b} -> \frac{a}{b}
    latex = re.sub(r'\\frac\s*', r'\\frac', latex)
    
    # Normalize common commands
    latex = re.sub(r'\\left\s*', r'\\left', latex)
    latex = re.sub(r'\\right\s*', r'\\right', latex)
    
    # Remove unnecessary braces: {x} -> x (single characters)
    latex = re.sub(r'\{([a-zA-Z0-9])\}', r'\1', latex)
    
    return latex.strip()


def tokenize_latex(latex: str) -> List[str]:
    """
    Tokenize LaTeX string into meaningful tokens.
    
    Args:
        latex: LaTeX string
    
    Returns:
        List of tokens
    """
    # Pattern to match LaTeX tokens
    pattern = r'(\\[a-zA-Z]+|\\.|[{}()\[\]^_]|[a-zA-Z0-9]+|[+\-*/=<>!,.])'
    
    tokens = re.findall(pattern, latex)
    return tokens


def exact_match(prediction: str, reference: str, normalize: bool = True) -> float:
    """
    Compute exact match score.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
        normalize: Whether to normalize before comparison
    
    Returns:
        1.0 if match, 0.0 otherwise
    """
    if normalize:
        prediction = normalize_latex(prediction)
        reference = normalize_latex(reference)
    
    return 1.0 if prediction == reference else 0.0


def edit_distance_score(prediction: str, reference: str, normalize: bool = True) -> float:
    """
    Compute normalized edit distance (Levenshtein distance).
    Returns similarity score (1 - normalized_distance).
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
        normalize: Whether to normalize LaTeX before comparison
    
    Returns:
        Similarity score between 0 and 1
    """
    if normalize:
        prediction = normalize_latex(prediction)
        reference = normalize_latex(reference)
    
    if not reference:
        return 1.0 if not prediction else 0.0
    
    distance = editdistance.eval(prediction, reference)
    max_len = max(len(prediction), len(reference))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)


def token_edit_distance(prediction: str, reference: str) -> float:
    """
    Compute edit distance at token level rather than character level.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        Similarity score between 0 and 1
    """
    pred_tokens = tokenize_latex(normalize_latex(prediction))
    ref_tokens = tokenize_latex(normalize_latex(reference))
    
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    distance = editdistance.eval(pred_tokens, ref_tokens)
    max_len = max(len(pred_tokens), len(ref_tokens))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)


def bleu_score(prediction: str, reference: str, n_gram: int = 4) -> float:
    """
    Compute BLEU score for LaTeX generation.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
        n_gram: Maximum n-gram order
    
    Returns:
        BLEU score between 0 and 1
    """
    pred_tokens = tokenize_latex(normalize_latex(prediction))
    ref_tokens = tokenize_latex(normalize_latex(reference))
    
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    if not pred_tokens:
        return 0.0
    
    # Use smoothing for short sequences
    smoothing = SmoothingFunction()
    
    # Compute BLEU with smoothing
    weights = tuple([1.0 / n_gram] * n_gram)
    
    try:
        score = sentence_bleu(
            [ref_tokens],
            pred_tokens,
            weights=weights,
            smoothing_function=smoothing.method1
        )
    except Exception:
        score = 0.0
    
    return score


def token_accuracy(prediction: str, reference: str) -> float:
    """
    Compute token-level accuracy.
    Measures the proportion of correctly predicted tokens.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        Token accuracy between 0 and 1
    """
    pred_tokens = tokenize_latex(normalize_latex(prediction))
    ref_tokens = tokenize_latex(normalize_latex(reference))
    
    if not ref_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    # Count matching tokens
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Find common tokens
    common = sum((pred_counter & ref_counter).values())
    
    # Accuracy based on reference length
    accuracy = common / len(ref_tokens)
    
    return min(1.0, accuracy)


def f1_score_tokens(prediction: str, reference: str) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score at token level.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_tokens = tokenize_latex(normalize_latex(prediction))
    ref_tokens = tokenize_latex(normalize_latex(reference))
    
    if not ref_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    
    if not ref_tokens:
        return 0.0, 0.0, 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    common = sum((pred_counter & ref_counter).values())
    
    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(ref_tokens) if ref_tokens else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def compute_all_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute all metrics for a single prediction-reference pair.
    
    Args:
        prediction: Predicted LaTeX
        reference: Ground truth LaTeX
    
    Returns:
        Dictionary of metric names to values
    """
    precision, recall, f1 = f1_score_tokens(prediction, reference)
    
    return {
        "exact_match": exact_match(prediction, reference),
        "bleu": bleu_score(prediction, reference),
        "edit_distance": edit_distance_score(prediction, reference),
        "token_edit_distance": token_edit_distance(prediction, reference),
        "token_accuracy": token_accuracy(prediction, reference),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def compute_corpus_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute aggregate metrics over a corpus.
    
    Args:
        predictions: List of predicted LaTeX strings
        references: List of ground truth LaTeX strings
    
    Returns:
        Dictionary of metric names to average values
    """
    assert len(predictions) == len(references), \
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
    
    if not predictions:
        return {}
    
    # Collect individual metrics
    all_metrics = [
        compute_all_metrics(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    
    # Average
    metric_names = all_metrics[0].keys()
    averaged = {
        name: sum(m[name] for m in all_metrics) / len(all_metrics)
        for name in metric_names
    }
    
    return averaged


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format metrics dictionary as a readable string.
    """
    lines = ["Evaluation Metrics:"]
    lines.append("-" * 40)
    
    for name, value in metrics.items():
        formatted_name = name.replace("_", " ").title()
        lines.append(f"  {formatted_name}: {value:.{precision}f}")
    
    return "\n".join(lines)


class MetricTracker:
    """
    Track and accumulate metrics during evaluation.
    """
    
    def __init__(self):
        self.predictions: List[str] = []
        self.references: List[str] = []
        self._cached_metrics: Optional[Dict[str, float]] = None
    
    def add(self, prediction: str, reference: str):
        """Add a prediction-reference pair."""
        self.predictions.append(prediction)
        self.references.append(reference)
        self._cached_metrics = None
    
    def add_batch(self, predictions: List[str], references: List[str]):
        """Add a batch of predictions and references."""
        self.predictions.extend(predictions)
        self.references.extend(references)
        self._cached_metrics = None
    
    def compute(self) -> Dict[str, float]:
        """Compute metrics over all accumulated examples."""
        if self._cached_metrics is None:
            self._cached_metrics = compute_corpus_metrics(
                self.predictions,
                self.references
            )
        return self._cached_metrics
    
    def reset(self):
        """Reset the tracker."""
        self.predictions = []
        self.references = []
        self._cached_metrics = None
    
    def __len__(self) -> int:
        return len(self.predictions)
    
    def __str__(self) -> str:
        return format_metrics(self.compute())


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    test_cases = [
        ("x^2 + y^2 = z^2", "x^2 + y^2 = z^2"),  # Exact match
        ("x^{2} + y^{2} = z^{2}", "x^2 + y^2 = z^2"),  # Equivalent
        ("\\frac{a}{b}", "\\frac{a}{b}"),  # Fraction
        ("\\frac {a} {b}", "\\frac{a}{b}"),  # Fraction with spaces
        ("x + y", "x - y"),  # One character different
        ("\\int_0^1 x dx", "\\int_0^1 x \\, dx"),  # Minor difference
    ]
    
    for pred, ref in test_cases:
        print(f"\nPrediction: {pred}")
        print(f"Reference:  {ref}")
        metrics = compute_all_metrics(pred, ref)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
