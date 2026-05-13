"""
Evaluation Metrics for Excipic.

Primary metric: NDCG@10 (drives early stopping and checkpoint selection).
Additional: Recall@K, Hit@K, MRR, Jaccard@K, Exact Match.
All reported at K = 5, 10, 15.
"""
import numpy as np
import torch
from typing import Dict, List, Optional


def compute_ndcg(predicted: List[int], target: set, k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    Args:
        predicted: Ranked list of predicted excipient indices.
        target: Set of ground-truth excipient indices.
        k: Cutoff.

    Returns:
        NDCG@K score in [0, 1].
    """
    predicted = predicted[:k]

    # DCG
    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in target:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG (all relevant items ranked first)
    ideal_len = min(len(target), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_recall(predicted: List[int], target: set, k: int) -> float:
    """Recall@K: fraction of target items found in top-K predictions."""
    predicted_set = set(predicted[:k])
    if len(target) == 0:
        return 0.0
    return len(predicted_set & target) / len(target)


def compute_hit(predicted: List[int], target: set, k: int) -> float:
    """Hit@K: 1 if at least one target item is in top-K predictions."""
    predicted_set = set(predicted[:k])
    return 1.0 if len(predicted_set & target) > 0 else 0.0


def compute_mrr(predicted: List[int], target: set) -> float:
    """MRR: 1/rank of the first correct prediction."""
    for i, p in enumerate(predicted):
        if p in target:
            return 1.0 / (i + 1)
    return 0.0


def compute_jaccard(predicted: List[int], target: set, k: int) -> float:
    """Jaccard@K: |intersection| / |union| of predicted@K and target."""
    predicted_set = set(predicted[:k])
    intersection = predicted_set & target
    union = predicted_set | target
    if len(union) == 0:
        return 0.0
    return len(intersection) / len(union)


def compute_exact_match(predicted: List[int], target: set) -> float:
    """Exact Match: 1 if predicted set == target set exactly."""
    predicted_set = set(predicted[:len(target)])
    return 1.0 if predicted_set == target else 0.0


def filter_special_tokens(indices: List[int], special_indices: set) -> List[int]:
    """Remove special tokens (PAD, BOS, EOS, UNK) from predictions."""
    return [i for i in indices if i not in special_indices]


class ExcipicEvaluator:
    """
    Computes all evaluation metrics for Excipic predictions.

    Usage:
        evaluator = ExcipicEvaluator(pad_idx=0, bos_idx=1, eos_idx=2, unk_idx=3)
        for batch in dataloader:
            predictions = model.predict(batch)
            evaluator.update(predictions, batch)
        metrics = evaluator.compute()
    """

    def __init__(self, pad_idx: int, bos_idx: int, eos_idx: int, unk_idx: int,
                 k_values: List[int] = None):
        self.special_indices = {pad_idx, bos_idx, eos_idx, unk_idx}
        self.k_values = k_values or [5, 10, 15]

        self.all_ndcg = {k: [] for k in self.k_values}
        self.all_recall = {k: [] for k in self.k_values}
        self.all_hit = {k: [] for k in self.k_values}
        self.all_mrr = []
        self.all_jaccard = {k: [] for k in self.k_values}
        self.all_exact_match = []

    def reset(self):
        """Reset accumulated metrics."""
        for k in self.k_values:
            self.all_ndcg[k] = []
            self.all_recall[k] = []
            self.all_hit[k] = []
            self.all_jaccard[k] = []
        self.all_mrr = []
        self.all_exact_match = []

    def update(
        self,
        predicted_ids: torch.Tensor,
        target_multi_hot: torch.Tensor,
    ):
        """
        Update metrics with a batch of predictions.

        Args:
            predicted_ids: [B, T] generated excipient indices
            target_multi_hot: [B, V] ground-truth multi-hot vectors
        """
        B = predicted_ids.size(0)

        for b in range(B):
            # Get predicted list (remove special tokens)
            pred_list = predicted_ids[b].cpu().tolist()
            pred_list = filter_special_tokens(pred_list, self.special_indices)

            # Get target set (from multi-hot)
            target_set = set(
                torch.where(target_multi_hot[b] > 0)[0].cpu().tolist()
            )
            target_set -= self.special_indices

            if len(target_set) == 0:
                continue

            # Compute metrics at each K
            for k in self.k_values:
                self.all_ndcg[k].append(compute_ndcg(pred_list, target_set, k))
                self.all_recall[k].append(compute_recall(pred_list, target_set, k))
                self.all_hit[k].append(compute_hit(pred_list, target_set, k))
                self.all_jaccard[k].append(compute_jaccard(pred_list, target_set, k))

            self.all_mrr.append(compute_mrr(pred_list, target_set))
            self.all_exact_match.append(compute_exact_match(pred_list, target_set))

    def compute(self) -> Dict[str, float]:
        """
        Compute averaged metrics.

        Returns:
            Dict of metric_name → value.
        """
        metrics = {}

        for k in self.k_values:
            if self.all_ndcg[k]:
                metrics[f"ndcg_at_{k}"] = np.mean(self.all_ndcg[k])
            if self.all_recall[k]:
                metrics[f"recall_at_{k}"] = np.mean(self.all_recall[k])
            if self.all_hit[k]:
                metrics[f"hit_at_{k}"] = np.mean(self.all_hit[k])
            if self.all_jaccard[k]:
                metrics[f"jaccard_at_{k}"] = np.mean(self.all_jaccard[k])

        if self.all_mrr:
            metrics["mrr"] = np.mean(self.all_mrr)
        if self.all_exact_match:
            metrics["exact_match"] = np.mean(self.all_exact_match)

        return metrics
