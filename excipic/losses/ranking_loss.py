"""
ListNet Ranking Loss.

Optimizes ranking quality by minimizing the cross-entropy between
the predicted probability distribution and the target distribution
over excipient rankings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ListNetLoss(nn.Module):
    """
    ListNet loss for learning-to-rank.

    Computes cross-entropy between the top-1 probability distributions
    of predicted scores and ground-truth relevance scores.

    This aligns with NDCG optimization.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted scores [B, T, V] (decoder output)
            targets: Target indices [B, T]
            mask: Valid position mask [B, T] (True for valid positions)

        Returns:
            Scalar loss.
        """
        B, T, V = logits.shape
        device = logits.device

        if mask is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Create relevance scores from target sequence
        # Earlier positions get higher relevance (ranking matters)
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            valid_len = mask[b].sum().item()
            if valid_len == 0:
                continue

            for t in range(int(valid_len)):
                pred_dist = F.log_softmax(logits[b, t] / self.temperature, dim=-1)
                target_idx = targets[b, t]

                # Cross-entropy at this position
                loss = -pred_dist[target_idx]
                total_loss = total_loss + loss
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss


class ListMLELoss(nn.Module):
    """
    ListMLE loss — likelihood of the target permutation.

    More directly optimizes ranking quality than ListNet.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, T, V] decoder logits
            targets: [B, T] target excipient indices
            mask: [B, T] valid positions

        Returns:
            Scalar loss.
        """
        B, T, V = logits.shape
        device = logits.device

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            valid_len = mask[b].sum().item()
            if valid_len <= 1:
                continue

            # For each position, compute log-probability of the correct item
            # given that previously selected items are removed
            selected = set()
            for t in range(int(valid_len)):
                scores = logits[b, t].clone()

                # Mask previously selected items
                for prev_idx in selected:
                    scores[prev_idx] = float("-inf")

                log_prob = F.log_softmax(scores, dim=-1)
                target_idx = targets[b, t].item()

                total_loss = total_loss - log_prob[target_idx]
                selected.add(target_idx)
                count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss

