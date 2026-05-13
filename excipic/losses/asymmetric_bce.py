"""
Asymmetric Binary Cross-Entropy Loss.

Designed for severe class imbalance where positive examples are sparse.
Uses different focusing parameters for positive and negative samples.
"""
import torch
import torch.nn as nn


class AsymmetricBCELoss(nn.Module):
    """
    Asymmetric BCE Loss with focusing.

    For excipient prediction:
        - ~598 possible excipients, only ~9 are positive per sample
        - gamma_neg = 4 → down-weight easy negatives aggressively
        - gamma_pos = 1 → mild focusing on hard positives

    Based on: "Asymmetric Loss For Multi-Label Classification" (Ridnik et al.)
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0,
                 clip: float = 0.05, eps: float = 1e-8):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples.
            gamma_pos: Focusing parameter for positive samples.
            clip: Probability margin for hard thresholding negatives.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Raw predictions [B, V] (before sigmoid)
            targets: Multi-hot targets [B, V]

        Returns:
            Scalar loss.
        """
        # Sigmoid probabilities
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric clipping for negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # Basic cross-entropy terms
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=self.eps))

        # Asymmetric focusing
        if self.gamma_pos > 0:
            # Focus on hard positives (low probability)
            pt_pos = probs_pos * targets + (1 - probs_pos) * (1 - targets)
            focal_weight_pos = (1 - probs_pos).pow(self.gamma_pos)
            loss_pos = loss_pos * focal_weight_pos

        if self.gamma_neg > 0:
            # Focus on hard negatives (high probability)
            focal_weight_neg = probs.pow(self.gamma_neg)
            loss_neg = loss_neg * focal_weight_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()
