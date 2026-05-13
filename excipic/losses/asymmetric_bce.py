"""
Asymmetric Binary Cross-Entropy Loss.

Designed for severe class imbalance where positive examples are sparse.
Uses different focusing parameters for positive and negative samples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Sanitize inputs to avoid NaNs/Infs propagating through BCE math.
        targets = torch.nan_to_num(targets.float(), nan=0.0, posinf=1.0, neginf=0.0)
        targets = targets.clamp_(0.0, 1.0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        logits = logits.clamp(-30.0, 30.0)

        # Sigmoid probabilities (safe due to logits clamping)
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Asymmetric clipping for negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        # Numerically stable BCE terms.
        # log(sigmoid(x)) = -softplus(-x), log(1-sigmoid(x)) = -softplus(x)
        log_p = -F.softplus(-logits)
        log_1m_p = torch.log(probs_neg.clamp(min=self.eps))
        loss_pos = targets * log_p
        loss_neg = (1 - targets) * log_1m_p

        # Asymmetric focusing
        if self.gamma_pos > 0:
            # Focus on hard positives (low probability)
            focal_weight_pos = (1 - probs_pos).pow(self.gamma_pos)
            loss_pos = loss_pos * focal_weight_pos

        if self.gamma_neg > 0:
            # Focus on hard negatives (high probability)
            focal_weight_neg = probs.pow(self.gamma_neg)
            loss_neg = loss_neg * focal_weight_neg

        loss = -(loss_pos + loss_neg)
        return loss.mean()
