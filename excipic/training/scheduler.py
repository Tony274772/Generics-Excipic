"""
Cosine Decay Scheduler with Linear Warmup.
"""
import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_warmup_scheduler(optimizer, num_warmup_steps: int,
                                 num_total_steps: int, min_lr_ratio: float = 0.01):
    """
    Create a cosine decay scheduler with linear warmup.

    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps: Number of warmup steps (linear increase).
        num_total_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as fraction of initial LR.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_total_steps - num_warmup_steps)
            )
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
