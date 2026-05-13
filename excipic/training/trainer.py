"""
Excipic Trainer.

Handles the full training loop with:
    - Differential learning rates per module
    - Mixed precision (AMP)
    - Gradient clipping
    - Evaluation and early stopping on NDCG@10
    - Checkpoint saving/loading
    - GPU/CPU memory monitoring
    - Epoch-wise progress tracking
    - Logging
"""
import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from ..losses.ranking_loss import ListNetLoss
from ..losses.asymmetric_bce import AsymmetricBCELoss
from ..metrics.evaluation import ExcipicEvaluator
from .scheduler import get_cosine_warmup_scheduler

logger = logging.getLogger(__name__)


class ExcipicTrainer:
    """
    Training loop for the Excipic model.

    Supports:
        - Per-module learning rates (AdamW)
        - Cosine warmup scheduling
        - Mixed-precision training (fp16/bf16)
        - Combined loss: 0.6*Ranking + 0.3*AsymBCE + 0.1*PropertyMSE
        - Early stopping on validation NDCG@10
    """

    def __init__(self, model, config, preprocessor):
        self.model = model
        self.config = config
        self.train_config = config.training
        self.preprocessor = preprocessor

        self.device = torch.device(
            self.train_config.device if torch.cuda.is_available() else "cpu"
        )

        # Loss functions
        self.ranking_loss = ListNetLoss(temperature=1.0)
        self.bce_loss = AsymmetricBCELoss(
            gamma_neg=self.train_config.gamma_neg,
            gamma_pos=self.train_config.gamma_pos,
        )
        self.property_loss = nn.MSELoss()

        # Loss weights
        self.w_rank = self.train_config.ranking_loss_weight
        self.w_bce = self.train_config.bce_loss_weight
        self.w_prop = self.train_config.property_loss_weight

        # Evaluator
        self.evaluator = ExcipicEvaluator(
            pad_idx=preprocessor.excipient_to_idx[preprocessor.PAD_TOKEN],
            bos_idx=preprocessor.excipient_to_idx[preprocessor.BOS_TOKEN],
            eos_idx=preprocessor.excipient_to_idx[preprocessor.EOS_TOKEN],
            unk_idx=preprocessor.excipient_to_idx[preprocessor.UNK_TOKEN],
        )

        # Mixed precision
        self.use_amp = self.train_config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Build optimizer and scheduler (after model is on device)
        self.optimizer = None
        self.scheduler = None

        # Tracking
        self.best_metric = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.current_epoch = 0

    def setup_optimizer(self, num_training_steps: int):
        """
        Create optimizer with per-module learning rates and scheduler.
        """
        tc = self.train_config

        # Group parameters by module with different LRs
        param_groups = [
            {
                "params": self.model.molecular_encoder.parameters(),
                "lr": tc.lr_molecular_encoder,
                "name": "molecular_encoder",
            },
            {
                "params": self.model.dosage_encoder.parameters(),
                "lr": tc.lr_dosage_encoder,
                "name": "dosage_encoder",
            },
            {
                "params": self.model.fusion.parameters(),
                "lr": tc.lr_fusion,
                "name": "fusion",
            },
            {
                "params": self.model.decoder.parameters(),
                "lr": tc.lr_decoder,
                "name": "decoder",
            },
            {
                "params": self.model.property_head.parameters(),
                "lr": tc.lr_property_head,
                "name": "property_head",
            },
            {
                "params": list(self.model.excipient_embedder.parameters())
                          + list(self.model.excipient_graph.parameters()),
                "lr": tc.lr_excipient_graph,
                "name": "excipient_graph",
            },
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=tc.weight_decay,
        )

        # Scheduler
        num_warmup = int(tc.warmup_fraction * num_training_steps)
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer,
            num_warmup_steps=num_warmup,
            num_total_steps=num_training_steps,
        )

        logger.info(
            f"Optimizer: AdamW with {len(param_groups)} param groups, "
            f"warmup={num_warmup}/{num_training_steps} steps"
        )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {
            "total": 0.0, "ranking": 0.0, "bce": 0.0, "property": 0.0
        }
        num_batches = 0
        batch_times = []

        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()
            # Move to device
            batch = self._to_device(batch)

            # Forward
            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(batch)

                # Ranking loss (decoder)
                ranking_loss = self.ranking_loss(
                    logits=outputs["decoder_logits"],
                    targets=batch["decoder_target"],
                    mask=batch["decoder_mask"],
                )

                # BCE loss (multi-hot)
                bce_loss = self.bce_loss(
                    logits=outputs["multi_hot_logits"],
                    targets=batch["multi_hot"],
                )

                # Property loss
                property_loss = self.property_loss(
                    outputs["property_preds"],
                    batch["api_targets"],
                )

                # Combined loss
                total_loss = (
                    self.w_rank * ranking_loss
                    + self.w_bce * bce_loss
                    + self.w_prop * property_loss
                )

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_config.gradient_clip,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Track losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["ranking"] += ranking_loss.item()
            epoch_losses["bce"] += bce_loss.item()
            epoch_losses["property"] += property_loss.item()
            num_batches += 1
            self.global_step += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Log with memory usage (only every 5 epochs or final step)
            if (batch_idx + 1) % self.train_config.log_interval == 0:

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()
        self.evaluator.reset()

        eval_losses = {
            "total": 0.0, "ranking": 0.0, "bce": 0.0, "property": 0.0
        }
        num_batches = 0

        for batch in val_loader:
            batch = self._to_device(batch)

            with autocast("cuda", enabled=self.use_amp):
                outputs = self.model(batch)

                ranking_loss = self.ranking_loss(
                    outputs["decoder_logits"],
                    batch["decoder_target"],
                    batch["decoder_mask"],
                )
                bce_loss = self.bce_loss(
                    outputs["multi_hot_logits"],
                    batch["multi_hot"],
                )
                property_loss = self.property_loss(
                    outputs["property_preds"],
                    batch["api_targets"],
                )
                total_loss = (
                    self.w_rank * ranking_loss
                    + self.w_bce * bce_loss
                    + self.w_prop * property_loss
                )

            eval_losses["total"] += total_loss.item()
            eval_losses["ranking"] += ranking_loss.item()
            eval_losses["bce"] += bce_loss.item()
            eval_losses["property"] += property_loss.item()
            num_batches += 1

            # Generate predictions for metrics
            predictions = self.model.predict(batch, temperature=0.0, max_len=15)
            self.evaluator.update(
                predicted_ids=predictions["predictions"],
                target_multi_hot=batch["multi_hot"],
            )

        # Average losses
        for key in eval_losses:
            eval_losses[key] /= max(num_batches, 1)

        # Compute metrics
        metrics = self.evaluator.compute()
        metrics.update({f"val_loss_{k}": v for k, v in eval_losses.items()})

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Full training loop with progress tracking.

        Returns:
            Best validation metrics.
        """
        num_epochs = num_epochs or self.train_config.num_epochs

        # Setup optimizer
        num_training_steps = len(train_loader) * num_epochs
        self.setup_optimizer(num_training_steps)

        logger.info(f"Starting training: {num_epochs} epochs, {num_training_steps} total steps | Early stop patience: {self.train_config.patience}\n")

        best_metrics = {}
        epoch_times = []

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            logger.info(f"[Epoch {self.current_epoch}/{num_epochs}]", end=" ")

            # Train
            train_losses = self.train_epoch(train_loader)
            train_time = time.time() - epoch_start
            epoch_times.append(train_time)

            logger.info(
                f"[Epoch {self.current_epoch}/{num_epochs}] Loss: {train_losses['total']:.4f} | {train_time:.1f}s"
            )

            # Evaluate
            if (epoch + 1) % self.train_config.eval_interval == 0:
                eval_start = time.time()
                metrics = self.evaluate(val_loader)
                eval_time = time.time() - eval_start

                primary = metrics.get(self.train_config.primary_metric, 0.0)

                logger.info(
                    f"✓ Val Summary | "
                    f"NDCG@10: {metrics.get('ndcg_at_10', 0):.4f} | "
                    f"Recall@10: {metrics.get('recall_at_10', 0):.4f} | "
                    f"Hit@10: {metrics.get('hit_at_10', 0):.4f} | "
                    f"MRR: {metrics.get('mrr', 0):.4f} | "
                    f"Jaccard@10: {metrics.get('jaccard_at_10', 0):.4f} | "
                    f"Val Time: {eval_time:.1f}s"
                )

                # Check improvement
                if primary > self.best_metric:
                    improvement = primary - self.best_metric
                    self.best_metric = primary
                    self.patience_counter = 0
                    best_metrics = metrics.copy()

                    # Save best checkpoint
                    self.save_checkpoint("best_model.pt")
                    logger.info(
                        f"⭐ NEW BEST! {self.train_config.primary_metric}: "
                        f"{primary:.4f} (+{improvement:.4f}) | Checkpoint saved"
                    )
                else:
                    self.patience_counter += 1
                    patience_remaining = self.train_config.patience - self.patience_counter
                    logger.info(
                        f"⏱️  No improvement | Patience: {self.patience_counter}/{self.train_config.patience} "
                        f"({patience_remaining} epochs remaining before early stop)"
                    )

                # Time estimation
                if epoch_times and epoch < num_epochs - 1:
                    avg_epoch_time = sum(epoch_times) / len(epoch_times)
                    remaining_epochs = num_epochs - self.current_epoch
                    estimated_remaining = avg_epoch_time * remaining_epochs / 60  # in minutes
                    logger.info(
                        f"⏱️  Estimated time remaining: {estimated_remaining:.1f} minutes "
                        f"({remaining_epochs} epochs × {avg_epoch_time:.1f}s avg)"
                    )

                # Early stopping
                if self.patience_counter >= self.train_config.patience:
                    logger.info(
                        f"\n{'🛑' * 40}"
                    )
                    logger.info(
                        f"🛑 EARLY STOPPING TRIGGERED at Epoch {self.current_epoch}/{num_epochs}"
                    )
                    logger.info(
                        f"   Best {self.train_config.primary_metric}: {self.best_metric:.4f}"
                    )
                    logger.info(
                        f"   Patience exhausted: {self.train_config.patience} epochs with no improvement"
                    )
                    logger.info(
                        f"{'🛑' * 40}\n"
                    )
                    break

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.current_epoch}.pt")

        logger.info(
            f"\n{'═' * 80}"
        )
        logger.info(
            f"✅ TRAINING COMPLETE"
        )
        logger.info(
            f"   Total Time: {sum(epoch_times) / 60:.1f} minutes"
        )
        logger.info(
            f"   Epochs Trained: {self.current_epoch}/{num_epochs}"
        )
        logger.info(
            f"   Best {self.train_config.primary_metric}: {self.best_metric:.4f}"
        )
        logger.info(
            f"{'═' * 80}\n"
        )
        return best_metrics

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.paths.checkpoint_dir, filename)
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.paths.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        if self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}: {path}")

    def _to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        moved = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                moved[key] = val.to(self.device)
            elif hasattr(val, "to"):
                # PyG Batch object
                moved[key] = val.to(self.device)
            else:
                moved[key] = val
        return moved

    def _get_memory_info(self) -> str:
        """Get GPU/CPU memory information."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)  # MB
            return f"GPU Mem: {allocated:.0f}MB/{reserved:.0f}MB (Peak: {max_allocated:.0f}MB)"
        else:
            return "CPU Mode (No GPU Memory)"
