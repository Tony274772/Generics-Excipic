"""
Excipic — Main Training Entry Point.

Usage:
    python train.py                    # Full training
    python train.py --debug            # Debug mode (small subset, 2 epochs)
    python train.py --epochs 50        # Custom epochs
    python train.py --resume checkpoint_epoch_10.pt  # Resume training
"""
import argparse
import logging
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from excipic.config import ExcipicConfig
from excipic.data.preprocessing import ExcipicPreprocessor
from excipic.data.dataset import ExcipicDataset, collate_fn
from excipic.models.excipic_model import ExcipicModel
from excipic.training.trainer import ExcipicTrainer


def _get_device_info() -> dict:
    """Get detailed device information for GPU or CPU."""
    if torch.cuda.is_available():
        device_idx = 0
        device_name = torch.cuda.get_device_name(device_idx)
        device_props = torch.cuda.get_device_properties(device_idx)
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)  # Convert to GB
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        # Check if RTX 4050
        is_rtx_4050 = "4050" in device_name or "RTX 4050" in device_name
        
        return {
            "device_type": "GPU",
            "device_name": device_name,
            "total_memory_gb": total_memory,
            "compute_capability": compute_capability,
            "is_rtx_4050": is_rtx_4050,
            "amp_dtype": "float16" if compute_capability >= "6.0" else "float32",
        }
    else:
        return {
            "device_type": "CPU",
            "device_name": torch.version.__version__,
            "total_memory_gb": 0,
            "compute_capability": "N/A",
            "is_rtx_4050": False,
            "amp_dtype": "float32",
        }


def setup_logging(output_dir: str):
    """Configure logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")

    # Force stdout to use utf-8 to avoid UnicodeEncodeError in Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Train Excipic model")
    parser.add_argument("--debug", action="store_true", help="Debug mode with small data")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Config
    config = ExcipicConfig()
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.debug:
        config.training.num_epochs = 2
        config.training.log_interval = 5

    setup_logging(config.paths.output_dir)
    logger = logging.getLogger("excipic.train")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 80)
    logger.info("  EXCIPIC — Pharmaceutical Excipient Recommendation System")
    logger.info("=" * 80)
    
    # ────────── DEVICE INFORMATION ──────────
    device_info = _get_device_info()
    logger.info(f"\n🖥️  DEVICE INFORMATION:")
    logger.info(f"   Device Type: {device_info['device_type']}")
    logger.info(f"   Device Name: {device_info['device_name']}")
    if device_info['device_type'] == 'GPU':
        logger.info(f"   GPU Memory: {device_info['total_memory_gb']:.2f} GB")
        logger.info(f"   Compute Capability: {device_info['compute_capability']}")
        logger.info(f"   Mixed Precision: {device_info['amp_dtype']}")
        if device_info['is_rtx_4050']:
            logger.info(f"   ✓ RTX 4050 Detected!")
        else:
            logger.warning(f"   ⚠ Not RTX 4050 (configured for RTX 4050)")
    else:
        logger.warning(f"   ⚠ CPU Mode - Training will be SLOW. GPU recommended!")
    
    # ────────── TRAINING CONFIGURATION ──────────
    logger.info(f"\n⚙️  TRAINING CONFIGURATION:")
    logger.info(f"   Batch size: {config.training.batch_size}")
    logger.info(f"   Epochs: {config.training.num_epochs}")
    logger.info(f"   Patience (Early Stop): {config.training.patience} epochs")
    logger.info(f"   Primary Metric: {config.training.primary_metric}")
    logger.info(f"   Loss Weights: Ranking={config.training.ranking_loss_weight}, "
                f"BCE={config.training.bce_loss_weight}, "
                f"Property={config.training.property_loss_weight}")
    logger.info(f"   Learning Rates: "
                f"Mol={config.training.lr_molecular_encoder:.0e}, "
                f"Dosage={config.training.lr_dosage_encoder:.0e}, "
                f"Fusion={config.training.lr_fusion:.0e}, "
                f"Decoder={config.training.lr_decoder:.0e}, "
                f"PropHead={config.training.lr_property_head:.0e}, "
                f"Graph={config.training.lr_excipient_graph:.0e}")
    logger.info("=" * 80)

    # ── 1. Preprocess data ─────────────────────────────────────────────────
    logger.info("\n── Step 1: Preprocessing ──")
    cache_path = os.path.join(config.paths.output_dir, "preprocessed_cache.pkl")

    preprocessor = ExcipicPreprocessor(config)
    train_formulations, val_formulations, test_formulations = preprocessor.preprocess(
        cache_path=cache_path
    )

    logger.info(f"Excipient vocabulary size: {preprocessor.num_excipients}")
    logger.info(f"Dosage form vocabulary size: {preprocessor.num_dosage_forms}")
    logger.info(f"Train: {len(train_formulations)}")
    logger.info(f"Val:   {len(val_formulations)}")
    logger.info(f"Test:  {len(test_formulations)}")

    # ── 2. Create datasets and dataloaders ─────────────────────────────────
    logger.info("\n── Step 2: Creating Datasets ──")

    if args.debug:
        train_formulations = train_formulations[:100]
        val_formulations = val_formulations[:50]
        test_formulations = test_formulations[:50]
        logger.info("DEBUG MODE: using small data subset")

    max_seq_len = config.decoder.max_seq_len
    graph_cache = {}  # shared cache for SMILES → graph conversion

    train_dataset = ExcipicDataset(
        train_formulations, preprocessor, max_seq_len=max_seq_len,
        graph_cache=graph_cache,
    )
    val_dataset = ExcipicDataset(
        val_formulations, preprocessor, max_seq_len=max_seq_len,
        graph_cache=graph_cache,
    )
    test_dataset = ExcipicDataset(
        test_formulations, preprocessor, max_seq_len=max_seq_len,
        graph_cache=graph_cache,
    )

    tc = config.training
    train_loader = DataLoader(
        train_dataset, batch_size=tc.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=tc.num_workers,
        pin_memory=tc.pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tc.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=tc.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=tc.num_workers,
        pin_memory=tc.pin_memory,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches:   {len(val_loader)}")
    logger.info(f"Test batches:  {len(test_loader)}")

    # ── 3. Build model ─────────────────────────────────────────────────────
    logger.info("\n── Step 3: Building Model ──")
    device = torch.device(tc.device if torch.cuda.is_available() else "cpu")

    # Update config dimensions from actual data
    from excipic.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
    config.molecular_encoder.node_input_dim = ATOM_FEATURE_DIM
    config.molecular_encoder.edge_input_dim = BOND_FEATURE_DIM
    config.dosage_encoder.num_dosage_forms = preprocessor.num_dosage_forms
    config.excipient_graph.num_excipients = preprocessor.num_excipients

    model = ExcipicModel(config, preprocessor)

    # Initialize (build excipient embeddings + set graph structure)
    model.initialize(device, pmi_matrix=preprocessor.pmi_matrix)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # ── 4. Train ───────────────────────────────────────────────────────────
    logger.info("\n── Step 4: Training ──")

    trainer = ExcipicTrainer(model, config, preprocessor)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    start_time = time.time()
    best_metrics = trainer.train(train_loader, val_loader)
    total_time = time.time() - start_time

    logger.info(f"\nTraining completed in {total_time / 60:.1f} minutes")
    logger.info(f"Best validation metrics:")
    for key, val in sorted(best_metrics.items()):
        logger.info(f"  {key}: {val:.4f}")

    # ── 5. Test evaluation ─────────────────────────────────────────────────
    logger.info("\n── Step 5: Test Evaluation ──")

    # Load best model
    trainer.load_checkpoint("best_model.pt")
    test_metrics = trainer.evaluate(test_loader)

    logger.info(f"\nTest set metrics:")
    for key, val in sorted(test_metrics.items()):
        logger.info(f"  {key}: {val:.4f}")

    # Save final results
    import json
    results = {
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
        "training_time_minutes": total_time / 60,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_size": len(train_formulations),
        "val_size": len(val_formulations),
        "test_size": len(test_formulations),
    }
    results_path = os.path.join(config.paths.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    logger.info("\n" + "=" * 60)
    logger.info("  Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
