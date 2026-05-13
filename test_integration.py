"""Full integration test for the shared vocab update."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch
from excipic.config import ExcipicConfig
from excipic.data.preprocessing import ExcipicPreprocessor
from excipic.data.dataset import ExcipicDataset, collate_fn
from excipic.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from excipic.models.excipic_model import ExcipicModel
from excipic.losses.ranking_loss import ListNetLoss
from excipic.losses.asymmetric_bce import AsymmetricBCELoss
from excipic.metrics.evaluation import ExcipicEvaluator
from excipic.training.trainer import ExcipicTrainer
from torch.utils.data import DataLoader


def main():
    config = ExcipicConfig()

    # Verify paths
    print(f"Vocab JSON: {config.paths.excipient_vocab_json}")
    assert os.path.exists(config.paths.excipient_vocab_json), "Shared vocab not found!"

    # Step 1: Preprocessing with shared vocab
    print("\n=== Step 1: Preprocessing ===")
    preprocessor = ExcipicPreprocessor(config)
    train_f, val_f, test_f = preprocessor.preprocess()

    print(f"Vocab size: {preprocessor.num_excipients}")
    print(f"PAD idx: {preprocessor.excipient_to_idx[preprocessor.PAD_TOKEN]}")
    print(f"BOS idx: {preprocessor.excipient_to_idx[preprocessor.BOS_TOKEN]}")
    print(f"EOS idx: {preprocessor.excipient_to_idx[preprocessor.EOS_TOKEN]}")
    print(f"Train: {len(train_f)}, Val: {len(val_f)}, Test: {len(test_f)}")

    # Check a sample formulation
    sample = train_f[0]
    print(f"\nSample excipient indices: {sample['excipient_indices'][:5]}...")
    for idx in sample["excipient_indices"][:3]:
        name = preprocessor.excipient_names.get(idx, "?")
        unii = preprocessor.excipient_uniis.get(idx, "?")
        print(f"  idx={idx}: {name} (UNII={unii})")

    # Step 2: Dataset + DataLoader
    print("\n=== Step 2: Dataset ===")
    dataset = ExcipicDataset(train_f[:8], preprocessor, max_seq_len=config.decoder.max_seq_len)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"Batch graph: {batch['graph'].num_graphs} graphs")
    print(f"decoder_input: {batch['decoder_input'].shape}")
    print(f"multi_hot: {batch['multi_hot'].shape}")
    print(f"multi_hot sum (excipients per sample): {batch['multi_hot'].sum(dim=1).tolist()}")

    # Step 3: Model
    print("\n=== Step 3: Model ===")
    config.molecular_encoder.node_input_dim = ATOM_FEATURE_DIM
    config.molecular_encoder.edge_input_dim = BOND_FEATURE_DIM
    config.dosage_encoder.num_dosage_forms = preprocessor.num_dosage_forms
    config.excipient_graph.num_excipients = preprocessor.num_excipients

    device = torch.device("cpu")
    model = ExcipicModel(config, preprocessor)
    model.initialize(device, pmi_matrix=preprocessor.pmi_matrix)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Model vocab_size: {model.vocab_size}")
    print(f"Model pad_idx: {model.pad_idx}, bos_idx: {model.bos_idx}, eos_idx: {model.eos_idx}")

    # Step 4: Forward pass
    print("\n=== Step 4: Forward pass ===")
    outputs = model(batch)
    print(f"decoder_logits: {outputs['decoder_logits'].shape}")
    print(f"property_preds: {outputs['property_preds'].shape}")
    print(f"multi_hot_logits: {outputs['multi_hot_logits'].shape}")
    print(f"fusion: {outputs['fusion'].shape}")

    # Step 5: Generate predictions
    print("\n=== Step 5: Generate ===")
    preds = model.predict(batch, temperature=0.7, max_len=10)
    print(f"predictions: {preds['predictions'].shape}")

    # Step 6: Training step
    print("\n=== Step 6: Training step ===")
    ranking_loss = ListNetLoss()
    bce_loss = AsymmetricBCELoss(gamma_neg=4.0, gamma_pos=1.0)
    prop_loss = torch.nn.MSELoss()

    model.train()
    outputs = model(batch)
    r = ranking_loss(outputs["decoder_logits"], batch["decoder_target"], batch["decoder_mask"])
    b = bce_loss(outputs["multi_hot_logits"], batch["multi_hot"])
    p = prop_loss(outputs["property_preds"], batch["api_targets"])
    total = 0.6 * r + 0.3 * b + 0.1 * p
    total.backward()
    print(f"Loss: {total.item():.4f} (R:{r.item():.4f} B:{b.item():.4f} P:{p.item():.4f})")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED WITH SHARED VOCAB!")
    print("=" * 50)


if __name__ == "__main__":
    main()
