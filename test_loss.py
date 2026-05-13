import torch
from excipic.config import ExcipicConfig
from excipic.data.preprocessing import ExcipicPreprocessor
from excipic.data.dataset import ExcipicDataset, collate_fn
from excipic.models.excipic_model import ExcipicModel
from excipic.training.trainer import ExcipicTrainer
from torch.utils.data import DataLoader
from excipic.data.graph_builder import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

def main():
    config = ExcipicConfig()
    config.training.batch_size = 4
    preprocessor = ExcipicPreprocessor(config)
    train_f, _, _ = preprocessor.preprocess("excipic/outputs/preprocessed_cache.pkl")
    dataset = ExcipicDataset(train_f[:4], preprocessor, max_seq_len=30)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    config.molecular_encoder.node_input_dim = ATOM_FEATURE_DIM
    config.molecular_encoder.edge_input_dim = BOND_FEATURE_DIM
    config.dosage_encoder.num_dosage_forms = preprocessor.num_dosage_forms
    config.excipient_graph.num_excipients = preprocessor.num_excipients
    
    model = ExcipicModel(config, preprocessor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.initialize(device, pmi_matrix=preprocessor.pmi_matrix)
    model.to(device)
    
    trainer = ExcipicTrainer(model, config, preprocessor)
    batch = next(iter(loader))
    batch = trainer._to_device(batch)
    
    outputs = model(batch)
    ranking_loss = trainer.ranking_loss(outputs["decoder_logits"], batch["decoder_target"], batch["decoder_mask"])
    bce_loss = trainer.bce_loss(outputs["multi_hot_logits"], batch["multi_hot"])
    property_loss = trainer.property_loss(outputs["property_preds"], batch["api_targets"])
    
    print(f"Ranking Loss: {ranking_loss.item()}")
    print(f"BCE Loss: {bce_loss.item()}")
    print(f"Property Loss: {property_loss.item()}")
    
    total = trainer.w_rank * ranking_loss + trainer.w_bce * bce_loss + trainer.w_prop * property_loss
    print(f"Total Loss: {total.item()}")

if __name__ == "__main__":
    main()
