"""
Excipic Configuration — All hyperparameters and paths.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    """File paths."""
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
    excipic_dir: str = os.path.dirname(__file__)
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    api_features: str = ""
    excipient_smiles: str = ""
    excipient_vocab_json: str = ""  # shared vocab for cross-architecture comparisons
    output_dir: str = os.path.join(os.path.dirname(__file__), "outputs")
    checkpoint_dir: str = os.path.join(os.path.dirname(__file__), "outputs", "checkpoints")

    def __post_init__(self):
        self.train_csv = os.path.join(self.data_dir, "train.csv")
        self.val_csv = os.path.join(self.data_dir, "val.csv")
        self.test_csv = os.path.join(self.data_dir, "test.csv")
        self.api_features = os.path.join(self.data_dir, "api_features.csv")
        self.excipient_smiles = os.path.join(self.data_dir, "excipinets_Smiles.csv")
        self.excipient_vocab_json = os.path.join(
            self.excipic_dir, "excipient_unii_vocab.json"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@dataclass
class MolecularEncoderConfig:
    """GATv2 molecular encoder."""
    num_layers: int = 5
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.15
    attention_dropout: float = 0.10
    use_virtual_node: bool = True
    # Node feature dimensions (will be set dynamically)
    node_input_dim: int = 89  # atom features after encoding
    edge_input_dim: int = 12  # bond features after encoding


@dataclass
class DosageEncoderConfig:
    """Dosage form encoder."""
    num_dosage_forms: int = 56  # will be set dynamically
    embedding_dim: int = 256
    descriptor_dim: int = 20  # from api_features.csv (raw descriptor count)
    projection_dim: int = 512
    mlp_layers: int = 2
    dropout: float = 0.20


@dataclass
class FusionConfig:
    """Cross-attention fusion transformer."""
    num_layers: int = 4
    num_heads: int = 8
    hidden_dim: int = 512
    ffn_dim: int = 2048
    dropout: float = 0.15
    attention_dropout: float = 0.10
    fusion_output_dim: int = 1536  # [mol_CLS ; dosage_emb ; elementwise_product]


@dataclass
class PropertyHeadConfig:
    """Physicochemical property prediction head."""
    input_dim: int = 1536
    hidden_dims: List[int] = field(default_factory=lambda: [768, 256])
    num_properties: int = 20  # matches api_features.csv descriptor count
    dropout: float = 0.20


@dataclass
class ExcipientGraphConfig:
    """Excipient knowledge graph + GCNII."""
    num_excipients: int = 953  # set dynamically (950 from shared vocab + 3 special tokens)
    embedding_dim: int = 512
    gcn_layers: int = 3
    gcn_hidden_dim: int = 512
    gcn_dropout: float = 0.10
    # Word2Vec config for excipients without SMILES
    w2v_dim: int = 128
    w2v_window: int = 5
    w2v_min_count: int = 1


@dataclass
class DecoderConfig:
    """Autoregressive set decoder with pointer attention."""
    num_layers: int = 4
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.15
    max_seq_len: int = 40  # max excipients to predict (max is 36 in new data)
    temperature: float = 0.7
    beam_size: int = 5


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Data is pre-split into train/val/test CSVs

    # Optimization
    batch_size: int = 16
    num_epochs: int = 100
    gradient_clip: float = 1.0

    # Learning rates (per module) - reduced to prevent NaN
    lr_molecular_encoder: float = 5e-5
    lr_dosage_encoder: float = 1e-4
    lr_fusion: float = 5e-5
    lr_decoder: float = 5e-5
    lr_property_head: float = 1e-4
    lr_excipient_graph: float = 2e-5

    # AdamW
    weight_decay: float = 0.01

    # Scheduler
    warmup_fraction: float = 0.05

    # Loss weights
    ranking_loss_weight: float = 0.6
    bce_loss_weight: float = 0.3
    property_loss_weight: float = 0.1

    # Asymmetric BCE
    gamma_neg: float = 4.0
    gamma_pos: float = 1.0

    # Early stopping
    patience: int = 10
    primary_metric: str = "ndcg_at_10"

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # or "bfloat16"

    # Device
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True

    # Logging
    log_interval: int = 50
    eval_interval: int = 1  # evaluate every N epochs


@dataclass
class ExcipicConfig:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    molecular_encoder: MolecularEncoderConfig = field(default_factory=MolecularEncoderConfig)
    dosage_encoder: DosageEncoderConfig = field(default_factory=DosageEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    property_head: PropertyHeadConfig = field(default_factory=PropertyHeadConfig)
    excipient_graph: ExcipientGraphConfig = field(default_factory=ExcipientGraphConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
