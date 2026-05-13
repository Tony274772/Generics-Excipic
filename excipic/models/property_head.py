"""
Physicochemical Property Prediction Head.

Auxiliary task that predicts molecular properties from the fusion vector.
Improves generalization and chemistry understanding.
"""
import torch
import torch.nn as nn


class PropertyHead(nn.Module):
    """
    MLP head for predicting physicochemical properties.

    Architecture:
        1536 → 768 → 256 → num_properties

    Properties predicted (matching api_features.csv):
        MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors,
        NumRotatableBonds, NumAromaticRings, NumAliphaticRings,
        RingCount, FractionCSP3, HeavyAtomCount, NumValenceElectrons,
        MolMR, LabuteASA, BalabanJ, BertzCT, HallKierAlpha,
        NumSaturatedRings, NumHeteroatoms, NHOHCount
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.input_dim
        for h_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, config.num_properties))
        self.mlp = nn.Sequential(*layers)

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fusion: Fusion vector [B, input_dim (1536)]

        Returns:
            Property predictions [B, num_properties]
        """
        return self.mlp(fusion)
