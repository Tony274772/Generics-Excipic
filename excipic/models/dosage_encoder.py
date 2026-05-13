"""
Dosage Form Encoder.

Encodes dosage form category (learned embedding) and API physicochemical
descriptors (MLP projection) into a unified representation.
"""
import torch
import torch.nn as nn


class DosageEncoder(nn.Module):
    """
    Dosage form encoder with learned embeddings and descriptor projection.

    Architecture:
        - Learned embedding table for dosage form categories
        - 2-layer MLP for RDKit descriptor features
        - Fusion: concatenate + project to output dim
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Dosage form embedding
        self.dosage_embedding = nn.Embedding(
            num_embeddings=config.num_dosage_forms,
            embedding_dim=config.embedding_dim,
        )

        # Descriptor MLP projection
        self.descriptor_mlp = nn.Sequential(
            nn.Linear(config.descriptor_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.projection_dim, config.projection_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Final projection: embedding_dim + projection_dim → projection_dim (=512)
        self.output_proj = nn.Sequential(
            nn.Linear(config.embedding_dim + config.projection_dim, config.projection_dim),
            nn.GELU(),
            nn.LayerNorm(config.projection_dim),
        )

    def forward(
        self,
        dosage_idx: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            dosage_idx: Dosage form indices [B]
            descriptors: API descriptor features [B, descriptor_dim]

        Returns:
            Dosage embedding [B, projection_dim (512)]
        """
        # Embed dosage form
        dosage_emb = self.dosage_embedding(dosage_idx)  # [B, embedding_dim]

        # Project descriptors
        desc_proj = self.descriptor_mlp(descriptors)  # [B, projection_dim]

        # Concatenate and project
        combined = torch.cat([dosage_emb, desc_proj], dim=-1)  # [B, embedding_dim + projection_dim]
        output = self.output_proj(combined)  # [B, projection_dim]

        return output
