"""
Cross-Attention Fusion Transformer.

Learns interactions between molecular embeddings and dosage form context
via bidirectional cross-attention.
"""
import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with Pre-LayerNorm.

    Molecule attends to dosage context, and vice versa.
    """

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.15, attention_dropout: float = 0.10):
        super().__init__()

        # Molecule → Dosage cross-attention
        self.mol_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.mol_norm1 = nn.LayerNorm(hidden_dim)
        self.mol_norm2 = nn.LayerNorm(hidden_dim)

        # Dosage → Molecule cross-attention
        self.dos_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dos_norm1 = nn.LayerNorm(hidden_dim)
        self.dos_norm2 = nn.LayerNorm(hidden_dim)

        # FFN for molecule stream
        self.mol_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # FFN for dosage stream
        self.dos_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, mol_emb: torch.Tensor, dos_emb: torch.Tensor
    ) -> tuple:
        """
        Args:
            mol_emb: Molecule embedding [B, 1, D] or [B, D]
            dos_emb: Dosage embedding [B, 1, D] or [B, D]

        Returns:
            Updated (mol_emb, dos_emb), both [B, 1, D].
        """
        # Ensure 3D
        if mol_emb.dim() == 2:
            mol_emb = mol_emb.unsqueeze(1)
        if dos_emb.dim() == 2:
            dos_emb = dos_emb.unsqueeze(1)

        # Molecule attends to dosage (Pre-LayerNorm)
        residual = mol_emb
        mol_normed = self.mol_norm1(mol_emb)
        dos_normed_for_mol = self.dos_norm1(dos_emb)
        mol_cross, _ = self.mol_cross_attn(
            query=mol_normed, key=dos_normed_for_mol, value=dos_normed_for_mol
        )
        mol_emb = residual + self.dropout(mol_cross)

        # Molecule FFN
        residual = mol_emb
        mol_emb = residual + self.mol_ffn(self.mol_norm2(mol_emb))

        # Dosage attends to molecule (Pre-LayerNorm)
        residual = dos_emb
        dos_normed = self.dos_norm1(dos_emb)
        mol_normed_for_dos = self.mol_norm1(mol_emb)
        dos_cross, _ = self.dos_cross_attn(
            query=dos_normed, key=mol_normed_for_dos, value=mol_normed_for_dos
        )
        dos_emb = residual + self.dropout(dos_cross)

        # Dosage FFN
        residual = dos_emb
        dos_emb = residual + self.dos_ffn(self.dos_norm2(dos_emb))

        return mol_emb, dos_emb


class FusionTransformer(nn.Module):
    """
    Cross-attention fusion transformer.

    Fuses molecular and dosage representations via N layers of
    bidirectional cross-attention, then produces a final fusion vector:

        fusion = [mol_CLS ; dosage_emb ; mol_CLS * dosage_emb]

    Output dimension: 3 * hidden_dim = 1536.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm_mol = nn.LayerNorm(config.hidden_dim)
        self.final_norm_dos = nn.LayerNorm(config.hidden_dim)

        # Fusion projection: 3 * hidden_dim → fusion_output_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.fusion_output_dim),
            nn.GELU(),
            nn.LayerNorm(config.fusion_output_dim),
        )

    def forward(
        self, mol_emb: torch.Tensor, dos_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mol_emb: Molecular encoder output [B, hidden_dim]
            dos_emb: Dosage encoder output [B, hidden_dim]

        Returns:
            Fusion vector [B, fusion_output_dim (1536)]
        """
        for layer in self.layers:
            mol_emb, dos_emb = layer(mol_emb, dos_emb)

        # Squeeze back to 2D
        if mol_emb.dim() == 3:
            mol_emb = mol_emb.squeeze(1)
        if dos_emb.dim() == 3:
            dos_emb = dos_emb.squeeze(1)

        mol_emb = self.final_norm_mol(mol_emb)
        dos_emb = self.final_norm_dos(dos_emb)

        # Fusion: concat + elementwise product
        elementwise = mol_emb * dos_emb
        fusion = torch.cat([mol_emb, dos_emb, elementwise], dim=-1)  # [B, 3*D]
        fusion = self.fusion_proj(fusion)  # [B, fusion_output_dim]

        return fusion
