"""
Dual-Path Excipient Embedder.

For excipients WITH SMILES:  SMILES → GATv2 molecular graph → embedding
For excipients WITHOUT SMILES: Word2Vec (index-based) → projection → embedding

Both paths produce a 512-dim embedding, which is then used in the
excipient knowledge graph and set decoder.

NOTE: The shared vocabulary is NAME-based — the same UNII can map to multiple
indices (e.g., "STARCH, CORN" idx=4 and "starch, corn" idx=126 both share
UNII O8232NY3SJ). When a UNII has SMILES, all indices sharing that UNII
get the same SMILES-based embedding.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..data.graph_builder import smiles_to_graph, ATOM_FEATURE_DIM, BOND_FEATURE_DIM

logger = logging.getLogger(__name__)


class ExcipientEmbedder(nn.Module):
    """
    Dual-path excipient embedding module.

    Path A (SMILES available):
        SMILES → molecular graph → lightweight GATv2 → readout → projection → 512-dim

    Path B (no SMILES):
        Word2Vec embedding (pretrained, frozen) → MLP projection → 512-dim

    All excipient embeddings are precomputed and stored as an embedding matrix.
    """

    def __init__(self, config, excipient_smiles: Dict[str, str],
                 w2v_embeddings: Dict[int, np.ndarray],
                 excipient_uniis: Dict[int, str],
                 num_excipients: int):
        """
        Args:
            config: ExcipientGraphConfig
            excipient_smiles: Dict mapping UNII → SMILES
            w2v_embeddings: Dict mapping vocab_idx → Word2Vec embedding array
            excipient_uniis: Dict mapping vocab_idx → UNII
            num_excipients: Total vocabulary size (including special tokens)
        """
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_excipients = num_excipients

        # Track which path each index uses
        self.has_smiles: Dict[int, bool] = {}
        self.smiles_map: Dict[int, str] = {}  # idx → SMILES string

        # Map: for each vocab index, if its UNII has SMILES, use SMILES path
        for idx, unii in excipient_uniis.items():
            if unii and unii in excipient_smiles:
                self.has_smiles[idx] = True
                self.smiles_map[idx] = excipient_smiles[unii]

        # ── Path A: SMILES-based embedding ──
        from torch_geometric.nn import GATv2Conv, global_mean_pool

        self.smiles_node_proj = nn.Linear(ATOM_FEATURE_DIM, 256)
        self.smiles_edge_proj = nn.Linear(BOND_FEATURE_DIM, 256)

        self.smiles_gat1 = GATv2Conv(256, 64, heads=4, concat=True, edge_dim=256,
                                      dropout=0.1, add_self_loops=True)
        self.smiles_gat2 = GATv2Conv(256, 64, heads=4, concat=True, edge_dim=256,
                                      dropout=0.1, add_self_loops=True)
        self.smiles_norm1 = nn.LayerNorm(256)
        self.smiles_norm2 = nn.LayerNorm(256)
        self.smiles_proj = nn.Sequential(
            nn.Linear(256, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
        )

        # ── Path B: Word2Vec-based embedding ──
        self.w2v_proj = nn.Sequential(
            nn.Linear(config.w2v_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(config.embedding_dim),
        )

        # Store Word2Vec embeddings as a buffer
        w2v_matrix = torch.zeros(num_excipients, config.w2v_dim)
        for idx, emb in w2v_embeddings.items():
            if idx < num_excipients:
                w2v_matrix[idx] = torch.from_numpy(emb)
        self.register_buffer("w2v_matrix", w2v_matrix)

        # Learnable embedding for special tokens (PAD=0, BOS, EOS)
        self.special_embedding = nn.Embedding(3, config.embedding_dim)

        # Final embedding matrix (computed during build_embeddings)
        self.register_buffer(
            "embedding_matrix",
            torch.zeros(num_excipients, config.embedding_dim),
        )
        self._embeddings_built = False

    def build_embeddings(self, device: torch.device):
        """
        Precompute all excipient embeddings.
        Call this once after loading the model / at the start of training.
        """
        logger.info("Building excipient embeddings...")
        self.eval()

        with torch.no_grad():
            embeddings = torch.zeros(
                self.num_excipients, self.embedding_dim, device=device
            )

            # idx 0 → PAD (special token 0)
            embeddings[0] = self.special_embedding(
                torch.tensor(0, device=device)
            )

            # Path A: SMILES-based
            smiles_count = 0
            # Cache: same SMILES string → same embedding (avoid recomputation)
            smiles_cache: Dict[str, torch.Tensor] = {}
            for idx, smiles in self.smiles_map.items():
                if smiles in smiles_cache:
                    embeddings[idx] = smiles_cache[smiles]
                    smiles_count += 1
                    continue
                graph = smiles_to_graph(smiles)
                if graph is not None:
                    graph = graph.to(device)
                    emb = self._encode_smiles(graph)
                    embeddings[idx] = emb.squeeze(0)
                    smiles_cache[smiles] = emb.squeeze(0)
                    smiles_count += 1

            # Path B: Word2Vec-based (for those without SMILES)
            w2v_count = 0
            for idx in range(1, self.num_excipients):
                if idx not in self.has_smiles:
                    # Check if this is BOS or EOS (last two indices)
                    if idx >= self.num_excipients - 2:
                        # BOS = special_embedding(1), EOS = special_embedding(2)
                        special_idx = idx - (self.num_excipients - 2) + 1
                        embeddings[idx] = self.special_embedding(
                            torch.tensor(special_idx, device=device)
                        )
                    else:
                        w2v_emb = self.w2v_matrix[idx].unsqueeze(0).to(device)
                        emb = self.w2v_proj(w2v_emb)
                        embeddings[idx] = emb.squeeze(0)
                        w2v_count += 1

            self.embedding_matrix = embeddings

        self._embeddings_built = True
        logger.info(
            f"Built embeddings: {smiles_count} via SMILES, "
            f"{w2v_count} via Word2Vec, 3 special tokens "
            f"(total {self.num_excipients})"
        )

    def _encode_smiles(self, graph) -> torch.Tensor:
        """Encode a single molecular graph through the lightweight GATv2."""
        from torch_geometric.nn import global_mean_pool

        x = self.smiles_node_proj(graph.x)
        edge_attr = self.smiles_edge_proj(graph.edge_attr) if graph.edge_attr.size(0) > 0 else None

        # Layer 1
        residual = x
        x = self.smiles_norm1(x)
        if edge_attr is not None:
            x = self.smiles_gat1(x, graph.edge_index, edge_attr=edge_attr)
        else:
            x = self.smiles_gat1(x, graph.edge_index)
        x = torch.nn.functional.gelu(x)
        x = x + residual

        # Layer 2
        residual = x
        x = self.smiles_norm2(x)
        if edge_attr is not None:
            x = self.smiles_gat2(x, graph.edge_index, edge_attr=edge_attr)
        else:
            x = self.smiles_gat2(x, graph.edge_index)
        x = torch.nn.functional.gelu(x)
        x = x + residual

        # Global readout
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = global_mean_pool(x, batch)  # [1, 256]

        return self.smiles_proj(pooled)  # [1, 512]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up excipient embeddings by index.

        Args:
            indices: Excipient indices [*] (any shape)

        Returns:
            Embeddings [*, embedding_dim]
        """
        if not self._embeddings_built:
            raise RuntimeError(
                "Excipient embeddings not built. Call build_embeddings() first."
            )
        return self.embedding_matrix[indices]

    def get_all_embeddings(self) -> torch.Tensor:
        """Return the full embedding matrix [num_excipients, embedding_dim]."""
        if not self._embeddings_built:
            raise RuntimeError(
                "Excipient embeddings not built. Call build_embeddings() first."
            )
        return self.embedding_matrix
