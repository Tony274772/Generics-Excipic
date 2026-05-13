"""
Excipient Knowledge Graph with GCNII.

Builds a co-occurrence graph over excipients and propagates information
using GCNII (deep GCN with initial residual connections).
"""
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
from torch_geometric.utils import dense_to_sparse

logger = logging.getLogger(__name__)


class ExcipientKnowledgeGraph(nn.Module):
    """
    Excipient co-occurrence knowledge graph with GCNII.

    Takes excipient embeddings (from ExcipientEmbedder) and refines them
    using graph propagation over the co-occurrence structure.

    GCNII is used because:
        - Supports deeper propagation (avoids oversmoothing)
        - Initial residual connections preserve node identity
        - Stable training
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.gcn_hidden_dim

        # Input projection (from excipient embedding dim to GCN hidden)
        self.input_proj = nn.Linear(config.embedding_dim, hidden_dim)

        # GCNII layers
        self.gcn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(config.gcn_layers):
            self.gcn_layers.append(
                GCN2Conv(
                    channels=hidden_dim,
                    alpha=0.1,  # initial residual weight
                    theta=0.5,  # identity mapping weight
                    layer=i + 1,
                    shared_weights=True,
                    normalize=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.gcn_dropout)

        # Graph structure (set during initialization)
        self.register_buffer("edge_index", torch.zeros(2, 0, dtype=torch.long))
        self.register_buffer("edge_weight", torch.zeros(0, dtype=torch.float))

    def set_graph(self, pmi_matrix: np.ndarray, threshold: float = 0.0):
        """
        Build the graph structure from the PMI co-occurrence matrix.

        Args:
            pmi_matrix: PMI matrix [N, N] (positive PMI values).
            threshold: Minimum PMI value to include as an edge.
        """
        # Threshold to create sparse graph
        adj = torch.from_numpy(pmi_matrix).float()
        adj[adj <= threshold] = 0

        # Convert to sparse edge_index + edge_weight
        edge_index, edge_weight = dense_to_sparse(adj)

        # Re-register buffers to ensure they are on the correct device
        device = self.input_proj.weight.device if hasattr(self, 'input_proj') else torch.device('cpu')
        self.register_buffer("edge_index", edge_index.to(device))
        self.register_buffer("edge_weight", edge_weight.to(device))

        logger.info(
            f"Excipient graph: {adj.size(0)} nodes, "
            f"{edge_index.size(1)} edges (threshold={threshold})"
        )

    def forward(self, excipient_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Propagate excipient embeddings through the knowledge graph.

        Args:
            excipient_embeddings: [N, embedding_dim] from ExcipientEmbedder

        Returns:
            Refined embeddings [N, gcn_hidden_dim]
        """
        # Project to hidden dim
        x = self.input_proj(excipient_embeddings)
        x_0 = x  # initial representation for GCNII residual

        # GCNII layers
        for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.layer_norms)):
            residual = x
            x = norm(x)
            x = gcn(x, x_0, self.edge_index, self.edge_weight)
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + residual  # additional residual

        x = self.output_norm(x)
        return x
