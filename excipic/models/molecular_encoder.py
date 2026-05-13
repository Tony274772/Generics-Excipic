"""
GATv2 Molecular Encoder.

Converts molecular graphs (atoms + bonds) into a fixed-dim embedding.
Uses GATv2 with virtual node, residual connections, and LayerNorm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.utils import to_dense_batch


class VirtualNodeUpdate(nn.Module):
    """
    Virtual node aggregation — collects global info from all nodes
    and broadcasts it back.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, vnode: torch.Tensor, batch: torch.Tensor):
        """
        Args:
            x: Node features [N, hidden_dim]
            vnode: Virtual node features [B, hidden_dim]
            batch: Batch assignment [N]

        Returns:
            Updated x and vnode.
        """
        # Aggregate all node features per graph → update virtual node
        agg = global_add_pool(x, batch)  # [B, hidden_dim]
        vnode = self.norm(vnode + self.mlp(agg))

        # Broadcast virtual node back to all nodes
        x = x + vnode[batch]
        return x, vnode


class MolecularEncoder(nn.Module):
    """
    GATv2 molecular encoder.

    Architecture:
        - Input projection (node features → hidden_dim)
        - N GATv2 layers with residual + LayerNorm
        - Virtual node for global information flow
        - Attention-weighted readout → graph-level [CLS] embedding
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim
        num_heads = config.num_heads

        # Input projections
        self.node_proj = nn.Linear(config.node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(config.edge_input_dim, hidden_dim)

        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(config.num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=config.attention_dropout,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                    concat=True,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

            # FFN after attention
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(config.dropout),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(hidden_dim))

        # Virtual node
        self.use_virtual_node = config.use_virtual_node
        if self.use_virtual_node:
            self.vnode_embedding = nn.Embedding(1, hidden_dim)
            self.vnode_updates = nn.ModuleList([
                VirtualNodeUpdate(hidden_dim, config.dropout)
                for _ in range(config.num_layers)
            ])

        # Attention readout
        self.readout_att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, node_input_dim]
            edge_index: Edge index [2, E]
            edge_attr: Edge features [E, edge_input_dim]
            batch: Batch assignment [N]

        Returns:
            Graph-level embeddings [B, hidden_dim]
        """
        # Input projection
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        # Initialize virtual node
        if self.use_virtual_node:
            batch_size = batch.max().item() + 1
            vnode = self.vnode_embedding.weight.expand(batch_size, -1)

        # GATv2 layers
        for i in range(len(self.gat_layers)):
            # Virtual node broadcast
            if self.use_virtual_node:
                x, vnode = self.vnode_updates[i](x, vnode, batch)

            # Pre-norm + GATv2 attention + residual
            residual = x
            x = self.layer_norms[i](x)
            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr)
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + residual

            # FFN + residual
            residual = x
            x = self.ffn_norms[i](x)
            x = self.ffn_layers[i](x)
            x = x + residual

        # Attention readout: weighted sum of node features
        att_weights = self.readout_att(x)  # [N, 1]
        # Mask attention per graph using to_dense_batch
        x_dense, mask = to_dense_batch(x, batch)  # [B, max_N, hidden_dim]
        att_dense, _ = to_dense_batch(att_weights.squeeze(-1), batch)  # [B, max_N]

        att_dense = att_dense.masked_fill(~mask, float("-inf"))
        att_dense = F.softmax(att_dense, dim=-1).unsqueeze(-1)  # [B, max_N, 1]

        graph_emb = (x_dense * att_dense).sum(dim=1)  # [B, hidden_dim]
        graph_emb = self.output_norm(graph_emb)

        return graph_emb
