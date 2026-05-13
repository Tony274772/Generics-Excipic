"""
Excipic Full Model.

End-to-end model combining:
    1. Molecular Encoder (GATv2)
    2. Dosage Encoder
    3. Cross-Attention Fusion Transformer
    4. Physicochemical Property Head
    5. Excipient Embedder (dual-path)
    6. Excipient Knowledge Graph (GCNII)
    7. Autoregressive Set Decoder (Pointer Transformer)
"""
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .molecular_encoder import MolecularEncoder
from .dosage_encoder import DosageEncoder
from .fusion import FusionTransformer
from .property_head import PropertyHead
from .excipient_embedder import ExcipientEmbedder
from .excipient_graph import ExcipientKnowledgeGraph
from .set_decoder import PointerTransformerDecoder

logger = logging.getLogger(__name__)


class ExcipicModel(nn.Module):
    """
    Full Excipic model.

    Forward pass:
        SMILES graph → MolecularEncoder → mol_emb (512)
        Dosage form + descriptors → DosageEncoder → dos_emb (512)
        (mol_emb, dos_emb) → FusionTransformer → fusion (1536)
        fusion → PropertyHead → property predictions
        ExcipientEmbedder → ExcipientKG → refined excipient embeddings
        (fusion, excipient_embs) → SetDecoder → ranked excipient predictions
    """

    def __init__(self, config, preprocessor):
        super().__init__()
        self.config = config

        # 1. Molecular Encoder
        self.molecular_encoder = MolecularEncoder(config.molecular_encoder)

        # 2. Dosage Encoder
        dosage_cfg = config.dosage_encoder
        dosage_cfg.num_dosage_forms = preprocessor.num_dosage_forms
        self.dosage_encoder = DosageEncoder(dosage_cfg)

        # 3. Fusion Transformer
        self.fusion = FusionTransformer(config.fusion)

        # 4. Property Head
        self.property_head = PropertyHead(config.property_head)

        # 5. Excipient Embedder (dual-path)
        # Build index-based W2V embeddings
        w2v_embeddings = {}
        for idx in range(preprocessor.num_excipients):
            w2v_embeddings[idx] = preprocessor.get_w2v_embedding(idx)

        self.excipient_embedder = ExcipientEmbedder(
            config=config.excipient_graph,
            excipient_smiles=preprocessor.excipient_smiles,
            w2v_embeddings=w2v_embeddings,
            excipient_uniis=preprocessor.excipient_uniis,
            num_excipients=preprocessor.num_excipients,
        )

        # 6. Excipient Knowledge Graph
        self.excipient_graph = ExcipientKnowledgeGraph(config.excipient_graph)

        # 7. Set Decoder
        self.decoder = PointerTransformerDecoder(
            config=config.decoder,
            vocab_size=preprocessor.num_excipients,
        )

        # Store special token indices
        self.pad_idx = preprocessor.excipient_to_idx[preprocessor.PAD_TOKEN]
        self.bos_idx = preprocessor.excipient_to_idx[preprocessor.BOS_TOKEN]
        self.eos_idx = preprocessor.excipient_to_idx[preprocessor.EOS_TOKEN]
        self.vocab_size = preprocessor.num_excipients

        # Refined excipient embeddings (updated each forward pass or periodically)
        self._refined_embeddings: Optional[torch.Tensor] = None

    def initialize(self, device: torch.device, pmi_matrix=None):
        """
        Initialize components that need setup after model creation.
        Call this before training.
        """
        # Build excipient embeddings
        self.to(device)
        self.excipient_embedder.build_embeddings(device)

        # Set knowledge graph structure
        if pmi_matrix is not None:
            self.excipient_graph.set_graph(pmi_matrix, threshold=0.0)

        # Compute refined embeddings
        self._update_excipient_embeddings()

        logger.info("Model initialized successfully")

    def _update_excipient_embeddings(self):
        """Propagate excipient embeddings through knowledge graph."""
        raw_embs = self.excipient_embedder.get_all_embeddings()  # [V, 512]
        self._refined_embeddings = self.excipient_graph(raw_embs)  # [V, 512]

    def _excipient_lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up refined excipient embeddings by index."""
        # Use raw embeddings for lookup (refined used only for pointer keys)
        return self.excipient_embedder(indices)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            batch: Dict from ExcipicDataset collate_fn containing:
                - graph: Batched PyG graph
                - dosage_idx: [B]
                - api_features: [B, D_desc]
                - decoder_input: [B, T]
                - decoder_target: [B, T]
                - decoder_mask: [B, T]
                - multi_hot: [B, V]
                - api_targets: [B, D_desc]

        Returns:
            Dict with:
                - decoder_logits: [B, T, V]
                - property_preds: [B, num_properties]
                - multi_hot_logits: [B, V]
                - fusion: [B, 1536]
        """
        graph = batch["graph"]
        device = graph.x.device

        # 1. Molecular encoding
        mol_emb = self.molecular_encoder(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
        )  # [B, 512]

        # 2. Dosage encoding
        dos_emb = self.dosage_encoder(
            dosage_idx=batch["dosage_idx"],
            descriptors=batch["api_features"],
        )  # [B, 512]

        # 3. Fusion
        fusion = self.fusion(mol_emb, dos_emb)  # [B, 1536]

        # 4. Property prediction
        property_preds = self.property_head(fusion)  # [B, num_properties]

        # 5. Update excipient embeddings through KG
        self._update_excipient_embeddings()
        refined_embs = self._refined_embeddings  # [V, 512]

        # 6. Decoder
        decoder_logits = self.decoder(
            decoder_input_ids=batch["decoder_input"],
            fusion_context=fusion,
            excipient_embeddings=refined_embs,
            excipient_lookup_fn=self._excipient_lookup,
        )  # [B, T, V]

        # 7. Multi-hot prediction (BCE branch) — simple linear from fusion
        # Use the fusion vector + average excipient context
        multi_hot_logits = torch.matmul(
            fusion,
            nn.functional.linear(refined_embs, torch.eye(refined_embs.size(-1), device=device)[:fusion.size(-1)]).transpose(0, 1)
        ) if False else self._compute_multi_hot_logits(fusion, refined_embs)

        return {
            "decoder_logits": decoder_logits,
            "property_preds": property_preds,
            "multi_hot_logits": multi_hot_logits,
            "fusion": fusion,
        }

    def _compute_multi_hot_logits(
        self, fusion: torch.Tensor, excipient_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-hot logits by projecting fusion to excipient space
        and computing similarity.
        """
        # Project fusion down to excipient dim
        # fusion: [B, 1536], excipient_embs: [V, 512]
        # Simple approach: use first 512 dims of fusion as query
        query = fusion[:, :512]  # [B, 512]
        logits = torch.matmul(query, excipient_embs.transpose(0, 1))  # [B, V]
        return logits

    @torch.no_grad()
    def predict(
        self,
        batch: Dict[str, Any],
        temperature: float = 0.7,
        beam_search: bool = False,
        beam_size: int = 5,
        max_len: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference: generate excipient predictions.

        Returns:
            Dict with:
                - predictions: [B, max_len] generated excipient indices
                - property_preds: [B, num_properties]
        """
        self.eval()
        graph = batch["graph"]
        device = graph.x.device

        # Encode
        mol_emb = self.molecular_encoder(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            batch=graph.batch,
        )
        dos_emb = self.dosage_encoder(
            dosage_idx=batch["dosage_idx"],
            descriptors=batch["api_features"],
        )
        fusion = self.fusion(mol_emb, dos_emb)
        property_preds = self.property_head(fusion)

        self._update_excipient_embeddings()
        refined_embs = self._refined_embeddings

        B = fusion.size(0)

        if beam_search and B == 1:
            predictions = self.decoder.beam_search(
                fusion_context=fusion,
                excipient_embeddings=refined_embs,
                excipient_lookup_fn=self._excipient_lookup,
                bos_idx=self.bos_idx,
                eos_idx=self.eos_idx,
                pad_idx=self.pad_idx,
                beam_size=beam_size,
                max_len=max_len,
            )
        else:
            predictions = self.decoder.generate(
                fusion_context=fusion,
                excipient_embeddings=refined_embs,
                excipient_lookup_fn=self._excipient_lookup,
                bos_idx=self.bos_idx,
                eos_idx=self.eos_idx,
                pad_idx=self.pad_idx,
                temperature=temperature,
                max_len=max_len,
            )

        return {
            "predictions": predictions,
            "property_preds": property_preds,
        }
