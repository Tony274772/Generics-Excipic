"""
ExcipicDataset — PyTorch Dataset for formulation data.

Each sample contains:
    - Molecular graph (PyG Data) for the API
    - Dosage form index
    - API descriptor features (normalized)
    - Target excipient sequence (ordered indices)
    - Multi-hot excipient labels (for BCE loss)
    - API property targets (for auxiliary head)
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph_builder import smiles_to_graph

logger = logging.getLogger(__name__)


class ExcipicDataset(Dataset):
    """
    Dataset for Excipic model training.

    Each item returns a dict with all tensors needed for forward pass + loss computation.
    """

    def __init__(
        self,
        formulations: List[Dict[str, Any]],
        preprocessor,
        max_seq_len: int = 30,
        graph_cache: Optional[Dict[str, Data]] = None,
    ):
        """
        Args:
            formulations: List of parsed formulation records.
            preprocessor: ExcipicPreprocessor instance (for vocab info).
            max_seq_len: Maximum excipient sequence length for the decoder.
            graph_cache: Optional dict to cache SMILES→graph conversions.
        """
        self.formulations = formulations
        self.preprocessor = preprocessor
        self.max_seq_len = max_seq_len
        self.graph_cache = graph_cache if graph_cache is not None else {}

        self.pad_idx = preprocessor.excipient_to_idx[preprocessor.PAD_TOKEN]
        self.bos_idx = preprocessor.excipient_to_idx[preprocessor.BOS_TOKEN]
        self.eos_idx = preprocessor.excipient_to_idx[preprocessor.EOS_TOKEN]
        self.vocab_size = preprocessor.num_excipients

    def __len__(self) -> int:
        return len(self.formulations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        form = self.formulations[idx]

        # 1. Molecular graph
        smiles = form["api_smiles"]
        if smiles in self.graph_cache:
            graph = self.graph_cache[smiles]
        else:
            graph = smiles_to_graph(smiles)
            if graph is None:
                # Fallback: minimal graph
                graph = Data(
                    x=torch.zeros((1, 161), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 12), dtype=torch.float),
                )
            self.graph_cache[smiles] = graph

        # Clone to avoid in-place modification issues
        graph = graph.clone()

        # 2. Dosage form
        dosage_idx = torch.tensor(form["dosage_form_idx"], dtype=torch.long)

        # 3. API descriptor features (normalized)
        api_features = torch.tensor(
            form["api_features_normalized"], dtype=torch.float
        )

        # 4. API raw features (for property prediction targets)
        api_targets = torch.tensor(form["api_features"], dtype=torch.float)

        # 5. Target excipient sequence (for autoregressive decoder)
        exc_indices = form["excipient_indices"]

        # Decoder input: [BOS] + excipients
        decoder_input = [self.bos_idx] + exc_indices
        # Decoder target: excipients + [EOS]
        decoder_target = exc_indices + [self.eos_idx]

        # Pad to max_seq_len
        dec_in_len = len(decoder_input)
        dec_tgt_len = len(decoder_target)

        if dec_in_len > self.max_seq_len:
            decoder_input = decoder_input[: self.max_seq_len]
            dec_in_len = self.max_seq_len
        if dec_tgt_len > self.max_seq_len:
            decoder_target = decoder_target[: self.max_seq_len]
            dec_tgt_len = self.max_seq_len

        # Pad
        decoder_input_padded = decoder_input + [self.pad_idx] * (self.max_seq_len - dec_in_len)
        decoder_target_padded = decoder_target + [self.pad_idx] * (self.max_seq_len - dec_tgt_len)

        decoder_input_tensor = torch.tensor(decoder_input_padded, dtype=torch.long)
        decoder_target_tensor = torch.tensor(decoder_target_padded, dtype=torch.long)
        decoder_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        decoder_mask[:dec_tgt_len] = True

        # 6. Multi-hot label vector (for BCE loss branch)
        multi_hot = torch.zeros(self.vocab_size, dtype=torch.float)
        for ei in exc_indices:
            multi_hot[ei] = 1.0

        # 7. Dose info
        dose_mg = torch.tensor(form["dose_mg"], dtype=torch.float)

        return {
            "graph": graph,
            "dosage_idx": dosage_idx,
            "api_features": api_features,
            "api_targets": api_targets,
            "decoder_input": decoder_input_tensor,
            "decoder_target": decoder_target_tensor,
            "decoder_mask": decoder_mask,
            "multi_hot": multi_hot,
            "dose_mg": dose_mg,
            "num_excipients": torch.tensor(len(exc_indices), dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collation function for ExcipicDataset.

    Handles PyG graph batching alongside regular tensor batching.
    """
    from torch_geometric.data import Batch

    graphs = [item["graph"] for item in batch]
    graph_batch = Batch.from_data_list(graphs)

    return {
        "graph": graph_batch,
        "dosage_idx": torch.stack([item["dosage_idx"] for item in batch]),
        "api_features": torch.stack([item["api_features"] for item in batch]),
        "api_targets": torch.stack([item["api_targets"] for item in batch]),
        "decoder_input": torch.stack([item["decoder_input"] for item in batch]),
        "decoder_target": torch.stack([item["decoder_target"] for item in batch]),
        "decoder_mask": torch.stack([item["decoder_mask"] for item in batch]),
        "multi_hot": torch.stack([item["multi_hot"] for item in batch]),
        "dose_mg": torch.stack([item["dose_mg"] for item in batch]),
        "num_excipients": torch.stack([item["num_excipients"] for item in batch]),
    }
