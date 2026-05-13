"""
Excipic Inference Module.

Load a trained model and predict excipient recommendations
for new API + dosage form inputs.
"""
import json
import logging
from typing import Dict, List, Optional

import sys
import os

import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from excipic.config import ExcipicConfig
from excipic.data.graph_builder import smiles_to_graph, compute_rdkit_descriptors
from excipic.data.preprocessing import ExcipicPreprocessor
from excipic.models.excipic_model import ExcipicModel

logger = logging.getLogger(__name__)


class ExcipicPredictor:
    """
    Inference wrapper for Excipic model.

    Usage:
        predictor = ExcipicPredictor.from_checkpoint("outputs/checkpoints/best_model.pt")
        results = predictor.predict(smiles="CCO", dosage_form="TABLET")
    """

    def __init__(self, model: ExcipicModel, preprocessor: ExcipicPreprocessor,
                 config: ExcipicConfig, device: torch.device):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, cache_path: str = None,
                        device: str = "cuda"):
        """Load predictor from a saved checkpoint."""
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        config = ExcipicConfig()
        preprocessor = ExcipicPreprocessor(config)

        # Load preprocessed data (we only need vocab/graph info, not formulations)
        if cache_path:
            _ = preprocessor.preprocess(cache_path=cache_path)
        else:
            _ = preprocessor.preprocess()

        # Build model
        model = ExcipicModel(config, preprocessor)
        model.initialize(device, pmi_matrix=preprocessor.pmi_matrix)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        logger.info(f"Loaded model from {checkpoint_path}")
        return cls(model, preprocessor, config, device)

    @torch.no_grad()
    def predict(
        self,
        smiles: str,
        dosage_form: str,
        api_unii: Optional[str] = None,
        top_k: int = 15,
        beam_search: bool = True,
        temperature: float = 0.7,
    ) -> List[Dict[str, str]]:
        """
        Predict excipient recommendations.

        Args:
            smiles: API molecule SMILES string
            dosage_form: Dosage form (e.g., "TABLET")
            api_unii: Optional API UNII for feature lookup
            top_k: Number of excipients to recommend
            beam_search: Use beam search decoding
            temperature: Sampling temperature (for non-beam-search)

        Returns:
            List of dicts: [{"rank": 1, "name": "...", "unii": "..."}, ...]
        """
        self.model.eval()

        # Build molecular graph
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        graph = graph.to(self.device)

        # Batch dimension
        from torch_geometric.data import Batch
        graph_batch = Batch.from_data_list([graph])

        # Dosage form
        dosage_form_upper = dosage_form.upper()
        dosage_idx = self.preprocessor.dosage_to_idx.get(dosage_form_upper, 0)
        dosage_tensor = torch.tensor([dosage_idx], dtype=torch.long, device=self.device)

        # API features
        if api_unii and api_unii in self.preprocessor.api_features:
            raw_features = self.preprocessor.api_features[api_unii]
        else:
            raw_features = compute_rdkit_descriptors(smiles)

        normalized_features = (
            (raw_features - self.preprocessor.api_feature_mean)
            / self.preprocessor.api_feature_std
        )
        features_tensor = torch.tensor(
            normalized_features, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        api_targets = torch.tensor(
            raw_features, dtype=torch.float, device=self.device
        ).unsqueeze(0)

        # Build batch
        batch = {
            "graph": graph_batch,
            "dosage_idx": dosage_tensor,
            "api_features": features_tensor,
            "api_targets": api_targets,
        }

        # Predict
        predictions = self.model.predict(
            batch,
            temperature=temperature,
            beam_search=beam_search,
            beam_size=self.config.decoder.beam_size,
            max_len=top_k,
        )

        # Decode predictions
        pred_indices = predictions["predictions"][0].cpu().tolist()
        results = []
        special_indices = {
            self.preprocessor.excipient_to_idx[self.preprocessor.PAD_TOKEN],
            self.preprocessor.excipient_to_idx[self.preprocessor.BOS_TOKEN],
            self.preprocessor.excipient_to_idx[self.preprocessor.EOS_TOKEN],
        }

        rank = 1
        for idx in pred_indices:
            if idx in special_indices:
                continue

            name = self.preprocessor.excipient_names.get(idx, "")
            unii = self.preprocessor.excipient_uniis.get(idx, "")
            if not name:
                continue

            results.append({
                "rank": rank,
                "unii": unii,
                "name": name,
            })
            rank += 1

            if rank > top_k:
                break

        return results
