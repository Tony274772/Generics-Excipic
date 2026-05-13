"""
Data preprocessing pipeline for Excipic.

Loads the shared excipient vocabulary from excipient_unii_vocab.json,
then parses train.csv / val.csv / test.csv, builds dosage/feature tables,
and prepares all data structures needed for training.

The shared vocabulary ensures all team architectures use identical
excipient index mappings for fair comparison.
"""
import json
import logging
import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .graph_builder import compute_rdkit_descriptors, smiles_to_graph

logger = logging.getLogger(__name__)


class ExcipicPreprocessor:
    """
    Preprocesses data using the shared excipient vocabulary.

    The excipient vocabulary is loaded from excipient_unii_vocab.json (shared
    across all team architectures). This ensures identical index mappings
    so that models can be compared fairly.

    Vocab layout:
        idx 0: <PAD>  (shared vocab reserves 0 for PAD/UNK)
        idx 1..950: excipients from shared vocab (name-level, same UNII may
                     appear multiple times with different names)
        idx 951: <BOS> (appended by us for decoder)
        idx 952: <EOS> (appended by us for decoder)
    """

    # Special tokens for the decoder
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"  # maps to same index as PAD (0)
    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self, config):
        self.config = config
        self.paths = config.paths

        # ── Vocabularies ──
        # Primary lookup: (name, unii) → idx  (from shared vocab)
        self.excipient_to_idx: Dict[str, int] = {}  # composite key "name||unii" → idx
        self.idx_to_excipient: Dict[int, str] = {}  # idx → composite key
        self.excipient_names: Dict[int, str] = {}    # idx → name
        self.excipient_uniis: Dict[int, str] = {}    # idx → unii
        self.unii_to_indices: Dict[str, List[int]] = {}  # unii → [idx, ...]

        # Backward compat aliases
        self.excipient_unii_to_idx: Dict[str, int] = {}  # unii → first idx (for embedder)
        self.excipient_smiles: Dict[str, str] = {}  # unii → SMILES

        self.dosage_to_idx: Dict[str, int] = {}
        self.idx_to_dosage: Dict[int, str] = {}

        # Feature tables
        self.api_features: Dict[str, np.ndarray] = {}
        self.api_feature_mean: Optional[np.ndarray] = None
        self.api_feature_std: Optional[np.ndarray] = None

        # Co-occurrence / KG
        self.cooccurrence_matrix: Optional[np.ndarray] = None
        self.pmi_matrix: Optional[np.ndarray] = None

        # Word2Vec
        self.w2v_model: Optional[Word2Vec] = None

        # Parsed data (separate splits)
        self.train_formulations: List[Dict[str, Any]] = []
        self.val_formulations: List[Dict[str, Any]] = []
        self.test_formulations: List[Dict[str, Any]] = []

        # Shared vocab metadata
        self._shared_vocab_size: int = 0  # excipient entries in shared vocab

    @property
    def num_excipients(self) -> int:
        """Total vocab size including special tokens."""
        return len(self.excipient_to_idx)

    @property
    def num_dosage_forms(self) -> int:
        return len(self.dosage_to_idx)

    @property
    def vocab_size(self) -> int:
        return self.num_excipients

    def preprocess(
        self, cache_path: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Run the full preprocessing pipeline.

        Returns:
            Tuple of (train_formulations, val_formulations, test_formulations).
        """
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached preprocessed data from {cache_path}")
            return self._load_cache(cache_path)

        logger.info("Starting preprocessing pipeline...")

        # 1. Load shared excipient vocabulary
        self._load_shared_vocab()

        # 2. Load all CSV splits
        train_df = self._load_csv(self.paths.train_csv, "train")
        val_df = self._load_csv(self.paths.val_csv, "val")
        test_df = self._load_csv(self.paths.test_csv, "test")
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # 3. Build dosage form vocabulary (from ALL splits)
        self._build_dosage_vocab(all_df)

        # 4. Load API features
        self._load_api_features()

        # 5. Load excipient SMILES
        self._load_excipient_smiles()

        # 6. Build co-occurrence graph (from TRAIN only — avoid leakage)
        self._build_cooccurrence(train_df)

        # 7. Train Word2Vec on excipient co-occurrences (TRAIN only)
        self._train_word2vec(train_df)

        # 8. Parse formulations for each split
        self.train_formulations = self._parse_formulations(train_df, "train")
        self.val_formulations = self._parse_formulations(val_df, "val")
        self.test_formulations = self._parse_formulations(test_df, "test")

        logger.info(
            f"Preprocessing complete: "
            f"train={len(self.train_formulations)}, "
            f"val={len(self.val_formulations)}, "
            f"test={len(self.test_formulations)}, "
            f"{self.num_excipients} vocab entries "
            f"({self._shared_vocab_size} excipients + special tokens), "
            f"{self.num_dosage_forms} dosage forms"
        )

        if cache_path:
            self._save_cache(cache_path)

        return self.train_formulations, self.val_formulations, self.test_formulations

    # ─────────────────────────────────────────────────────────────────────────
    # Shared vocabulary loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_shared_vocab(self):
        """
        Load the shared excipient vocabulary from excipient_unii_vocab.json.

        Layout after loading:
            idx 0: <PAD> / <UNK>
            idx 1..N: excipients from shared vocab (N = total_excipients)
            idx N+1: <BOS>
            idx N+2: <EOS>
        """
        vocab_path = self.paths.excipient_vocab_json
        logger.info(f"Loading shared excipient vocabulary from {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        metadata = vocab_data["metadata"]
        excipients = vocab_data["excipients"]
        self._shared_vocab_size = metadata["total_excipients"]

        logger.info(
            f"Shared vocab: {self._shared_vocab_size} excipients, "
            f"pad_idx=0, built_from={metadata.get('built_from', '?')}"
        )

        # idx 0 → PAD / UNK
        self.excipient_to_idx[self.PAD_TOKEN] = 0
        self.excipient_to_idx[self.UNK_TOKEN] = 0  # UNK shares PAD index
        self.idx_to_excipient[0] = self.PAD_TOKEN
        self.excipient_names[0] = self.PAD_TOKEN
        self.excipient_uniis[0] = ""

        # Load excipient entries
        for entry in excipients:
            idx = entry["idx"]  # 1-based from the JSON
            unii = entry["unii"]
            name = entry["name"]

            # Composite key: "name||unii" to handle same UNII different names
            composite_key = f"{name}||{unii}"
            self.excipient_to_idx[composite_key] = idx
            self.idx_to_excipient[idx] = composite_key
            self.excipient_names[idx] = name
            self.excipient_uniis[idx] = unii

            # Track all indices per UNII
            if unii not in self.unii_to_indices:
                self.unii_to_indices[unii] = []
            self.unii_to_indices[unii].append(idx)

            # Backward compat: first occurrence of each UNII
            if unii not in self.excipient_unii_to_idx:
                self.excipient_unii_to_idx[unii] = idx

        # Append BOS and EOS after all excipient entries
        bos_idx = self._shared_vocab_size + 1
        eos_idx = self._shared_vocab_size + 2
        self.excipient_to_idx[self.BOS_TOKEN] = bos_idx
        self.excipient_to_idx[self.EOS_TOKEN] = eos_idx
        self.idx_to_excipient[bos_idx] = self.BOS_TOKEN
        self.idx_to_excipient[eos_idx] = self.EOS_TOKEN
        self.excipient_names[bos_idx] = self.BOS_TOKEN
        self.excipient_names[eos_idx] = self.EOS_TOKEN
        self.excipient_uniis[bos_idx] = ""
        self.excipient_uniis[eos_idx] = ""

        logger.info(
            f"Vocab loaded: {self.num_excipients} total entries "
            f"(PAD=0, excipients=1..{self._shared_vocab_size}, "
            f"BOS={bos_idx}, EOS={eos_idx})"
        )

    def _match_excipient(self, name: str, unii: str) -> Optional[int]:
        """
        Match a (name, unii) pair from formulation data to the shared vocab.

        Tries exact composite key first, then falls back to UNII-only match.
        Returns the vocab index, or None if not found.
        """
        # Exact match: "name||unii"
        composite_key = f"{name}||{unii}"
        if composite_key in self.excipient_to_idx:
            return self.excipient_to_idx[composite_key]

        # Fallback: match by UNII (use the first matching index)
        if unii in self.unii_to_indices:
            return self.unii_to_indices[unii][0]

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Data loading helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _load_csv(self, path: str, split_name: str) -> pd.DataFrame:
        logger.info(f"Loading {split_name}: {path}")
        df = pd.read_csv(path)
        logger.info(f"  {split_name}: {len(df)} rows")
        return df

    def _build_dosage_vocab(self, df: pd.DataFrame):
        logger.info("Building dosage form vocabulary...")
        dosage_forms = df["primary_dosage_form"].dropna().str.upper().unique()
        dosage_forms = sorted(dosage_forms)
        self.dosage_to_idx = {form: idx for idx, form in enumerate(dosage_forms)}
        self.idx_to_dosage = {v: k for k, v in self.dosage_to_idx.items()}
        logger.info(f"Dosage form vocabulary: {self.num_dosage_forms} forms")

    def _load_api_features(self):
        logger.info(f"Loading API features from {self.paths.api_features}")
        df = pd.read_csv(self.paths.api_features)
        feature_cols = [c for c in df.columns if c not in ("api_unii", "smiles")]

        for _, row in df.iterrows():
            unii = row["api_unii"]
            features = row[feature_cols].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            self.api_features[unii] = features

        all_features = np.stack(list(self.api_features.values()))
        self.api_feature_mean = np.mean(all_features, axis=0)
        self.api_feature_std = np.std(all_features, axis=0)
        self.api_feature_std[self.api_feature_std < 1e-8] = 1.0

        logger.info(
            f"Loaded features for {len(self.api_features)} APIs, "
            f"{len(feature_cols)} descriptors"
        )

    def _load_excipient_smiles(self):
        logger.info(f"Loading excipient SMILES from {self.paths.excipient_smiles}")
        df = pd.read_csv(self.paths.excipient_smiles)

        for _, row in df.iterrows():
            unii = row.get("unii", "")
            smiles = row.get("smiles", "")
            if unii and isinstance(smiles, str) and smiles and smiles != "Not found":
                if unii in self.excipient_unii_to_idx:
                    self.excipient_smiles[unii] = smiles

        logger.info(
            f"Excipient SMILES: {len(self.excipient_smiles)} with SMILES "
            f"(of {len(self.excipient_unii_to_idx)} unique UNIIs in vocab)"
        )

    def _build_cooccurrence(self, df: pd.DataFrame):
        """Build excipient co-occurrence matrix using vocab indices."""
        logger.info("Building excipient co-occurrence matrix...")

        n = self.num_excipients
        cooccurrence = np.zeros((n, n), dtype=np.float32)
        frequency = np.zeros(n, dtype=np.float32)

        total_formulations = 0
        for row in df["inactive_ingredients"].dropna():
            try:
                exc_list = json.loads(row)
                indices = []
                for e in exc_list:
                    idx = self._match_excipient(e.get("name", ""), e.get("unii", ""))
                    if idx is not None and idx > 0:
                        indices.append(idx)

                for idx in indices:
                    frequency[idx] += 1
                for i, idx_i in enumerate(indices):
                    for j, idx_j in enumerate(indices):
                        if i != j:
                            cooccurrence[idx_i, idx_j] += 1
                total_formulations += 1
            except (json.JSONDecodeError, TypeError):
                continue

        self.cooccurrence_matrix = cooccurrence

        logger.info("Computing PMI matrix...")
        pmi = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                if cooccurrence[i, j] > 0 and frequency[i] > 0 and frequency[j] > 0:
                    p_ij = cooccurrence[i, j] / total_formulations
                    p_i = frequency[i] / total_formulations
                    p_j = frequency[j] / total_formulations
                    pmi_val = np.log(p_ij / (p_i * p_j) + 1e-10)
                    pmi[i, j] = max(pmi_val, 0)
                    pmi[j, i] = pmi[i, j]

        self.pmi_matrix = pmi
        logger.info(
            f"Co-occurrence matrix: {n}x{n}, "
            f"non-zero PMI edges: {(pmi > 0).sum() // 2}"
        )

    def _train_word2vec(self, df: pd.DataFrame):
        """Train Word2Vec on excipient co-occurrences (using vocab indices as tokens)."""
        logger.info("Training Word2Vec on excipient co-occurrences...")

        sentences = []
        for row in df["inactive_ingredients"].dropna():
            try:
                exc_list = json.loads(row)
                sentence = []
                for e in exc_list:
                    idx = self._match_excipient(e.get("name", ""), e.get("unii", ""))
                    if idx is not None and idx > 0:
                        sentence.append(str(idx))  # Word2Vec needs string tokens
                if len(sentence) >= 2:
                    sentences.append(sentence)
            except (json.JSONDecodeError, TypeError):
                continue

        cfg = self.config.excipient_graph
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=cfg.w2v_dim,
            window=cfg.w2v_window,
            min_count=cfg.w2v_min_count,
            workers=4,
            epochs=30,
            seed=42,
        )

        logger.info(
            f"Word2Vec trained: {len(self.w2v_model.wv)} token embeddings, "
            f"dim={cfg.w2v_dim}"
        )

    def get_w2v_embedding(self, idx: int) -> np.ndarray:
        """Get Word2Vec embedding for an excipient index."""
        dim = self.config.excipient_graph.w2v_dim
        key = str(idx)
        if self.w2v_model is not None and key in self.w2v_model.wv:
            return self.w2v_model.wv[key].astype(np.float32)
        return np.zeros(dim, dtype=np.float32)

    def _parse_formulations(self, df: pd.DataFrame, split_name: str) -> List[Dict]:
        """Parse each row into a structured formulation record."""
        logger.info(f"Parsing {split_name} formulations...")

        formulations = []
        skipped = 0

        for _, row in df.iterrows():
            api_smiles = row.get("api_smiles", None)
            if not api_smiles or not isinstance(api_smiles, str) or pd.isna(api_smiles):
                skipped += 1
                continue

            # Extract excipient list and match to shared vocab
            try:
                inactive = row["inactive_ingredients"]
                exc_list = json.loads(inactive) if isinstance(inactive, str) else []
            except (json.JSONDecodeError, TypeError):
                exc_list = []

            excipient_indices = []
            for e in exc_list:
                idx = self._match_excipient(e.get("name", ""), e.get("unii", ""))
                if idx is not None and idx > 0:
                    excipient_indices.append(idx)

            if len(excipient_indices) < 1:
                skipped += 1
                continue

            # Dosage form
            dosage_form = str(row.get("primary_dosage_form", "")).upper()
            dosage_idx = self.dosage_to_idx.get(dosage_form, 0)

            # API features
            api_unii = row.get("api_unii", "")
            api_feat = self.api_features.get(api_unii, None)
            if api_feat is None:
                api_feat = compute_rdkit_descriptors(api_smiles)

            api_feat_normalized = (api_feat - self.api_feature_mean) / self.api_feature_std

            dose_mg = float(row.get("dose_mg", 0.0)) if pd.notna(row.get("dose_mg")) else 0.0

            formulations.append({
                "api_smiles": api_smiles,
                "api_unii": api_unii,
                "dosage_form_idx": dosage_idx,
                "dosage_form": dosage_form,
                "api_features": api_feat,
                "api_features_normalized": api_feat_normalized,
                "excipient_indices": excipient_indices,
                "dose_mg": dose_mg,
            })

        logger.info(
            f"  {split_name}: {len(formulations)} valid formulations, skipped {skipped}"
        )
        return formulations

    # ─────────────────────────────────────────────────────────────────────────
    # Cache
    # ─────────────────────────────────────────────────────────────────────────

    def _save_cache(self, cache_path: str):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        data = {
            "train_formulations": self.train_formulations,
            "val_formulations": self.val_formulations,
            "test_formulations": self.test_formulations,
            "excipient_to_idx": self.excipient_to_idx,
            "idx_to_excipient": self.idx_to_excipient,
            "excipient_names": self.excipient_names,
            "excipient_uniis": self.excipient_uniis,
            "unii_to_indices": self.unii_to_indices,
            "excipient_unii_to_idx": self.excipient_unii_to_idx,
            "excipient_smiles": self.excipient_smiles,
            "dosage_to_idx": self.dosage_to_idx,
            "idx_to_dosage": self.idx_to_dosage,
            "api_features": self.api_features,
            "api_feature_mean": self.api_feature_mean,
            "api_feature_std": self.api_feature_std,
            "cooccurrence_matrix": self.cooccurrence_matrix,
            "pmi_matrix": self.pmi_matrix,
            "w2v_model": self.w2v_model,
            "_shared_vocab_size": self._shared_vocab_size,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved cache to {cache_path}")

    def _load_cache(self, cache_path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        self.train_formulations = data["train_formulations"]
        self.val_formulations = data["val_formulations"]
        self.test_formulations = data["test_formulations"]
        self.excipient_to_idx = data["excipient_to_idx"]
        self.idx_to_excipient = data["idx_to_excipient"]
        self.excipient_names = data["excipient_names"]
        self.excipient_uniis = data["excipient_uniis"]
        self.unii_to_indices = data["unii_to_indices"]
        self.excipient_unii_to_idx = data["excipient_unii_to_idx"]
        self.excipient_smiles = data["excipient_smiles"]
        self.dosage_to_idx = data["dosage_to_idx"]
        self.idx_to_dosage = data["idx_to_dosage"]
        self.api_features = data["api_features"]
        self.api_feature_mean = data["api_feature_mean"]
        self.api_feature_std = data["api_feature_std"]
        self.cooccurrence_matrix = data["cooccurrence_matrix"]
        self.pmi_matrix = data["pmi_matrix"]
        self.w2v_model = data["w2v_model"]
        self._shared_vocab_size = data["_shared_vocab_size"]

        logger.info(
            f"Loaded cache: train={len(self.train_formulations)}, "
            f"val={len(self.val_formulations)}, test={len(self.test_formulations)}"
        )
        return self.train_formulations, self.val_formulations, self.test_formulations
