"""
Bemis-Murcko scaffold-based data splitting.

Ensures that molecules with the same scaffold are in the same split,
preventing chemistry leakage between train/val/test.
"""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric

logger = logging.getLogger(__name__)


def get_scaffold(smiles: str, generic: bool = False) -> str:
    """
    Get the Bemis-Murcko scaffold for a SMILES string.

    Args:
        smiles: SMILES string.
        generic: If True, return a generic scaffold (all atoms → carbon, all bonds → single).

    Returns:
        Scaffold SMILES string, or empty string if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    try:
        scaffold = GetScaffoldForMol(mol)
        if generic:
            scaffold = MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def scaffold_split(
    smiles_list: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split data indices by Bemis-Murcko scaffold.

    Molecules with the same scaffold go into the same split.
    Scaffolds are sorted by size (largest first) and greedily assigned
    to the smallest split until ratios are met.

    Args:
        smiles_list: List of SMILES strings (one per sample).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    rng = np.random.RandomState(seed)

    # Group indices by scaffold
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi, generic=True)
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (largest groups first for deterministic assignment)
    scaffolds = list(scaffold_to_indices.keys())
    scaffold_sizes = [len(scaffold_to_indices[s]) for s in scaffolds]
    sorted_order = np.argsort(scaffold_sizes)[::-1]

    n = len(smiles_list)
    train_cutoff = train_ratio * n
    val_cutoff = (train_ratio + val_ratio) * n

    train_indices = []
    val_indices = []
    test_indices = []

    for i in sorted_order:
        scaffold = scaffolds[i]
        indices = scaffold_to_indices[scaffold]
        rng.shuffle(indices)

        if len(train_indices) < train_cutoff:
            train_indices.extend(indices)
        elif len(train_indices) + len(val_indices) < val_cutoff:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    # Shuffle within each split
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    logger.info(
        f"Scaffold split: train={len(train_indices)}, "
        f"val={len(val_indices)}, test={len(test_indices)}, "
        f"scaffolds={len(scaffold_to_indices)}"
    )

    return train_indices, val_indices, test_indices
