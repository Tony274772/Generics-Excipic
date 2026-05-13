"""
SMILES → PyTorch Geometric molecular graph.

Converts SMILES strings into graph representations using RDKit,
with atom node features and bond edge features.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from torch_geometric.data import Data


# ── Atom feature extraction ──────────────────────────────────────────────────

ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # 1-118
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-2, -1, 0, 1, 2, 3],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": [0, 1, 2, 3, 4],
    "valence": [0, 1, 2, 3, 4, 5, 6],
    "chiral_tag": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
}


def one_hot(value, choices):
    """One-hot encoding with an extra 'unknown' bit."""
    encoding = [0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    encoding[idx] = 1
    return encoding


def get_atom_features(atom):
    """Extract feature vector for a single atom."""
    features = []
    features += one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
    features += one_hot(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
    features += one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
    features += one_hot(atom.GetHybridization(), ATOM_FEATURES["hybridization"])
    features += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
    features += one_hot(atom.GetTotalValence(), ATOM_FEATURES["valence"])
    features += one_hot(atom.GetChiralTag(), ATOM_FEATURES["chiral_tag"])
    features.append(int(atom.GetIsAromatic()))
    features.append(int(atom.IsInRing()))
    return features


# Total atom feature dim:
# 119 + 8 + 7 + 6 + 6 + 8 + 5 + 1 + 1 = 161
# Actually: (118+1) + (7+1) + (6+1) + (5+1) + (5+1) + (7+1) + (4+1) + 1 + 1
ATOM_FEATURE_DIM = (
    (118 + 1) + (7 + 1) + (6 + 1) + (5 + 1) + (5 + 1) + (7 + 1) + (4 + 1) + 1 + 1
)  # = 161


# ── Bond feature extraction ─────────────────────────────────────────────────

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def get_bond_features(bond):
    """Extract feature vector for a single bond."""
    features = []
    features += one_hot(bond.GetBondType(), BOND_TYPES)
    features += one_hot(bond.GetStereo(), BOND_STEREO)
    features.append(int(bond.GetIsConjugated()))
    features.append(int(bond.IsInRing()))
    return features


BOND_FEATURE_DIM = (4 + 1) + (4 + 1) + 1 + 1  # = 12


# ── SMILES → PyG Data ────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str) -> Data:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Returns None if the SMILES is invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge features (undirected → add both directions)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # Handle single-atom molecules
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    data.num_nodes = x.size(0)
    return data


def compute_rdkit_descriptors(smiles: str) -> np.ndarray:
    """
    Compute RDKit molecular descriptors for a SMILES string.
    Returns a numpy array of descriptor values matching api_features.csv columns.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(20, dtype=np.float32)

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.MolMR(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0.0,
        Descriptors.BertzCT(mol),
        Descriptors.HallKierAlpha(mol),
        rdMolDescriptors.CalcNumSaturatedRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        Descriptors.NHOHCount(mol),
    ]
    return np.array(descriptors, dtype=np.float32)


# Descriptor names matching api_features.csv (minus api_unii and smiles columns)
DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "RingCount", "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons",
    "MolMR", "LabuteASA", "BalabanJ", "BertzCT", "HallKierAlpha",
    "NumSaturatedRings", "NumHeteroatoms", "NHOHCount",
]
NUM_DESCRIPTORS = len(DESCRIPTOR_NAMES)
