# Excipic Architecture & Implementation Guide

# Goal

Build a multimodal pharmaceutical AI system that recommends a ranked set of valid excipients for a given:

- API molecule (SMILES)
- Dosage form

The task is:

```text
(API + dosage form) → ranked excipient set
```

This is NOT a standard classification problem.

The correct formulation framing is:

- ranked retrieval
- structured set prediction
- graph-based recommendation

---

# Why NOT Use Simple Multi-Label Classification?

A standard classifier predicts:

```text
784 independent yes/no labels
```

But excipients are NOT independent.

Example:

- MCC often appears with Mg Stearate
- PVP often appears with Crospovidone

Also:

- output size varies
- ordering matters
- co-occurrence matters

Therefore:

```text
Excipic = ranked set retrieval problem
```

Best architecture:

```text
Dual-stream GNN
+ Cross-attention fusion
+ Excipient graph reasoning
+ Autoregressive set decoder
```

---

# High-Level Architecture

```text
SMILES ---------------------> Molecular GNN (GATv2)
                                       |
Dosage Form -------------> Dosage Encoder
                                       |
                    Cross-Attention Fusion Transformer
                                       |
                   Physicochemical Property Head
                                       |
             Excipient Knowledge Graph + GNN
                                       |
               Autoregressive Set Decoder
                                       |
                 Ranked Excipient Predictions
```

---

# 1. Molecular Encoder

# Architecture

Use:

```text
GATv2 (Graph Attention Network v2)
```

because:

- better than vanilla GAT
- avoids static attention collapse
- strong for molecular property learning
- state-of-the-art in chemistry GNNs

---

# Molecular Graph Construction

SMILES are converted using RDKit:

- atoms → nodes
- bonds → edges

---

# Node Features

Use:

| Feature | Include |
|---|---|
| Atomic number | Yes |
| Formal charge | Yes |
| Hybridization | Yes |
| Aromaticity | Yes |
| Chirality | Yes |
| Degree | Yes |
| Ring membership | Yes |
| Valence | Yes |

---

# Edge Features

| Feature | Include |
|---|---|
| Bond type | Yes |
| Stereo | Yes |
| Conjugation | Yes |
| Ring bond | Yes |

---

# Recommended GATv2 Configuration

| Parameter | Value |
|---|---|
| Layers | 5 |
| Hidden dim | 512 |
| Attention heads | 8 |
| Dropout | 0.15 |
| Attention dropout | 0.10 |
| Residual connections | Yes |
| LayerNorm | Yes |
| Virtual node | Yes |

---

# Best Activation Function

Use:

```text
GELU
```

Reason:

- smoother than ReLU
- better transformer/GNN compatibility
- stable gradients
- widely used in modern architectures

Apply GELU after:
- GATv2 outputs
- MLP blocks
- projection layers

---

# Recommended Optimizer

Use:

```text
AdamW
```

for ALL neural modules.

Reason:
- better weight decay handling
- standard for transformers/GNNs
- stable convergence

---

# Molecular Encoder Optimizer

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-2 |
| Scheduler | Cosine decay |
| Warmup | 5% steps |

---

# Why Pretraining Is Needed

Your formulation dataset is small.

Without pretraining:
- model overfits
- poor generalization

Pretraining teaches chemistry first.

---

# Best Pretraining Strategy

Use:

```text
Self-supervised molecular pretraining
```

on:
- ChEMBL
- PubChem
- ZINC

---

# Pretraining Tasks

## 1. Masked Atom Prediction

Like BERT:

```text
C [MASK] O
```

predict missing atom.

---

## 2. Contrastive Learning

Two augmented versions of same molecule should have similar embeddings.

Augmentations:
- atom masking
- bond deletion
- subgraph masking

---

# Best Existing Pretrained Models

Recommended:

| Model | Recommendation |
|---|---|
| GraphMVP | Excellent |
| GROVER | Excellent |
| MolCLR | Excellent |
| Hu et al. pretrained GNN | Strong |
| ChemBERTa | Transformer alternative |

---

# Recommended Choice

Use:

```text
GraphMVP pretrained GNN
```

OR

```text
MolCLR pretrained GAT/GIN
```

---

# How To Get Pretrained Models

## GraphMVP

GitHub:
https://github.com/chao1224/GraphMVP

Contains:
- pretrained molecular GNN checkpoints
- chemistry pretraining pipeline

---

## MolCLR

GitHub:
https://github.com/yuyangw/MolCLR

---

# Fine-Tuning Strategy

Load pretrained weights ONLY for:

```text
Molecular GNN encoder
```

Everything else:
- initialized randomly
- trained on Excipic dataset

---

# 2. Dosage Form Encoder

# Input

- dosage form category
- RDKit descriptor vector

---

# Recommended Embedding Strategy

## Learned Embedding Table

NOT one-hot encoding.

Example:

```text
TABLET → embedding vector
CAPSULE → embedding vector
```

---

# Descriptor Features

Use:

- MolWt
- LogP
- TPSA
- HBD
- HBA
- Rotatable bonds
- Ring counts
- etc.

---

# MLP Projection

Project descriptors to:

```text
512 dimensions
```

---

# Dosage Encoder Configuration

| Component | Value |
|---|---|
| Embedding dim | 256 |
| Projection dim | 512 |
| MLP layers | 2 |
| Activation | GELU |
| Dropout | 0.20 |

---

# Optimizer

Use same:

```text
AdamW
```

Learning rate:

```text
3e-4
```

because this block trains from scratch.

---

# 3. Cross-Attention Fusion Transformer

# Purpose

Learn interactions between:
- chemistry
- dosage requirements

---

# Why Cross-Attention?

Instead of simple concatenation:

```text
[molecule ; dosage]
```

cross-attention allows:

```text
Molecule attends to dosage form
Dosage attends to molecule
```

Example:

```text
Hydrophobic API + tablet
→ surfactants become important
```

---

# Recommended Transformer Configuration

| Parameter | Value |
|---|---|
| Layers | 4 |
| Attention heads | 8 |
| Hidden dim | 512 |
| FFN dim | 2048 |
| Dropout | 0.15 |
| Attention dropout | 0.10 |
| Activation | GELU |
| Norm | Pre-LayerNorm |

---

# Final Fusion Vector

Concatenate:

```text
[mol_CLS ; dosage_emb ; elementwise_product]
```

Final size:

```text
1536 dimensions
```

---

# Optimizer

Use:

```text
AdamW
```

| Parameter | Value |
|---|---|
| LR | 1e-4 |
| Weight decay | 0.01 |

---

# 4. Physicochemical Property Head

# Purpose

Auxiliary chemistry supervision.

Predict:
- solubility
- LogP
- HLB
- melting point
- charge
- permeability

This improves:
- generalization
- chemistry understanding
- unseen API performance

---

# Architecture

Simple MLP:

```text
1536
→ 768
→ 256
→ property outputs
```

---

# Activation

Use:

```text
GELU
```

---

# Dropout

| Layer | Dropout |
|---|---|
| Hidden layers | 0.20 |

---

# Loss

Use:

```text
MSELoss
```

for continuous properties.

---

# 5. Excipient Knowledge Graph

# Purpose

Learn excipient relationships.

Example:

```text
MCC ↔ Mg Stearate
PVP ↔ Crospovidone
```

---

# Excipient Embeddings

Each excipient gets:
- molecular embedding
- role embedding
- co-occurrence embedding

---

# Pre-Encoding

Use SAME pretrained molecular GNN backbone.

Freeze after warmup.

---

# Recommended Excipient Encoder

| Parameter | Value |
|---|---|
| Backbone | Shared GATv2 |
| Shared weights | Yes |
| Frozen after Phase 1 | Yes |
| Embedding dim | 512 |

---

# Co-Occurrence Graph

Build graph from:
- formulation co-occurrence counts

Edge weights:
- PMI
- normalized frequency
- Jaccard co-occurrence

---

# Graph Network

Use:

```text
GCNII
```

Reason:
- deeper graph propagation
- stable training
- avoids oversmoothing

---

# GCN Configuration

| Parameter | Value |
|---|---|
| Layers | 3 |
| Hidden dim | 512 |
| Dropout | 0.10 |
| Activation | GELU |

---

# Optimizer

```text
AdamW
```

Learning rate:

```text
5e-5
```

because graph priors should update slowly.

---

# 6. Set Decoder (MOST IMPORTANT)

# Why Pointer Network?

Excipients are:
- variable-length
- ordered recommendations
- dependent labels

Pointer networks naturally model:
- ranking
- dependencies
- set generation

---

# Decoder Configuration

Use:

```text
Transformer decoder + pointer attention
```

---

# Recommended Parameters

| Parameter | Value |
|---|---|
| Decoder layers | 4 |
| Hidden dim | 512 |
| Heads | 8 |
| Dropout | 0.15 |
| Activation | GELU |

---

# Stopping Criterion

Predict:
- STOP token
OR
- max length = 15

---

# Sampling Strategy

During inference:

Use:
- beam search = 5
- temperature = 0.7

---

# Optimizer

```text
AdamW
```

| Parameter | Value |
|---|---|
| LR | 1e-4 |
| Weight decay | 0.01 |

---

# 7. Loss Functions

# Primary Loss

Use:

```text
LambdaRank / ListNet
```

Reason:
- directly optimizes ranking quality
- aligns with NDCG

---

# Secondary Loss

Use:

```text
Asymmetric BCE
```

Reason:
- severe class imbalance
- sparse positives

Recommended gamma:

| Parameter | Value |
|---|---|
| gamma_neg | 4 |
| gamma_pos | 1 |

---

# Auxiliary Loss

Use:

```text
MSELoss
```

for property head.

---

# Final Combined Loss

```text
Total Loss =
0.6 * Ranking Loss
+ 0.3 * Asymmetric BCE
+ 0.1 * Property MSE
```

---

# 8. Evaluation Metrics

# PRIMARY METRIC

Use:

```text
NDCG@10
```

This should drive:
- early stopping
- best checkpoint selection

---

# Required Metrics

Report:

| Metric | Purpose |
|---|---|
| Recall@K | Coverage |
| NDCG@K | Ranking quality |
| Hit@K | At least one correct |
| MRR | First-hit quality |
| Jaccard@K | Set overlap quality |
| Exact Match | Perfect formulation recovery |

---

# Jaccard Similarity

Use:

```text
Intersection / Union
```

for:
- @5
- @10
- @15

---

# Exact Match

Strict metric:

```text
Predicted set == True set
```

Useful but very hard.

---

# Recommended K Values

Report all metrics at:

```text
K = 5, 10, 15
```

---

# 9. Data Split Strategy

# NEVER Use Random Split

Random split causes:
- chemistry leakage
- overly optimistic metrics

---

# Correct Strategy

Use:

```text
Bemis-Murcko scaffold split
```

---

# Recommended Split

| Split | Percentage |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

# Temporal Holdout

Final test should additionally include:

```text
future FDA approvals
```

Use:
- marketing start date
OR
- earliest approval date

---

# 10. Three-Phase Training

# Phase 1 — Pretraining

Train ONLY molecular GNN.

Dataset:
- ChEMBL
- PubChem
- ZINC

Tasks:
- masked atom prediction
- contrastive learning

---

# Phase 2 — Fine-Tuning

Train full Excipic architecture.

Unfreeze:
- molecular encoder
- fusion transformer
- decoder

---

# Recommended Fine-Tuning LRs

| Module | LR |
|---|---|
| Pretrained GNN | 5e-5 |
| Transformer | 1e-4 |
| Decoder | 1e-4 |
| Property head | 3e-4 |

---

# Phase 3 — RLHF-Style Fine-Tuning (Optional)

Use:
- pharmacist rankings
- heuristic preference scores
- incompatibility penalties

Train:
- reward model
- PPO optimization

Optional for V1.

---

# 11. Recommended Training Infrastructure

# Hardware

Recommended:
- A100 40GB
- H100
- 4090 (smaller batch sizes)

---

# Mixed Precision

Use:

```text
bf16
```

OR

```text
fp16
```

---

# Batch Size

| GPU | Batch |
|---|---|
| 24GB | 16 |
| 40GB | 32 |
| 80GB | 64 |

---

# Gradient Clipping

Use:

```text
1.0
```

---

# Early Stopping

Monitor:

```text
Validation NDCG@10
```

Patience:

```text
10 epochs
```

---

# 12. Recommended Libraries

| Purpose | Library |
|---|---|
| Molecular processing | RDKit |
| GNNs | PyTorch Geometric |
| Transformers | HuggingFace / PyTorch |
| Ranking losses | pytorchltr |
| Metrics | torchmetrics |
| Graphs | DGL / PyG |

---

# 13. Final Recommended Stack

| Component | Best Choice |
|---|---|
| Molecular GNN | GATv2 |
| Pretraining | GraphMVP / MolCLR |
| Optimizer | AdamW |
| Activation | GELU |
| Main Metric | NDCG@10 |
| Graph propagation | GCNII |
| Decoder | Pointer Transformer |
| Split | Scaffold split |
| Final holdout | Temporal |

---

# 14. Recommended V1 Scope

For first implementation:

Build:

```text
Pretrained GATv2
+ dosage encoder
+ cross-attention fusion
+ excipient graph
+ pointer decoder
+ scaffold split
+ NDCG optimization
```

Skip initially:
- RLHF
- PPO
- reward models

---

# Final Goal

The model should learn:

```text
chemistry
+
dosage constraints
+
excipient relationships
+
ranking priorities
```

to produce:

```text
industry-grade excipient recommendations
```
