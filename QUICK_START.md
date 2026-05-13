# Quick Start Guide - Enhanced Training

## What's New?

Your training code now displays:

1. ✅ **GPU/CPU Detection** - Shows device type and specifications
2. ✅ **RTX 4050 Verification** - Confirms RTX 4050 is being used
3. ✅ **Epoch-wise Requirements** - All important metrics shown per epoch
4. ✅ **Memory Monitoring** - GPU memory usage tracked each batch
5. ✅ **Early Stopping Info** - Clear patience counter display

---

## Starting Training

### Basic command:

```bash
python train.py
```

### Custom settings:

```bash
python train.py --epochs 50 --batch-size 32
```

### Debug mode (fast testing):

```bash
python train.py --debug
```

### Resume from checkpoint:

```bash
python train.py --resume checkpoint_epoch_10.pt
```

---

## What You'll See

### ① At Startup (Device & Config Info)

```
================================================================================
  EXCIPIC — Pharmaceutical Excipient Recommendation System
================================================================================

🖥️  DEVICE INFORMATION:
   Device Type: GPU
   Device Name: NVIDIA RTX 4050
   GPU Memory: 12.00 GB
   Compute Capability: 8.9
   Mixed Precision: float16
   ✓ RTX 4050 Detected!

⚙️  TRAINING CONFIGURATION:
   Batch size: 16
   Epochs: 100
   Patience (Early Stop): 10 epochs
   Primary Metric: ndcg_at_10
   Loss Weights: Ranking=0.6, BCE=0.3, Property=0.1
   Learning Rates: Mol=1e-04, Dosage=3e-04, Fusion=1e-04, Decoder=1e-04, PropHead=3e-04, Graph=5e-05
================================================================================
```

### ② During Training (Each Batch)

```
  [50/200] Step 5000 | Loss: 0.4521 (R:0.3421 B:0.0892 P:0.0208) | LR: 1.50e-04 | Time: 0.32s/batch | GPU Mem: 2048MB/2560MB (Peak: 3100MB)
```

### ③ At End of Each Epoch

```
📈 Train Summary | Loss: 0.4521 (Ranking:0.3421 BCE:0.0892 Property:0.0208) | Epoch Time: 124.5s

✓ Val Summary | NDCG@10: 0.6234 | Recall@10: 0.7891 | Hit@10: 0.8234 | MRR: 0.5234 | Jaccard@10: 0.4521 | Val Time: 45.3s

⭐ NEW BEST! ndcg_at_10: 0.6234 (+0.0234) | Checkpoint saved

⏱️  Estimated time remaining: 45.2 minutes (8 epochs × 124.5s avg)
```

### ④ Early Stopping (No Improvement)

```
⏱️  No improvement | Patience: 3/10 (7 epochs remaining before early stop)
```

### ⑤ When Early Stopping Triggers

```
🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑
🛑 EARLY STOPPING TRIGGERED at Epoch 25/100
   Best ndcg_at_10: 0.7234
   Patience exhausted: 10 epochs with no improvement
🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑🛑
```

### ⑥ Final Summary

```
════════════════════════════════════════════════════════════════════════════════
✅ TRAINING COMPLETE
   Total Time: 127.3 minutes
   Epochs Trained: 75/100
   Best ndcg_at_10: 0.7456
════════════════════════════════════════════════════════════════════════════════
```

---

## Key Metrics Displayed

| Metric                 | Location           | Meaning                             |
| ---------------------- | ------------------ | ----------------------------------- |
| **Device Type**        | Startup            | GPU or CPU                          |
| **GPU Memory**         | Startup            | Total VRAM available                |
| **RTX 4050 Detection** | Startup            | ✓ if RTX 4050, ⚠ otherwise          |
| **Batch Time**         | Each log interval  | seconds per batch (lower is better) |
| **GPU Mem**            | Each log interval  | Allocated/Reserved (Peak)           |
| **Loss (Total)**       | Each batch & epoch | Combined loss value                 |
| **Loss (Ranking)**     | Each batch & epoch | Ranking loss only (60%)             |
| **Loss (BCE)**         | Each batch & epoch | Multi-hot loss only (30%)           |
| **Loss (Property)**    | Each batch & epoch | Property prediction loss (10%)      |
| **NDCG@10**            | Validation         | Ranking quality (primary metric)    |
| **Recall@10**          | Validation         | Coverage of true excipients         |
| **Hit@10**             | Validation         | % of correct predictions            |
| **MRR**                | Validation         | Mean Reciprocal Rank                |
| **Jaccard@10**         | Validation         | Set similarity metric               |
| **Patience**           | Validation         | Epochs until early stop             |

---

## Understanding the Output

### Loss Components

- **Ranking Loss (R)**: How well the model ranks excipients
- **BCE Loss (B)**: Multi-hot classification accuracy
- **Property Loss (P)**: API property prediction accuracy

### GPU Memory

- **Allocated**: Actually used by model/data
- **Reserved**: Pre-allocated pool (higher to prevent fragmentation)
- **Peak**: Maximum ever used (helps identify memory issues)

### Early Stopping

- **Patience Counter**: How many epochs without improvement
- **Threshold**: Training stops when counter reaches 10 epochs
- Shows remaining epochs before automatic stop

---

## Common Issues

### ⚠️ "CPU Mode - Training will be SLOW"

- GPU not detected
- Check NVIDIA drivers: `nvidia-smi`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### ⚠️ "Not RTX 4050 (configured for RTX 4050)"

- Different GPU detected
- Adjust detection in `_get_device_info()` if needed
- Training will still work, just with different GPU

### 💾 GPU Memory Growing in Peak

- Possible memory leak
- Check batch size isn't too large
- Reduce batch size if needed: `python train.py --batch-size 8`

### ⏱️ Estimated time is way off

- Happens if first few epochs are slower
- Estimation improves after few epochs
- Based on average of completed epochs

---

## Configuration Adjustments

Edit `excipic/config.py` to modify:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 16                    # Change batch size
    num_epochs: int = 100                   # Change max epochs
    patience: int = 10                      # Change early stop patience
    primary_metric: str = "ndcg_at_10"      # Change primary metric
    ranking_loss_weight: float = 0.6        # Change loss weights
    bce_loss_weight: float = 0.3
    property_loss_weight: float = 0.1
    lr_molecular_encoder: float = 1e-4      # Change learning rates
    lr_dosage_encoder: float = 3e-4
    # ... other module LRs ...
```

---

## Tips for Best Results

1. **Watch GPU Memory**: Check if Peak memory keeps growing (memory leak)
2. **Monitor Batch Time**: If increasing, may indicate training instability
3. **Check Patience Counter**: Training ends automatically after N epochs with no improvement
4. **Verify RTX 4050**: Ensure ✓ is shown at startup
5. **Save Best Model**: Checkpoint automatically saved when NDCG@10 improves
6. **Resume Training**: Use `--resume best_model.pt` to continue from best point

---

## Files Modified

- ✅ `excipic/train.py` - Added GPU detection and startup info display
- ✅ `excipic/training/trainer.py` - Added memory monitoring and epoch-wise logging
- 📄 `TRAINING_ENHANCEMENTS.md` - Detailed documentation (this file)

---

## Support

For detailed information, see `TRAINING_ENHANCEMENTS.md`
